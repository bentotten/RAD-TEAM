import numpy as np
import torch
from torch.optim import Adam
import gym # type: ignore
import time
import os
import core # type: ignore
import ppo_tools # type: ignore

from RADTEAM_core import StatisticStandardization, CNNBase

from gym.utils.seeding import _int_list_from_bigint, hash_seed # type: ignore
from rl_tools.logx import EpochLogger # type: ignore
from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params,synchronize, mpi_avg_grads, sync_params_stats # type: ignore
from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,mpi_statistics_vector, num_procs, mpi_min_max_scalar # type: ignore

BATCHED_UPDATE = False

def ppo(env_fn, actor_critic=CNNBase, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, alpha=0, clip_ratio=0.2, pi_lr=3e-4, mp_mm=[5,5],
        vf_lr=3e-4, train_pi_iters=40, train_v_iters=40, lam=0.9, max_ep_len=120, save_gif=False,
        target_kl=0.07, logger_kwargs=dict(), save_freq=500, render= False,dims=None, load_model=0):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set Pytorch random seed
    torch.manual_seed(seed)

    # Instantiate environment
    env = env_fn()
    #ac_kwargs['seed'] = seed
    # ac_kwargs['pad_dim'] = 2
    ac_kwargs["id"] = 0
    ac_kwargs["action_space"] = env.detectable_directions  # Usually 8
    ac_kwargs["observation_space"] = env.observation_space.shape[0]  # Also known as state dimensions: The dimensions of the observation returned from the environment. Usually 11
    ac_kwargs["detector_step_size"] = env.step_size  # Usually 100 cm
    ac_kwargs["environment_scale"] = env.scale
    ac_kwargs["bounds_offset"] = env.observation_area
    ac_kwargs["grid_bounds"] = env.scaled_grid_max    
    ac_kwargs["steps_per_episode"] = 120
    ac_kwargs["number_of_agents"] = 1
    ac_kwargs["enforce_boundaries"] = env.enforce_grid_boundaries

    obs_dim = env.observation_space.shape[0]

    #Instantiate A2C
    ac = actor_critic(**ac_kwargs)
    
    if load_model != 0:
        ac.load_state_dict(torch.load('model.pt'))           
    
    # Sync params across processes
    sync_params(ac.pi)
    sync_params(ac.critic)
    #sync_params(ac.model)

    #PFGRU args, from Ma et al. 2020
    bp_args = {
        'bp_decay' : 0.1,
        'l2_weight':1.0, 
        'l1_weight':0.0,
        'elbo_weight':1.0,
        'area_scale':env.search_area[2][1]}

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.critic])
    logger.log('\nNumber of parameters: \t pi: %d, critic: %d \t'%var_counts)

    # Set up trajectory buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    #buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, ac_kwargs['hidden_sizes_rec'][0])
    buf = ppo_tools.PPOBuffer(observation_dimension=obs_dim, max_size=local_steps_per_epoch, max_episode_length=120, number_agents=1)
    
    save_gif_freq = epochs // 3
    if proc_id() == 0:
        print(f'Local steps per epoch: {local_steps_per_epoch}')

    def sample(data, minibatch=1):
        """Get sample indexes of episodes to train on"""

        # Randomize and sample observation batch indexes
        ep_length = data["ep_len"].item()
        indexes = np.arange(0, ep_length, dtype=np.int32)
        number_of_samples = int((ep_length / minibatch))
        return np.random.choice(
            indexes, size=number_of_samples, replace=False
        )  # Uniform
 
    def compute_loss_pi(data, pi_maps, step):
        act = data['act']
        adv = data['adv']
        logp_old = data['logp']

        # Policy loss
        logp, dist_entropy = ac.step_keep_gradient_for_actor(actor_mapstack=pi_maps[step], action_taken=act[step])
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv # Alpha and entropy here?
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = dist_entropy.mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data, v_maps, step):
        ret = torch.unsqueeze(data['ret'][step], 0)        
        value = ac.step_keep_gradient_for_critic(critic_mapstack=v_maps[step])
        loss = optimization.MSELoss(value, ret)
        
        return loss

    def compute_batched_losses_pi(agent, sample, data, mapstacks_buffer, minibatch=None):
        """Simulates batched processing through CNN. Wrapper for computing single-batch loss for pi"""

        # TODO make more concise
        # Due to linear layer in CNN, this must be run individually
        pi_loss_list = []
        kl_list = []
        entropy_list = []
        clip_fraction_list = []

        # Get sampled returns from actor and critic
        for index in sample:
            # Reset existing episode maps
            agent.reset()
            single_pi_l, single_pi_info = compute_loss_pi(
                data=data, index=index, map_stack=mapstacks_buffer[index]
            )

            pi_loss_list.append(single_pi_l)
            kl_list.append(single_pi_info["kl"])
            entropy_list.append(single_pi_info["entropy"])
            clip_fraction_list.append(single_pi_info["clip_fraction"])

        # pi_loss, kl, entropy, clip_fraction - removed dict to improve speed
        return torch.stack(pi_loss_list).mean(), np.mean(kl_list), np.mean(entropy_list),  np.mean(clip_fraction_list)

    def compute_batched_losses_critic(agent, data, map_buffer_maps, sample):
        """Simulates batched processing through CNN. Wrapper for single-batch computing critic loss"""
        critic_loss_list = []

        # Get sampled returns from actor and critic
        for index in sample:
            # Reset existing episode maps
            agent.reset()
            critic_loss_list.append(compute_loss_v(data=data, map_stack=map_buffer_maps[index], index=index))

        # take mean of everything for batch update
        return torch.stack(critic_loss_list).mean()
    
    optimization = ppo_tools.OptimizationStorage(
        pi_optimizer=Adam(ac.pi.parameters(), lr=pi_lr),
        critic_optimizer= Adam(ac.critic.parameters(), lr=vf_lr), 
        #model_optimizer=Adam(ac.model.parameters(), lr=vf_lr), 
        MSELoss=torch.nn.MSELoss(reduction="mean"),
    )    

    def update(env, args, loss_fcn=optimization.MSELoss):
        """Update for the localization and A2C modules"""
        #data = buf.get(logger=logger)
        ac.set_mode("train")
        data, pi_maps, v_maps = buf.get() # TODO use arrays not dicts for faster processing

        #Update function if using the PFGRU, fcn. performs multiple updates per call
        #ac.model.train()
        #loss_mod = update_model(data, args, loss=loss_fcn)

        #Update function if using the regression GRU
        #loss_mod = update_loc_rnn(data,env,loss)

        sample_indexes = sample(data=data)

        for kk in range(train_pi_iters):
            if BATCHED_UPDATE:
                optimization.pi_optimizer.zero_grad()           
                                     
                loss_pi, kl, entropy, clip_fraction = compute_batched_losses_pi(agent=ac, data=data, sample=sample_indexes, mapstacks_buffer=pi_maps)
                pi_info = dict(kl = kl, ent = entropy, cf = clip_fraction)

                if kl < 1.5 * target_kl:
                    logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    loss_pi.backward()
                    optimization.pi_optimizer.step()
                else:
                    kk = train_pi_iters # Avoid messy for-loop breaking
                
            elif not BATCHED_UPDATE:
                for step in sample_indexes:
                    ac.reset()
                    optimization.pi_optimizer.zero_grad()
                    
                    loss_pi, pi_info = compute_loss_pi(data, pi_maps, step)
 
                    if kl < 1.5 * target_kl:
                        logger.log('Early stopping at step %d due to reaching max kl.'%i)
                        loss_pi.backward()
                        optimization.pi_optimizer.step()
                    else:
                        kk = train_pi_iters # Avoid messy for-loop breaking 

        # Update value function 
        for i in range(train_v_iters):
            if BATCHED_UPDATE:
                optimization.critic_optimizer.zero_grad()
                critic_loss = compute_batched_losses_critic(agent=ac, data=data, sample=sample_indexes, map_buffer_maps=v_maps)
                
                critic_loss.backward()
                optimization.critic_optimizer.step()
                
            else:
                for step in sample_indexes:
                    ac.reset()

                    optimization.critic_optimizer.zero_grad()
                    loss_v = compute_loss_v(data, v_maps, step)
                    loss_v.backward()
                    mpi_avg_grads(ac.critic)    # average grads across MPI processes
                    optimization.critic_optimizer.step()            
        
        #Reduce learning rate
        #pi_scheduler.step()
        optimization.pi_scheduler.step()                
        optimization.critic_scheduler.step()        
        #model_scheduler.step()
        #optimization.model_scheduler.step()

        logger.store(StopIter=kk)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info['ent'], pi_info['cf']

        # Log changes from update
        kl, ent, cf = (
            pi_info["kl"],
            pi_info["ent"],
            pi_info["cf"],
        )

        logger.store(
            LossPi=loss_pi.item(),
            LossV=loss_v.item(),
            LossModel=0, # loss_mod
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            LocLoss=0,
            VarExplain=0,
        )
        
        ac.set_mode("val")
        ########################


    # Prepare for interaction with environment
    start_time = time.time()
    o, _, _, _ = env.reset()
    o = o[0]
    ep_ret, ep_len, done_count, a = 0, 0, 0, -1

    ep_ret_ls = []
    oob = 0

    ac.set_mode("eval")

    
    # Main loop: collect experience in env and update/log each epoch
    print(f'Proc id: {proc_id()} -> Starting main training loop!', flush=True)
    for epoch in range(epochs):
        #Reset hidden state
        #hidden = ac.reset_hidden()
        hidden = []
        for t in range(local_steps_per_epoch):
            #Standardize input using running statistics per episode
            obs_std = o
            
            #compute action and logp (Actor), compute value (Critic)
            result, heatmap_stack = ac.step({0: obs_std}, hidden=hidden)
            next_o, r, d, _ = env.step({0: result.action})
            next_o, r, d = next_o[0], r['individual_reward'][0], d[0]
            ep_ret += r
            ep_len += 1
            ep_ret_ls.append(ep_ret)

            logger.store(VVals=result.state_value)
            buf.store(
                obs=obs_std,
                act=result.action,
                val=result.state_value,
                logp=result.action_logprob,
                rew=r,
                src=env.src_coords,
                full_observation={0: obs_std},
                heatmap_stacks=heatmap_stack,
                terminal=d,
            )
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            
            if terminal or epoch_ended:
                if d and not timeout:
                    done_count += 1
                if env.get_agent_outOfBounds_count(id=0) > 0:
                    # Log if agent went out of bounds
                    oob += 1
                if epoch_ended and not(terminal):
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps and time {t}.', flush=True)

                if timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    
                    result, _ = ac.step({0: obs_std}, hidden=hidden)
                    v = result.state_value
                    
                    if epoch_ended:
                        #Set flag to sample new environment parameters
                        env.epoch_end = True
                else:
                    v = 0
                #buf.finish_path(v)
                buf.GAE_advantage_and_rewardsToGO(v)
                
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    buf.store_episode_length(episode_length=ep_len)

                if epoch_ended and render and (epoch % save_gif_freq == 0 or ((epoch + 1 ) == epochs)):
                    #Check agent progress during training
                    if proc_id() == 0 and epoch != 0:
                        env.render(save_gif=save_gif,path=logger.output_dir,epoch_count=epoch,
                                   ep_rew=ep_ret_ls)
                
                ep_ret_ls = []
                if not env.epoch_end:
                    #Reset detector position and episode tracking
                    #hidden = ac.reset_hidden()
                    o, _, _, _ = env.reset()
                    o = o[0]
                    ep_ret, ep_len, a = 0, 0, -1    
                else:
                    #Sample new environment parameters, log epoch results
                    oob += env.get_agent_outOfBounds_count(id=0)
                    logger.store(DoneCount=done_count, OutOfBound=oob)
                    done_count = 0; 
                    oob = 0
                    o, _, _, _ = env.reset()
                    o = o[0]
                    ep_ret, ep_len, a = 0, 0, -1

                # Clear maps for next episode
                ac.reset()

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            fpath = "pyt_save"
            fpath = os.path.join(logger.output_dir, fpath)
            os.makedirs(fpath, exist_ok=True)
            ac.save(checkpoint_path=fpath)            
        
            pass


        # Perform PPO update!
        update(env, bp_args)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossModel', average_only=True)
        logger.log_tabular('LocLoss', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('DoneCount', sum_only=True)
        logger.log_tabular('OutOfBound', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gym_rad_search:RadSearchMulti-v1')
    parser.add_argument('--hid_gru', type=int, default=[24],help='A2C GRU hidden state size')
    parser.add_argument('--hid_pol', type=int, default=[32],help='Actor linear layer size') 
    parser.add_argument('--hid_val', type=int, default=[32],help='Critic linear layer size') 
    parser.add_argument('--hid_rec', type=int, default=[24],help='PFGRU hidden state size')
    parser.add_argument('--l_pol', type=int, default=1,help='Number of layers for Actor MLP')
    parser.add_argument('--l_val', type=int, default=1,help='Number of layers for Critic MLP')
    parser.add_argument('--gamma', type=float, default=0.99,help='Reward attribution for advantage estimator')
    parser.add_argument('--seed', '-s', type=int, default=2,help='Random seed control')
    parser.add_argument('--cpu', type=int, default=1,help='Number of cores/environments to train the agent with')
    parser.add_argument('--steps_per_epoch', type=int, default=480,help='Number of timesteps per epoch per cpu. Default is equal to 4 episodes per cpu per epoch.')      
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs to train the agent')
    parser.add_argument('--exp_name', type=str,default='alpha01_tkl07_val01_lam09_npart40_lr3e-4_proc10_obs-1_iter40_blr5e-3_2_tanh',help='Name of experiment for saving')
    parser.add_argument('--dims', type=list, default=[[0.0,0.0],[1500.0,0.0],[1500.0,1500.0],[0.0,1500.0]],
                        help='Dimensions of radiation source search area in cm, decreased by area_obs param. to ensure visilibity graph setup is valid.')
    parser.add_argument('--area_obs', type=list, default=[100.0,100.0], help='Interval for each obstruction area in cm')
    parser.add_argument('--obstruct', type=int, default=0, 
                        help='Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions')
    parser.add_argument('--net_type',type=str, default='rnn', help='Choose between recurrent neural network A2C or MLP A2C, option: rnn, mlp') 
    parser.add_argument('--alpha',type=float,default=0.1, help='Entropy reward term scaling') 
    parser.add_argument('--load_model', type=int, default=0)
    
    args = parser.parse_args()

    #Change mini-batch size, only been tested with size of 1
    args.batch = 1

    #Save directory and experiment name
    args.env_name = 'stage_1'
    args.exp_name = (f'{args.exp_name}')

    init_dims = {
        'bbox':args.dims,
        'observation_area':args.area_obs, 
        'obstruction_count':args.obstruct,
        "number_agents": 1, 
        "enforce_grid_boundaries": True,
        "DEBUG": True,
        "TEST": 1
        }
    max_ep_step = 120
    if args.cpu > 1:
        #max cpus, steps in batch must be greater than the max eps steps times num. of cpu
        tot_epoch_steps = args.cpu * args.steps_per_epoch
        args.steps_per_epoch = tot_epoch_steps if tot_epoch_steps > args.steps_per_epoch else args.steps_per_epoch
        print(f'Sys cpus (avail, using): ({os.cpu_count()},{args.cpu}), Steps set to {args.steps_per_epoch}')
        mpi_fork(args.cpu)  # run parallel code with mpi
    
    #Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed((1+proc_id())*args.seed))[0]
    rng = np.random.default_rng(robust_seed)
    init_dims['np_random'] = rng

    #Setup logger for tracking training metrics
    from rl_tools.run_utils import setup_logger_kwargs # type: ignore
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,data_dir='../../models/train',env_name=args.env_name)
    
    ac_kwargs = dict(
        #hidden_sizes_pol=[args.hid_pol]*args.l_pol,
        #hidden_sizes_val=[args.hid_val]*args.l_val, 
        predictor_hidden_size=args.hid_rec[0], 
        #hidden=[args.hid_gru], 
        #net_type=args.net_type,
        #batch_s=args.batch
    )
    
    #Run ppo training function
    ppo(lambda : gym.make(args.env,**init_dims), actor_critic=CNNBase,
        ac_kwargs=ac_kwargs, gamma=args.gamma, alpha=args.alpha,
        seed=robust_seed, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,dims= init_dims,
        logger_kwargs=logger_kwargs,render=False, save_gif=False, load_model=args.load_model)
    