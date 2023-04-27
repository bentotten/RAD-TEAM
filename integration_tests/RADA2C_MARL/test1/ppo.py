import numpy as np
import torch
from torch.optim import Adam
import gym # type: ignore
import time
import os
import core # type: ignore
import ppo_tools # type: ignore

from RADTEAM_core import StatisticStandardization 

from gym.utils.seeding import _int_list_from_bigint, hash_seed # type: ignore
from rl_tools.logx import EpochLogger # type: ignore
from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params,synchronize, mpi_avg_grads, sync_params_stats # type: ignore
from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,mpi_statistics_vector, num_procs, mpi_min_max_scalar # type: ignore



def update(ac, env, args, buf, train_pi_iters, train_v_iters, optimization, logger, clip_ratio, target_kl, alpha):
    """Update for the localization and A2C modules"""
    
    def update_a2c(data, env_sim, minibatch=None,iter=None):
        observation_idx = 11
        action_idx = 14
        logp_old_idx = 13
        advantage_idx = 11
        return_idx = 12
        source_loc_idx = 15
        
        ep_form= data['ep_form']
        pi_info = dict(kl=[], ent=[], cf=[], val= np.array([]), val_loss = [])
        ep_select = np.random.choice(np.arange(0,len(ep_form)),size=int(minibatch),replace=False)
        ep_form = [ep_form[idx] for idx in ep_select]
        loss_sto = torch.zeros((len(ep_form),4),dtype=torch.float32)
        loss_arr_buff = torch.zeros((len(ep_form),1),dtype=torch.float32)
        loss_arr = torch.autograd.Variable(loss_arr_buff)

        for ii,ep in enumerate(ep_form):
            #For each set of episodes per process from an epoch, compute loss 
            trajectories = ep[0]
            hidden = ac.reset_hidden()
            obs, act, logp_old, adv, ret, src_tar = trajectories[:,:observation_idx], trajectories[:,action_idx],trajectories[:,logp_old_idx], \
                                                     trajectories[:,advantage_idx], trajectories[:,return_idx,None], trajectories[:,source_loc_idx:].clone()
            #Calculate new log prob.
            pi, val, logp, loc = ac.grad_step(obs, act, hidden=hidden)
            logp_diff = logp_old - logp 
            ratio = torch.exp(logp - logp_old)

            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
            clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)

            #Useful extra info
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).detach().mean().item()
            approx_kl = logp_diff.detach().mean().item()
            ent = pi.entropy().detach().mean().item()
            
            #val_loss = loss(val,ret)
            val_loss = optimization.MSELoss(val, ret)

            loss_arr[ii] = -(torch.min(ratio * adv, clip_adv).mean() - 0.01*val_loss + alpha * ent)
            loss_sto[ii,0] = approx_kl; loss_sto[ii,1] = ent; loss_sto[ii,2] = clipfrac; loss_sto[ii,3] = val_loss.detach()
            

        mean_loss = loss_arr.mean()
        means = loss_sto.mean(axis=0)
        loss_pi, approx_kl, ent, clipfrac, loss_val = mean_loss, means[0].detach(), means[1].detach(), means[2].detach(), means[3].detach()
        pi_info['kl'].append(approx_kl), pi_info['ent'].append(ent), pi_info['cf'].append(clipfrac), pi_info['val_loss'].append(loss_val)
        
        #Average KL across processes 
        kl = mpi_avg(pi_info['kl'][-1])
        if kl.item() < 1.5 * target_kl:
            #pi_optimizer.zero_grad() 
            optimization.pi_optimizer.zero_grad()
            
            loss_pi.backward()
            #Average gradients across processes
            mpi_avg_grads(ac.pi)
            
            #pi_optimizer.step()
            optimization.pi_optimizer.step()
            term = False
        else:
            term = True
            if proc_id() == 0:
                logger.log('Terminated at %d steps due to reaching max kl.'%iter)

        pi_info['kl'], pi_info['ent'], pi_info['cf'], pi_info['val_loss'] = pi_info['kl'][0].numpy(), pi_info['ent'][0].numpy(), pi_info['cf'][0].numpy(), pi_info['val_loss'][0].numpy()
        loss_sum_new = loss_pi
        return loss_sum_new, pi_info, term, (env_sim.search_area[2][1]*loc-(src_tar)).square().mean().sqrt()

    def update_model(data, args):
        #Update the PFGRU, see Ma et al. 2020 for more details
        ep_form= data['ep_form']
        model_loss_arr_buff = torch.zeros((len(ep_form),1),dtype=torch.float32)
        source_loc_idx = 15
        o_idx = 3

        for jj in range(train_v_iters):
            model_loss_arr_buff.zero_()
            model_loss_arr = torch.autograd.Variable(model_loss_arr_buff)
            for ii,ep in enumerate(ep_form):
                sl = len(ep[0])
                hidden = ac.reset_hidden()[0]
                src_tar =  ep[0][:,source_loc_idx:].clone()
                src_tar[:,:2] = src_tar[:,:2]/args['area_scale']
                obs_t = torch.as_tensor(ep[0][:,:o_idx], dtype=torch.float32)
                loc_pred = torch.empty_like(src_tar)
                particle_pred = torch.empty((sl,ac.model.num_particles,src_tar.shape[1]))
                
                bpdecay_params = np.exp(args['bp_decay'] * np.arange(sl))
                bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
                for zz,meas in enumerate(obs_t):
                    loc, hidden = ac.model(meas, hidden)
                    particle_pred[zz] = ac.model.hid_obs(hidden[0])
                    loc_pred[zz,:] = loc

                bpdecay_params = torch.FloatTensor(bpdecay_params)
                bpdecay_params = bpdecay_params.unsqueeze(-1)
                l2_pred_loss = torch.nn.functional.mse_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction='none') * bpdecay_params
                l1_pred_loss = torch.nn.functional.l1_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction='none') * bpdecay_params
                
                l2_loss = torch.sum(l2_pred_loss)
                l1_loss = 10*torch.mean(l1_pred_loss)

                pred_loss = args['l2_weight'] * l2_loss + args['l1_weight'] * l1_loss

                total_loss = pred_loss
                particle_pred = particle_pred.transpose(0, 1).contiguous()

                particle_gt = src_tar.repeat(ac.model.num_particles, 1, 1)
                l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
                l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

                # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
                # other more complicated distributions could be used to improve the performance
                y_prob_l2 = torch.exp(-l2_particle_loss).view(ac.model.num_particles, -1, sl, 2)
                l2_particle_loss = - y_prob_l2.mean(dim=0).log()

                y_prob_l1 = torch.exp(-l1_particle_loss).view(ac.model.num_particles, -1, sl, 2)
                l1_particle_loss = - y_prob_l1.mean(dim=0).log()

                xy_l2_particle_loss = torch.mean(l2_particle_loss)
                l2_particle_loss = xy_l2_particle_loss

                xy_l1_particle_loss = torch.mean(l1_particle_loss)
                l1_particle_loss = 10 * xy_l1_particle_loss

                belief_loss = args['l2_weight'] * l2_particle_loss + args['l1_weight'] * l1_particle_loss
                total_loss = total_loss + args['elbo_weight'] * belief_loss

                model_loss_arr[ii] = total_loss
            
            model_loss = model_loss_arr.mean()
            
            #model_optimizer.zero_grad()
            optimization.model_optimizer.zero_grad()
            
            model_loss.backward()

            #Average gradients across the processes
            mpi_avg_grads(ac.model)
            torch.nn.utils.clip_grad_norm_(ac.model.parameters(), 5)
            
            #model_optimizer.step() 
            optimization.model_optimizer.step()
            
        return model_loss
           
    #data = buf.get(logger=logger)
    data = buf.get()

    #Update function if using the PFGRU, fcn. performs multiple updates per call
    ac.model.train()
    loss_mod = update_model(data, args)

    #Update function if using the regression GRU
    #loss_mod = update_loc_rnn(data,env,loss)

    ac.model.eval()
    min_iters = len(data['ep_form'])
    kk = 0; term = False

    # Train policy with multiple steps of gradient descent (mini batch)
    while (not term and kk < train_pi_iters):
        #Early stop training if KL-div above certain threshold
        pi_l, pi_info, term, loc_loss = update_a2c(data, env, minibatch=min_iters,iter=kk)
        kk += 1
    
    #Reduce learning rate
    #pi_scheduler.step()
    optimization.pi_scheduler.step()                
    
    #model_scheduler.step()
    optimization.model_scheduler.step()

    # logger.store(StopIter=kk)

    # Log changes from update
    kl, ent, cf, loss_v = pi_info['kl'], pi_info['ent'], pi_info['cf'], pi_info['val_loss']

    # logger.store(LossPi=pi_l.item(), LossV=loss_v.item(), LossModel= loss_mod.item(),
    #                 KL=kl, Entropy=ent, ClipFrac=cf,
    #                 LocLoss=loc_loss, VarExplain=0)
    # Returns:
    # actor_loss[id], 
    # critic_loss[id],
    # model_loss[id],                
    # stop_iteration[id],
    # kl[id], 
    # entropy[id], 
    # location_loss[id]    
    
    return pi_l, loss_v, loss_mod, kk, kl, ent, loc_loss


def ppo(env_fn, actor_critic=core.RNNModelActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, alpha=0, clip_ratio=0.2, pi_lr=3e-4, mp_mm=[5,5],
        vf_lr=5e-3, train_pi_iters=40, train_v_iters=15, lam=0.9, max_ep_len=120, save_gif=False,
        target_kl=0.07, logger_kwargs=dict(), save_freq=500, render= False,dims=None, load_model=0, number_of_agents=1):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    
    # Set up general logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set up individual loggers
    agent_loggers = [EpochLogger(**logger_kwargs) for _ in range(number_of_agents)]
    
    #Set Pytorch random seed
    torch.manual_seed(seed)

    # Instantiate environment
    env = env_fn()
    ac_kwargs['seed'] = seed
    ac_kwargs['pad_dim'] = 2

    obs_dim = env.observation_space.shape[0]

    #Instantiate A2C
    #ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    agents = [actor_critic(env.observation_space, env.action_space, **ac_kwargs) for _ in range(number_of_agents)]
    
    if load_model != 0:
        for id in range(len(agents)):
            agents[id].load_state_dict(torch.load('model.pt'))           
    
    # Sync params across processes
    # sync_params(ac)

    #PFGRU args, from Ma et al. 2020
    bp_args = {
        'bp_decay' : 0.1,
        'l2_weight':1.0, 
        'l1_weight':0.0,
        'elbo_weight':1.0,
        'area_scale':env.search_area[2][1]}

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [agents[0].pi, agents[0].model])
    logger.log('\nNumber of parameters: \t pi: %d, model: %d \t'%var_counts)

    # Set up trajectory buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    #buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, ac_kwargs['hidden_sizes_rec'][0])
    buffer = [
        ppo_tools.PPOBuffer(observation_dimension=obs_dim, max_size=local_steps_per_epoch, max_episode_length=120, number_agents=1)
        for _ in range(len(agents))
    ]
    
    save_gif_freq = epochs // 3
    if proc_id() == 0:
        print(f'Local steps per epoch: {local_steps_per_epoch}')

    optimization = [
        ppo_tools.OptimizationStorage(
            pi_optimizer=Adam(agents[id].pi.parameters(), lr=pi_lr),
            #critic_optimizer= Adam(ac.critic.parameters(), lr=pi_lr),  # TODO change this to own learning rate
            model_optimizer=Adam(agents[id].model.parameters(), lr=vf_lr),  # TODO change this to correct name (for PFGRU)
            MSELoss=torch.nn.MSELoss(reduction="mean"),
            critic_flag=False,
        )   
    for id in range(number_of_agents)]

    # Set up model saving
    for id in range(len(agents)):
        agent_loggers[id].setup_pytorch_saver(agents[id])

    # Prepare for interaction with environment
    start_time = time.time()
    o, _, _, _ = env.reset()
    ep_ret, ep_len, done_count, a = 0, 0, 0, -1

    stat_buffers = list()
    for id in range(len(agents)):
        stat_buffers.append(StatisticStandardization())
        stat_buffers[id].update(o[id][0])
    
    ep_ret_ls = []
    reduce_v_iters = True
    for id in range(len(agents)):
        agents[id].model.eval()
        
    hidden = [None for _ in range(len(agents))]
    
    # Main loop: collect experience in env and update/log each epoch
    print(f'Proc id: {proc_id()} -> Starting main training loop!', flush=True)
    for epoch in range(epochs):
        #Reset hidden state
        for id in range(len(agents)):
            hidden[id] = agents[id].reset_hidden()
            agents[id].pi.logits_net.v_net.eval()
            
        for t in range(local_steps_per_epoch):
            # Artifact - overwrites o because python
            obs_std = o

            # obs_std[0] = stat_buff.standardize(o[0])
            actions = {id: None for id in range(len(agents))}
            values = []
            logprobs = []
            
            for id in range(len(agents)):
                obs_std[id][0] = stat_buffers[id].standardize(o[id][0])

            for id in range(len(agents)):
                obs_std[id][0] = stat_buffers[id].standardize(o[id][0])            
                #compute action and logp (Actor), compute value (Critic)
                actions[id], v, logp, hidden[id], _ = agents[id].step(obs_std, hidden=hidden[id], id=id, obs_count=number_of_agents)
                values.append(v)
                logprobs.append(logp)
                
            next_o, r, d, _ = env.step(actions)
            d = True if True in d.values() else False
            
            ep_ret += r['team_reward']
            ep_len += 1
            ep_ret_ls.append(ep_ret)

            #buf.store(obs_std, a, r, v, logp, env.src_coords)
            for id in range(len(agents)):
                buffer[id].store(
                    obs=obs_std[id], 
                    act=actions[id],
                    rew=r['team_reward'],
                    val=values[id], 
                    logp=logprobs[id], 
                    src=env.src_coords, 
                    full_observation=obs_std, 
                    heatmap_stacks=None, 
                    terminal=d
                    )
            
            logger.store(VVals=v[0]) # TODO only taking first agents v

            # Update obs (critical!)
            o = next_o

            #Update running mean and std
            for id in range(len(agents)):
                stat_buffers[id].update(o[id][0])            

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            
            if terminal or epoch_ended:
                if d and not timeout:
                    done_count += 1

                if epoch_ended and not(terminal):
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps and time {t}.', flush=True)

                if timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    #obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8)
                    values = []
                    for id in range(len(agents)):
                        obs_std[id][0] = stat_buffers[id].standardize(o[id][0])                    
                    
                        _, v, _, _, _ = agents[id].step(obs_std[id], hidden=hidden[id])
                        values.append(v)
                    if epoch_ended:
                        #Set flag to sample new environment parameters
                        env.epoch_end = True
                else:
                    values = [0 for _ in range(len(agents))]
                # buf.GAE_advantage_and_rewardsToGO(v)
                for id in range(len(agents)):
                    buffer[id].GAE_advantage_and_rewardsToGO(values[id])
                
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    # buf.store_episode_length(episode_length=ep_len)
                    for id in range(len(agents)):
                        buffer[id].store_episode_length(episode_length=ep_len)
                    
                if epoch_ended and render and (epoch % save_gif_freq == 0 or ((epoch + 1 ) == epochs)):
                    #Check agent progress during training
                    if proc_id() == 0 and epoch != 0:
                        env.render(save_gif=save_gif,path=logger.output_dir,epoch_count=epoch,
                                   ep_rew=ep_ret_ls)
                
                ep_ret_ls = []
                # stat_buff.reset()
                for id in range(len(agents)):
                    stat_buffers[id].reset()
                
                if not env.epoch_end:
                    #Reset detector position and episode tracking
                    # hidden = ac.reset_hidden()
                    for id in range(len(agents)):
                        hidden[id] = agents[id].reset_hidden()
                        
                    o, _, _, _ = env.reset()
                    ep_ret, ep_len, a = 0, 0, -1    
                else:
                    #Sample new environment parameters, log epoch results
                    logger.store(DoneCount=done_count)
                    done_count = 0
                    o, _, _, _ = env.reset()
                    ep_ret, ep_len, a = 0, 0, -1

                # stat_buff.update(o[0])
                for id in range(len(agents)):
                    stat_buffers[id].update(o[id][0])

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            for id in range(len(agents)):
                agent_loggers[id].save_state(None, None)

        #Reduce localization module training iterations after 100 epochs to speed up training
        if reduce_v_iters and epoch > 99:
            train_v_iters = 5
            reduce_v_iters = False

        # Perform PPO update!
        # update(agent, env, bp_args)
        stop_iteration = np.zeros((len(agents)))
        kl = np.zeros((len(agents)))
        entropy = np.zeros((len(agents)))
        clip_frac = np.zeros((len(agents)))
        actor_loss = np.zeros((len(agents)))                
        critic_loss = np.zeros((len(agents)))
        model_loss = np.zeros((len(agents)))
        location_loss = np.zeros((len(agents)))
        
        for id in range(len(agents)):
            (
                actor_loss[id], 
                critic_loss[id],
                model_loss[id],                
                stop_iteration[id],
                kl[id], 
                entropy[id], 
                location_loss[id]
            ) = update(
                ac=agents[id], 
                env=env,
                args=bp_args,
                buf=buffer[id],
                train_pi_iters=train_pi_iters,
                train_v_iters=train_v_iters,
                optimization=optimization[id],
                logger=logger,
                clip_ratio=clip_ratio,
                target_kl=target_kl,
                alpha=alpha
                )
        
        logger.store(StopIter=stop_iteration.mean().item())

        # Log changes from update
        kl, ent, cf, loss_v = kl.mean().item(), entropy.mean().item(), clip_frac.mean().item(), critic_loss.mean().item()
        loss_pi = actor_loss.mean().item()
        loss_mod = model_loss.mean().item()
        loc_loss = location_loss.mean().item()

        logger.store(
            LossPi=loss_pi,
            LossV=loss_v, 
            LossModel= loss_mod,
            KL=kl, 
            Entropy=ent, 
            ClipFrac=cf,
            LocLoss=loc_loss, 
            VarExplain=0
            )            
        

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
        # logger.log_tabular('OutOfBound', average_only=True)
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
    parser.add_argument('--agents', type=int, default=1)
    
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
        "number_agents": args.agents, 
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
    
    #Run ppo training function
    ppo(lambda : gym.make(args.env,**init_dims), actor_critic=core.RNNModelActorCritic,
        ac_kwargs=dict(hidden_sizes_pol=[args.hid_pol]*args.l_pol,hidden_sizes_val=[args.hid_val]*args.l_val,
        hidden_sizes_rec=args.hid_rec, hidden=[args.hid_gru], net_type=args.net_type,batch_s=args.batch), gamma=args.gamma, alpha=args.alpha,
        seed=robust_seed, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,dims= init_dims,
        logger_kwargs=logger_kwargs,render=False, save_gif=False, load_model=args.load_model, number_of_agents=args.agents)
    