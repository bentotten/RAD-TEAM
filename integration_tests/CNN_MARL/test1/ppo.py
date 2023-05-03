import numpy as np
import torch
from torch.optim import Adam
import gym  # type: ignore
import time
import os
import ppo_tools  # type: ignore
import argparse

from RADTEAM_core import CNNBase

from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
from rl_tools.logx import EpochLogger  # type: ignore
from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, mpi_avg_grads  # type: ignore
from rl_tools.mpi_tools import mpi_fork, proc_id, num_procs  # type: ignore

BATCHED_UPDATE = True


# def update(env, args, loss_fcn=optimization.MSELoss):
def update(
    ac,
    args,
    buf,
    train_pi_iters,
    train_v_iters,
    train_pfgru_iters,
    optimization,
    logger,
    clip_ratio,
    target_kl,
    alpha,
    number_agents,
    id,
):
    """Update for the localization and A2C modules"""

    def sample(data, minibatch=1):
        """Get sample indexes of episodes to train on"""

        # Randomize and sample observation batch indexes
        ep_length = data["ep_len"].item()
        indexes = np.arange(0, ep_length, dtype=np.int32)
        number_of_samples = int((ep_length / minibatch))
        return np.random.choice(indexes, size=number_of_samples, replace=False)  # Uniform

    def compute_loss_pi(
        agent,
        data,
        index,
        map_stack,
    ):
        """
        Compute loss for actor network. Loss is the difference between the probability of taking the action according
        to the current policy and the probability of taking the action according to the old policy, multiplied by the
        advantage of the action.

        Process:
            #. Calculate how much the policy has changed:  ratio = policy_new / policy_old
            #. Take log form of this:  ratio = [log(policy_new) - log(policy_old)].exp()
            #. Calculate Actor loss as the minimum of two functions:
                #. p1 = ratio * advantage
                #. p2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
                #. actor_loss = min(p1, p2)

        :param data: (array) data from PPO buffer. Contains:
            * obs: (tensor) Unused: batch of observations from the PPO buffer. Currently only used to ensure map buffer
                observations are correct.
            * act: (tensor) batch of actions taken.
            * adv: (tensor) batch of advantages cooresponding to actions. These are the difference between the expected
                reward for taking that action and the true reward (See: TD Error, GAE).
            * logp: (tensor) batch of action logprobabilities.
            * loc_pred: (tensor) batch of predicted location by PFGRU.
            * ep_len: (tensor[int]) single dimension int of length of episode.
            * ep_form: (List) # Basically a list of all episodes, that then contain a single-element list of a tensor
                representing the observation.

        :param index: (int) If doing a single observation at a time, index for data[]
        """
        # NOTE: Not using observation tensor, using internal map buffer
        _, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Get action probabilities and entropy for an state's mapstack and action, then put the action probabilities on the CPU (if on the GPU)
        action_logprobs, dist_entropy = agent.step_keep_gradient_for_actor(map_stack[index], act[index])
        action_logprobs = action_logprobs.cpu()

        # Get how much change is about to be made, then clip it if it exceeds our threshold (PPO-CLIP)
        # NOTE: Loss will be averaged in the wrapper function, not here, as this is for a single observation/mapstack
        ratio = torch.exp(action_logprobs - logp_old[index])
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv[index]  # Objective surrogate
        loss_pi = -(torch.min(ratio * adv[index], clip_adv))

        # Useful extra info
        approx_kl = (logp_old[index] - action_logprobs).item()
        ent = dist_entropy.item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).item()
        # pi_info = dict(kl=approx_kl, entropy=ent, clip_fraction=clipfrac)

        # return loss_pi, pi_info
        return loss_pi, approx_kl, ent, clipfrac

    def compute_loss_critic(
        agent,
        index,
        data,
        map_stack,
    ):
        """Compute loss for state-value approximator (critic network) using MSE. Calculates the MSE of the
        predicted state value from the critic and the true state value

        data (array): data from PPO buffer
            ret (tensor): batch of returns

        map_stack (tensor): Either a single observations worth of maps, or a batch of maps
        index (int): If doing a single observation at a time, index for data[]

        Adapted from https://github.com/nikhilbarhate99/PPO-PyTorch

        Calculate critic loss with MSE between returns and critic value
            critic_loss = (R - V(s))^2
        """
        true_return = data["ret"][index]

        # Compare predicted return with true return and use MSE to indicate loss
        predicted_value = agent.step_keep_gradient_for_critic(map_stack[index])
        critic_loss = optimization.MSELoss(torch.squeeze(predicted_value), true_return)
        return critic_loss

    def compute_batched_losses_pi(agent, sample, data, mapstacks_buffer):
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
            loss_pi, approx_kl, ent, clipfrac = compute_loss_pi(
                agent=agent, data=data, index=index, map_stack=mapstacks_buffer
            )

            pi_loss_list.append(loss_pi)
            kl_list.append(approx_kl)
            entropy_list.append(ent)
            clip_fraction_list.append(clipfrac)

        # pi_loss, kl, entropy, clip_fraction - removed dict to improve speed
        return (
            torch.stack(pi_loss_list).mean(),
            np.mean(kl_list),
            np.mean(entropy_list),
            np.mean(clip_fraction_list),
        )

    def compute_batched_losses_critic(agent, data, map_buffer_maps, sample):
        """Simulates batched processing through CNN. Wrapper for single-batch computing critic loss"""
        critic_loss_list = []

        # Get sampled returns from actor and critic
        for index in sample:
            # Reset existing episode maps
            agent.reset()
            critic_loss_list.append(compute_loss_critic(agent=agent, data=data, map_stack=map_buffer_maps, index=index))

        # take mean of everything for batch update
        return torch.stack(critic_loss_list).mean()

    def update_model(data, args):
        # Update the PFGRU, see Ma et al. 2020 for more details
        ep_form = data["ep_form"]
        model_loss_arr_buff = torch.zeros((len(ep_form), 1), dtype=torch.float32)
        # source_loc_idx = 15
        source_loc_idx = 4 + (11 * number_agents)

        observation_stop = 11 * number_agents

        for jj in range(train_pfgru_iters):
            model_loss_arr_buff.zero_()
            model_loss_arr = torch.autograd.Variable(model_loss_arr_buff)
            for ii, ep in enumerate(ep_form):
                sl = len(ep[0])
                hidden = ac.reset_hidden()[0]
                src_tar = ep[0][:, source_loc_idx:].clone()
                src_tar[:, :2] = src_tar[:, :2] / args["area_scale"]

                observations_slice = ep[0][:, 0:observation_stop]
                obs_for_pfgru = torch.zeros((len(observations_slice), number_agents * 3))
                for offset in range(number_agents):
                    slice_obs = observations_slice[:, (offset * 11):(offset * 11 + 3)]
                    obs_for_pfgru[:, offset * 3:offset * 3 + 3] = slice_obs

                obs_t = obs_for_pfgru

                loc_pred = torch.empty_like(src_tar)
                particle_pred = torch.empty((sl, ac.model.num_particles, src_tar.shape[1]))

                bpdecay_params = np.exp(args["bp_decay"] * np.arange(sl))
                bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
                for zz, meas in enumerate(obs_t):
                    loc, hidden = ac.model(meas, hidden)
                    particle_pred[zz] = ac.model.hid_obs(hidden[0])
                    loc_pred[zz, :] = loc

                bpdecay_params = torch.FloatTensor(bpdecay_params)
                bpdecay_params = bpdecay_params.unsqueeze(-1)
                l2_pred_loss = (
                    torch.nn.functional.mse_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction="none")
                    * bpdecay_params
                )
                l1_pred_loss = (
                    torch.nn.functional.l1_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction="none")
                    * bpdecay_params
                )

                l2_loss = torch.sum(l2_pred_loss)
                l1_loss = 10 * torch.mean(l1_pred_loss)

                pred_loss = args["l2_weight"] * l2_loss + args["l1_weight"] * l1_loss

                total_loss = pred_loss
                particle_pred = particle_pred.transpose(0, 1).contiguous()

                particle_gt = src_tar.repeat(ac.model.num_particles, 1, 1)
                l2_particle_loss = (
                    torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction="none") * bpdecay_params
                )
                l1_particle_loss = (
                    torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction="none") * bpdecay_params
                )

                # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
                # other more complicated distributions could be used to improve the performance
                y_prob_l2 = torch.exp(-l2_particle_loss).view(ac.model.num_particles, -1, sl, 2)
                l2_particle_loss = -y_prob_l2.mean(dim=0).log()

                y_prob_l1 = torch.exp(-l1_particle_loss).view(ac.model.num_particles, -1, sl, 2)
                l1_particle_loss = -y_prob_l1.mean(dim=0).log()

                xy_l2_particle_loss = torch.mean(l2_particle_loss)
                l2_particle_loss = xy_l2_particle_loss

                xy_l1_particle_loss = torch.mean(l1_particle_loss)
                l1_particle_loss = 10 * xy_l1_particle_loss

                belief_loss = args["l2_weight"] * l2_particle_loss + args["l1_weight"] * l1_particle_loss
                total_loss = total_loss + args["elbo_weight"] * belief_loss

                model_loss_arr[ii] = total_loss

            model_loss = model_loss_arr.mean()

            # model_optimizer.zero_grad()
            optimization.model_optimizer.zero_grad()

            model_loss.backward()

            # Average gradients across the processes
            mpi_avg_grads(ac.model)
            torch.nn.utils.clip_grad_norm_(ac.model.parameters(), 5)

            # model_optimizer.step()
            optimization.model_optimizer.step()

        return model_loss

    ###################################################################################
    # data = buf.get(logger=logger)
    ac.set_mode("train")
    data, pi_maps, v_maps = buf.get()  # TODO use arrays not dicts for faster processing

    # Update function if using the PFGRU, fcn. performs multiple updates per call
    # ac.model.train()
    # loss_mod = update_model(data, args, loss=loss_fcn)

    # Update function if using the regression GRU
    # loss_mod = update_loc_rnn(data,env,loss)

    sample_indexes = sample(data=data)
    kk = 0
    kl_bound_flag = False

    # Update Actor
    while kk < train_pi_iters and not kl_bound_flag:
        if BATCHED_UPDATE:
            optimization.pi_optimizer.zero_grad()

            loss_pi, kl, entropy, clip_fraction = compute_batched_losses_pi(
                agent=ac, data=data, sample=sample_indexes, mapstacks_buffer=pi_maps
            )

            if kl < 1.5 * target_kl:
                loss_pi.backward()
                optimization.pi_optimizer.step()
            else:
                logger.log("Early stopping at update iteration %d due to reaching max kl." % kk)
                kl_bound_flag = True
                break

        elif not BATCHED_UPDATE:
            for step in sample_indexes:
                ac.reset()
                optimization.pi_optimizer.zero_grad()

                loss_pi, kl, entropy, clip_fraction = compute_loss_pi(
                    agent=ac, data=data, map_stack=pi_maps, index=step
                )

                loss_pi.backward()
                optimization.pi_optimizer.step()
            if kl < 1.5 * target_kl:
                logger.log("Early stopping at update iteration %d due to reaching max kl." % kk)
                kl_bound_flag = True
                break
        else:
            raise ValueError("Batched update problem")
        kk += 1

    pi_info = dict(kl=kl, ent=entropy, cf=clip_fraction)  # Just for last step

    # Update Critic
    for i in range(train_v_iters):
        if BATCHED_UPDATE:
            optimization.critic_optimizer.zero_grad()
            loss_v = compute_batched_losses_critic(agent=ac, data=data, sample=sample_indexes, map_buffer_maps=v_maps)

            loss_v.backward()
            optimization.critic_optimizer.step()

        elif not BATCHED_UPDATE:
            for step in sample_indexes:
                ac.reset()

                optimization.critic_optimizer.zero_grad()
                loss_v = compute_loss_critic(agent=ac, index=step, data=data, map_stack=v_maps)
                loss_v.backward()
                mpi_avg_grads(ac.critic)  # average grads across MPI processes
                optimization.critic_optimizer.step()
        else:
            raise ValueError("Batched update problem")

    # Update predictor model (iterations happen within function)
    loss_mod = update_model(data, args)

    # Reduce learning rate
    # pi_scheduler.step()
    optimization.pi_scheduler.step()
    optimization.critic_scheduler.step()
    # model_scheduler.step()
    optimization.model_scheduler.step()

    logger.store(StopIter=kk)

    # Log changes from update
    kl, ent, cf = pi_info["kl"], pi_info["ent"], pi_info["cf"]

    # Log changes from update
    kl, ent, cf = (
        pi_info["kl"],
        pi_info["ent"],
        pi_info["cf"],
    )

    ac.set_mode("eval")

    return loss_pi, loss_v, loss_mod, kk, kl, ent, cf


def ppo(
    env_fn,
    actor_critic=CNNBase,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    alpha=0,
    clip_ratio=0.2,
    pi_lr=3e-4,
    mp_mm=[5, 5],
    vf_lr=3e-4,
    pfgru_lr=5e-3,
    train_pi_iters=40,
    train_v_iters=40,
    train_pfgru_iters=15,
    reduce_pfgru_iters=True,
    lam=0.9,
    max_ep_len=120,
    save_gif=False,
    target_kl=0.07,
    logger_kwargs=dict(),
    save_freq=500,
    render=False,
    dims=None,
    load_model=0,
    number_of_agents=1,
):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set Pytorch random seed
    torch.manual_seed(seed)

    # Set up individual loggers
    # agent_loggers = [EpochLogger(**logger_kwargs) for _ in range(number_of_agents)]

    # Instantiate environment
    env = env_fn()

    ac_kwargs["id"] = 0
    ac_kwargs["action_space"] = env.detectable_directions  # Usually 8
    ac_kwargs["observation_space"] = env.observation_space.shape[
        0
    ]  # Also known as state dimensions: The dimensions of the observation returned from the environment. Usually 11
    ac_kwargs["detector_step_size"] = env.step_size  # Usually 100 cm
    ac_kwargs["environment_scale"] = env.scale
    ac_kwargs["bounds_offset"] = env.observation_area
    ac_kwargs["grid_bounds"] = env.scaled_grid_max
    ac_kwargs["steps_per_episode"] = 120
    ac_kwargs["number_of_agents"] = number_of_agents
    ac_kwargs["enforce_boundaries"] = env.enforce_grid_boundaries

    obs_dim = env.observation_space.shape[0]

    # Instantiate A2C
    # ac = actor_critic(**ac_kwargs)
    observation_size = obs_dim * number_of_agents
    agents = [actor_critic(**ac_kwargs) for _ in range(number_of_agents)]

    for ac in agents:
        logger.save_config(ac.get_config(), text="_agent", quiet=True)

    # TODO Make work with RAD-TEAM
    # if load_model != 0:
    #     for id in range(len(agents)):
    #         agents[id].load_state_dict(torch.load('model.pt'))

    # Sync params across processes
    # sync_params(ac.pi)
    # sync_params(ac.critic)
    # sync_params(ac.model)

    # PFGRU args, from Ma et al. 2020
    bp_args = {
        "bp_decay": 0.1,
        "l2_weight": 1.0,
        "l1_weight": 0.0,
        "elbo_weight": 1.0,
        "area_scale": env.search_area[2][1],
    }

    # Count variables
    var_counts = tuple(ppo_tools.count_vars(module) for module in [ac.pi, ac.critic, ac.model])
    logger.log("\nNumber of parameters: \t pi: %d, critic: %d predictor: %d\t" % var_counts)

    # Set up trajectory buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    # buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, ac_kwargs['hidden_sizes_rec'][0])
    # buf = ppo_tools.PPOBuffer(observation_dimension=obs_dim, max_size=local_steps_per_epoch, max_episode_length=120, number_agents=1)
    buffer = [
        ppo_tools.PPOBuffer(
            observation_dimension=obs_dim,
            max_size=local_steps_per_epoch,
            max_episode_length=120,
            number_agents=number_of_agents,
        )
        for _ in range(len(agents))
    ]

    save_gif_freq = epochs // 3
    if proc_id() == 0:
        print(f"Local steps per epoch: {local_steps_per_epoch}")

    # optimization = ppo_tools.OptimizationStorage(
    #     pi_optimizer=Adam(ac.pi.parameters(), lr=pi_lr),
    #     critic_optimizer= Adam(ac.critic.parameters(), lr=vf_lr),
    #     model_optimizer=Adam(ac.model.parameters(), lr=vf_lr),
    #     MSELoss=torch.nn.MSELoss(reduction="mean"),
    # )

    optimization = [
        ppo_tools.OptimizationStorage(
            pi_optimizer=Adam(agents[id].pi.parameters(), lr=pi_lr),
            critic_optimizer=Adam(agents[id].critic.parameters(), lr=vf_lr),
            model_optimizer=Adam(agents[id].model.parameters(), lr=pfgru_lr),
            MSELoss=torch.nn.MSELoss(reduction="mean"),
        )
        for id in range(number_of_agents)
    ]

    # Prepare for interaction with environment
    start_time = time.time()
    o, _, _, _ = env.reset()

    if not isinstance(o, dict):
        raise ValueError("Env not giving observations in dict. Must be in a dict.")

    ep_ret, ep_len, done_count, a = 0, 0, 0, -1
    ep_ret_ls = []
    ep_count = 0
    hidden = [None for _ in range(len(agents))]

    ac.set_mode("eval")

    # Main loop: collect experience in env and update/log each epoch
    print(f"Proc id: {proc_id()} -> Starting main training loop!", flush=True)
    for epoch in range(epochs):
        # Reset hidden state
        for id in range(len(agents)):
            hidden[id] = agents[id].reset_hidden()

        for t in range(local_steps_per_epoch):
            # Artifact - overwrites o because python
            obs_std = o

            actions = {id: None for id in range(len(agents))}
            values = []
            logprobs = []
            heatmap_stacks = []

            # compute action and logp (Actor), compute value (Critic)
            for id in range(len(agents)):
                result, heatmap_stack = agents[id].step(obs_std, hidden=hidden[id])
                actions[id] = result.action
                values.append(result.state_value)
                logprobs.append(result.action_logprob)
                heatmap_stacks.append(heatmap_stack)

            next_o, r, d, _ = env.step(actions)
            d = True if True in d.values() else False

            ep_ret += r["team_reward"]
            ep_len += 1
            ep_ret_ls.append(ep_ret)

            for id in range(len(agents)):
                buffer[id].store(
                    obs=obs_std[id],
                    act=actions[id],
                    val=values[id],
                    logp=logprobs[id],
                    rew=r["team_reward"],
                    src=env.src_coords,
                    full_observation=obs_std,
                    heatmap_stacks=heatmap_stacks[id],
                    terminal=d,
                )

            logger.store(VVals=result.state_value)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if d and not timeout:
                    done_count += 1

                if epoch_ended and not (terminal):
                    print(
                        f"Warning: trajectory cut off by epoch at {ep_len} steps and time {t}.",
                        flush=True,
                    )

                if timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    values = []
                    for id in range(len(agents)):
                        result, _ = ac.step(obs_std, hidden=hidden[id])
                        values.append(result.state_value)

                    if epoch_ended:
                        # Set flag to sample new environment parameters
                        env.epoch_end = True
                else:
                    values = [0 for _ in range(len(agents))]

                # Calculate Advantage and complete filling buffers
                for id in range(len(agents)):
                    buffer[id].GAE_advantage_and_rewardsToGO(values[id])

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    for id in range(len(agents)):
                        buffer[id].store_episode_length(episode_length=ep_len)

                if epoch_ended and render and (epoch % save_gif_freq == 0 or ((epoch + 1) == epochs)):
                    # Check agent progress during training
                    # Render gif
                    env.render(
                        path=logger.output_dir,
                        epoch_count=epoch,
                        episode_count=ep_count,
                        silent=False,
                    )
                    # Render environment image
                    env.render(
                        path=logger.output_dir,
                        epoch_count=epoch,
                        just_env=True,
                        episode_count=ep_count,
                        silent=False,
                    )
                if epoch_ended:
                    ep_count = 0

                # Reset returns list
                ep_ret_ls = []
                if not env.epoch_end:
                    # Reset detector position and episode tracking
                    for id in range(len(agents)):
                        hidden[id] = agents[id].reset_hidden()
                    o, _, _, _ = env.reset()
                    ep_ret, ep_len, a = 0, 0, -1
                else:
                    # Sample new environment parameters, log epoch results
                    logger.store(DoneCount=done_count)
                    done_count = 0
                    oob = 0
                    o, _, _, _ = env.reset()
                    ep_ret, ep_len, a = 0, 0, -1

                # Clear maps for next episode
                ac.reset()

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            for id in range(len(agents)):
                fpath = f"{id}agent"
                fpath = os.path.join(logger.output_dir, fpath)
                os.makedirs(fpath, exist_ok=True)
                ac.save(checkpoint_path=fpath)

        # Reduce localization module training iterations after 100 epochs to speed up training
        if reduce_pfgru_iters and epoch > 99:
            train_pfgru_iters = 5
            reduce_pfgru_iters = False

        # Perform PPO update!
        # update(agent, env, bp_args)
        stop_iteration = np.zeros((len(agents)))
        kl = np.zeros((len(agents)))
        entropy = np.zeros((len(agents)))
        clip_frac = np.zeros((len(agents)))
        actor_loss = np.zeros((len(agents)))
        critic_loss = np.zeros((len(agents)))
        model_loss = np.zeros((len(agents)))

        for id in range(len(agents)):
            (
                actor_loss[id],
                critic_loss[id],
                model_loss[id],
                stop_iteration[id],
                kl[id],
                entropy[id],
                clip_frac[id],
            ) = update(
                ac=agents[id],
                args=bp_args,
                buf=buffer[id],
                train_pi_iters=train_pi_iters,
                train_v_iters=train_v_iters,
                train_pfgru_iters=train_pfgru_iters,
                optimization=optimization[id],
                logger=logger,
                clip_ratio=clip_ratio,
                target_kl=target_kl,
                alpha=alpha,
                number_agents=number_of_agents,
                id=id,
            )

        logger.store(StopIter=stop_iteration.mean().item())

        # Log changes from update
        kl, ent, cf, loss_v = (
            kl.mean().item(),
            entropy.mean().item(),
            clip_frac.mean().item(),
            critic_loss.mean().item(),
        )
        loss_pi = actor_loss.mean().item()
        loss_mod = model_loss.mean().item()

        logger.store(
            LossPi=loss_pi,
            LossV=loss_v,
            LossModel=loss_mod,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            VarExplain=0,
        )

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("LossModel", average_only=True)
        # logger.log_tabular('LocLoss', average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("DoneCount", sum_only=True)
        # logger.log_tabular('OutOfBound', average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_rad_search:RadSearchMulti-v1")
    parser.add_argument("--hid_gru", type=int, default=[24], help="A2C GRU hidden state size")
    parser.add_argument("--hid_pol", type=int, default=[32], help="Actor linear layer size")
    parser.add_argument("--hid_val", type=int, default=[32], help="Critic linear layer size")
    parser.add_argument("--hid_rec", type=int, default=[24], help="PFGRU hidden state size")
    parser.add_argument("--l_pol", type=int, default=1, help="Number of layers for Actor MLP")
    parser.add_argument("--l_val", type=int, default=1, help="Number of layers for Critic MLP")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Reward attribution for advantage estimator",
    )
    parser.add_argument("--seed", "-s", type=int, default=2, help="Random seed control")
    parser.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Number of cores/environments to train the agent with",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=480,
        help="Number of timesteps per epoch per cpu. Default is equal to 4 episodes per cpu per epoch.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the agent")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="alpha01_tkl07_val01_lam09_npart40_lr3e-4_proc10_obs-1_iter40_blr5e-3_2_tanh",
        help="Name of experiment for saving",
    )
    parser.add_argument(
        "--dims",
        type=list,
        default=[[0.0, 0.0], [2700.0, 0.0], [2700.0, 2700.0], [0.0, 2700.0]],
        help="Dimensions of radiation source search area in cm, decreased by area_obs param. to ensure visilibity graph setup is valid.",
    )
    parser.add_argument(
        "--area_obs",
        type=list,
        default=[200.0, 500.0],
        help="Interval for each obstruction area in cm",
    )
    parser.add_argument(
        "--obstruct",
        type=int,
        default=-1,
        help="Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="rnn",
        help="Choose between recurrent neural network A2C or MLP A2C, option: rnn, mlp",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Entropy reward term scaling")
    parser.add_argument("--load_model", type=int, default=0)
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument(
        "--enforce-boundaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicate if each agent should have their own critic or a global.",
    )
    parser.add_argument(
        "--DEBUG",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Indicate if each agent should have their own critic or a global.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Indicate if each agent should have their own critic or a global.",
    )
    args = parser.parse_args()

    # Change mini-batch size, only been tested with size of 1
    args.batch = 1

    # Save directory and experiment name
    args.env_name = "stage_1"
    args.exp_name = f"{args.exp_name}"

    init_dims = {
        "bbox": args.dims,
        "observation_area": args.area_obs,
        "obstruction_count": args.obstruct,
        "number_agents": args.agents,
        "enforce_grid_boundaries": args.enforce_boundaries,
        "DEBUG": args.DEBUG,
        "TEST": args.test,
    }
    max_ep_step = 120
    if args.cpu > 1:
        # max cpus, steps in batch must be greater than the max eps steps times num. of cpu
        tot_epoch_steps = args.cpu * args.steps_per_epoch
        args.steps_per_epoch = tot_epoch_steps if tot_epoch_steps > args.steps_per_epoch else args.steps_per_epoch
        print(f"Sys cpus (avail, using): ({os.cpu_count()},{args.cpu}), Steps set to {args.steps_per_epoch}")
        mpi_fork(args.cpu)  # run parallel code with mpi

    # Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed((1 + proc_id()) * args.seed))[0]
    rng = np.random.default_rng(robust_seed)
    init_dims["np_random"] = rng

    # Setup logger for tracking training metrics
    from rl_tools.run_utils import setup_logger_kwargs  # type: ignore

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir="../../models/train", env_name=args.env_name)

    ac_kwargs = dict(
        # hidden_sizes_pol=[args.hid_pol]*args.l_pol,
        # hidden_sizes_val=[args.hid_val]*args.l_val,
        predictor_hidden_size=args.hid_rec[0],
        # hidden=[args.hid_gru],
        # net_type=args.net_type,
        # batch_s=args.batch
    )

    # Run ppo training function
    ppo(
        lambda: gym.make(args.env, **init_dims),
        actor_critic=CNNBase,
        ac_kwargs=ac_kwargs,
        gamma=args.gamma,
        alpha=args.alpha,
        seed=robust_seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        dims=init_dims,
        logger_kwargs=logger_kwargs,
        render=False,
        save_gif=False,
        load_model=args.load_model,
        number_of_agents=args.agents,
    )
