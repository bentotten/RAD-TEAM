'''
Built from https://github.com/nikhilbarhate99/PPO-PyTorch
'''
import os
import sys
import glob
import time
from datetime import datetime

import torch
import numpy as np
import numpy.random as npr
import numpy.typing as npt
from typing import Any, List, Literal, NewType, Optional, TypedDict, cast, get_args, Dict
from typing_extensions import TypeAlias

import gym
# import roboschool
from gym_rad_search.envs import RadSearch  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

from vanilla_PPO import PPO as van_PPO # vanilla_PPO
from CNN_PPO import PPO as PPO

from dataclasses import dataclass, field
from typing_extensions import TypeAlias

import copy

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DEBUG = True
CNN = True  # TODO remove after done
SCOOPERS_IMPLEMENTATION = False
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Scaling
# TODO get from env instead, remove from global
DET_STEP = 100.0  # detector step size at each timestep in cm/s
DET_STEP_FRAC = 71.0  # diagonal detector step size in cm/s
DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC = 78.0  # Diagonal detector-obstruction range measurement threshold in cm


def convert_nine_to_five_action_space(action):
    ''' Converts 4 direction + idle action space to 9 dimensional equivelant
        Environment action values:
        -1: idle
        0: left
        1: up and left
        2: upfrom gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
        3: up and right
        4: right
        5: down and right
        6: down
        7: down and left

        Cardinal direction action values:
        -1: idle
        0: left
        1: up
        2: right
        3: down
    '''
    match action:
        # Idle
        case -1:
            return -1
        # Left
        case 0:
            return 0
        # Up
        case 1:
            return 2
        # Right
        case 2:
            return 4
        # Down
        case 3:
            return 6
        case _:
            raise Exception('Action is not within valid [-1,3] range.')
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore


################################### Training ###################################

def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "radppo-v2"

    # max_ep_len = 1000                   # max timesteps in one episode
    #training_timestep_bound = int(3e6)   # break training loop if timeteps > training_timestep_bound TODO DELETE
    epochs = int(3e6)  # Actual epoch will be a maximum of this number + max_ep_len
    steps_per_epoch = 3000
    max_ep_len = 120                      # max timesteps in one episode
    #training_timestep_bound = 100  # Change to epoch count DELETE ME

    # print avg reward in the interval (in num timesteps)
    #print_freq = max_ep_len * 3
    print_freq = max_ep_len * 100
    # log avg rewardfrom gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore in the interval (in num timesteps)
    log_freq = max_ep_len * 2
    # save model frequency (in num timesteps)
    save_model_freq = int(1e5)

    # starting std for action distribution (Multivariate Normal)
    action_std = 0.6
    # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    action_std_decay_rate = 0.05
    # minimum action_std (stop decay after action_std <= min_action_std)
    min_action_std = 0.1
    # action_std decay frequency (in num timesteps)
    action_std_decay_freq = int(2.5e5)
    #####################################################

    # Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    steps_per_epoch = 480
    K_epochs = 80               # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lamda = 0.95            # smoothing parameter for Generalize Advantage Estimate (GAE) calculations
    beta: float = 0.005     # TODO look up what this is doing

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0        # set random seed if required (0 = no random seed)
    
    #####################################################

    ###################### logging ######################
    
    # For render
    render = True
    save_gif_freq = 1

    # log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    log_f_name = log_dir + 'PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    # change this to prevent overwriting weights in same env_name folder
    run_num_pretrained = 0

    # directoself.step(action=None, action_list=None)ry = "PPO_preTrained"
    directory = "RAD_PPO"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    print(f"Current directory: {os.getcwd()}")
    #####################################################

    ################ Setup Environment ################

    print("training environment name : " + env_name)
    
    # Generate a large random seed and random generator object for reproducibility
    #robust_seed = _int_list_from_bigint(hash_seed(seed))[0] # TODO get this to work
    #rng = npr.default_rng(robust_seed)
    # Pass np_random=rng, to env creation

    obstruction_count = 0
    number_of_agents = 1
    env: RadSearch = RadSearch(number_agents=number_of_agents, seed=random_seed, obstruction_count=obstruction_count)
    
    resolution_accuracy = 1 * 1/env.scale  
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if DEBUG:
        epochs = 1   # Actual epoch will be a maximum of this number + max_ep_len
        max_ep_len = 20                      # max timesteps in one episode # TODO delete me after fixing
        steps_per_epoch = 20
        K_epochs = 4
                     
        obstruction_count = 0 #TODO error with 7 obstacles
        number_of_agents = 1
        
        seed = 0
        random_seed = _int_list_from_bigint(hash_seed(seed))[0]
        
        log_freq = 2000
        
        render = False
        
        #bbox = tuple(tuple(((0.0, 0.0), (2000.0, 0.0), (2000.0, 2000.0), (0.0, 2000.0))))  
        
        #env: RadSearch = RadSearch(DEBUG=DEBUG, number_agents=number_of_agents, seed=random_seed, obstruction_count=obstruction_count, bbox=bbox) 
        env: RadSearch = RadSearch(DEBUG=DEBUG, number_agents=number_of_agents, seed=random_seed, obstruction_count=obstruction_count)         
        
        # How much unscaling to do. State returnes scaled coordinates for each agent. 
        # A resolution_accuracy value of 1 here means no unscaling, so all agents will fit within 1x1 grid
        resolution_accuracy = 0.01 * 1/env.scale  
        #resolution_accuracy = 1 * 1/env.scale  
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    env.render(
        just_env=True,
        path=directory,
        save_gif=True
    )
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n
    
    # Search area
    search_area = env.search_area[2][1]
    
    # Scaled grid dimensions
    scaled_grid_bounds = (1, 1)  # Scaled to match return from env.step(). Can be reinflated with resolution_accuracy


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    #print("max training timesteps : ", training_timestep_bound)
    print("max training epochs : ", epochs)    
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("Grid space bounds : ", scaled_grid_bounds)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(steps_per_epoch) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize PPO agents
    ppo_agents = {i:
        PPO(
            state_dim=state_dim, 
            action_dim=action_dim, 
            grid_bounds=scaled_grid_bounds, 
            lr_actor=lr_actor, 
            lr_critic=lr_critic,
            gamma=gamma, 
            K_epochs=K_epochs, 
            eps_clip=eps_clip,
            resolution_accuracy=resolution_accuracy,
            steps_per_epoch=steps_per_epoch,
            id=i,
            lamda=lamda,
            beta = beta,
            random_seed= _int_list_from_bigint(hash_seed(seed))[0]
            ) 
        for i in range(number_of_agents)
        }
    
    # TODO move to a unit test
    if number_of_agents > 1:
        ppo_agents[0].maps.buffer.adv_buf[0] = 1
        assert ppo_agents[1].maps.buffer.adv_buf[0] != 1, "Singleton pattern in buffer class"
        ppo_agents[0].maps.buffer.adv_buf[0] = 0.0


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # TODO Need to log for each agent?
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    total_time_step = 0

    # Initial values
    source_coordinates = np.array(env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
    episode_return = {id: 0 for id in ppo_agents}
    episode_return_buffer = []  # TODO can probably get rid of this, unless want to keep for logging
    out_of_bounds_count = np.zeros(number_of_agents)
    success_count = 0
    steps_in_episode = 0
    #local_steps_per_epoch = int(steps_per_epoch / num_procs()) # TODO add this after everything is working
    local_steps_per_epoch = steps_per_epoch
    
    # state = env.reset()['state'] # All agents begin in same location
    # Returns aggregate_observation_result, aggregate_reward_result, aggregate_done_result, aggregate_info_result
    # Unpack relevant results. This primes the pump for agents to choose an action.    
    observations = env.reset().observation # All agents begin in same location, only need one state
        
    # Training loop
    #while total_time_step < training_timestep_bound:
    #while steps_in_epoch < epochs:
    for epoch_counter in range(epochs):
        
        # Put actor into evaluation mode
        # TODO why?
        for agent in ppo_agents.values(): 
            agent.policy.eval()

        for steps_in_epoch in range(local_steps_per_epoch):
            
            # TODO From philippe - is this necessary and should it be added? Appears to make observations between 0 and 1
            #Standardize input using running statistics per episode
            # obs_std = o
            # obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8)            
            
            # Get actions
            if CNN:
                agent_action_returns = {id: agent.select_action(observations, id) for id, agent in ppo_agents.items()} # TODO is this running the same state twice for every step?                
            else:
                # Vanilla FFN
                agent_action_returns = {id: agent.select_action(observations[id].state) -1 for id, agent in ppo_agents.items()} # TODO is this running the same state twice for every step?
            
            if number_of_agents == 1:
                assert ppo_agents[0].maps.others_locations_map.max() == 0.0
            
            # Convert actions to include -1 as "idle" option
            # TODO Make this work in the env calculation for actions instead of here, and make 0 the idle state            
            # TODO REMOVE convert_nine_to_five_action_space AFTER WORKING WITH DIAGONALS
            if CNN:
                if SCOOPERS_IMPLEMENTATION:
                    include_idle_action_list = {id: action.action - 1 for id, action in agent_action_returns.items()}                    
                    action_list = {id: convert_nine_to_five_action_space(action) for id, action in include_idle_action_list.items()}
                else:
                    action_list = {id: action.action - 1 for id, action in agent_action_returns.items()}      
            else:
                # Vanilla FFN
                action_list = agent_action_returns
            
            # Sanity check
            # Ensure no item is above 7 or below -1
            for action in action_list.values():
                assert action < 8 and action >= -1

            next_results = env.step(action_list=action_list, action=None)

            # Unpack information
            next_observations = next_results.observation
            rewards = next_results.reward
            successes = next_results.success
            infos = next_results.info
                
            # Sanity Check
            # Ensure Agent moved in a direction
            for id in ppo_agents:
                # Because of collision avoidance, this assert will not hold true for multiple agents
                if number_of_agents == 1 and action_list[id] != -1 and not infos[id]["out_of_bounds"] and not infos[id]['blocked']:
                    assert (next_observations[id][1] != observations[id][1] or next_observations[id][2] !=  observations[id][2]), "Agent coodinates did not change when should have"

            # Incremement Counters and save new cumulative return
            for id in rewards:
                episode_return[id] += rewards[id]

            episode_return_buffer.append(episode_return)
            total_time_step += 1
            steps_in_episode += 1            
            
            # saving prior state, and current reward/is_terminals etc
            if CNN:
                for id, agent in ppo_agents.items():
                    obs: npt.NDArray[Any] = observations[id]
                    rew: npt.NDArray[np.float32] = rewards[id]
                    terminal: npt.NDArray[np.bool] = successes[id]                                        
                    act: npt.NDArray[np.int32] = agent_action_returns[id].action           
                    val: npt.NDArray[np.float32] = agent_action_returns[id].state_value      
                    logp: npt.NDArray[np.float32] = agent_action_returns[id].action_logprob
                    src: npt.NDArray[np.float32] = source_coordinates                    
                
                    agent.store(
                        obs = obs,
                        act = act,
                        rew = rew,
                        val = val,
                        logp = logp,
                        src = src,
                        terminal = terminal,
                    )

            # Update obs (critical!)
            observations = next_observations
            
            # Check if there was a success
            sucess_counter = 0
            for id in successes:
                if successes[id] == True:
                    sucess_counter += 1

            # Check if some agents went out of bounds
            for id in infos:
                if 'out_of_bounds' in infos[id] and infos[id]['out_of_bounds'] == True:
                        out_of_bounds_count[id] += 1
                                    
            # Stopping conditions for episode
            timeout = steps_in_episode == max_ep_len
            terminal = sucess_counter > 0 or timeout
            epoch_ended = steps_in_epoch == local_steps_per_epoch - 1
            
            if terminal or epoch_ended:
                if terminal and not timeout:
                    success_count += 1

                if epoch_ended and not (terminal):
                    print(f"Warning: trajectory cut off by epoch at {steps_in_episode} steps in episode, at epoch count {steps_in_epoch}.", flush=True)

                if timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    
                    # TODO Philippes normalizing thing, see if we want this
                    #obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8)
                    
                    # TODO Investigate why all state_values are identical
                    agent_state_values = {id: agent.select_action(observations, id).state_value for id, agent in ppo_agents.items()}
                                                            
                    if epoch_ended:
                        # Set flag to sample new environment parameters
                        env.epoch_end = True # TODO make multi-agent?
                else:
                    agent_state_values = {id: 0 for id in ppo_agents}        
                
                # Finish the path and compute advantages
                for id, agent in ppo_agents.items():
                    agent.maps.buffer.finish_path_and_compute_advantages(agent_state_values[id])
                    
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    agent.maps.buffer.store_episode_length(steps_in_episode)

                if (epoch_ended and render and (epoch_counter % save_gif_freq == 0 or ((epoch_counter + 1) == epochs))):
                    # Render agent progress during training
                    #if proc_id() == 0 and epoch != 0:
                    if epoch_counter != 0:
                        for agent in ppo_agents.values():
                            agent.render(
                                add_value_text=True, 
                                savepath=directory,
                                epoch_count=epoch_counter,
                            )                   
                        env.render(
                            save_gif=True,
                            path=directory,
                            epoch_count=epoch_counter,
                            )
                if DEBUG and render:
                    for agent in ppo_agents.values():
                        agent.render(
                            add_value_text=True, 
                            savepath=directory,
                            epoch_count=epoch_counter,
                        )                   
                    env.render(
                        save_gif=True,
                        path=directory,
                        epoch_count=epoch_counter,
                        )                     

                episode_return_buffer = []
                # Reset the environment
                if not epoch_ended: 
                    # Reset detector position and episode tracking
                    # hidden = self.ac.reset_hidden()
                    pass 
                else:
                    # Sample new environment parameters, log epoch results
                    #oob += env.oob_count
                    #logger.store(DoneCount=done_count, OutOfBound=oob)
                    success_count = 0
                    out_of_bounds_count = np.zeros(number_of_agents)

                # Unpack relevant results. This primes the pump for agents to choose an action.                
                observations = env.reset().observation
                source_coordinates = np.array(env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
                episode_return = {id: 0 for id in ppo_agents}
                steps_in_episode = 0
                
                # Update stats buffer for normalizer
                #stat_buff.update(o[0])

        # TODO for eventual merger with radppo
        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs-1):
        #     logger.save_state(None, None)
        #     pass

        # TODO implement this
        # # Reduce localization module training iterations after 100 epochs to speed up training
        # if reduce_v_iters and epoch_counter > 99:
        #     train_v_iters = 5
        #     reduce_v_iters = False

        # Perform PPO update!
        #self.update(env, bp_args) 

        # update PPO agents
        if CNN:
            for id, agent in ppo_agents.items():
                agent.update(search_area)
        
        # Vanilla FFN
        else:
            for id, agent in ppo_agents.items():
                agent.buffer.rewards.append(rewards[id])
                agent.buffer.is_terminals.append(successes[id])
                ####
                # update PPO agent
                agent.update()
                                    
        # printing average reward
        if total_time_step % print_freq == 0:
            print("Epoch : {} \t\t  Total Timestep: {} \t\t Cumulative Returns: {} \t\t Out of bounds count: {}".format(
                epoch_counter, total_time_step, episode_return_buffer, out_of_bounds_count))
            
        # TODO Logging logger goes here
                 

        # Save model weights
        # if total_time_step % save_model_freq == 0:
        #     print(
        #         "--------------------------------------------------------------------------------------------")
        #     print("saving model at : " + checkpoint_path)
        #     # Render last episode
        #     print("TEST RENDER - Delete Me Later")
        #     episode_rewards = {id: render_buffer_rewards[-max_ep_len:] for id, agent in ppo_agents.items()}
        #     env.render(
        #         save_gif=True,
        #         path=directory,
        #         epoch_count=epoch_counter,
        #         episode_rewards=episode_rewards
        #     )
        #     #ppo_agent.save(checkpoint_path)
        #     print("model saved")
        #     print("Elapsed Time  : ", datetime.now().replace(
        #         microsecond=0) - start_time)
        #     print(
        #         "--------------------------------------------------------------------------------------------")

    # Render last episode
    env.render(
        save_gif=True,
        path=directory,
        epoch_count=epoch_counter,
    )
    for agent in ppo_agents.values():
        agent.render(add_value_text=True, savepath=directory)   

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
