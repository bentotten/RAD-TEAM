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

from vanilla_PPO import PPO as van_PPO # vanilla_PPO
from CNN_PPO import PPO as PPO

from dataclasses import dataclass, field
from typing_extensions import TypeAlias

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DEBUG = True
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
        2: up
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

################################### Training ###################################

def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "radppo-v2"

    # max_ep_len = 1000                   # max timesteps in one episode
    #training_timestep_bound = int(3e6)   # break training loop if timeteps > training_timestep_bound TODO DELETE
    epochs = 62  # Actual epoch will be a maximum of this number + max_ep_len
    #epochs = int(3e6)  # Actual epoch will be a maximum of this number + max_ep_len
    #max_ep_len = 120                      # max timesteps in one episode
    max_ep_len = 10                      # max timesteps in one episode # TODO delete me after fixing
    #training_timestep_bound = 100  # Change to epoch count DELETE ME

    # print avg reward in the interval (in num timesteps)
    #print_freq = max_ep_len * 3
    print_freq = max_ep_len * 100
    # log avg reward in the interval (in num timesteps)
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
    #update_timestep = max_ep_len * 4      # update policy every n timesteps # TODO Change to epochs
    update_timestep = 480     # update policy every n timesteps # TODO Change to epochs
    K_epochs = 80               # update policy for K epochs in one PPO update
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if DEBUG:
        update_timestep = 2
        K_epochs = 4
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 1         # set random seed if required (0 = no random seed)
    
    resolution_accuracy = 100  # Round agent coordinates to nearest 100 for transition to map
    resolution_accuracy = 10
    #####################################################

    ###################### logging ######################

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
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    CNN = True  # TODO remove after done
    if DEBUG:
        obstruction_count = 1
        bbox = tuple(tuple(((0.0, 0.0), (2000.0, 0.0), (2000.0, 2000.0), (0.0, 2000.0))))  
        #observation_area = tuple((20.0, 50.0))
        env: RadSearch = RadSearch(DEBUG=DEBUG, number_agents=number_of_agents, seed=random_seed, obstruction_count=obstruction_count, bbox=bbox) 
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
    
    # Grid dimensions
    grid_bounds = (env.bbox[2][0] / DET_STEP, env.bbox[2][1] / DET_STEP)  # Scale to match env returned state coords for each agent


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
    print("Grid space bounds : ", grid_bounds)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
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

    # initialize a PPO agent
    ppo_agents = {i:
        PPO(
            state_dim=state_dim, 
            action_dim=action_dim, 
            grid_bounds=grid_bounds, 
            lr_actor=lr_actor, 
            lr_critic=lr_critic, 
            gamma=gamma, 
            K_epochs=K_epochs, 
            eps_clip=eps_clip,
            resolution_accuracy=resolution_accuracy,
            id=i
            ) 
        for i in range(number_of_agents)
        }

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # TODO Need to log for each agent?
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    total_time_step = 0
    i_episode = 0

    epoch_counter = 0

    # training loop
    #while total_time_step < training_timestep_bound:
    while epoch_counter < epochs:
    
        # TODO why is state 11 long?
        # state = env.reset()['state'] # All agents begin in same location
        results = env.reset() # All agents begin in same location, only need one state
        
        #print("Source location: ", env.src_coords)
        #print("Agent location: ", env.agents[0].det_coords)
        
        # TODO turn into array for each agent ID (maybe a dict or tuple)
        current_ep_reward_sample = 0
        epoch_counter += 1

        # Sanity check
        prior_state = []
        for result in results.values():
            prior_state.append([result.state[1], result.state[2]])  

        for _ in range(max_ep_len):
            if DEBUG:
                print("Training [state]: ", results[0].state)
            if CNN:
                raw_action_list = {id: agent.select_action(results, id) -1 for id, agent in ppo_agents.items()} # TODO is this running the same state twice for every step?
            else:
                raw_action_list = {id: agent.select_action(results[id].state) -1 for id, agent in ppo_agents.items()} # TODO is this running the same state twice for every step?
            
            # TODO Make this work in the env calculation for actions instead of here, and make 0 the idle state
            # Convert actions to include -1 as "idle" option
            # TODO REMOVE THIS AFTER WORKING WITH DIAGONALS
            if CNN:
                action_list = {id: convert_nine_to_five_action_space(action) for id, action in raw_action_list.items()}
            else:
                action_list = raw_action_list
            
            # Ensure no item is above 7 or below -1
            for action in action_list.values():
                assert action < 8 and action >= -1

            #state, reward, done, _
            results = env.step(action_list=action_list, action=None) 

            # TODO Put more thought into how to handle this for CNN
            #      Moved to environment
            # if CNN:
            #     for id, result in results.items():
            #         if result.error["out_of_bounds"]:
            #             print("WARNING: Out of bounds not implemented for CNN! Resetting location to last known.")
            #             result.state[1] = prior_state[id][0]
            #             result.state[2] = prior_state[id][1]
                
            # Ensure Agent moved in a direction
            for id, result in results.items():
                if action_list[id] != -1 and not result.error["out_of_bounds"]:
                    assert (result.state[1] != prior_state[id][0] or result.state[2] != prior_state[id][1]), "Agent coodinates did not change when should have"
                    pass
                prior_state[id][0] = result.state[1]
                prior_state[id][1] = result.state[2]

            total_time_step += 1
            # TODO make work with averaged rewards from all agent, not just first agent
            current_ep_reward_sample += results[0].reward # Just take first agents rewards for now

            # saving reward and is_terminals
            # TODO move out of episode
            # Vanilla
            if CNN:
                for id, agent in ppo_agents.items():
                    agent.maps.buffer.rewards.append(results[id].reward)
                    agent.maps.buffer.is_terminals.append(results[id].done)
                    ####
                    # update PPO agent
                    if total_time_step % update_timestep == 0:
                        agent.update()
                        epoch_counter += 1
            else:
                for id, agent in ppo_agents.items():
                    agent.buffer.rewards.append(results[id].reward)
                    agent.buffer.is_terminals.append(results[id].done)
                    ####
                    # update PPO agent
                    if total_time_step % update_timestep == 0:
                        agent.update()
                        epoch_counter += 1
                                    
            # log in logging file
            # TODO Log each agent instead of one
            if total_time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(
                    i_episode, total_time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            # TODO print each agent instead of one
            if total_time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t (Agent 0) Average Reward : {}".format(
                    i_episode, total_time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            # TODO Save each agent model instead of one?
            # if total_time_step % save_model_freq == 0:
            #     print(
            #         "--------------------------------------------------------------------------------------------")
            #     print("saving model at : " + checkpoint_path)
            #     # TODO make multi-agent
            #     print("TODO SAVE EACH AGENT HERE")
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

            # TODO Find better way to do this
            done = False
            for result in results.values():
                if result.done:
                    done = True
                    break
            # break; if the episode is over
            if done:
                break

        # TODO Print array instead
        print_running_reward += current_ep_reward_sample
        print_running_episodes += 1

        log_running_reward += current_ep_reward_sample
        log_running_episodes += 1

        i_episode += 1
        
        # TODO Delete me
        if DEBUG:
            env.render(
                save_gif=True,
                path=directory,
                epoch_count=epoch_counter,
                #episode_rewards=episode_rewards
            )

    # Render last episode
    # episode_rewards = {id: render_buffer_rewards[-max_ep_len:] for id, agent in ppo_agents.items()}
    env.render(
        save_gif=True,
        path=directory,
        epoch_count=epoch_counter,
        #episode_rewards=episode_rewards
    )

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
