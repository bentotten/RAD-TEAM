'''
Implementation of "Target Localization using Multi-Agent Deep Reinforcement Learning with Proximal Policy Optimization" by Alagha et al.
Partially adapted from:
    - towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
    - github.com/nikhilbarhate99/PPO-PyTorch

'''
from os import stat, path, mkdir, getcwd

from numpy import dtype
import numpy as np
import numpy.typing as npt
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import pytorch_lightning as pl

from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Union, Literal, NewType, Optional, TypedDict, cast, get_args, Dict
from typing_extensions import TypeAlias

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.streamplot import Grid

# Maps
Point: TypeAlias = NewType("Point", tuple[float, float])  # Array indicies to access a GridSquare
Map: TypeAlias = NewType("Map", npt.NDArray[np.float32]) # 2D array that holds gridsquare values

DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm for inflating step size to obstruction

# TODO move this somewhere ... else lol
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## Helper Functions ##################################

def discount_cumsum(
    x: npt.NDArray[np.float64], discount: float
) -> npt.NDArray[np.float64]:
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

################################## PPO Policy ##################################
@dataclass()
class RolloutBuffer():
    # Outdated - TODO remove
    actions: List = field(default_factory=list)
    states: List = field(default_factory=list)
    logprobs: List = field(default_factory=list)
    state_values: List = field(default_factory=list)
    rewards: List = field(default_factory=list)
    is_terminals: List = field(default_factory=list)
    mapstacks: List = field(default_factory=list) #TODO change to tensor?

    readings: Dict[Any, list] = field(default_factory=dict)
    
    def __post_init__(self):
        pass
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.mapstacks[:]
        self.readings.clear()


@dataclass()
class MapsBuffer:        
    '''
    4 maps: 
        1. Location Map: a 2D matrix showing the individual agent's location.
        2. Map of Other Locations: a grid showing the number of agents located in each grid element (excluding current agent).
        3. Readings map: a grid of the last reading collected in each grid square - unvisited squares are given a reading of 0.
        4. Visit Counts Map: a grid of the number of visits to each grid square from all agents combined.
    '''
    # Parameters
    grid_bounds: tuple = field(default_factory= lambda: (1,1))
    x_limit_scaled: int = field(init=False)
    y_limit_scaled: int = field(init=False)
    resolution_accuracy: int = field(default=100)
    map_dimensions: Tuple = field(init=False)
    obstacle_state_offset: int = field(default=3) # Number of initial elements in state return that do not indicate there is an obstacle. 
    
    # TODO make work with max_step_count so boundaries dont need to be enforced on grid. Basically take the grid bounds and add the max step count to make the map sizes
    
    # Maps
    location_map: Map = field(init=False)
    others_locations_map: Map = field(init=False)
    readings_map: Map = field(init=False)
    visit_counts_map: Map = field(init=False)
    obstacles_map: Map = field(init=False)
    
    # Buffers
    buffer: RolloutBuffer = field(default_factory=lambda: RolloutBuffer())

    def __post_init__(self):
        '''
        Scaled maps
        '''
        self.map_dimensions = (int(self.grid_bounds[0] * self.resolution_accuracy), int(self.grid_bounds[1] * self.resolution_accuracy))
        self.x_limit_scaled: int = self.map_dimensions[0]
        self.y_limit_scaled: int = self.map_dimensions[1] 
        self.clear()

    
    def clear(self):
        self.location_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow - potentially change to torch?
        self.others_locations_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.readings_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.visit_counts_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.obstacles_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.buffer.clear()
        
    def state_to_map(self, observation, id):
        '''
        state: observations from environment for all agents
        id: ID of agent to reference in state object 
        '''
        
        # TODO Remove redundant calculations
        
        # Process state for current agent's locations map
        scaled_coordinates = (int(observation[id].state[1] * self.resolution_accuracy), int(observation[id].state[2] * self.resolution_accuracy))        
        # Capture current and reset previous location
        if self.buffer.states:
            last_state = self.buffer.states[-1][id].state
            scaled_last_coordinates = (int(last_state[1] * self.resolution_accuracy), int(last_state[2] * self.resolution_accuracy))
            x_old = int(scaled_last_coordinates[0])
            y_old = int(scaled_last_coordinates[1])
            self.location_map[x_old][y_old] -= 1 # In case agents are at same location, usually the start-point
            assert self.location_map[x_old][y_old] > -1, "location_map grid coordinate reset where agent was not present. The map location that was reset was already at 0."
        # Set new location
        x = int(scaled_coordinates[0])
        y = int(scaled_coordinates[1])
        self.location_map[x][y] = 1.0 
        
        # Process state for other agent's locations map

        for other_agent_id in observation:
            # Do not add current agent to other_agent map
            if other_agent_id != id:
                others_scaled_coordinates = (int(observation[other_agent_id].state[1] * self.resolution_accuracy), int(observation[other_agent_id].state[2] * self.resolution_accuracy))
                # Capture current and reset previous location
                if self.buffer.states:
                    last_state = self.buffer.states[-1][other_agent_id].state
                    scaled_last_coordinates = (int(last_state[1] * self.resolution_accuracy), int(last_state[2] * self.resolution_accuracy))
                    x_old = int(scaled_last_coordinates[0])
                    y_old = int(scaled_last_coordinates[1])
                    self.others_locations_map[x_old][y_old] -= 1 # In case agents are at same location, usually the start-point
                    assert self.others_locations_map[x_old][y_old] > -1, "Location map grid coordinate reset where agent was not present"
        
                # Set new location
                x = int(others_scaled_coordinates[0])
                y = int(others_scaled_coordinates[1])
                self.others_locations_map[x][y] += 1.0  # Initial agents begin at same location        
                 
        # Process state for readings_map
        for agent_id in observation:
            scaled_coordinates = (int(observation[agent_id].state[1] * self.resolution_accuracy), int(observation[agent_id].state[2] * self.resolution_accuracy))            
            x = int(scaled_coordinates[0])
            y = int(scaled_coordinates[1])
            unscaled_coordinates = (observation[agent_id].state[1], observation[agent_id].state[2])
                        
            assert len(self.buffer.readings[unscaled_coordinates]) > 0
            # TODO onsider using a particle filter for resampling            
            estimated_reading = np.median(self.buffer.readings[unscaled_coordinates])
            self.readings_map[x][y] = estimated_reading  # Initial agents begin at same location

        # Process state for visit_counts_map
        for agent_id in observation:
            scaled_coordinates = (int(observation[agent_id].state[1] * self.resolution_accuracy), int(observation[agent_id].state[2] * self.resolution_accuracy))            
            x = int(scaled_coordinates[0])
            y = int(scaled_coordinates[1])

            self.visit_counts_map[x][y] += 1
            
        # Process state for obstacles_map 
        # Agent detects obstructions within 110 cm of itself
        for agent_id in observation:
            scaled_agent_coordinates = (int(observation[agent_id].state[1] * self.resolution_accuracy), int(observation[agent_id].state[2] * self.resolution_accuracy))            
            if np.count_nonzero(observation[agent_id].state[self.obstacle_state_offset:]) > 0:
                indices = np.flatnonzero(observation[agent_id].state[self.obstacle_state_offset::]).astype(int)
                for index in indices:
                    real_index = int(index + self.obstacle_state_offset)
                    
                    # Inflate to actual distance, then convert and round with resolution_accuracy
                    inflated_distance = (-(observation[agent_id].state[real_index] * DIST_TH - DIST_TH))
                    
                    # scaled_obstacle_distance = int(inflated_distance / self.resolution_accuracy)
                    # step: int = field(init=False)
                    # match index:
                        # Access the obstacle detection portion of state and see what direction an obstacle is in
                        # These offset indexes correspond to:
                        # 0: left
                        # 1: up and left
                        # 2: up
                        # 3: up and right
                        # 4: right
                        # 5: down and right
                        # 6: down
                        # 7: down and left                    
                    #     # 0: Left
                    #     case 0:
                    #         step = (-1 ,0)
                    #     # 1: up and left
                    #     case 1:
                    #         step = (-scaled_obstacle_distance, scaled_obstacle_distance)
                    #     # 2: up
                    #     case 2:
                    #         step = (0, scaled_obstacle_distance)
                    #     # 3: up and right
                    #     case 3:
                    #         step = (scaled_obstacle_distance, scaled_obstacle_distance)
                    #     # 4: right
                    #     case 4:
                    #         step = (scaled_obstacle_distance, 0)                        
                    #     # 5: down and right
                    #     case 5:
                    #         step = (scaled_obstacle_distance, -scaled_obstacle_distance)                           
                    #     # 6: down
                    #     case 6:
                    #         step = (0, -scaled_obstacle_distance)                           
                    #     # 7: down and left
                    #     case 7:
                    #         step = (-scaled_obstacle_distance, -scaled_obstacle_distance)                                                     
                    #     case _:
                    #         raise Exception('Obstacle index is not within valid [0,7] range.')                         
                    # x = int(scaled_agent_coordinates[0] + step[0])
                    # y = int(scaled_agent_coordinates[1] + step[1])
                    x = int(scaled_coordinates[0])
                    y = int(scaled_coordinates[1])
                    
                    # Semi-arbritrary, but should make the number higher as the agent gets closer to the object, making heatmap look more correct
                    self.obstacles_map[x][y] = DIST_TH - inflated_distance
        
        return self.location_map, self.others_locations_map, self.readings_map, self.visit_counts_map, self.obstacles_map


class Actor(nn.Module):
    def __init__(self, map_dim, state_dim, batches: int=1, map_count: int=5, action_dim: int=5, global_critic: bool=False):
        super(Actor, self).__init__()
        
        # TODO get to work with 5 maps, adding obstacles_map
        ''' Actor Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 1
                2. (map_count) number of channels: 5 input maps
                3. Height: grid height
                4. Width: grid width
            
                5 maps: 
                    1. Location Map: a 2D matrix showing the agents location.
                    2. Map of Other Locations: a 2D matrix showing the number of agents located in each grid element (excluding current agent).
                    3. Readings map: a 2D matrix showing the last reading collected in each grid element. Grid elements that have not been visited are given a reading of 0.
                    4. Visit Counts Map: a 2D matrix showing the number of visits each grid element has received from all agents combined.
                    5. Obstacle Map: a 2D matrix of obstacles detected by agents
                    
            Critic Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 1 maps
                2. (map_count) number of channels: 5 input maps, same as Actor
                3. Height: grid height
                4. Width: grid width

        '''

        assert map_dim[0] > 0 and map_dim[0] == map_dim[1], 'Map dimensions mismatched. Must have equal x and y bounds.'
        
        channels = map_count
        pool_output = int(((map_dim[0]-2) / 2) + 1) # Get maxpool output height/width and floor it

        # Actor network
        self.step1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1)  # output tensor with shape (batchs, 8, Height, Width)
        self.relu = nn.ReLU()
        self.step2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output height and width is floor(((Width - Size)/ Stride) +1)
        self.step3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1) 
        #nn.ReLU()
        self.step4 = nn.Flatten(start_dim=0, end_dim= -1) # output tensor with shape (1, x)
        self.step5 = nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32) 
        #nn.ReLU()
        self.step6 = nn.Linear(in_features=32, out_features=16) 
        #nn.ReLU()
        self.step7 = nn.Linear(in_features=16, out_features=5) # TODO eventually make '5' action_dim instead
        self.softmax = nn.Softmax(dim=0)  # Put in range [0,1] 
        self.global_critic = global_critic

        # TODO uncomment after ready to combine
        self.actor = nn.Sequential(
                        nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (4, 8, Height, Width)
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # output tensor with shape (4, 8, 2, 2)
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),  # output tensor with shape (4, 16, 2, 2)
                        nn.ReLU(),
                        nn.Flatten(start_dim=0, end_dim= -1),  # output tensor with shape (1, x)
                        nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32), # output tensor with shape (32)
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=16), # output tensor with shape (16)
                        nn.ReLU(),
                        nn.Linear(in_features=16, out_features=5), # output tensor with shape (5)
                        nn.Softmax(dim=0)  # Put in range [0,1]
                    )

        # If decentralized critic
        # TODO uncomment after ready to combine
        if not self.global_critic:
            self.local_critic = nn.Sequential(
                        # Starting shape (batch_size, 4, Height, Width)
                        nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (batch_size, 8, Height, Width)
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # output tensor with shape (batch_size, 8, x, x) x is the floor(((Width - Size)/ Stride) +1)
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),  # output tensor with shape (batch_size, 16, 2, 2)
                        nn.ReLU(),
                        nn.Flatten(start_dim=0, end_dim= -1),  # output tensor with shape (1, x)
                        nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32), # output tensor with shape (32)
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=16), # output tensor with shape (16)
                        nn.ReLU(),
                        nn.Linear(in_features=16, out_features=1), # output tensor with shape (1)
                        nn.Tanh(), #TODO Did the original maker implement softmax here?
                        #nn.Softmax(dim=0)  #TODO Did the original maker implement softmax here?
                    )

    def test(self, state_map_stack): 
        print("Starting shape, ", state_map_stack.size())
        x = self.step1(state_map_stack) # conv1
        x = self.relu(x)
        print("shape, ", x.size()) 
        x = self.step2(x) # Maxpool
        print("shape, ", x.size()) 
        x = self.step3(x) # conv2
        x = self.relu(x)
        print("shape, ", x.size()) 
        x = self.step4(x) # Flatten
        print("shape, ", x.size()) 
        x = self.step5(x) # linear
        x = self.relu(x) 
        print("shape, ", x.size()) 
        x = self.step6(x) # linear
        x = self.relu(x)
        print("shape, ", x.size()) 
        x = self.step7(x) # Output layer
        print("shape, ", x.size()) 
        x = self.softmax(x)
        
        print(x)
        pass

    def forward(self):
        raise NotImplementedError
    
    def act(self, state_map_stack):
        # Select Action from Actor
        action_probs = self.actor(state_map_stack)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Get q-value from critic
        #state_value = self.local_critic(state_map_stack) if not self.global_critic else global_critic(state_map_stack) # TODO implement global critic
        state_value = self.local_critic(state_map_stack)
        
        return action.detach(), action_logprob.detach(), state_value.detach()
    
    def evaluate(self, state_map_stack, action):
        
        # TODO Works without unsqueezing, investigate why
        for map_stack in state_map_stack:
            single_map_stack = torch.unsqueeze(map_stack, dim=0) 
            self.test(single_map_stack)
                
            action_probs = self.actor(state_map_stack)
            
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_values = self.local_critic(state_map_stack)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, grid_bounds, lr_actor, lr_critic, gamma, K_epochs, eps_clip, resolution_accuracy, id, lmbda=0.95, random_seed=None):
        '''
        state_dim: The dimensions of the return from the environment
        action_dim: How many actions the actor chooses from
        grid_bounds: The grid bounds for the state returned by the environment. For RAD-PPO, this is (1, 1). This value will be scaled by the resolution_accuracy variable
        lr_actor: learning rate for actor neural network
        lr_critic: learning rate for critic neural network
        gamma: discount rate for expected return and Generalize Advantage Estimate (GAE) calculations
        K_epochs:
        eps_clip: 
        resolution_accuracy: How much to scale the convolution maps by (higher rate means more accurate, but more memory usage)
        lmbda: smoothing parameter for Generalize Advantage Estimate (GAE) calculations
        '''
        # Testing
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)            
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.lmbda = lmbda  # Smoothing parameter for GAE calculation
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.resolution_accuracy = resolution_accuracy
        self.id = id
        
        # Initialize
        self.maps = MapsBuffer(grid_bounds, resolution_accuracy)

        self.policy = Actor(map_dim=self.maps.map_dimensions, state_dim=state_dim, action_dim=action_dim).to(device) 
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.local_critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = Actor(map_dim=self.maps.map_dimensions, state_dim=state_dim, action_dim=action_dim).to(device)  # TODO Really slow
        self.policy_old.load_state_dict(self.policy.state_dict()) # TODO Really slow
        
        self.MseLoss = nn.MSELoss()
        
        # Rendering
        self.render_counter = 0        

    def select_action(self, state, id):    
        
        # Add intensity readings to a list if reading has not been seen before at that location. 
        for observation in state.values():
            key = (observation.state[1], observation.state[2])
            if key in self.maps.buffer.readings:
                if observation.state[0] not in self.maps.buffer.readings[key]:
                    self.maps.buffer.readings[key].append(observation.state[0])
            else:
                self.maps.buffer.readings[key] = [observation.state[0]]
            assert observation.state[0] in self.maps.buffer.readings[key], "Observation not recorded into readings buffer"

        with torch.no_grad():
            (
                location_map,
                others_locations_map,
                readings_map,
                visit_counts_map,
                obstacles_map
            ) = self.maps.state_to_map(state, id)
            
            # TODO add obstacles_map
            map_stack = torch.stack([torch.tensor(location_map), torch.tensor(others_locations_map), torch.tensor(readings_map), torch.tensor(visit_counts_map),  torch.tensor(obstacles_map)]) # Convert to tensor
            
            # Add to mapstack buffer to eventually be converted into tensor with minibatches
            self.maps.buffer.mapstacks.append(map_stack)
            
            # Add single minibatch for action selection
            map_stack = torch.unsqueeze(map_stack, dim=0) 
            
            #state = torch.FloatTensor(state).to(device) # Convert to tensor TODO already a tensor, is this necessary?
            #action, action_logprob = self.policy_old.act(state) # Choose action
            action, action_logprob, state_value = self.policy_old.act(map_stack) # Choose action # TODO why old policy?
        
        self.maps.buffer.states.append(state)
        self.maps.buffer.actions.append(action)
        self.maps.buffer.logprobs.append(action_logprob)
        self.maps.buffer.state_values.append(state_value)

        return action.item()

    def calculate_advantages(self, rewards=None):
        ''' Advantage is roughly "how much better off will the actor be if a particular action is taken over the course of the episode"
        Generalized Advantage Estimation (GAE)
        
        gamma = discount rate
        lambda = smoothing parameter (0.95 suggested in the paper)
        
        1. Loop from last step backwards to step 0 of this batch
        2. Get delta:
            Delta = reward_t + (discount_rate * value_{t+1} * terminal_t) - value_t
            If terminal for any timestep t, then terminal_t = 0, else terminal_t = 1. Because this module is episodic and a terminal value means the episode has reset,
            we do not want to apply rewards from the reset to the calculations
        3. Get GAE:
            GAE = delta + (discount_rate  * smoothing_parameter * terminal_t * GAE_{t+1})
        4. Get return values:
            Return_values = GAE_t + value_t
        5. Reverse return_values to get back in original order
        '''
        if not rewards: rewards = self.maps.buffer.rewards
        returns = [] 
        gae = 0
        # Slow way
        # TODO something is wrong here; believe state should always be one ahead
        #for i in reversed(range(len(rewards))):
        for i in reversed(range(len(rewards)-1)):
            mask = 0 if self.maps.buffer.is_terminals[i] else 1 
            delta = rewards[i] + (self.gamma * self.maps.buffer.state_values[i + 1] * mask) - self.maps.buffer.state_values[i] # TODO does the discount factor need to be gamma^t?  
            gae = delta + (self.gamma * self.lmbda * mask * gae) 
            returns.insert(0, gae + self.maps.buffer.state_values[i]) # Puts back in correct order
            
        # TODO Get to work with discount_cumsum() instead
        # From OpenAI
        # path_slice = slice(self.path_start_idx, self.ptr)
        # rews = np.append(self.rew_buf[path_slice], last_val)
        # vals = np.append(self.val_buf[path_slice], last_val)
        
        # # the next two lines implement GAE-Lambda advantage calculation
        # # gamma determines scale of value function, introduces bias regardless of VF accuracy
        # # lambda introduces bias when VF is inaccurate
        # deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # # the next line computes rewards-to-go, to be targets for the value function
        # self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        # self.path_start_idx = self.ptr

        adv = np.array(returns) - self.maps.buffer.state_values[:-1] # TODO this throws a warning ...
        normalized_returns = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return returns, normalized_returns

    def calculate_loss_critic():
        '''
        Calculate Critic loss (Critic loss is nothing but the usual mean squared error loss with the Returns)
        '''
        raise Exception('Not implemented yet')
        
    def calculate_loss_actor():
        '''
        Calculate how much the policy has changed, as in a ratio of the new policy over the old policy, in log form for easier computation
        Decide update tolerance using the clipping parameter epsilon to ensure only make the maximum of epsilon% change to our policy at a time. 
        Calculate Actor loss using the minimum between the policy change ratio * the advantage and the clipping parameter * the advantage
        Calculate total loss (using a discount factor to bring them to the same order of magnitude)
        An entropy term is optional, but it encourages our actor model to explore different policies and the degree to which we want to experiment can be controlled by an entropy beta parameter.
        '''
        raise Exception('Not implemented yet')

    def calculate_total_clipped_loss():
        def loss(y_true, y_pred):
            newpolicy_probs = y_pred
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            critic_loss = K.mean(K.square(rewards - values))
            total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
                -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
            return total_loss

    def update(self):
        # TODO I believe this is wrong; see vanilla_PPO.py TODO comment
        # Monte Carlo estimate of returns
        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(self.maps.buffer.rewards), reversed(self.maps.buffer.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        returns, normalized_advantages = self.calculate_advantages()

        # TODO for memory efficiency, could remap state buffers here instead of storing them
        # convert list to tensor
        old_maps = torch.squeeze(torch.stack(self.maps.buffer.mapstacks, dim=0)).detach().to(device)
        print(old_maps.shape)
        old_actions = torch.squeeze(torch.stack(self.maps.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.maps.buffer.logprobs, dim=0)).detach().to(device)
        # TODO throws error
        #returns_tensor = torch.squeeze(returns, dim=-1).detach().to(device)
        #normalized_advantages_tensor = torch.squeeze(normalized_advantages, dim=0).detach().to(device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_maps, old_actions) # Actor-critic

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict()) # Actor-critic

        # clear buffer
        self.maps.clear() # TODO clear buffer or maps buffer?
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path) # Actor-critic
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic
        
    def render(self, savepath=getcwd(), save_map=True, add_value_text=False, interpolation_method='nearest', epoch_count: int=0):
        # TODO x and y are swapped - investigate if reading that way or a part of  imshow()
        if save_map:
            if not path.isdir(str(savepath) + "/heatmaps/"):
                mkdir(str(savepath) + "/heatmaps/")
        else:
            plt.show()                
     
        loc_transposed = self.maps.location_map.T # TODO this seems expensive
        other_transposed = self.maps.others_locations_map.T 
        readings_transposed = self.maps.readings_map.T
        visits_transposed = self.maps.visit_counts_map.T
        obstacles_transposed = self.maps.obstacles_map.T
     
        fig, (loc_ax, other_ax, intensity_ax, visit_ax, obs_ax) = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
        
        loc_ax.imshow(loc_transposed, cmap='viridis', interpolation=interpolation_method)
        loc_ax.set_title('Agent Location')
        loc_ax.invert_yaxis()        
        
        other_ax.imshow(other_transposed, cmap='viridis', interpolation=interpolation_method)
        other_ax.set_title('Other Agent Locations') 
        other_ax.invert_yaxis()  
        
        intensity_ax.imshow(readings_transposed, cmap='viridis', interpolation=interpolation_method)
        intensity_ax.set_title('Radiation Intensity')
        intensity_ax.invert_yaxis()
        
        visit_ax.imshow(visits_transposed, cmap='viridis', interpolation=interpolation_method)
        visit_ax.set_title('Visit Counts') 
        visit_ax.invert_yaxis()
        
        obs_ax.imshow(obstacles_transposed, cmap='viridis', interpolation=interpolation_method)
        obs_ax.set_title('Obstacles Detected (cm from Agent)') 
        obs_ax.invert_yaxis()
        
        #divider = make_axes_locatable(loc_ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)             
        #fig.colorbar(loc_ax, cax=cax, orientation='vertical')
        
        # Add values to gridsquares if value is greater than 0 #TODO if large grid, this will be slow
        if add_value_text:
            for i in range(loc_transposed.shape[0]):
                for j in range(loc_transposed.shape[1]):
                    if loc_transposed[i, j] > 0: 
                        loc_ax.text(j, i, loc_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)
                    if other_transposed[i, j] > 0: 
                        other_ax.text(j, i, other_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)
                    if readings_transposed[i, j] > 0:
                        intensity_ax.text(j, i, readings_transposed[i, j].astype(int), ha="center", va="center", color="black", size=4)
                    if visits_transposed[i, j] > 0:
                        visit_ax.text(j, i, visits_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)
                    if obstacles_transposed[i, j] > 0:
                        obs_ax.text(j, i, obstacles_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)                        
        
        fig.savefig(f'{str(savepath)}/heatmaps/heatmap_agent{self.id}_epoch_{epoch_count}-{self.render_counter}.png')
        
        self.render_counter += 1
        plt.close(fig)