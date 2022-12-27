from os import stat
from matplotlib.streamplot import Grid
from numpy import dtype
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import pytorch_lightning as pl

from dataclasses import dataclass, field, asdict
from typing import Any, List, Union, Literal, NewType, Optional, TypedDict, cast, get_args, Dict
from typing_extensions import TypeAlias

# Maps
Point: TypeAlias = NewType("Point", tuple[float, float])  # Array indicies to access a GridSquare
GridSquare: TypeAlias = NewType("GridSquare", float)  # Value stored in a map location
Map: TypeAlias = NewType("Map", List[List[GridSquare]])  # 2D array that holds gridsquare values TODO switch to npt.NDArray[] or tensor

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


################################## PPO Policy ##################################
@dataclass()
class RolloutBuffer():
    # Outdated - TODO remove
    actions: List = field(default_factory=list)
    states: List = field(default_factory=list)
    logprobs: List = field(default_factory=list)
    rewards: List = field(default_factory=list)
    is_terminals: List = field(default_factory=list)
    # TODO add buffer for history
    
    def __post_init__(self):
        pass
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


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
    grid_bounds: List = field(default_factory=List)
    x_limit_scaled: int = field(init=False)
    y_limit_scaled: int = field(init=False)
    resolution_accuracy: int = field(default=100)
    
    # Maps
    location_map: Map = field(init=False)
    others_locations_map: Map = field(init=False)
    readings_map: Map = field(init=False)
    visit_counts_map: Map = field(init=False)
    # TODO insert obstacles map
    
    # Buffers
    buffer: RolloutBuffer = field(default=RolloutBuffer())

    def __post_init__(self):
        '''
        Scaled maps
        '''
        self.x_limit_scaled = int(self.grid_bounds[0] * self.resolution_accuracy)
        self.y_limit_scaled = int(self.grid_bounds[1] * self.resolution_accuracy)
        
        self.location_map: Map = Map([[GridSquare(0.0)] * self.x_limit_scaled] * self.y_limit_scaled)  # TODO rethink this, this is very slow
        self.others_locations_map: Map = Map([[GridSquare(0.0)] * self.x_limit_scaled] * self.y_limit_scaled)  # TODO rethink this, this is very slow
        self.readings_map: Map = Map([[GridSquare(0.0)] * self.x_limit_scaled] * self.y_limit_scaled)  # TODO rethink this, this is very slow
        self.visit_counts_map: Map = Map([[GridSquare(0.0)] * self.x_limit_scaled] * self.y_limit_scaled)  # TODO rethink this, this is very slow
    
    def clear(self):
        del self.location_map[:]
        del self.others_locations_map[:]
        del self.readings_map[:]
        del self.visit_counts_map[:]
        self.buffer.clear()
        
    def state_to_map(self, state):
        # Capture current and reset previous location
        if self.buffer.states:
            last_state = self.buffer.states[-1]
            self.location_map[int(last_state[1])][int(last_state[2])]
            print(self.location_map[int(last_state[1])][int(last_state[2])])
        
        # Set new location
        self.location_map[int(state[1])][int(state[2])] = GridSquare(1) # Convert to Gridsquare datatype
        print(self.location_map[int(state[1])][int(state[2])])
        # Insert state
        
        print(self.buffer.states)
        
        return self.location_map, self.others_locations_map, self.readings_map, self.visit_counts_map


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim: int =5, global_critic: bool=False):
        super(Actor, self).__init__()
        
        ''' Actor Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 4 maps
                2. number of channels: 3 for RGB (red green blue) heatmap 
                3. Height: grid height
                4. Width: grid width
            
                4 maps: 
                    1. Location Map: a 2D matrix showing the agents location.
                    2. Map of Other Locations: a 2D matrix showing the number of agents located in each grid element (excluding current agent).
                    3. Readings map: a 2D matrix showing the last reading collected in each grid element. Grid elements that have not been visited are given a reading of 0.
                    4. Visit Counts Map: a 2D matrix showing the number of visits each grid element has received from all agents combined.
                    
            Critic Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 3 maps
                2. number of channels: 3 for RGB (red green blue) heatmap 
                3. Height: grid height
                4. Width: grid width
                
                4 maps: 
                    1. Location Map: a 2D matrix showing the agents location.
                    2. Map of Other Locations: a 2D matrix showing the number of agents located in each grid element (excluding current agent).
                    3. Readings map: a 2D matrix showing the last reading collected in each grid element. Grid elements that have not been visited are given a reading of 0.
                    4. Visit Counts Map: a 2D matrix showing the number of visits each grid element has received from all agents combined.
        '''
        def test(): 
            # Define the activation function
            #relu = nn.ReLU()
            
            # Define the first convolutional layer
            #conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)  # output tensor with shape (4, 8, 5, 5)
            
            # Define the maxpool layer
            #maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # output tensor with shape (4, 8, 2, 2)

            # Define the second convolution layer
            #conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)  # output tensor with shape (4, 16, 2, 2)

            # Define flattening function
            #flat = lambda x: x.view(x.size(0), -1)  # output tensor with shape (4, 64)
            #flat = lambda x: x.flatten(start_dim=0, end_dim= -1)  # output tensor with shape (1, 256)

            # Define the first linear layer
            #linear1 = nn.Linear(in_features=16*2*2*4, out_features=32) # output tensor with shape (32)
            
            # Define the second linear layer
            # linear2 = nn.Linear(in_features=32, out_features=16) # output tensor with shape (16)
            
            # # Define the output layer
            # output = nn.Linear(in_features=16, out_features=5) # output tensor with shape (5)
            
            # # Define the softmax function
            # softm = nn.Softmax(dim=0)
            
            # Apply the convolutional layer to the input tensor
            # x = relu(conv1(x))  # output tensor with shape (4, 8, 5, 5)
            # print(x.size())
            
            # # Apply maxpool layer
            # x = maxpool(x)  # output tensor with shape (4, 8, 2, 2)
            # print(x.size())
            
            # # Apply the second convolutional layer
            # x = relu(conv2(x))  # output tensor with shape (4, 16, 2, 2)
            # print(x.size())

            # # Flatten the output tensor of the convolutional layer to a 1D tensor
            # x = flat(x)  # output tensor with shape (1, 256)
            # print(x.size())
            
            # # Apply the linear layer to the flattened output tensor
            # x = relu(linear1(x))  # output tensor with shape (32)
            # print(x.size())
            
            # # Apply the second linear layer to the flattened output tensor
            # x = relu(linear2(x))  # output tensor with shape (16)
            # print(x.size())
            
            # # Apply the output layer
            # x = output(x)  # output tensor with shape (5)
            # print(x.size())
            # print(x)
            
            # test = torch.softmax(x, dim=0)
            # print(test.size())
            # print(test)
            
            # x = softm(x)
            # print(x.size())
            # print(x)
            ######################
            pass

        # Actor network
        self.step1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)  # output tensor with shape (4, 8, Height, Width)
        self.relu = nn.ReLU()
        self.step2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output tensor with shape (4, 8, 2, 2)
        self.step3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1) # output tensor with shape (4, 16, 2, 2)
        #nn.ReLU()
        self.step4 = nn.Flatten(start_dim=0, end_dim= -1)  # output tensor with shape (1, 256)
        self.step5 = nn.Linear(in_features=16*2*2*4, out_features=32) # output tensor with shape (32)
        #nn.ReLU()
        self.step6 = nn.Linear(in_features=32, out_features=16) # output tensor with shape (16)
        #nn.ReLU()
        self.step7 = nn.Linear(in_features=16, out_features=5) # output tensor with shape (5) # TODO eventually make '5' action_dim instead
        self.softmax = nn.Softmax()  # Put in range [0,1] 

        # TODO uncomment after ready to combine
        self.actor = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (4, 8, Height, Width)
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # output tensor with shape (4, 8, 2, 2)
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),  # output tensor with shape (4, 16, 2, 2)
                        nn.ReLU(),
                        nn.Flatten(start_dim=0, end_dim= -1),  # output tensor with shape (1, 256)
                        nn.Linear(in_features=16*2*2*4, out_features=32), # output tensor with shape (32)
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=16), # output tensor with shape (16)
                        nn.ReLU(),
                        nn.Linear(in_features=16, out_features=5), # output tensor with shape (5)
                        nn.Softmax()  # Put in range [0,1]
                    )

        # If decentralized critic
        # TODO uncomment after ready to combine
        if not global_critic:
            self.local_critic = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (4, 8, Height, Width)
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # output tensor with shape (4, 8, 2, 2)
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),  # output tensor with shape (4, 16, 2, 2)
                        nn.ReLU(),
                        nn.Flatten(start_dim=0, end_dim= -1),  # output tensor with shape (1, 256)
                        nn.Linear(in_features=16*2*2*4, out_features=32), # output tensor with shape (32)
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=16), # output tensor with shape (16)
                        nn.ReLU(),
                        nn.Linear(in_features=16, out_features=1), # output tensor with shape (1)
                        nn.Softmax()  # Put in range [0,1] TODO Is this needed for critic?
                    )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state_map_stack):
        print("Starting shape, ", state_map_stack.size())
        x = self.step1(state_map_stack) # conv1
        x = self.relu(x)
        print("1st conv shape, ", x.size()) 
        x = self.step2(x) # Maxpool
        x = self.step3(x) # conv2
        x = self.relu(x)
        x = self.step4(x) # Flatten
        x = self.step5(x) # linear
        x = self.relu(x) 
        x = self.step6(x) # linear
        x = self.relu(x)
        x = self.step7(x) # Output layer
        x = self.softmax(x)
        
        print(x)
        
        action_probs = self.actor(state_map_stack)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        #robust_seed = _int_list_from_bigint(hash_seed(seed))[0] # TODO get this to work

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.local_critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, grid_bounds, lr_actor, lr_critic, gamma, K_epochs, eps_clip, resolution_accuracy):
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.resolution_accuracy = resolution_accuracy
        
        # Initialize
        self.maps = MapsBuffer(grid_bounds, resolution_accuracy)

        self.policy = Actor(state_dim, action_dim).to(device) 
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.local_critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = Actor(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # Process state
        state[1] = state[1] * self.resolution_accuracy
        state[2] = state[2] * self.resolution_accuracy
        with torch.no_grad():
            (
                location_map,
                others_locations_map,
                readings_map,
                visit_counts_map
            ) = self.maps.state_to_map(state)
            map_stack = torch.stack([torch.tensor(location_map), torch.tensor(others_locations_map), torch.tensor(readings_map), torch.tensor(visit_counts_map)]) # Convert to tensor
            state = torch.FloatTensor(state).to(device) # Convert to tensor
            
            #action, action_logprob = self.policy_old.act(state) # Choose action
            action, action_logprob = self.policy_old.act(map_stack) # Choose action
        
        self.maps.buffer.states.append(state)
        self.maps.buffer.actions.append(action)
        self.maps.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.maps.buffer.rewards), reversed(self.maps.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.maps.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.maps.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.maps.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions) # Actor-critic

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