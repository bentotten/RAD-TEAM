'''
Implementation of "Target Localization using Multi-Agent Deep Reinforcement Learning with Proximal Policy Optimization" by Alagha et al.

'''
from os import stat, path, mkdir, getcwd
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
from typing import Any, List, Tuple, Union, Literal, NewType, Optional, TypedDict, cast, get_args, Dict
from typing_extensions import TypeAlias

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Maps
Point: TypeAlias = NewType("Point", tuple[float, float])  # Array indicies to access a GridSquare
Map: TypeAlias = NewType("Map", npt.NDArray[np.float32]) # 2D array that holds gridsquare values

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
    mapstacks: List = field(default_factory=list) #TODO change to tensor?

    readings: Dict[Any, list] = field(default_factory=dict)
    
    def __post_init__(self):
        pass
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
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
    
    # Maps
    location_map: Map = field(init=False)
    others_locations_map: Map = field(init=False)
    readings_map: Map = field(init=False)
    visit_counts_map: Map = field(init=False)
    # TODO insert obstacles map
    
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
            self.readings_map[x][y] += estimated_reading  # Initial agents begin at same location

        # Process state for visit_counts_map
        for agent_id in observation:
            scaled_coordinates = (int(observation[agent_id].state[1] * self.resolution_accuracy), int(observation[agent_id].state[2] * self.resolution_accuracy))            
            x = int(scaled_coordinates[0])
            y = int(scaled_coordinates[1])

            self.visit_counts_map[x][y] += 1
        
        return self.location_map, self.others_locations_map, self.readings_map, self.visit_counts_map


class Actor(nn.Module):
    def __init__(self, map_dim, state_dim, batches: int=1, action_dim: int=5, global_critic: bool=False):
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

        assert map_dim[0] > 0 and map_dim[0] == map_dim[1], 'Map dimensions mismatched. Must have equal x and y bounds.'
        
        pool_output = int(((map_dim[0]-2) / 2) + 1) # Get maxpool output height/width and floor it

        # Actor network
        self.step1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)  # output tensor with shape (batchs, 8, Height, Width)
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

        # TODO uncomment after ready to combine
        self.actor = nn.Sequential(
                        nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (4, 8, Height, Width)
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
        if not global_critic:
            self.local_critic = nn.Sequential(
                        # Starting shape (batch_size, 4, Height, Width)
                        nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (batch_size, 8, Height, Width)
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
                        nn.Softmax(dim=0)  # Put in range [0,1] TODO Is this needed for critic?
                    )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state_map_stack):
        # print("Starting shape, ", state_map_stack.size())
        # x = self.step1(state_map_stack) # conv1
        # x = self.relu(x)
        # print("shape, ", x.size()) 
        # x = self.step2(x) # Maxpool
        # print("shape, ", x.size()) 
        # x = self.step3(x) # conv2
        # x = self.relu(x)
        # print("shape, ", x.size()) 
        # x = self.step4(x) # Flatten
        # print("shape, ", x.size()) 
        # x = self.step5(x) # linear
        # x = self.relu(x) 
        # print("shape, ", x.size()) 
        # x = self.step6(x) # linear
        # x = self.relu(x)
        # print("shape, ", x.size()) 
        # x = self.step7(x) # Output layer
        # print("shape, ", x.size()) 
        # x = self.softmax(x)
        
        # print(x)
        
        action_probs = self.actor(state_map_stack)
        # print(x)
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
    def __init__(self, state_dim, action_dim, grid_bounds, lr_actor, lr_critic, gamma, K_epochs, eps_clip, resolution_accuracy, id):
        # Hyperparameters
        self.gamma = gamma  # Discount factor
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
                visit_counts_map
            ) = self.maps.state_to_map(state, id)
            
            map_stack = torch.stack([torch.tensor(location_map), torch.tensor(others_locations_map), torch.tensor(readings_map), torch.tensor(visit_counts_map)]) # Convert to tensor
            
            # Add to mapstack buffer to eventually be converted into tensor with minibatches
            self.maps.buffer.mapstacks.append(map_stack)
            
            # Add single minibatch for action selection
            map_stack = torch.unsqueeze(map_stack, dim=0) 
            
            #state = torch.FloatTensor(state).to(device) # Convert to tensor TODO already a tensor, is this necessary?
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

        # TODO for memory efficiency, could remap state buffers here instead of storing them

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.maps.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.maps.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.maps.buffer.logprobs, dim=0)).detach().to(device)
        
        ### DELETE 
        print(self.maps.buffer.mapstacks)
        print(torch.max(self.maps.buffer.mapstacks[0]))
        print(torch.max(self.maps.buffer.mapstacks[1]))
        print(torch.min(self.maps.buffer.mapstacks[0]))
        print(torch.min(self.maps.buffer.mapstacks[1]))
        
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
        
    def render(self, savepath=getcwd(), save_map=True, add_value_text=False, interpolation_method='nearest'):  
        if save_map:
            if not path.isdir(str(savepath) + "/heatmaps/"):
                mkdir(str(savepath) + "/heatmaps/")
        else:
            plt.show()                
     
        fig, (loc_ax, other_ax, intensity_ax, visit_ax) = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
        
        loc_ax.imshow(self.maps.location_map, cmap='viridis', interpolation=interpolation_method)
        loc_ax.set_title('Agent Location')
        
        other_ax.imshow(self.maps.others_locations_map, cmap='viridis', interpolation=interpolation_method)
        other_ax.set_title('Other Agent Locations') 
        
        intensity_ax.imshow(self.maps.readings_map, cmap='viridis', interpolation=interpolation_method)
        intensity_ax.set_title('Radiation Intensity') 
        
        visit_ax.imshow(self.maps.visit_counts_map, cmap='viridis', interpolation=interpolation_method)
        visit_ax.set_title('Visit Counts') 
        
        #divider = make_axes_locatable(loc_ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)             
        #fig.colorbar(loc_ax, cax=cax, orientation='vertical')
        
        # Add values to gridsquares if value is greater than 0 #TODO if large grid, this will be slow
        if add_value_text:
            for i in range(self.maps.location_map.shape[0]):
                for j in range(self.maps.location_map.shape[1]):
                    if self.maps.location_map[i, j] > 0: 
                        loc_ax.text(j, i, self.maps.location_map[i, j].astype(int), ha="center", va="center", color="b", size=6)
                    if self.maps.others_locations_map[i, j] > 0: 
                        other_ax.text(j, i, self.maps.others_locations_map[i, j].astype(int), ha="center", va="center", color="b", size=6)
                    if self.maps.readings_map[i, j] > 0:
                        intensity_ax.text(j, i, self.maps.readings_map[i, j].astype(int), ha="center", va="center", color="b", size=6)
                    if self.maps.visit_counts_map[i, j] > 0:
                        visit_ax.text(j, i, self.maps.visit_counts_map[i, j].astype(int), ha="center", va="center", color="b", size=6)
        
        fig.savefig(f'{str(savepath)}/heatmaps/agent{self.id}_heatmaps_{self.render_counter}.png')
        
        self.render_counter += 1
        plt.close(fig)
        
        def foo(self, savepath):
            # # fig
            # # loc_map = plt.imshow(self.maps.location_map, cmap='hot', interpolation='nearest')
            # # loc_map.colorbar()  
            # # loc_map.savefig
            # # plt.close()

            # plt.imshow(self.maps.others_locations_map, cmap='hot', interpolation='nearest')
            # plt.colorbar()  
            # plt.savefig(f'{str(savepath)}/heatmaps/others_map_{self.render_counter}.png')
            # plt.close()
            
            # plt.imshow(self.maps.readings_map, cmap='hot', interpolation='nearest')
            # plt.colorbar()  
            # plt.savefig(f'{str(savepath)}/heatmaps/readings_map_{self.render_counter}.png')
            # plt.close()

            # plt.imshow(self.maps.visit_counts_map, cmap='hot', interpolation='nearest')
            # plt.colorbar()  
            # plt.savefig(f'{str(savepath)}/heatmaps/visit_counts_map_{self.render_counter}.png')
            # plt.close()        
            
            print("rendered")
            self.render_counter += 1
            #exit()             
