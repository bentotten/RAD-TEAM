import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import pytorch_lightning as pl

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
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        # TODO add buffer for history
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        # TODO add buffer for history


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, global_critic: bool=False):
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
                        nn.Linear(in_features=16, out_features=5), # output tensor with shape (5)
                        nn.Softmax()  # Put in range [0,1]
                    )
        
    def set_action_std(self, new_action_std):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling Actor::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        raise NotImplementedError


    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs = self.actor(state)
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
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = Actor(state_dim, action_dim, action_std_init).to(device) 
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.local_critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = Actor(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        raise NotImplementedError


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        raise NotImplementedError

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state) # Actor-critic
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

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
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path) # Actor-critic
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic