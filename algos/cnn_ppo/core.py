import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


import matplotlib.pyplot as plt

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
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


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        obs = obs
        act = act
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        obs = obs
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        act = act
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class CNNSharedNet(nn.Module):
    def __init__(self, observation_space, hidden_sizes):
        super(CNNSharedNet, self).__init__()
        pretrained_CNN = 'resnet'+str(hidden_sizes[0])
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', pretrained_CNN, pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        if observation_space.shape[0] == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_sizes[1])

    def forward(self, x):
        return self.resnet(x)
#class CNNSharedNet(nn.Module):
#    def __init__(self, observation_space, hidden_sizes):
#        super(CNNSharedNet, self).__init__()
#        input_shape = observation_space.shape
#        self.conv = nn.Sequential(
#            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3),
#            nn.ReLU(),
#            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#            nn.ReLU(),
#            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#            nn.ReLU()
#        )
#
#        conv_out_size = self._get_conv_out(observation_space)
#
#        self.linear = nn.Sequential(nn.Linear(conv_out_size, 128),
#                                    nn.ReLU(),
#                                    nn.Linear(128, hidden_sizes[1]))
#
#    def _get_conv_out(self,observation_space):
#        shape = observation_space.shape
#        o = self.conv(torch.zeros(1, *shape))
#        return int(np.prod(o.size()))
#
#    def forward(self, x):
#        x = self.conv(x).view(x.size()[0], -1)
#        return self.linear(x)
class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(18,64,64), activation=nn.Tanh):
        super().__init__()
        # shared network
        self.shared = CNNSharedNet(observation_space, hidden_sizes)
        hidden_sizes.pop(0)
        dummy_obs_dim_to_be_replaced = 1
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(dummy_obs_dim_to_be_replaced, action_space.shape[0], hidden_sizes, activation)
            self.pi.mu_net[0] = self.shared
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(dummy_obs_dim_to_be_replaced, action_space.n, hidden_sizes, activation)
            self.pi.logits_net[0] = self.shared
        # build value function
        self.v  = MLPCritic(dummy_obs_dim_to_be_replaced, hidden_sizes, activation)
        self.v.v_net[0] = self.shared

    def step(self, obs):
        obs = obs
        with torch.no_grad():
            pi = self.pi._distribution(obs.unsqueeze(0))
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            a = a[0]
            v = self.v(obs.unsqueeze(0))
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

