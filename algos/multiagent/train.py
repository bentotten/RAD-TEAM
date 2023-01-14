'''
Originally built from https://github.com/nikhilbarhate99/PPO-PyTorch, however has been merged with RAD-PPO
'''
import os
import sys
import glob
import time
from datetime import datetime

import torch
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np
import numpy.random as npr
import numpy.typing as npt
from typing import Any, List, Literal, NewType, Optional, TypedDict, cast, get_args, Dict, NamedTuple, Type, Union
from typing_extensions import TypeAlias

import gym
from gym_rad_search.envs import rad_search_env # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch, StepResult  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

from vanilla_PPO import PPO as van_PPO # vanilla_PPO
from CNN_PPO import PPO as PPO

from dataclasses import dataclass, field
from typing_extensions import TypeAlias

import copy

from cgitb import reset

import core
from epoch_logger import EpochLogger, EpochLoggerKwargs, setup_logger_kwargs

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

################################### Functions for algorithm/implementation conversions ###################################

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


class AgentStepReturn(NamedTuple):
    action: Union[npt.NDArray, None]
    value:  Union[npt.NDArray, None]
    logprob:  Union[npt.NDArray, None]
    hidden:  Union[torch.Tensor, None]
    out_prediction:  Union[npt.NDArray, None]


class BpArgs(NamedTuple):
    bp_decay: float
    l2_weight: float
    l1_weight: float
    elbo_weight: float
    area_scale: float

@dataclass
class OptimizationStorage:
    train_pi_iters: int        
    train_v_iters: int    
    pi_optimizer: torch.optim
    model_optimizer: torch.optim
    clip_ratio: float
    alpha: float
    target_kl: float
    pi_scheduler: torch.optim.lr_scheduler = field(init=False)
    model_scheduler: torch.optim.lr_scheduler = field(init=False)    
    loss: torch.nn.modules.loss.MSELoss = field(default_factory= (lambda: torch.nn.MSELoss(reduction="mean")))          
        
    def __post_init__(self):        
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, step_size=100, gamma=0.99
        )
        self.model_scheduler = torch.optim.lr_scheduler.StepLR(
            self.model_optimizer, step_size=100, gamma=0.99
        )        


@dataclass
class PPOBuffer:
    obs_dim: core.Shape  # Observation space dimensions
    max_size: int  # Max steps per epoch

    obs_buf: npt.NDArray[np.float32] = field(init=False)
    act_buf: npt.NDArray[np.float32] = field(init=False)
    adv_buf: npt.NDArray[np.float32] = field(init=False)
    rew_buf: npt.NDArray[np.float32] = field(init=False)
    ret_buf: npt.NDArray[np.float32] = field(init=False)
    val_buf: npt.NDArray[np.float32] = field(init=False)
    source_tar: npt.NDArray[np.float32] = field(init=False)
    logp_buf: npt.NDArray[np.float32] = field(init=False)
    obs_win: npt.NDArray[np.float32] = field(init=False) # TODO where is this used?
    obs_win_std: npt.NDArray[np.float32] = field(init=False) # TODO where is this used?

    gamma: float = 0.99
    lam: float = 0.90
    beta: float = 0.005
    ptr: int = 0
    path_start_idx: int = 0

    """
    A buffer for storing histories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __post_init__(self):
        self.obs_buf: npt.NDArray[np.float32] = np.zeros(
            core.combined_shape(self.max_size, self.obs_dim), dtype=np.float32
        )
        self.act_buf: npt.NDArray[np.float32] = np.zeros(
            core.combined_shape(self.max_size), dtype=np.float32
        )
        self.adv_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.rew_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.ret_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.val_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.source_tar: npt.NDArray[np.float32] = np.zeros(
            (self.max_size, 2), dtype=np.float32
        )
        self.logp_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.obs_win: npt.NDArray[np.float32] = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_win_std: npt.NDArray[np.float32] = np.zeros(
            self.obs_dim, dtype=np.float32
        )
        
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

    def store(
        self,
        obs: npt.NDArray[np.float32],
        act: npt.NDArray[np.float32],
        rew: npt.NDArray[np.float32],
        val: npt.NDArray[np.float32],
        logp: npt.NDArray[np.float32],
        src: npt.NDArray[np.float32],
    ) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        obs: observation (Usually the one returned from environment for previous step)
        act: action taken 
        rew: reward from environment
        val: state-value from critic
        logp: log probability from actor
        src: source coordinates
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr, :] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.source_tar[self.ptr] = src
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: int = 0) -> None:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # gamma determines scale of value function, introduces bias regardless of VF accuracy
        # lambda introduces bias when VF is inaccurate
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, logger: EpochLogger) -> dict[str, Union[torch.Tensor, list]]:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf: npt.NDArray[np.float32] = (self.adv_buf - adv_mean) / adv_std
        # ret_mean, ret_std = self.ret_buf.mean(), self.ret_buf.std()
        # self.ret_buf = (self.ret_buf) / ret_std
        # obs_mean, obs_std = self.obs_buf.mean(), self.obs_buf.std()
        # self.obs_buf_std_ind[:,1:] = (self.obs_buf[:,1:] - obs_mean[1:]) / (obs_std[1:])

        epLens: list[int] = logger.epoch_dict["EpLen"]
        numEps = len(epLens)
        epLenTotal = sum(epLens)
        data = dict(
            obs=torch.as_tensor(self.obs_buf, dtype=torch.float32),
            act=torch.as_tensor(self.act_buf, dtype=torch.float32),
            ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
            adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
            logp=torch.as_tensor(self.logp_buf, dtype=torch.float32),
            loc_pred=torch.as_tensor(self.obs_win_std, dtype=torch.float32),
            ep_len=torch.as_tensor(epLenTotal, dtype=torch.float32),
            ep_form = []
        )

        if logger:
            epLenSize = (
                # If they're equal then we don't need to do anything
                # Otherwise we need to add one to make sure that numEps is the correct size
                numEps
                + int(epLenTotal != len(self.obs_buf))
            )
            obs_buf = np.hstack(
                (
                    self.obs_buf,
                    self.adv_buf[:, None],
                    self.ret_buf[:, None],
                    self.logp_buf[:, None],
                    self.act_buf[:, None],
                    self.source_tar,
                )
            )
            epForm: list[list[torch.Tensor]] = [[] for _ in range(epLenSize)]
            slice_b: int = 0
            slice_f: int = 0
            jj: int = 0
            # TODO: This is essentially just a sliding window over obs_buf; use a built-in function to do this
            for ep_i in epLens:
                slice_f += ep_i
                epForm[jj].append(
                    torch.as_tensor(obs_buf[slice_b:slice_f], dtype=torch.float32)
                )
                slice_b += ep_i
                jj += 1
            if slice_f != len(self.obs_buf):
                epForm[jj].append(
                    torch.as_tensor(obs_buf[slice_f:], dtype=torch.float32)
                )

            data["ep_form"] = epForm

        return data


################################### Training ###################################
@dataclass
class PPO:
    env: RadSearch
    logger_kwargs: EpochLoggerKwargs
    seed: int = field(default= 0)
    max_ep_len: int = field(default= 120)
    steps_per_epoch: int = field(default= 4000)
    epochs: int = field(default= 50)
    save_freq: int = field(default= 500)
    render: bool = field(default= False)
    save_gif: bool = field(default= False)
    gamma: float = field(default= 0.99)
    alpha: float = field(default= 0)
    clip_ratio: float = field(default= 0.2)
    pi_lr: float = field(default= 3e-4)
    mp_mm: tuple[int, int] = field(default= (5, 5))
    vf_lr: float = field(default= 5e-3)
    train_pi_iters: int = field(default= 40)
    train_v_iters: int = field(default= 15)
    lam: float = field(default= 0.9)
    number_of_agents: int = field(default= 1)
    target_kl: float = field(default= 0.07)
    ac_kwargs: dict[str, Any] = field(default_factory= lambda: dict())
    actor_critic: Type[core.RNNModelActorCritic] = field(default=core.RNNModelActorCritic)
    start_time: float = field(default_factory= lambda: time.time())
    
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Base code from OpenAI:
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    Args:
        env : An environment satisfying the OpenAI Gym API.
        
        logger_kwargs: Arguments for the logging mechanism for saving models and saving/printing progress for each agent
        
        seed (int): Seed for random number generators.
        
        max_ep_len (int): Maximum length of trajectory / episode / rollout / steps.        

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of total epochs of interaction (equivalent to
            number of policy updates) to perform.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        render (bool): Indicates whether to render last episode
        
        save_gif (bool): Indicates whether to save render of last episode

        gamma (float): Discount factor for calculating expected return. (Always between 0 and 1.)
        
        alpha (float): Entropy reward term scaling.        

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. Basically if the policy wants to perform too large
            an update, it goes with a clipped value instead.

        pi_lr (float): Learning rate for Actor/policy optimizer.

        vf_lr (float): Learning rate for Critic/state-value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on actor policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            critic state-value function per epoch.

        lam (float): Lambda for GAE-Lambda advantage estimator calculations. (Always between 0 and 1,
            close to 1.)

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.) This is a part of PPO.        

        ac_kwargs (dict): Any kwargs appropriate for the Actor-Critic object
            provided to PPO.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                        | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                        | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                        | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                        | a batch of distributions describing
                                        | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                        | actions is given). Tensor containing
                                        | the log probability, according to
                                        | the policy, of the provided actions.
                                        | If actions not given, will contain
                                        | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                        | for the provided observations. (Critical:
                                        | make sure to flatten this!)
            ===========  ================  ======================================
    """
    def __post_init__(self):          
        # Set Pytorch random seed
        torch.manual_seed(self.seed)

        # Set additional Actor-Critic variables
        self.ac_kwargs["seed"] = self.seed
        self.ac_kwargs["pad_dim"] = 2        

        # Set arguments for bootstrap particle filter in the Particle Filter Gated Recurrent Unit (PFGRU) for the source prediction neural networks, from Ma et al. 2020
        self.bp_args = BpArgs(
            bp_decay=0.1,
            l2_weight=1.0,
            l1_weight=0.0,
            elbo_weight=1.0,
            area_scale=self.env.search_area[2][
                1
            ],  # retrieves the height of the created environment
        )

        # Instantiate environment 
        self.obs_dim: int = self.env.observation_space.shape[0]
        self.act_dim: int = rad_search_env.A_SIZE
        self.save_gif_freq = self.epochs // 3        

        # Instantiate Actor-Critic (A2C) Agents
        #self.ac = actor_critic(obs_dim, act_dim, **ac_kwargs)
        self.agents = {i: self.actor_critic(self.obs_dim, self.act_dim, **self.ac_kwargs) for i in range(self.number_of_agents)}
        for agent in self.agents.values():
            agent.model.eval() # Sets PFGRU model into "eval" mode # TODO why not in the episode with the other agents?   
        
        # Count variables for actor/policy (pi) and PFGRU (model)
        # TODO rename all of these to make them consistent :V 
        self.pi_var_count, self.model_var_count = {}, {}
        for id, ac in self.agents.items():
            self.pi_var_count[id], self.model_var_count[id] = (
                core.count_vars(module) for module in [ac.pi, ac.model]
            )
        
        # Set up PPO trajectory buffers. This stores values to be later used for updating each agent after the conclusion of an epoch
        self.agent_buffers = {
            i: PPOBuffer(
                    obs_dim=self.obs_dim, max_size=self.steps_per_epoch, gamma=self.gamma, lam=self.lam,
                )
                for i in range(self.number_of_agents)
            }
        self.buf = self.agent_buffers # Adding for backwards compatibility # TODO verify .buf has been replaced and delete

        # Set up optimizers and learning rate decay for policy and localization modules in each agent. 
        #  Pi_scheduler and model_scheduler are set up in initialization.
        self.agent_optimizers = {
                i: OptimizationStorage(
                    train_pi_iters = self.train_pi_iters,                
                    train_v_iters = self.train_v_iters,                
                    pi_optimizer = Adam(self.agents[i].pi.parameters(), lr=self.pi_lr),
                    model_optimizer = Adam(self.agents[i].model.parameters(), lr=self.vf_lr),
                    loss = torch.nn.MSELoss(reduction="mean"),
                    clip_ratio = self.clip_ratio,
                    alpha = self.alpha,
                    target_kl = self.target_kl,             
                    )
                for i in range(self.number_of_agents)
            }
        
        # Setup statistics buffers for normalizing returns from environment
        self.stat_buffers = {i: core.StatBuff() for i in range(self.number_of_agents)}
        self.reduce_v_iters = True  # Reduces training iteration when further along to speed up training
                
        # Instatiate loggers and set up model saving
        logger_kwargs_set = {
            id: setup_logger_kwargs(
                exp_name=f"{id}_{self.logger_kwargs['exp_name']}",
                seed=self.logger_kwargs['seed'],
                data_dir=self.logger_kwargs['data_dir'],
                env_name=self.logger_kwargs['env_name']
            ) for id in self.agents
        }
        self.logger = {id: EpochLogger(**(logger_kwargs_set[id])) for id in self.agents}
        
        for id in self.agents:
            self.logger[id].save_config(locals())  # TODO THIS PICKLE DEPENDS ON THE DIRECTORY STRUCTURE!! Needs rewrite!      
            self.logger[id].log(
                f"\nNumber of parameters: \t actor policy (pi): {self.pi_var_count[0]}, particle filter gated recurrent unit (model): {self.model_var_count[0]} \t"
            )
            self.logger[id].setup_pytorch_saver(self.agents[id])

    def train(self):
        # Prepare environment and get initial values
        env = self.env
        source_coordinates = np.array(self.env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
        episode_return = {id: 0 for id in self.agents}
        episode_return_buffer = []  # TODO can probably get rid of this, unless want to keep for logging
        out_of_bounds_count = np.zeros(self.number_of_agents) # TODO consider changing to dict for consistency
        success_count = 0
        steps_in_episode = 0
        
        # Obsertvations aka States: for each agent, 11 dimensions: [intensity reading, x coord, y coord, 8 directions of distance detected to obstacle]
        observations, _,  _, _ = env.reset()  

        # Update stat buffers for all agent observations for later observation normalization
        for id in self.agents:
            self.stat_buffers[id].update(observations[id][0])
        
        # Removed features - migrating to pytorch lightning instead of mpi (God willing)
        #local_steps_per_epoch = int(steps_per_epoch / num_procs())    

        print(f"Starting main training loop!", flush=True)
        self.start_time = time.time()        
        # For a total number of epochs, Agent will choose an action using its policy and send it to the environment to take a step in it, yielding a new state observation.
        #   Agent will continue doing this until the episode concludes; a check will be done to see if Agent is at the end of an epoch or not - if so, the agent will use 
        #   its buffer to update/train its networks. Sometimes an epoch ends mid-episode - there is a finish_path() function that addresses this.
        for epoch in range(self.epochs):
            
            # Reset hidden states and sets Actor into "eval" mode 
            hidden = {id: ac.reset_hidden() for id, ac in self.agents.items()}
            for ac in self.agents.values():
                ac.pi.logits_net.v_net.eval() # TODO should the pfgru call .eval also?
            
            for t in range(self.steps_per_epoch):
                # Standardize prior observation of radiation intensity for the actor-critic input using running statistics per episode
                standardized_observations = {id: observations[id] for id in self.agents}
                for id in self.agents:
                    standardized_observations[id][0] = np.clip((observations[id][0] - self.stat_buffers[id].mu) / self.stat_buffers[id].sig_obs, -8, 8)     
                    
                # Actor: Compute action and logp (log probability); Critic: compute state-value
                # a, v, logp, hidden, out_pred = self.ac.step(obs_std, hidden=hidden) # TODO make multi-agent # TODO what is the hidden variable doing?                
                agent_thoughts = {id: None for id in self.agents}
                for id, agent in self.agents.items():
                    action, value, logprob, hidden, out_prediction = agent.step(standardized_observations[id], hidden=hidden[id])
                    
                    agent_thoughts[id] = AgentStepReturn(
                        action=action, value=value, logprob=logprob, hidden=hidden, out_prediction=out_prediction
                    )
                
                # Create action list to send to environment
                agent_action_decisions = {id: int(agent_thoughts[id].action.item()) for id, action in agent_thoughts.items()} 
                
                # TODO the above does not include idle action. After working, add an additional state space for 9 potential actions and uncomment:                 
                #agent_action_decisions = {id: int(action)-1 for id, action in agent_thoughts.items()} 
                
                # Ensure no item is above 7 or below -1
                for action in agent_action_decisions.values():
                    assert -1 <= action and action < 8            
                
                # Take step in environment
                #StepResult(observation=aggregate_observation_result, reward=aggregate_reward_result, success=aggregate_success_result, info=aggregate_info_result)
                observations, rewards, terminals, infos = env.step(action=1)
                
                # Incremement Counters and save new (individual) cumulative returns
                for id in rewards:
                    episode_return[id] += rewards[id]
                episode_return_buffer.append(episode_return)
                steps_in_episode += 1    

                for id, buffer in self.agent_buffers.items():
                    act: npt.NDArray[np.int32] = agent_action_returns[id].action           
                    val: npt.NDArray[np.float32] = agent_action_returns[id].state_value      
                    logp: npt.NDArray[np.float32] = agent_action_returns[id].action_logprob
                    src: npt.NDArray[np.float32] = source_coordinates                    
                
                    agent.store(
                        obs = observations[id],
                        rew = rewards[id],
                        terminal = terminals[id],                        
                        act = agent_thoughts['actions'],
                        val = val,
                        logp = logp,
                        src = src,
                    )

                self.buf.store(obs_std, a, r, v, logp, source_coordinates) # Feed prior observation to buffer # TODO make multi-agent?
                logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                # Update running mean and std
                stat_buff.update(o[0])

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if d and not timeout:
                        done_count += 1
                    #if env.out_of_bounds:
                    # Artifact - TODO decouple from rad_ppo agent
                    if 'out_of_bounds' in msg and msg['out_of_bounds'] == True:
                        # Log if agent went out of bounds
                        oob += 1
                    if epoch_ended and not (terminal):
                        print(
                            f"Warning: trajectory cut off by epoch at {ep_len} steps and time {t}.",
                            flush=True,
                        )

                    if timeout or epoch_ended:
                        # if trajectory didn't reach terminal state, bootstrap value target
                        obs_std[0] = np.clip(
                            (o[0] - stat_buff.mu) / stat_buff.sig_obs, -8, 8
                        )
                        _, v, _, _, _ = self.ac.step(obs_std, hidden=hidden) # TODO make multi-agent
                        if epoch_ended:
                            # Set flag to sample new environment parameters
                            env.epoch_end = True # TODO make multi-agent?
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)

                    if (
                        epoch_ended
                        and render
                        and (epoch % save_gif_freq == 0 or ((epoch + 1) == epochs))
                    ):
                        # Check agent progress during training
                        if epoch != 0:
                            env.render(
                                save_gif=save_gif,
                                path=str(logger.output_dir),
                                epoch_count=epoch,
                                #ep_rew=ep_ret_ls,
                            )

                    ep_ret_ls = []
                    stat_buff.reset()
                    if not env.epoch_end: # TODO make multi-agent
                        # Reset detector position and episode tracking
                        hidden = self.ac.reset_hidden() # TODO make multi-agent
                        o, ep_ret, ep_len, a = env.reset()[0].state, 0, 0, -1 # TODO make multi-agent
                        source_coordinates = np.array(env.src_coords, dtype="float32")
                    else:
                        # Sample new environment parameters, log epoch results
                        if 'out_of_bounds_count' in msg:
                            oob += msg['out_of_bounds_count']
                        logger.store(DoneCount=done_count, OutOfBound=oob)
                        done_count = 0
                        oob = 0
                        o, ep_ret, ep_len, a = env.reset()[0].state, 0, 0, -1 # TODO make multi-agent
                        source_coordinates = np.array(env.src_coords, dtype="float32")
                        
                    stat_buff.update(o[0])

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({}, None)
                pass

            # Reduce localization module training iterations after 100 epochs to speed up training
            if reduce_v_iters and epoch > 99:
                self.train_v_iters = 5
                reduce_v_iters = False

            # Perform PPO update!
            self.update(env, bp_args) # TODO make multi-agent

            # Log info about epoch
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.log_tabular("VVals", with_min_and_max=True)
            self.logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
            self.logger.log_tabular("LossPi", average_only=True)
            self.logger.log_tabular("LossV", average_only=True)
            self.logger.log_tabular("LossModel", average_only=True)
            self.logger.log_tabular("LocLoss", average_only=True)
            self.logger.log_tabular("Entropy", average_only=True)
            self.logger.log_tabular("KL", average_only=True)
            self.logger.log_tabular("ClipFrac", average_only=True)
            self.logger.log_tabular("DoneCount", sum_only=True)
            self.logger.log_tabular("OutOfBound", average_only=True)
            self.logger.log_tabular("StopIter", average_only=True)
            self.logger.log_tabular("Time", time.time() - start_time)
            self.logger.dump_tabular()
                        

def train_scaffolding():
    print("============================================================================================")

    # Using this to fold setup in VSCode
    if True:
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
        
        #################################################torchinfo####

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
            max_ep_len = 5                      # max timesteps in one episode # TODO delete me after fixing
            steps_per_epoch = 5
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
            resolution_accuracy = 0.01 * 1/env.scale  # Less accurate
            #resolution_accuracy = 1 * 1/env.scale   # More accurate
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
                agent.update()
        
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
