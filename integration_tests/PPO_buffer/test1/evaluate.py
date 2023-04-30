"""
Evaluate agents and update neural networks using simulation environment.
"""

import os
import sys
import time
import random
from datetime import datetime
import math
from statsmodels.stats.weightstats import DescrStatsW  # type: ignore

import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import numpy.random as npr
import numpy.typing as npt

from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params,synchronize, mpi_avg_grads, sync_params_stats # type: ignore
from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,mpi_statistics_vector, num_procs, mpi_min_max_scalar # type: ignore

from typing import (
    Any,
    List,
    Literal,
    NewType,
    Optional,
    TypedDict,
    cast,
    get_args,
    Dict,
    NamedTuple,
    Type,
    Union,
    Tuple,
)
from typing_extensions import TypeAlias
from dataclasses import dataclass, field

import json
import joblib  # type: ignore
import ray

# Simulation Environment
import gym  # type: ignore
from gym_rad_search.envs import rad_search_env  # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch, StepResult  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore


# Neural Networks
import core as RADA2C_core  # type: ignore


# Helpful functions
def median(data: List) -> np.float32:
    return np.median(data) if len(data) > 0 else np.nan


def variance(data: List) -> np.float32:
    return np.var(np.array(data) / len(data)) if len(data) > 0 else np.nan


@dataclass
class Results:
    episode_length: List[int] = field(default_factory=lambda: list())
    episode_return: List[float] = field(default_factory=lambda: list())
    intensity: List[float] = field(default_factory=lambda: list())
    background_intensity: List[float] = field(default_factory=lambda: list())
    success_count: List[int] = field(default_factory=lambda: list())


@dataclass
class MonteCarloResults:
    id: int
    completed_runs: int = field(init=False, default=0)
    successful: Results = field(default_factory=lambda: Results())
    unsuccessful: Results = field(default_factory=lambda: Results())
    total_episode_length: List[int] = field(default_factory=lambda: list())
    success_counter: int = field(default=0)
    total_episode_return: List[float] = field(default_factory=lambda: list())


@dataclass
class Metrics:
    medians: List = field(default_factory=lambda: list())
    variances: List = field(default_factory=lambda: list())


@dataclass
class Distribution:
    unique: Dict = field(default_factory=lambda: dict())
    counts: Dict = field(default_factory=lambda: dict())


# Uncomment when ready to run with Ray
# @ray.remote
@dataclass
class EpisodeRunner:

    id: int
    # env_sets: Dict
    current_dir: str

    env_name: str
    env_kwargs: Dict
    steps_per_episode: int
    team_mode: str
    resolution_multiplier: float

    render: bool
    save_gif_freq: int
    render_path: str

    model_path: str
    save_path_for_ac: str
    test_env_path: str = field(default="./evaluation/test_environments")
    save_path: str = field(default=".")
    seed: Union[int, None] = field(default=0)

    obstruction_count: int = field(default=0)
    enforce_boundaries: bool = field(default=False)
    actor_critic_architecture: str = field(default="cnn")
    number_of_agents: int = field(default=1)
    episodes: int = field(default=100)
    montecarlo_runs: int = field(default=100)
    snr: str = field(default="high")

    render_first_episode: bool = field(default=True)
    env_dict: Dict = field(default_factory=lambda: dict())    

    # Initialized elsewhere
    #: Object that holds agents
    agents: Dict[int, RADA2C_core.RNNModelActorCritic] = field(default_factory=lambda: dict())

    def __post_init__(self) -> None:
        # Change to correct directory
        os.chdir(self.current_dir)


        # Create own instatiation of environment
        self.env: rad_search_env = self.create_environment()

        # Get agent model paths and saved agent configurations
        agent_models = {}
        for child in os.scandir(self.model_path):
            if child.is_dir() and "agent" in child.name:
                agent_models[int(child.name[0])] = (child.path)  # Read in model path by id number. NOTE: Important that ID number is the first element of file name
            if child.is_dir() and "general" in child.name:
                general_config_path = child.path

        obj = json.load(open(f"{self.model_path}/config.json"))
        if 'self' in obj.keys():
            original_configs = list(obj["self"].values())[0]["ppo_kwargs"]["actor_critic_args"]
        else:
            original_configs = obj["ac_kwargs"] # Original project save format

        # Set up static A2C actor-critic args
        actor_critic_args = dict(
            hidden_sizes_pol=[[32]],
            hidden_sizes_val=[[32]],
            hidden_sizes_rec=[24],
            hidden=[[24]],
            net_type="rnn",
            batch_s=1,
        )

        if self.actor_critic_architecture != "cnn":
            assert self.team_mode == "individual"  # No global critic for RAD-A2C

        # Check current important parameters match parameters read in
        for arg in actor_critic_args:
            if arg != "no_critic" and arg != "GlobalCritic" and arg != "save_path":
                if (
                    type(original_configs[arg]) == int
                    or type(original_configs[arg]) == float
                    or type(original_configs[arg]) == bool
                ):
                    assert ( actor_critic_args[arg] == original_configs[arg]), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is str:
                    if arg == "net_type":
                        assert actor_critic_args[arg] == original_configs[arg]
                    else:
                        to_list = original_configs[arg].strip("][").split(" ")
                        config = np.array([float(x) for x in to_list], dtype=np.float32)
                        assert np.array_equal(config, actor_critic_args[arg]), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is list:
                    for a, b in zip(original_configs[arg], actor_critic_args[arg]):
                        assert (a == b), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                else:
                    assert (actor_critic_args[arg] == original_configs[arg]), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"

        # Initialize agents and load agent models
        actor_critic_args['observation_space'] = self.env.observation_space 
        actor_critic_args['action_space'] = self.env.action_space
        actor_critic_args['seed'] = self.seed
        actor_critic_args['pad_dim'] = 2

        self.agents[0] = RADA2C_core.RNNModelActorCritic(**actor_critic_args)
        self.agents[0].load_state_dict(torch.load('model.pt'))                     
                  

    def run(self) -> MonteCarloResults:
        # Prepare tracking buffers and counters
        episode_return: Dict[int, float] = {id: 0.0 for id in self.agents}
        steps_in_episode: int = 0
        terminal_counter: Dict[int, int] = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)
        run_counter = 0

        # Prepare results buffers
        results = MonteCarloResults(id=self.id)
        # For RAD-A2C Compatibility
        stat_buffers = dict()

        # Reset environment and save test env parameters
        observations, _, _, _  = self.env.reset()

        # Save env for refresh
        self.env_dict['env_0'] = [_ for _ in range(5)]
        self.env_dict['env_0'][0] = self.env.src_coords
        self.env_dict['env_0'][1] = self.env.agents[0].det_coords
        self.env_dict['env_0'][2] = self.env.intensity
        self.env_dict['env_0'][3] = self.env.bkg_intensity
        self.env_dict['env_0'][4] = [] # obstructions

        self.agents[0].pi.eval()
        self.agents[0].model.eval()

        # Prepare episode variables
        agent_thoughts: Dict = dict()
        hiddens: Dict[int, Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None]] = {id: self.agents[id].reset_hidden() for id in self.agents}  # For RAD-A2C compatibility

        initial_prediction = np.zeros((3,))
        for id, ac in self.agents.items():
            stat_buffers[id] = RADA2C_core.StatBuff()
            stat_buffers[id].update(observations[id][0])
            observations[id][0] = np.clip((observations[id][0]-stat_buffers[id].mu)/stat_buffers[id].sig_obs,-8,8)

        while run_counter < self.montecarlo_runs:
            # Get agent thoughts on current state. Actor: Compute action and logp (log probability); Critic: compute state-value
            agent_thoughts.clear()
            agent_thoughts[id] = {}
            for id, ac in self.agents.items():
                with torch.no_grad():
                    a, v, logp, hidden, out_pred = ac.step(observations[id], hiddens[id])
                    agent_thoughts[id]['action'] = a


                hiddens[id] = hidden
            # Create action list to send to environment
            agent_action_decisions = {id: int(agent_thoughts[id]['action']) for id in agent_thoughts}
            for action in agent_action_decisions.values():
                assert 0 <= action and action < int(self.env.number_actions)

            # Take step in environment - Note: will be missing last reward, rewards link to previous observation in env
            observations, rewards, terminals, _ = self.env.step(action=agent_action_decisions)

            for id, ac in self.agents.items():
                stat_buffers[id].update(observations[id][0])
                observations[id][0] = np.clip((observations[id][0]-stat_buffers[id].mu)/stat_buffers[id].sig_obs,-8,8)

            # Incremement Counters and save new (individual) cumulative returns
            for id in range(self.number_of_agents):

                episode_return[id] += np.array(rewards["individual_reward"][id], dtype="float32").item()

            steps_in_episode += 1

            # Tally up ending conditions
            # Check if there was a terminal state. Note: if terminals are introduced that only affect one agent but not all, this will need to be changed.
            terminal_reached_flag = False
            for id in terminal_counter:
                if terminals[id] == True and not timeout:
                    terminal_counter[id] += 1
                    terminal_reached_flag = True

            # Stopping conditions for episode
            timeout: bool = (steps_in_episode == self.steps_per_episode)  # Max steps per episode reached
            episode_over: bool = (terminal_reached_flag or timeout)  # Either timeout or terminal found

            if episode_over:
                self.process_render(run_counter=run_counter, id=self.id)

                # Save results
                if run_counter < 1:
                    if terminal_reached_flag:
                        results.successful.intensity.append(self.env.intensity)
                        results.successful.background_intensity.append(self.env.bkg_intensity)
                    else:
                        results.unsuccessful.intensity.append(self.env.intensity)
                        results.unsuccessful.background_intensity.append(self.env.bkg_intensity)
                results.total_episode_length.append(steps_in_episode)

                if terminal_reached_flag:
                    results.success_counter += 1
                    results.successful.episode_length.append(steps_in_episode)
                    results.successful.episode_return.append(episode_return[0])  # TODO change for individual mode                    
                else:
                    results.unsuccessful.episode_length.append(steps_in_episode)
                    results.unsuccessful.episode_return.append(episode_return[0])  # TODO change for individual mode

                results.total_episode_return.append(episode_return[0])  # TODO change for individual mode

                # Incremenet run counter
                run_counter += 1

                # Reset environment without performing an env.reset()
                episode_return = {id: 0.0 for id in self.agents}
                steps_in_episode = 0
                terminal_counter = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)

                observations = self.env.refresh_environment(env_dict=self.env_dict, id=0, num_obs=self.obstruction_count)

                # Reset stat buffer for RAD-A2C
                for id, ac in self.agents.items():
                    stat_buffers[id].reset()
                    stat_buffers[id].update(observations[id][0])
                    observations[id][0] = np.clip((observations[id][0]-stat_buffers[id].mu)/stat_buffers[id].sig_obs,-8,8)

        results.completed_runs = run_counter

        print(f"Finished episode {self.id}! Success count: {results.success_counter} out of {self.montecarlo_runs}")
        return results

    def create_environment(self) -> RadSearch:
        env = gym.make(self.env_name, **self.env_kwargs)
        env.reset()
        return env

    def getattr(self, attr):
        return getattr(self, attr)

    def say_hello(self):
        return self.id

    def process_render(self, run_counter: int, id: int) -> None:
        # Render
        save_time_triggered = (
            (run_counter % self.save_gif_freq == 0)
            if self.save_gif_freq != 0
            else False
        )
        time_to_save = save_time_triggered or (
            (run_counter + 1) == self.montecarlo_runs
        )
        if self.render and time_to_save:
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=True,
            )
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=True,
            )
        # Always render first episode
        if self.render and run_counter == 0 and self.render_first_episode:
          
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=True,
            )
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=True,
            )
            self.render_first_episode = False

        # Always render last episode
        if self.render and run_counter == self.montecarlo_runs - 1:
         
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=True,
            )
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=True,
            )


@dataclass
class evaluate_PPO:
    """
    Test existing model across random episodes for a set number of monte carlo runs per episode.
    """

    eval_kwargs: Dict

    # Initialized elsewhere
    #: Directory containing test environments. Each test environment file contains 1000 environment configurations.
    test_env_dir: str = field(init=False)
    #: Full path to file containing chosen test environment. Each test environment file contains 1000 environment configurations.
    test_env_path: str = field(init=False)
    #: Sets of environments for specifications. Comes in sets of 1000.
    environment_sets: Dict = field(init=False)
    #: runners
    runners: Dict = field(init=False)
    #: Number of monte carlo runs per episode configuration
    montecarlo_runs: int = field(init=False)

    def __post_init__(self) -> None:
        self.montecarlo_runs = self.eval_kwargs["montecarlo_runs"]

    def evaluate(self):
        """Driver"""
        start_time = time.time()

        self.runners = {
            i: EpisodeRunner(id=i, current_dir=os.getcwd(), **self.eval_kwargs)
            for i in range(self.eval_kwargs["episodes"])
        }
        
        full_results = [runner.run() for runner in self.runners.values()]

        print("Runtime: {}", time.time() - start_time)

        self.calc_stats(results=full_results)

        pass
    
    def calc_stats(self, results, mc=None):
        """
        Calculate results from the evaluation. Performance is determined by accuracy and speed.
        Accuracy: Median value of the count of successully found sources from each episode configuration. 
        Speed: The median value of all epsiode lengths from all successful runs.
        After review, for only 1000 values, a weighted median does not add enough benefit to warrant reimplementation.
        """
        if not mc:
            mc = self.montecarlo_runs
        
        success_counts = np.zeros(len(results))
        successful_episode_lengths = np.empty(len(results) * mc)
        successful_episode_lengths[:] = np.nan
        episode_returns = np.zeros(len(results) * mc)
        
        # Pointer for episode lengths, as shape is not known ahead of time
        ep_start_ptr = 0

        for ep_index, episode in enumerate(results):
            success_counts[ep_index] = episode.success_counter
            episode_returns[ep_index : ep_index + mc] = episode.total_episode_return
            
            successful_episode_lengths[ep_start_ptr : ep_start_ptr + len(episode.successful.episode_length)] = episode.successful.episode_length[:]
            ep_start_ptr = ep_start_ptr + len(episode.successful.episode_length)
            
        success_counts_median = np.median(sorted(success_counts))
        success_lengths_median = np.median(sorted(successful_episode_lengths))
        return_medidan = np.median(sorted(episode_returns))
        
        succ_std = round(np.std(success_counts_median), 3)
        len_std = round(np.std(success_lengths_median), 3)
        ret_std = round(np.std(return_medidan), 3)
        
        print(f"Accuracy - Median Success Counts: {success_counts_median} with std {succ_std}")
        print(f"Speed - Median Successful Episode Length: {success_lengths_median} with std {len_std}")        
        print(f"Learning - Median Episode Return: {return_medidan} with std {ret_std}")
        
    
if __name__ == "__main__":
    #Generate a large random seed and random generator object for reproducibility
    rng = np.random.default_rng(2) 
       
    
    env_kwargs = {
        'bbox': [[0.0,0.0],[1500.0,0.0],[1500.0,1500.0],[0.0,1500.0]],
        'observation_area': [100.0,100.0], 
        'obstruction_count': 0,
        "number_agents": 1, 
        "enforce_grid_boundaries": True,
        "DEBUG": True,
        "np_random": rng,        
        "TEST": 1
        }    
    
    eval_kwargs = dict(
        env_name='gym_rad_search:RadSearchMulti-v1',
        env_kwargs=env_kwargs,

        model_path=(lambda: os.getcwd())(),
        episodes=100,  # Number of episodes to test on [1 - 1000]
        montecarlo_runs=100,  # Number of Monte Carlo runs per episode (How many times to run/sample each episode setup) (mc_runs)
        actor_critic_architecture='rnn',  # Neural network type (control)
        snr="none",  # signal to noise ratio [none, low, medium, high]
        obstruction_count = 0,  # number of obstacles [0 - 7] (num_obs)
        steps_per_episode = 120,
        number_of_agents = 1,
        enforce_boundaries = True,
        resolution_multiplier = 0.01,
        team_mode =  "individual",
        render = False,
        save_gif_freq = 1,
        render_path = ".",
        save_path_for_ac = ".",
        seed = 2,
    )
    
    test = evaluate_PPO(eval_kwargs=eval_kwargs)
    
    test.evaluate()
    
    print('done')