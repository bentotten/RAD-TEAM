"""
Evaluate agents and update neural networks using simulation environment.
"""

import os
import time
import torch
import numpy as np

from typing import (
    List,
    Dict,
    Union,
    Tuple,
)
from dataclasses import dataclass, field

import json
import ray
import joblib
from statsmodels.stats.weightstats import DescrStatsW
import visilibity as vis

# Simulation Environment
import gym  # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch  # type: ignore

# Neural Networks
import RADTEAM_core as RADCNN_core  # type: ignore
import core as RADA2C_core

# import RADA2C_core as RADA2C_core  # type: ignore

# NOTE: Do not use Ray with env generator for random position generation; will create duplicates of identical episode configurations. Ok for TEST1
USE_RAY = False
CHECK_CONFIGS = False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@dataclass
class Results:
    episode_length: List[int] = field(default_factory=lambda: list())
    episode_return: List[float] = field(default_factory=lambda: list())
    intensity: List[float] = field(default_factory=lambda: list())
    background_intensity: List[float] = field(default_factory=lambda: list())


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


def refresh_env(env_dict, env, n, num_obs=0):
    """
    Load saved test environment parameters from dictionary
    into the current instantiation of environment
    """

    def to_vis_p(p) -> vis.Point:
        """
        Return a visilibity Point from a Point.
        """
        return vis.Point(p[0], p[1])

    def to_vis_poly(poly) -> vis.Polygon:
        """
        Return a visilibity Polygon from a Polygon.
        """
        return vis.Polygon(list(map(to_vis_p, poly)))

    def set_vis_coord(point, coords):
        point.set_x(coords[0])
        point.set_y(coords[1])
        return point

    EPSILON = 0.0000001

    key = "env_" + str(n)
    env.src_coords = env_dict[key][0]
    env.intensity = env_dict[key][2]
    env.bkg_intensity = env_dict[key][3]
    env.source = set_vis_coord(env.source, env.src_coords)

    for agent in env.agents.values():
        agent.reset()
        agent.det_coords = env_dict[key][1]
        agent.detector = set_vis_coord(agent.detector, agent.det_coords)

    if num_obs > 0:
        env.obs_coord = env_dict[key][4]
        env.obstruction_count = len(env_dict[key][4])
        env.poly = []
        env.line_segs = []
        for obs in env.obs_coord:
            geom = [vis.Point(float(obs[jj][0]), float(obs[jj][1])) for jj in range(len(obs))]
            poly = vis.Polygon(geom)
            env.poly.append(poly)
            env.line_segs.append(
                [
                    vis.Line_Segment(geom[0], geom[1]),
                    vis.Line_Segment(geom[0], geom[3]),
                    vis.Line_Segment(geom[2], geom[1]),
                    vis.Line_Segment(geom[2], geom[3]),
                ]
            )

        env.env_ls = [solid for solid in env.poly]
        env.env_ls.insert(0, to_vis_poly(env.walls))
        env.world = vis.Environment(env.env_ls)
        # Check if the environment is valid
        assert env.world.is_valid(EPSILON), "Environment is not valid"
        env.vis_graph = vis.Visibility_Graph(env.world, EPSILON)

    o, _, _, _ = env.step(-1)
    for id, agent in env.agents.items():
        agent.det_sto = [env_dict[key][1]]
        agent.meas_sto = [o[id][0]]
        agent.prev_det_dist = env.world.shortest_path(env.source, agent.detector, env.vis_graph, EPSILON).length()

    env.iter_count = 1
    return o, env


# Uncomment when ready to run with Ray
# @ray.remote
@dataclass
class RADTEAM_EpisodeRunner:
    """
    Remote function to execute requested number of episodes for requested number of monte carlo runs each episode for RADTEAM models
    """

    id: int
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
    test_env_path: str = field(default="./test_environments")
    save_path: str = field(default=".")
    seed: Union[int, None] = field(default=0)

    obstruction_count: int = field(default=0)
    enforce_boundaries: bool = field(default=False)
    actor_critic_architecture: str = field(default="cnn")
    number_of_agents: int = field(default=1)
    episodes: int = field(default=100)
    montecarlo_runs: int = field(default=100)
    snr: str = field(default="high")
    env_sets: Dict = field(default_factory=lambda: dict())

    render_first_episode: bool = field(default=True)
    PFGRU: bool = field(default=True)
    load_env: bool = field(default=True)

    # Initialized elsewhere
    #: Object that holds agents
    agents: Dict[int, RADCNN_core.CNNBase] = field(default_factory=lambda: dict())

    def __post_init__(self) -> None:
        # Change to correct directory
        os.chdir(self.current_dir)

        # Load or create test environments
        if self.load_env:
            self.env_sets = joblib.load(self.test_env_path)
        else:
            self.env_sets = None

        # Create own instatiation of environment
        self.env = self.create_environment()
        self.test_number = self.env.TEST

        if not USE_RAY:
            print(f"Evaluating: {self.number_of_agents} agents with obstruction count: {self.obstruction_count}")

        # Get agent model paths and saved agent configurations
        agent_models = {}
        for child in os.scandir(self.model_path):
            if child.is_dir() and ("agent" in child.name or "pyt_save" in child.name):
                agent_models[
                    int(child.name[0])
                ] = child.path  # Read in model path by id number. NOTE: Important that ID number is the first element of file name

        if CHECK_CONFIGS:
            obj = json.load(open(f"{self.model_path}/config_agent0.json"))
            if "self" in obj.keys():
                original_configs = list(obj["self"].values())[0]["ppo_kwargs"]["actor_critic_args"]
            else:
                original_configs = obj  # Original project save format

        # Set up static A2C actor-critic args
        actor_critic_args = dict(
            action_space=self.env.detectable_directions,
            # Also known as state dimensions: The dimensions of the observation returned from the environment
            observation_space=self.env.observation_space.shape[0],
            steps_per_episode=self.steps_per_episode,
            number_of_agents=self.number_of_agents,
            detector_step_size=self.env.step_size,
            environment_scale=self.env.scale,
            bounds_offset=self.env.observation_area,
            enforce_boundaries=self.enforce_boundaries,
            grid_bounds=self.env.scaled_grid_max,
            resolution_multiplier=self.resolution_multiplier,
            GlobalCritic=None,
            no_critic=True,
            save_path=self.save_path_for_ac,
            PFGRU=self.PFGRU,
        )

        # Check current important parameters match parameters read in
        if CHECK_CONFIGS:
            for arg in actor_critic_args:
                if arg != "no_critic" and arg != "GlobalCritic" and arg != "save_path":
                    if type(original_configs[arg]) == int or type(original_configs[arg]) == float or type(original_configs[arg]) == bool:
                        assert (
                            actor_critic_args[arg] == original_configs[arg]
                        ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                    elif type(original_configs[arg]) is str:
                        if arg == "net_type":
                            assert actor_critic_args[arg] == original_configs[arg]
                        else:
                            to_list = original_configs[arg].strip("][").split(" ")
                            config = np.array([float(x) for x in to_list], dtype=np.float32)
                            assert np.array_equal(
                                config, actor_critic_args[arg]
                            ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                    elif type(original_configs[arg]) is list:
                        for a, b in zip(original_configs[arg], actor_critic_args[arg]):
                            assert a == b, f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                    else:
                        assert (
                            actor_critic_args[arg] == original_configs[arg]
                        ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"

        # Initialize agents and load agent models
        for i in range(self.number_of_agents):
            self.agents[i] = RADCNN_core.CNNBase(id=i, **actor_critic_args)  # NOTE: No updates, do not need PPO
            self.agents[i].load(checkpoint_path=agent_models[i])

            # Sanity check
            assert self.agents[i].critic.is_mock_critic()

    def run(self) -> MonteCarloResults:
        # Prepare tracking buffers and counters
        episode_return = 0
        steps_in_episode: int = 0
        terminal_counter: Dict[int, int] = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)
        run_counter = 0

        # Prepare results buffers
        results = MonteCarloResults(id=self.id)

        # Reset environment and save test env parameters
        if self.load_env:
            #observations = self.env.refresh_environment(env_dict=self.env_sets, id=self.id)
            self.obstruction_count = len(self.env_sets[f"env_{self.id}"][4])
            observations = self.env.refresh_environment(env_dict=self.env_sets, n=self.id, num_obs=self.obstruction_count)
        else:
            observations, _, _, _ = self.env.reset()
            self.env_sets = {}
            # Save env for refresh
            self.env_sets[f"env_{self.id}"] = [_ for _ in range(5)]
            self.env_sets[f"env_{self.id}"][0] = self.env.src_coords
            self.env_sets[f"env_{self.id}"][1] = self.env.agents[0].det_coords
            self.env_sets[f"env_{self.id}"][2] = self.env.intensity
            self.env_sets[f"env_{self.id}"][3] = self.env.bkg_intensity
            self.env_sets[f"env_{self.id}"][4] = self.env.obs_coord.copy()

            self.obstruction_count = len(self.env_sets[f"env_{self.id}"][4])

        for agent in self.agents.values():
            agent.set_mode("eval")

        # Prepare episode variables
        agent_thoughts: Dict[int, RADCNN_core.ActionChoice] = dict()
        hiddens: Dict[int, Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None]] = {
            id: self.agents[id].reset_hidden() for id in self.agents
        }  # For RAD-A2C compatibility

        while run_counter < self.montecarlo_runs:
            # Get agent thoughts on current state. Actor: Compute action and logp (log probability); Critic: compute state-value
            agent_thoughts.clear()
            for id, ac in self.agents.items():
                with torch.no_grad():
                    agent_thoughts[id], _ = ac.step(observations, hiddens[id])

                hiddens[id] = agent_thoughts[id].hidden

            # Create action list to send to environment
            agent_action_decisions = {id: int(agent_thoughts[id].action) for id in agent_thoughts}
            for action in agent_action_decisions.values():
                assert 0 <= action and action < int(self.env.number_actions)

            # Take step in environment - Note: will be missing last reward, rewards link to previous observation in env
            observations, rewards, terminals, _ = self.env.step(action=agent_action_decisions)

            # Incremement Counters
            episode_return += rewards["team_reward"]
            steps_in_episode += 1

            # Tally up ending conditions
            # Check if there was a terminal state. Note: if terminals are introduced that only affect one agent but not all,
            # this will need to be changed.
            terminal_reached_flag = False
            for id in terminal_counter:
                if terminals[id] is True:
                    terminal_counter[id] += 1
                    terminal_reached_flag = True

            # Stopping conditions for episode
            timeout: bool = steps_in_episode == self.steps_per_episode  # Max steps per episode reached
            episode_over: bool = terminal_reached_flag or timeout  # Either timeout or terminal found

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
                    results.successful.episode_return.append(episode_return)
                else:
                    results.unsuccessful.episode_length.append(steps_in_episode)
                    results.unsuccessful.episode_return.append(episode_return)

                results.total_episode_return.append(episode_return)

                # Incremenet run counter
                run_counter += 1

                # Reset environment without performing an env.reset()
                episode_return = 0
                steps_in_episode = 0
                terminal_counter = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)

                observations = self.env.refresh_environment(env_dict=self.env_sets, n=self.id, num_obs=self.obstruction_count)

                # Reset agents
                for agent in self.agents.values():
                    agent.reset()

        results.completed_runs = run_counter

        print(f"Finished episode {self.id}! Success count: {results.success_counter} out of {self.montecarlo_runs}")
        return results

    def create_environment(self) -> RadSearch:
        if USE_RAY:
            silent = True
        else:
            silent = False
        env = gym.make(self.env_name, silent=silent, **self.env_kwargs)
        env.reset()
        return env

    def getattr(self, attr):
        return getattr(self, attr)

    def say_hello(self):
        return self.id

    def process_render(self, run_counter: int, id: int) -> None:
        silent = False
        # Render
        save_time_triggered = (run_counter % self.save_gif_freq == 0) if self.save_gif_freq != 0 else False
        time_to_save = save_time_triggered or ((run_counter + 1) == self.montecarlo_runs)
        if self.render and time_to_save:
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=silent,
            )            
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=silent,
            )

        # Always render first episode
        elif self.render and run_counter == 0 and self.render_first_episode:
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=silent,
            )
            self.render_first_episode = False            
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=silent,
            )

        # Always render last episode
        elif self.render and run_counter == self.montecarlo_runs - 1:
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=silent,
            )            
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=silent,
            )


# Uncomment when ready to run with Ray
# @ray.remote
@dataclass
class RADA2C_EpisodeRunner:
    """Episode runner for RADA2C Models"""

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
    env_sets: Dict = field(default_factory=lambda: dict())
    load_env: bool = field(default=True)

    # Initialized elsewhere
    #: Object that holds agents
    agents: Dict[int, RADA2C_core.RNNModelActorCritic] = field(default_factory=lambda: dict())

    def __post_init__(self) -> None:
        # Change to correct directory
        os.chdir(self.current_dir)

        # Create own instatiation of environment
        self.env = self.create_environment()

        # Load or create test environments
        if self.load_env:
            self.env_sets = joblib.load(self.test_env_path)
        else:
            self.env_sets = None

        # Get agent model paths and saved agent configurations
        agent_models = {}
        for child in os.scandir(self.model_path):
            if child.is_dir() and "agent" in child.name:
                agent_models[
                    int(child.name[0])
                ] = child.path  # Read in model path by id number. NOTE: Important that ID number is the first element of file name
            if child.is_dir() and "general" in child.name:
                general_config_path = child.path

        obj = json.load(open(f"{self.model_path}/config.json"))
        if "self" in obj.keys():
            original_configs = list(obj["self"].values())[0]["ppo_kwargs"]["actor_critic_args"]
        else:
            original_configs = obj["ac_kwargs"]  # Original project save format

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
                if type(original_configs[arg]) == int or type(original_configs[arg]) == float or type(original_configs[arg]) == bool:
                    assert (
                        actor_critic_args[arg] == original_configs[arg]
                    ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is str:
                    if arg == "net_type":
                        assert actor_critic_args[arg] == original_configs[arg]
                    else:
                        to_list = original_configs[arg].strip("][").split(" ")
                        config = np.array([float(x) for x in to_list], dtype=np.float32)
                        assert np.array_equal(
                            config, actor_critic_args[arg]
                        ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is list:
                    for a, b in zip(original_configs[arg], actor_critic_args[arg]):
                        assert a == b, f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                else:
                    assert (
                        actor_critic_args[arg] == original_configs[arg]
                    ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"

        # Initialize agents and load agent models
        actor_critic_args["observation_space"] = self.env.observation_space
        actor_critic_args["action_space"] = self.env.action_space
        actor_critic_args["seed"] = self.seed
        actor_critic_args["pad_dim"] = 2

        self.agents[0] = RADA2C_core.RNNModelActorCritic(**actor_critic_args)
        self.agents[0].load_state_dict(torch.load("pyt_save/model.pt"))

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
        if self.load_env:
            #observations = self.env.refresh_environment(env_dict=self.env_sets, id=self.id)
            self.obstruction_count = len(self.env_sets[f"env_{self.id}"][4])
            observations = self.env.refresh_environment(env_dict=self.env_sets, n=self.id, num_obs=self.obstruction_count)
        else:
            observations, _, _, _ = self.env.reset()
            self.env_sets = {}
            # Save env for refresh
            self.env_sets[f"env_{self.id}"] = [_ for _ in range(5)]
            self.env_sets[f"env_{self.id}"][0] = self.env.src_coords
            self.env_sets[f"env_{self.id}"][1] = self.env.agents[0].det_coords
            self.env_sets[f"env_{self.id}"][2] = self.env.intensity
            self.env_sets[f"env_{self.id}"][3] = self.env.bkg_intensity
            self.env_sets[f"env_{self.id}"][4] = self.env.obs_coord.copy()

            self.obstruction_count = len(self.env_sets[f"env_{self.id}"][4])

        self.agents[0].pi.eval()
        self.agents[0].model.eval()

        # Prepare episode variables
        agent_thoughts: Dict = dict()
        hiddens: Dict[int, Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None]] = {
            id: self.agents[id].reset_hidden() for id in self.agents
        }  # For RAD-A2C compatibility

        for id, ac in self.agents.items():
            stat_buffers[id] = RADA2C_core.StatBuff()
            stat_buffers[id].update(observations[id][0])
            observations[id][0] = np.clip(
                (observations[id][0] - stat_buffers[id].mu) / stat_buffers[id].sig_obs,
                -8,
                8,
            )

        while run_counter < self.montecarlo_runs:
            # Get agent thoughts on current state. Actor: Compute action and logp (log probability); Critic: compute state-value
            agent_thoughts.clear()
            agent_thoughts[id] = {}
            for id, ac in self.agents.items():
                with torch.no_grad():
                    a, v, logp, hidden, out_pred = ac.step(observations[id], hiddens[id])
                    agent_thoughts[id]["action"] = a

                hiddens[id] = hidden
            # Create action list to send to environment
            agent_action_decisions = {id: int(agent_thoughts[id]["action"]) for id in agent_thoughts}
            for action in agent_action_decisions.values():
                assert 0 <= action and action < int(self.env.number_actions)

            # Take step in environment - Note: will be missing last reward, rewards link to previous observation in env
            observations, rewards, terminals, _ = self.env.step(action=agent_action_decisions)

            for id, ac in self.agents.items():
                stat_buffers[id].update(observations[id][0])
                observations[id][0] = np.clip(
                    (observations[id][0] - stat_buffers[id].mu) / stat_buffers[id].sig_obs,
                    -8,
                    8,
                )

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
            timeout: bool = steps_in_episode == self.steps_per_episode  # Max steps per episode reached
            episode_over: bool = terminal_reached_flag or timeout  # Either timeout or terminal found

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

                # observations = self.env.refresh_environment(env_dict=self.env_sets, id=self.id)
                observations = self.env.refresh_environment(env_dict=self.env_sets, n=self.id, num_obs=self.obstruction_count)

                # Reset stat buffer for RAD-A2C
                for id, ac in self.agents.items():
                    stat_buffers[id].reset()
                    stat_buffers[id].update(observations[id][0])
                    observations[id][0] = np.clip(
                        (observations[id][0] - stat_buffers[id].mu) / stat_buffers[id].sig_obs,
                        -8,
                        8,
                    )

        results.completed_runs = run_counter

        print(f"Finished episode {self.id}! Success count: {results.success_counter} out of {run_counter}")
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
        silent = False
        # Render
        save_time_triggered = (run_counter % self.save_gif_freq == 0) if self.save_gif_freq != 0 else False
        time_to_save = save_time_triggered or ((run_counter + 1) == self.montecarlo_runs)
        if self.render and time_to_save:
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=silent,
            )            
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=silent,
            )

        # Always render first episode
        elif self.render and run_counter == 0 and self.render_first_episode:
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=silent,
            )
            self.render_first_episode = False            
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=silent,
            )

        # Always render last episode
        elif self.render and run_counter == self.montecarlo_runs - 1:
            # Render environment image
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                just_env=True,
                episode_count=id,
                silent=silent,
            )            
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=silent,
            )


@dataclass
class evaluate_PPO:
    """
    Test existing model across random episodes for a set number of monte carlo runs per episode.
    """

    eval_kwargs: Dict
    #: Path to save results to
    save_path: Union[str, None] = field(default=None)
    #: Load RADA2C or RADTEAM
    RADTEAM: bool = field(default=True)

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
        if not self.save_path:
            self.save_path = eval_kwargs["model_path"]  # type: ignore

        # get test envs
        if self.eval_kwargs["obstruction_count"] == -1:
            raise ValueError("Random sample of obstruction counts indicated. Please indicate a specific count between 1 and 5")
        self.test_env_dir = self.eval_kwargs["test_env_path"]
        self.test_env_path = self.test_env_dir + f"/test_env_obs{self.eval_kwargs['obstruction_count']}_{self.eval_kwargs['snr']}" if self.test_env_dir else None
        self.eval_kwargs["test_env_path"] = self.test_env_path

        # Initialize ray
        if USE_RAY:
            try:
                ray.init(address="auto")
            except:  # noqa
                print("Ray failed to initialize. Running on single server.")

    def evaluate(self):
        """Driver"""
        start_time = time.time()
        if USE_RAY:
            if self.RADTEAM:
                runners = {
                    i: RADTEAM_EpisodeRunner.remote(id=i, current_dir=os.getcwd(), **self.eval_kwargs) for i in range(self.eval_kwargs["episodes"])
                }
            else:
                runners = {
                    i: RADA2C_EpisodeRunner.remote(id=i, current_dir=os.getcwd(), **self.eval_kwargs) for i in range(self.eval_kwargs["episodes"])
                }
            full_results = ray.get([runner.run.remote() for runner in runners.values()])
        else:
            if self.RADTEAM:
                self.runners = {
                    i: RADTEAM_EpisodeRunner(id=i, current_dir=os.getcwd(), **self.eval_kwargs) for i in range(self.eval_kwargs["episodes"])
                }
            else:
                self.runners = {
                    i: RADA2C_EpisodeRunner(id=i, current_dir=os.getcwd(), **self.eval_kwargs) for i in range(self.eval_kwargs["episodes"])
                }
            full_results = [runner.run() for runner in self.runners.values()]

        print("Runtime: {}", time.time() - start_time)

        score = self.calc_stats(results=full_results)

        # Convert to raw results
        counter = 0
        raw_results = list()
        for index, result in enumerate(full_results):
            raw_results.append(dict())
            raw_results[index]["id"] = result.id
            raw_results[index]["completed_runs"] = result.completed_runs
            raw_results[index]["success_counter"] = result.success_counter
            raw_results[index]["total_episode_length"] = result.total_episode_length
            raw_results[index]["total_episode_return"] = result.total_episode_length
            raw_results[index]["successful"] = dict()

            raw_results[index]["successful"]["episode_length"] = result.successful.episode_length
            raw_results[index]["successful"]["episode_return"] = result.successful.episode_return
            raw_results[index]["successful"]["intensity"] = result.successful.intensity

            raw_results[index]["unsuccessful"] = dict()
            raw_results[index]["unsuccessful"]["episode_length"] = result.unsuccessful.episode_length
            raw_results[index]["unsuccessful"]["episode_return"] = result.unsuccessful.episode_return
            raw_results[index]["unsuccessful"]["intensity"] = result.unsuccessful.intensity

            counter += result.completed_runs

        with open(f"{self.save_path}/results.json", "w+") as f:
            f.write(json.dumps(score, indent=4, cls=NpEncoder))

        with open(f"{self.save_path}/results_raw.json", "w+") as f:
            f.write(json.dumps(raw_results, indent=4, cls=NpEncoder))

        # print(f"Total Runs: {counter}")
        # print(f"Accuracy - Median Success Counts: {score['accuracy'][0]['median']} with std {score['stdev']['accuracy']}")
        # print(f"Speed - Median Successful Episode Length: {score['speed'][0]['median']} with std {score['stdev']['speed']}")
        # print(f"Learning - Median Episode Return: {score['score'][0]['median']} with std {score['stdev']['score']}")

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

        # Pointers for episode
        ep_len_start_ptr = 0
        ep_ret_start_ptr = 0

        for ep_index, episode in enumerate(results):
            success_counts[ep_index] = episode.success_counter
            episode_returns[ep_ret_start_ptr : ep_ret_start_ptr + mc] = episode.total_episode_return
            ep_ret_start_ptr = ep_ret_start_ptr + mc

            successful_episode_lengths[
                ep_len_start_ptr : ep_len_start_ptr + len(episode.successful.episode_length)
            ] = episode.successful.episode_length[:]
            ep_len_start_ptr = ep_len_start_ptr + len(episode.successful.episode_length)

        final = dict(accuracy=dict(), super=dict(), score=dict())

        succ_std = round(np.nanstd(success_counts), 3)
        suc_low_whisker, suc_q1, suc_median, suc_q3, suc_high_whisker = self.calc_quartiles(success_counts)
        suc_median2 = np.nanmedian(sorted(success_counts))
        assert suc_median == suc_median2

        len_std = round(np.nanstd(successful_episode_lengths), 3)
        len_low_whisker, len_q1, len_median, len_q3, len_high_whisker = self.calc_quartiles(successful_episode_lengths)

        ret_std = round(np.nanstd(episode_returns), 3)
        ret_low_whisker, ret_q1, ret_median, ret_q3, ret_high_whisker = self.calc_quartiles(episode_returns)

        final["speed"] = [{"whislo": len_low_whisker, "q1": len_q1, "med": len_median, "q3": len_q3, "whishi": len_high_whisker, "fliers": []}]
        final["accuracy"] = [{"whislo": suc_low_whisker, "q1": suc_q1, "med": suc_median, "q3": suc_q3, "whishi": suc_high_whisker, "fliers": []}]
        final["score"] = [{"whislo": ret_low_whisker, "q1": ret_q1, "med": ret_median, "q3": ret_q3, "whishi": ret_high_whisker, "fliers": []}]
        final["stdev"] = {"speed": len_std, "accuracy": succ_std, "score": ret_std}

        return final

    def calc_quartiles(self, data):
        d1 = DescrStatsW(data)
        low_whisker, q1, median, q3, high_whisker = d1.quantile([0.025, 0.25, 0.5, 0.75, 0.975], return_pandas=False)
        return (low_whisker, q1, median, q3, high_whisker)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obstacles", type=int, default=0, help="Number of obstructions")
    parser.add_argument("--agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--test", type=str, default="FULL", help="Test to run (0 for no test)")
    parser.add_argument("--snr", type=str, default="none", help="Signal to noise ratio [none, low, med, high]")    
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max_dim", type=list, default=[1500, 1500])
    parser.add_argument("--render_freq", type=int, default=100)
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where results are saved. Ex: ../models/train/gru_8_acts/bpf/model_dir",
        default=".",  # noqa
    )    
    parser.add_argument(
        "--load_env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load from standardized saved environment. False will generate a new environment from seed (do not run with Ray!).",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render Gif",
    )
    parser.add_argument(
        "--rada2c",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a RADA2C model instead of RADTEAM",
    )
    args = parser.parse_args()

    # Go to data directory
    if args.data_dir == ".":
        args.data_dir = os.getcwd() + "/"
    args.data_dir = args.data_dir + "/" if args.data_dir[-1] != "/" else args.data_dir

    os.chdir(args.data_dir)

    number_of_agents = args.agents
    mode = "collaborative"  # No critic, ok to leave as collaborative for all tests
    render = args.render
    obstruction_count = args.obstacles

    if args.load_env:
        if args.test in ["1", "2", "3", "4"]:
            env_path = os.getcwd() + f"/saved_env/test_environments_TEST{args.test}"
        else:
            env_path = os.getcwd() + f"/saved_env/test_environments_{args.max_dim[0]}x{args.max_dim[1]}"
        assert os.path.isdir(env_path), f"{env_path} does not exist"
    else:
        env_path = None

    PFGRU = False
    seed = 2

    # Generate a large random seed and random generator object for reproducibility
    rng = np.random.default_rng(seed)
    env_kwargs = {
        "bbox": [[0.0, 0.0], [args.max_dim[0], 0.0], [args.max_dim[0], args.max_dim[1]], [0.0, args.max_dim[1]]],
        "observation_area": [100.0, 200.0],
        "obstruction_count": obstruction_count,
        "MIN_STARTING_DISTANCE": 500,
        "number_agents": number_of_agents,
        "enforce_grid_boundaries": True,
        "np_random": rng,
        "TEST": args.test,
    }

    # Set eval parameters according to which version we're running
    if not args.rada2c:
        eval_kwargs = dict(
            env_name="gym_rad_search:RadSearchMulti-v1",
            env_kwargs=env_kwargs,
            model_path=(lambda: os.getcwd())(),
            episodes=args.episodes,  # Number of episodes to test on [1 - 1000]
            montecarlo_runs=args.runs,  # Number of Monte Carlo runs per episode (How many times to run/sample each episode setup) (mc_runs)
            actor_critic_architecture="cnn",  # Neural network type (control)
            snr=args.snr,  # signal to noise ratio [none, low, medium, high]
            obstruction_count=obstruction_count,  # number of obstacles [0 - 7] (num_obs)
            steps_per_episode=120,
            number_of_agents=number_of_agents,
            enforce_boundaries=True,
            resolution_multiplier=0.01,
            team_mode=mode,
            render=render,
            save_gif_freq=args.render_freq,
            render_path=".",
            save_path_for_ac=".",
            seed=seed,
            PFGRU=PFGRU,
            load_env=args.load_env,
            test_env_path=env_path,
        )
        radteam = True

    elif args.rada2c:
        eval_kwargs = dict(
            env_name="gym_rad_search:RadSearchMulti-v1",
            env_kwargs=env_kwargs,
            model_path=(lambda: os.getcwd())(),
            episodes=args.episodes,  # Number of episodes to test on [1 - 1000]
            montecarlo_runs=args.runs,  # Number of Monte Carlo runs per episode (How many times to run/sample each episode setup) (mc_runs)
            actor_critic_architecture="rnn",  # Neural network type (control)
            snr=args.snr,  # signal to noise ratio [none, low, medium, high]
            obstruction_count=obstruction_count,  # number of obstacles [0 - 7] (num_obs)
            steps_per_episode=120,
            number_of_agents=1,
            enforce_boundaries=True,
            resolution_multiplier=0.01,
            team_mode="individual",
            render=render,
            save_gif_freq=args.render_freq,
            render_path=".",
            save_path_for_ac=".",
            seed=seed,
            load_env=args.load_env,
            test_env_path=env_path,
        )
        radteam = False

    test = evaluate_PPO(RADTEAM=radteam, eval_kwargs=eval_kwargs)

    test.evaluate()

    print("done")
