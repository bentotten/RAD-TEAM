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

# Simulation Environment
import gym  # type: ignore
from gym_rad_search.envs import rad_search_env  # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

# Neural Networks
import RADTEAM_core as RADCNN_core  # type: ignore

# import RADA2C_core as RADA2C_core  # type: ignore

# NOTE: Do not use Ray with env generator for random position generation; will create duplicates of identical episode configurations. Ok for TEST1
USE_RAY = False

ALL_ACKWARGS_SAVED = False


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


# Uncomment when ready to run with Ray
# @ray.remote
@dataclass
class EpisodeRunner:
    """
    Remote function to execute requested number of episodes for requested number of monte carlo runs each episode.

    Process from RAD-A2C:
    - 100 episodes classes:
        - [done] create environment
        - [done] refresh environment with test env
        - [done] create and upload agent
        - [done] Get initial environment observation
        - Do monte-carlo runs
            - Get action
            - Take step in env
            - Save return and increment steps-in-episode
            - If terminal or timeout:
                - Save det_sto from environment (why?)
                - If first monte-carlo:
                    - If terminal, save intensity/background intenity into "done" list
                    - If not terminal, save intensity/background intenity into "not done" list
                - If Terminal, increment done counter, add episode length to "done" list, and add episode return to "done" list
                - If not Terminal, add episode length to "not done" list, and add episode return to "not done" list
                - Render if desired
                - Refresh environment and reset episode tracking
                - ? #Reset model in action selection fcn. get_action(0)
                - ? #Get initial location prediction
        - Render
        - Save stats/results and return:
            mc_stats['dEpLen'] = d_ep_len
            mc_stats['ndEpLen'] = nd_ep_len
            mc_stats['dEpRet'] = d_ep_ret
            mc_stats['ndEpRet'] = nd_ep_ret
            mc_stats['dIntDist'] = done_dist_int
            mc_stats['ndIntDist'] = not_done_dist_int
            mc_stats['dBkgDist'] = done_dist_bkg
            mc_stats['ndBkgDist'] = not_done_dist_bkg
            mc_stats['DoneCount'] = np.array([done_count])
            mc_stats['TotEpLen'] = tot_ep_len
            mc_stats['LocEstErr'] = loc_est_err
            results = [loc_est_ls, FIM_bound, J_score_ls, det_ls]
            print(f'Finished episode {n}!, completed count: {done_count}')
            return (results,mc_stats)

    """

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
    env_dict: Dict = field(default_factory=lambda: dict())

    render_first_episode: bool = field(default=True)
    PFGRU: bool = field(default=True)

    # Initialized elsewhere
    #: Object that holds agents
    agents: Dict[int, RADCNN_core.CNNBase] = field(default_factory=lambda: dict())

    def __post_init__(self) -> None:
        # Change to correct directory
        os.chdir(self.current_dir)

        # Create own instatiation of environment
        self.env = self.create_environment()

        # Get agent model paths and saved agent configurations
        agent_models = {}
        for child in os.scandir(self.model_path):
            if child.is_dir() and "agent" in child.name:
                agent_models[
                    int(child.name[0])
                ] = (
                    child.path
                )  # Read in model path by id number. NOTE: Important that ID number is the first element of file name
            if child.is_dir() and "general" in child.name:
                general_config_path = child.path

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
            PFGRU=self.PFGRU
        )

        # Check current important parameters match parameters read in
        if ALL_ACKWARGS_SAVED:
            for arg in actor_critic_args:
                if arg != "no_critic" and arg != "GlobalCritic" and arg != "save_path":
                    if (
                        type(original_configs[arg]) == int
                        or type(original_configs[arg]) == float
                        or type(original_configs[arg]) == bool
                    ):
                        assert (
                            actor_critic_args[arg] == original_configs[arg]
                        ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                    elif type(original_configs[arg]) is str:
                        if arg == "net_type":
                            assert actor_critic_args[arg] == original_configs[arg]
                        else:
                            to_list = original_configs[arg].strip("][").split(" ")
                            config = np.array(
                                [float(x) for x in to_list], dtype=np.float32
                            )
                            assert np.array_equal(
                                config, actor_critic_args[arg]
                            ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                    elif type(original_configs[arg]) is list:
                        for a, b in zip(original_configs[arg], actor_critic_args[arg]):
                            assert (
                                a == b
                            ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                    else:
                        assert (
                            actor_critic_args[arg] == original_configs[arg]
                        ), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"

        # Initialize agents and load agent models
        for i in range(self.number_of_agents):
            self.agents[i] = RADCNN_core.CNNBase(
                id=i, **actor_critic_args
            )  # NOTE: No updates, do not need PPO
            self.agents[i].load(checkpoint_path=agent_models[i])

            # Sanity check
            assert self.agents[i].critic.is_mock_critic()

    def run(self) -> MonteCarloResults:
        # Prepare tracking buffers and counters
        episode_return = 0
        steps_in_episode: int = 0
        terminal_counter: Dict[int, int] = {
            id: 0 for id in self.agents
        }  # Terminal counter for the epoch (not the episode)
        run_counter = 0

        # Prepare results buffers
        results = MonteCarloResults(id=self.id)

        # Reset environment and save test env parameters
        observations, _, _, _ = self.env.reset()

        # Save env for refresh
        self.env_dict["env_0"] = [_ for _ in range(5)]
        self.env_dict["env_0"][0] = self.env.src_coords
        self.env_dict["env_0"][1] = self.env.agents[0].det_coords
        self.env_dict["env_0"][2] = self.env.intensity
        self.env_dict["env_0"][3] = self.env.bkg_intensity
        self.env_dict["env_0"][4] = []  # obstructions

        for agent in self.agents.values():
            agent.set_mode("eval")

        # Prepare episode variables
        agent_thoughts: Dict[int, RADCNN_core.ActionChoice] = dict()
        hiddens: Dict[
            int, Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None]
        ] = {
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
            agent_action_decisions = {
                id: int(agent_thoughts[id].action) for id in agent_thoughts
            }
            for action in agent_action_decisions.values():
                assert 0 <= action and action < int(self.env.number_actions)

            # Take step in environment - Note: will be missing last reward, rewards link to previous observation in env
            observations, rewards, terminals, _ = self.env.step(
                action=agent_action_decisions
            )

            # Incremement Counters
            episode_return += rewards["team_reward"]
            steps_in_episode += 1

            # Tally up ending conditions
            # Check if there was a terminal state. Note: if terminals are introduced that only affect one agent but not all, this will need to be changed.
            terminal_reached_flag = False
            for id in terminal_counter:
                if terminals[id] is True:
                    terminal_counter[id] += 1
                    terminal_reached_flag = True

            # Stopping conditions for episode
            timeout: bool = (
                steps_in_episode == self.steps_per_episode
            )  # Max steps per episode reached
            episode_over: bool = (
                terminal_reached_flag or timeout
            )  # Either timeout or terminal found

            if episode_over:
                self.process_render(run_counter=run_counter, id=self.id)

                # Save results
                if run_counter < 1:
                    if terminal_reached_flag:
                        results.successful.intensity.append(self.env.intensity)
                        results.successful.background_intensity.append(
                            self.env.bkg_intensity
                        )
                    else:
                        results.unsuccessful.intensity.append(self.env.intensity)
                        results.unsuccessful.background_intensity.append(
                            self.env.bkg_intensity
                        )
                results.total_episode_length.append(steps_in_episode)

                if terminal_reached_flag:
                    results.success_counter += 1
                    results.successful.episode_length.append(steps_in_episode)
                    results.successful.episode_return.append(
                        episode_return
                    )  
                else:
                    results.unsuccessful.episode_length.append(steps_in_episode)
                    results.unsuccessful.episode_return.append(
                        episode_return
                    )  

                results.total_episode_return.append(
                    episode_return
                ) 

                # Incremenet run counter
                run_counter += 1

                # Reset environment without performing an env.reset()
                episode_return = 0
                steps_in_episode = 0
                terminal_counter = {
                    id: 0 for id in self.agents
                }  # Terminal counter for the epoch (not the episode)

                observations = self.env.refresh_environment(
                    env_dict=self.env_dict, id=0, num_obs=self.obstruction_count
                )

                # Reset agents
                for agent in self.agents.values():
                    agent.reset()

        results.completed_runs = run_counter

        print(
            f"Finished episode {self.id}! Success count: {results.success_counter} out of {self.montecarlo_runs}"
        )
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
            # Render Agent heatmaps
            if self.actor_critic_architecture == "cnn":
                for id, ac in self.agents.items():
                    # TODO add episode counter
                    ac.render(
                        savepath=self.render_path,
                        episode_count=id,
                        epoch_count=run_counter,
                        add_value_text=True,
                    )
            # Render gif
            self.env.render(
                path=self.render_path,
                epoch_count=run_counter,
                episode_count=id,
                silent=False,
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
            # Render Agent heatmaps
            if self.actor_critic_architecture == "cnn":
                for id, ac in self.agents.items():
                    ac.render(
                        savepath=self.render_path,
                        epoch_count=run_counter,
                        add_value_text=True,
                        episode_count=id,
                    )
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
            # Render Agent heatmaps
            if self.actor_critic_architecture == "cnn":
                for id, ac in self.agents.items():
                    ac.render(
                        savepath=self.render_path,
                        epoch_count=run_counter,
                        add_value_text=True,
                        episode_count=id,
                    )
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
    #: Path to save results to
    save_path: Union[str, None] = field(default=None)

    def __post_init__(self) -> None:
        self.montecarlo_runs = self.eval_kwargs["montecarlo_runs"]
        if not self.save_path:
            self.save_path = eval_kwargs["model_path"]  # type: ignore
        # Initialize ray
        if USE_RAY:
            try:
                ray.init(address="auto")
            except:
                print("Ray failed to initialize. Running on single server.")

    def evaluate(self):
        """Driver"""
        start_time = time.time()
        if USE_RAY:
            # Uncomment when ready to run with Ray
            runners = {
                i: EpisodeRunner.remote(
                    id=i, current_dir=os.getcwd(), **self.eval_kwargs
                )
                for i in range(self.eval_kwargs["episodes"])
            }

            full_results = ray.get([runner.run.remote() for runner in runners.values()])
        else:
            # Uncomment when to run without Ray
            self.runners = {
                i: EpisodeRunner(id=i, current_dir=os.getcwd(), **self.eval_kwargs)
                for i in range(self.eval_kwargs["episodes"])
            }

            full_results = [runner.run() for runner in self.runners.values()]

        print("Runtime: {}", time.time() - start_time)

        score = self.calc_stats(results=full_results)
        with open(f"{self.save_path}/results.json", "w+") as f:
            f.write(json.dumps(score, indent=4))

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

            raw_results[index]["successful"][
                "episode_length"
            ] = result.successful.episode_length
            raw_results[index]["successful"][
                "episode_return"
            ] = result.successful.episode_return
            raw_results[index]["successful"]["intensity"] = result.successful.intensity

            raw_results[index]["unsuccessful"] = dict()
            raw_results[index]["unsuccessful"][
                "episode_length"
            ] = result.unsuccessful.episode_length
            raw_results[index]["unsuccessful"][
                "episode_return"
            ] = result.unsuccessful.episode_return
            raw_results[index]["unsuccessful"][
                "intensity"
            ] = result.unsuccessful.intensity

            counter += result.completed_runs

        with open(f"{self.save_path}/results_raw.json", "w+") as f:
            f.write(json.dumps(raw_results, indent=4))

        print(f"Total Runs: {counter}")
        print(
            f"Accuracy - Median Success Counts: {score['accuracy']['median']} with std {score['accuracy']['std']}"
        )
        print(
            f"Speed - Median Successful Episode Length: {score['speed']['median']} with std {score['speed']['std']}"
        )
        print(
            f"Learning - Median Episode Return: {score['score']['median']} with std {score['score']['std']}"
        )

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
            episode_returns[
                ep_ret_start_ptr : ep_ret_start_ptr + mc
            ] = episode.total_episode_return
            ep_ret_start_ptr = ep_ret_start_ptr + mc

            successful_episode_lengths[
                ep_len_start_ptr : ep_len_start_ptr
                + len(episode.successful.episode_length)
            ] = episode.successful.episode_length[:]
            ep_len_start_ptr = ep_len_start_ptr + len(episode.successful.episode_length)

        success_counts_median = np.nanmedian(sorted(success_counts))
        success_lengths_median = np.nanmedian(sorted(successful_episode_lengths))
        return_median = np.nanmedian(sorted(episode_returns))

        succ_std = round(np.nanstd(success_counts), 3)
        len_std = round(np.nanstd(successful_episode_lengths), 3)
        ret_std = round(np.nanstd(episode_returns), 3)

        return {
            "accuracy": {"median": success_counts_median, "std": succ_std},
            "speed": {"median": success_lengths_median, "std": len_std},
            "score": {"median": return_median, "std": ret_std},
        }


if __name__ == "__main__":

    number_of_agents = 1
    mode = "collaborative"  # No critic, ok to leave as collaborative for all tests
    render = False
    obstruction_count = 0

    PFGRU = False
    seed = 2
    # Generate a large random seed and random generator object for reproducibility
    rng = np.random.default_rng(seed)
    env_kwargs = {
        "bbox": [[0.0, 0.0], [2700.0, 0.0], [2700.0, 2700.0], [0.0, 2700.0]],
        "observation_area": [200.0, 500.0],
        "obstruction_count": obstruction_count,
        "number_agents": number_of_agents,
        "enforce_grid_boundaries": True,
        "np_random": rng,
    }

    eval_kwargs = dict(
        env_name="gym_rad_search:RadSearchMulti-v1",
        env_kwargs=env_kwargs,
        model_path=(lambda: os.getcwd())(),
        episodes=100,  # Number of episodes to test on [1 - 1000]
        montecarlo_runs=100,  # Number of Monte Carlo runs per episode (How many times to run/sample each episode setup) (mc_runs)
        actor_critic_architecture="cnn",  # Neural network type (control)
        snr="none",  # signal to noise ratio [none, low, medium, high]
        obstruction_count=obstruction_count,  # number of obstacles [0 - 7] (num_obs)
        steps_per_episode=120,
        number_of_agents=number_of_agents,
        enforce_boundaries=True,
        resolution_multiplier=0.01,
        team_mode=mode,
        render=render,
        save_gif_freq=1,
        render_path=".",
        save_path_for_ac=".",
        seed=seed,
        PFGRU=PFGRU
    )

    test = evaluate_PPO(eval_kwargs=eval_kwargs)

    test.evaluate()

    print("done")
