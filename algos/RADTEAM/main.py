import torch
import argparse
import os
import numpy as np
import gym  # type: ignore
from dataclasses import dataclass
from typing import Tuple, Literal
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
from rl_tools.run_utils import setup_logger_kwargs  # type: ignore

try:
    from ppo import ppo
    from RADTEAM_core import CNNBase
    from rl_tools.mpi_tools import mpi_fork, proc_id  # type: ignore
except ModuleNotFoundError:
    from algos.RADTEAM.ppo import ppo
    from algos.RADTEAM.RADTEAM_core import CNNBase
    from algos.RADTEAM.rl_tools.mpi_tools import mpi_fork, proc_id  # type: ignore
except:  # noqa
    raise Exception


def check_for_cuda() -> None:
    """ Check for CUDA; if exists, set device to CUDA-enabled device. """
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")


@dataclass
class CliArgs:
    """ Parameters passed in through the command line and their expected types """
    #: Name of gym environment to be used for simulations. ie: gym_rad_search:RadSearchMulti-v1.
    env: str
    #: RAD-A2C gated recurrent unit hidden state size.
    hid_gru: int
    #: RAD-A2C actor linear layer size.
    hid_pol: int
    #: RAD-A2C critic linear layer size.
    hid_val: int
    #: RAD-A2C Particle Filter Gated Recurrent Unit (PFGRU) hidden state size.
    hid_rec: int
    #: RAD-A2C number of layers for actor in Multi-Layer Perceptron (MLP) mode.
    l_pol: int
    #: RAD-A2C number of layers for critic in Multi-Layer Perceptron (MLP) mode.
    l_val: int
    #: Reward attribution for advantage estimator (used as a part of the Proximal Policy Optimization (PPO) learning algorithm).
    gamma: float
    # lam: float # TODO: Uncomment when all variables are available to CLI
    #: Random seed control for neural networks and simulation environment generation.
    seed: int
    #: Number of cores to run parallel episodes in each epoch for training (using MPI)
    cpu: int
    #: Number of timesteps before performing a gradient update with accumulated results from episodes. Note that an episode can be cut-off
    #   by an epoch end. This episode configuration will be repeated at the beginning of the next epoch, after learning has been applied.
    steps_per_epoch: int
    #: Maximum number of steps an agent can take in an episode
    steps_per_episode: int # TODO: Uncomment when all variables are available to CLI
    #: Number of epochs to train agent with.
    epochs: int
    #: The name of the experiment (for logging and saving purposes).
    exp_name: str
    #: The dimensions of radiation source search area in cm. This will be decreased by the area_obs parameter to ensure the simulation environment
    #  visilibity graph setup is valid.
    dims: Tuple[int, int]
    #: Area of obstructions. This determines how large obstacles are.
    area_obs: Tuple[int, int]
    #: Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7.
    obstruct: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7]
    #: RAD-A2C: Choose between a Recurrent Neural Network (RNN) or a MLP actor-critic. Options: 'rnn', 'mlp'
    net_type: str
    #: Entropy reward term scaling
    alpha: float
    #: Indicate true if agents are loading their neural network parameters from a file.
    load_model: bool
    #: Number of agents present on team.
    agents: int
    #: Save a gif of the last episode after this many epochs
    save_gif_freq: int
    #: Game mode: Cooperative is global critic and team reward and Collaborative is individual critic and inv reward.
    # TODO change this to CTDE and BTBE
    mode: str
    #: For unit testing, indicate which test to run (0-4). 0 indicates no test.
    test: int
    #: Whether to render gifs and static environment images or not
    render: bool
    #: Whether RADTEAM should use the particle filter module for source location prediction or not.
    PFGRU: bool
    #: Flag that indicates training mode, where model is updated with new learning
    training: bool

    # TODO uncomment these when these variables are available to be controlled from CLI
    # clip_ratio: float
    # target_kl: float
    # render: bool
    # agent_count: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # enforce_boundaries: bool
    # resolution_multiplier: float
    # global_critic: bool
    # minibatches: int
    # save_freq: int
    # actor_learning_rate: float
    # critic_learning_rate: float
    # pfgru_learning_rate: float
    # train_pi_iters: float
    # train_v_iters: float
    # train_pfgru_iters: float
    # DEBUG: bool


def create_parser() -> argparse.ArgumentParser:
    """
    Function to generate argument parser. This uses the argparse library to read command-line arguments into variables.
    :returns: An argument parser with command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_rad_search:RadSearchMulti-v1")
    parser.add_argument("--hid_gru", type=int, default=[24], help="RAD-A2C GRU hidden state size")
    parser.add_argument("--hid_pol", type=int, default=[32], help="RAD-A2C Actor linear layer size")
    parser.add_argument("--hid_val", type=int, default=[32], help="Critic linear layer size")
    parser.add_argument("--hid_rec", type=int, default=[24], help="PFGRU hidden state size")
    parser.add_argument("--l_pol", type=int, default=1, help="Number of layers for Actor MLP")
    parser.add_argument("--l_val", type=int, default=1, help="Number of layers for Critic MLP")
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward attribution for advantage estimator")
    parser.add_argument("--seed", "-s", type=int, default=2, help="Random seed control")
    parser.add_argument("--cpu", type=int, default=1, help="Number of cores/environments to train the agent with")
    parser.add_argument(
        "--steps_per_epoch", type=int, default=480, help="Number of timesteps per epoch per cpu. Default is equal to 4 episodes per cpu per epoch."
    )
    parser.add_argument(
        "--steps_per_episode", type=int, default=120, help="Number of timesteps per episode."
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train the agent")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="run",
        help="Name of experiment for saving",
    )
    parser.add_argument(
        "--dims",
        type=list,
        default=[[0.0, 0.0], [1500.0, 0.0], [1500.0, 1500.0], [0.0, 1500.0]],
        help="Dimensions of radiation source search area in cm, decreased by area_obs param. to ensure visilibity graph setup is valid.",
    )
    parser.add_argument("--area_obs", type=list, default=[100.0, 100.0], help="Interval for each obstruction area in cm")
    parser.add_argument(
        "--obstruct",
        type=int,
        default=-1,
        help="Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7",
    )
    parser.add_argument("--net_type", type=str, default="rnn", help="Choose between recurrent neural network A2C or MLP A2C, option: rnn, mlp")
    parser.add_argument("--alpha", type=float, default=0.1, help="Entropy reward term scaling")
    parser.add_argument("--load_model", type=int, default=0, help="Load parameters from saved model. 0 is false, 1 is true")
    parser.add_argument("--agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--save_gif_freq", type=int, default=-1, help="Gif frequency to Save")
    parser.add_argument(
        "--mode",
        type=str,
        default="cooperative",
        help="Game mode: Cooperative is global critic and team reward, Collaborative is individual critic and inv reward, Competative\
                              is individual zero-sum game",
    )
    parser.add_argument("--test", type=str, default="FULL", help="Test to run (0 for no test)")
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to render gifs and static images of the environment or not",
    )
    parser.add_argument(
        "--PFGRU",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether RADTEAM should use the particle filter module for source location prediction or not.",
    )
    parser.add_argument(
        "--training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flag that indicates training mode, where model is updated with new learning",
    )    
    return parser


def parse_args(parser: argparse.ArgumentParser) -> CliArgs:
    """Function to parse command line arguments into dataclass members. This ensures that arguments are correctly typed as we explicitly intended.

    :param parser: (argparse.ArgumentParser) The parser from argparse module with read-in arguments.
    :returns: Command line argument class-object containing read-in arguments
    """
    args = parser.parse_args()
    return CliArgs(
        env=args.env,
        hid_gru=args.hid_gru,
        hid_pol=args.hid_pol,
        hid_val=args.hid_val,
        hid_rec=args.hid_rec,
        l_pol=args.l_pol,
        l_val=args.l_val,
        gamma=args.gamma,
        seed=args.seed,
        cpu=args.cpu,
        steps_per_epoch=args.steps_per_epoch,
        steps_per_episode=args.steps_per_episode,
        epochs=args.epochs,
        exp_name=args.exp_name,
        dims=args.dims,
        area_obs=args.area_obs,
        obstruct=args.obstruct,
        net_type=args.net_type,
        alpha=args.alpha,
        load_model=args.load_model,
        agents=args.agents,
        save_gif_freq=args.save_gif_freq,
        mode=args.mode,
        test=args.test,
        render=args.render,
        PFGRU=args.PFGRU,
        training=args.training
    )


if __name__ == "__main__":
    check_for_cuda()

    # Get command line args
    args = parse_args(create_parser())

    # Add mini-batch size. NOTE: Has only been tested with size of 1
    args.batch = 1

    # Save directory and experiment name
    save_freq = 250
    args.env_name = "results"
    args.exp_name = f"{args.exp_name}"

    init_dims = {
        "bbox": args.dims,
        "observation_area": args.area_obs,
        "obstruction_count": args.obstruct,
        "number_agents": args.agents,
        "enforce_grid_boundaries": True,
        "TEST": args.test,
    }

    if args.cpu > 1:
        # max cpus, steps in batch must be greater than the max eps steps times num. of cpu
        tot_epoch_steps = args.cpu * args.steps_per_epoch
        args.steps_per_epoch = tot_epoch_steps if tot_epoch_steps > args.steps_per_epoch else args.steps_per_epoch
        print(f"Sys cpus (avail, using): ({os.cpu_count()},{args.cpu}), Steps set to {args.steps_per_epoch}")
        # run parallel code with mpi
        mpi_fork(args.cpu)

    # Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed((1 + proc_id()) * args.seed))[0]
    rng = np.random.default_rng(robust_seed)
    init_dims["np_random"] = rng

    # Setup logger for tracking training metrics
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir="../../models/train", env_name=args.env_name)

    ac_kwargs = dict(
        predictor_hidden_size=args.hid_rec[0],
    )

    # Run ppo training function
    if args.training:
        ppo(
            lambda: gym.make(args.env, **init_dims),
            actor_critic=CNNBase,
            ac_kwargs=ac_kwargs,
            gamma=args.gamma,
            alpha=args.alpha,
            seed=robust_seed,
            steps_per_epoch=args.steps_per_epoch,
            max_ep_len=args.steps_per_episode,
            epochs=args.epochs,
            dims=init_dims,
            logger_kwargs=logger_kwargs,
            render=args.render,
            save_gif=args.render,
            load_model=args.load_model,
            PFGRU=args.PFGRU,
            number_of_agents=args.agents,
            mode=args.mode,
            save_freq=save_freq,
            save_gif_freq=args.save_gif_freq
        )
    else:
        raise Exception("Training mode not indicated. Evaluation mode is currently run independent of this function, see README.")
