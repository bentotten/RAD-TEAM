from typing import (
    Any,
    List,
    Union,
    Literal,
    NewType,
    Optional,
    TypedDict,
    cast,
    get_args,
    Dict,
    NamedTuple,
    Tuple,
    List,
)

from dataclasses import dataclass, field
import os
import sys
import math

import numpy as np
import numpy.typing as npt
import numpy.random as npr

import gym  # type: ignore
from gym import spaces  # type: ignore

import visilibity as vis  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.animation as animation  # type: ignore
from matplotlib.ticker import FormatStrFormatter  # type: ignore
from matplotlib.animation import PillowWriter  # type: ignore
from matplotlib.patches import Polygon as PolygonPatches  # type: ignore
from matplotlib.markers import MarkerStyle  # type: ignore


from typing_extensions import TypeAlias


Point = NewType("Point", Tuple[float, float])
Polygon = NewType("Polygon", List[Point])
Interval = NewType("Interval", Tuple[float, float])
BBox = NewType("BBox", Tuple[Point, Point, Point, Point])
Colorcode = NewType("Colorcode", List[int])
Color = NewType("Color", npt.NDArray[np.float64])

Metadata = TypedDict(
    "Metadata", {"render.modes": List[str], "video.frames_per_second": int}
)

MAX_CREATION_TRIES = 1000000000

GLOBAL_REWARD = False # Beat the global minimum shortest path distance or get punished
PROPORTIONAL_REWARD = False if GLOBAL_REWARD else False # Get rewarded for improving your own shortest path, proportional to last time. Closest agent gets saved.
BASIC_REWARD = False if (GLOBAL_REWARD or PROPORTIONAL_REWARD) else True # 0 for every good step; prevents agent from gaining rewards by maximizing the episode length
ORIGINAL_REWARD = False if (GLOBAL_REWARD or PROPORTIONAL_REWARD or BASIC_REWARD) else True # +0.1 for every step that is closer than prev shortest path. Unfortunately rewards agent for extending episode

BASIC_SUC_AMOUNT = 1.0

# These actions correspond to:
# -1: stay idle
# 0: left
# 1: up and left
# 2: up
# 3: up and right
# 4: right
# 5: down and right
# 6: down
# 7: down and left
Action: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]
Directions: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7]

A_SIZE = len(get_args(Action))
DETECTABLE_DIRECTIONS = len(get_args(Directions))  # Ignores -1 idle state
DET_STEP = 100.0  # detector step size at each timestep in cm/s
DET_STEP_FRAC = 71.0  # diagonal detector step size in cm/s
DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC = (
    78.0  # Diagonal detector-obstruction range measurement threshold in cm #TODO unused
)
EPSILON = (
    0.0000001  # Parameter for Visilibity function to check if environment is valid
)
COLORS = [
    # Colorcode([148, 0, 211]), # Violet (Removed due to being too similar to indigo)
    Colorcode([255, 105, 180]),  # Pink
    Colorcode([75, 0, 130]),  # Indigo
    Colorcode([0, 0, 255]),  # Blue
    Colorcode([0, 255, 0]),  # Green
    Colorcode([255, 127, 0]),  # Orange
]
COLOR_FACTOR = 0.75  # How much to lighten previous rendered step by

# Rendering
ACTION_MAPPING: Dict = {
    0: "Left",
    1: "Up Left",
    2: "Up",
    3: "Up Right",
    4: "Right",
    5: "Down Right",
    6: "Down",
    7: "Down Left",
    8: "Idle",
}
FPS = 5


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def sum_p(p1: Point, p2: Point) -> Point:
    """
    Return the sum of the two points.
    """
    return Point((p1[0] + p2[0], p1[1] + p2[1]))


def sub_p(p1: Point, p2: Point) -> Point:
    """
    Return the difference of the two points.
    """
    return Point((p1[0] - p2[0], p1[1] - p2[1]))


def scale_p(p: Point, s: float) -> Point:
    """
    Return the scaled point p by the scalar s.
    """
    return Point((p[0] * s, p[1] * s))


def dist_p(p1: Point, p2: Point) -> float:
    """
    Return the distance between the two points.
    """

    def dist_sq_p(p1: Point, p2: Point) -> float:
        """
        Return the squared distance between the two points.
        """
        return float((p1[0] - p2[0]) ** 2) + float((p1[1] - p2[1]) ** 2)

    return math.sqrt(dist_sq_p(p1, p2))


def from_vis_p(p: vis.Point) -> Point:
    """
    Return a Point from a visilibity Point.
    """
    return Point((p.x(), p.y()))  # type: ignore


def to_vis_p(p: Point) -> vis.Point:
    """
    Return a visilibity Point from a Point.
    """
    return vis.Point(p[0], p[1])


def from_vis_poly(poly: vis.Polygon) -> Polygon:
    """
    Return a Polygon from a visilibity Polygon.
    """
    return Polygon(list(map(from_vis_p, poly)))


def to_vis_poly(poly: Polygon) -> vis.Polygon:
    """
    Return a visilibity Polygon from a Polygon.
    """
    return vis.Polygon(list(map(to_vis_p, poly)))


def count_matching_p(p1: Point, point_list: List[Point]) -> int:
    """
    Count number of times a Point appears in a list
    """
    count = 0
    for p2 in point_list:
        if p1[0] == p2[0] and p1[1] == p2[1]:
            count += 1
    return count


def get_step_size(action: Union[Action, int]) -> float:
    """
    Return the step size for the given action.
    """

    return DET_STEP if action % 2 == 0 else DET_STEP_FRAC


def get_y_step_coeff(action: Union[Action, int]) -> float:
    """
    action (Action): Scalar representing desired travel angle
    idle_action (Action): Action representing idle state (usually the maximum action)
    """
    # return math.sin(2 * math.pi * action / idle_action) if action != idle_action else 0
    return round(math.sin(math.pi * (1.0 - action / 4.0)))


# Get the new X coordinate for an arbritrary action angle
def get_x_step_coeff(action: Union[Action, int]) -> float:
    """
    action (Action): Scalar representing desired travel angle
    idle_action (Action): Action representing idle state (usually the maximum action)
    """
    # return math.cos(2 * math.pi * action / idle_action)
    return get_y_step_coeff((action + 6) % 8)


def get_step(action: Union[Action, int]) -> Point:
    """
    Return the step offset for the given action, scaled
        0: left
        1: up left
        2: up
        3: up right
        4: right
        5: down right
        6: down
        7: down left
        8: Idle
    """
    if action == A_SIZE - 1:  # if max action
        return Point((0.0, 0.0))
    else:
        return scale_p(
            Point((get_x_step_coeff(action=action), get_y_step_coeff(action=action))),
            get_step_size(action),
        )


def create_color(id: int) -> Color:
    """Pick initial Colorcode based on id number, then offset it"""
    specific_color: Colorcode = COLORS[id % (len(COLORS))]  #
    if id > (len(COLORS) - 1):
        offset: int = (
            id * 22
        ) % 255  # Create large offset for that base color, bounded by 255
        specific_color[id % 3] = (
            255 + specific_color[id % 3] - offset
        ) % 255  # Perform the offset
    return Color(np.array(specific_color) / 255)


def lighten_color(color: Color, factor: float) -> Color:
    """increase tint of a color"""
    scaled_color = color * 255  # return to original scale
    return Color(
        np.array(list(map(lambda c: (c + (255 - c) * factor) / 255, scaled_color)))
    )


def ping():
    return "PONG!"


class StepResult(NamedTuple):
    observation: Dict[int, npt.NDArray[np.float32]]
    reward: Dict[int, float]
    terminal: Dict[int, bool]
    info: Dict[int, Dict[Any, Any]]


@dataclass
class Agent:
    sp_dist: float = field(
        init=False
    )  # Shortest path distance between agent and source
    euc_dist: float = field(init=False)  # Crow-Flies distance between agent and source
    det_coords: Point = field(init=False)  # Detector Coordinates
    out_of_bounds: bool = field(init=False)
    out_of_bounds_count: int = field(init=False)
    collision: bool = field(init=False)
    intersect: bool = field(
        default=False
    )  # Check if line of sight is blocked by obstacle
    obstacle_blocking: bool = field(
        default=False
    )  # For position assertions and testing
    detector: vis.Point = field(init=False)  # Visilibity graph detector coordinates
    prev_det_dist: float = field(init=False)
    id: int = field(default=0)

    # Rendering
    marker_color: Color = field(init=False)
    det_sto: List[Point] = field(init=False)  # Coordinate history for episdoe
    meas_sto: List[float] = field(init=False)  # Measurement history for episode
    team_reward_sto: List[float] = field(init=False)  # Team Reward history for epsisode
    cum_reward_sto: List = field(init=False)  # Cumulative rewards tracker for episode
    action_sto: List = field(init=False)  # Stores actions for render
    terminal_sto: List = field(init=False)

    def __post_init__(self) -> None:
        self.marker_color: Color = create_color(self.id)
        self.reset()

    def reset(self) -> None:
        self.obstacle_blocking = False
        self.out_of_bounds = False
        self.out_of_bounds_count = 0
        self.det_sto: List[Point] = []  # Coordinate history for episdoe
        self.meas_sto: List[float] = []  # Measurement history for episode
        self.team_reward_sto: List[float] = []  # Team Reward history for epsisode
        self.cum_reward_sto: List = []  # Cumulative rewards tracker for episode
        self.action_sto: List = []
        self.terminal_sto: List = []


@dataclass
class RadSearch(gym.Env):
    """
    bbox is the "bounding box"

    Dimensions of radiation source search area in cm, decreased by observation_area param. to ensure visilibity graph setup is valid.

    observation_area: Interval for each obstruction area in cm. The actual search area will be the bounds box decreased by this amount. This is also used to offset obstacles from one another

    np_random: A random number generator

    obstruction_count: Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions
    """

    # Environment
    #    BBox = NewType("BBox", Tuple[Point, Point, Point, Point])
    bbox: BBox = field(
        default_factory=lambda: BBox(
            tuple((Point((0.0, 0.0)), Point((2700.0, 0.0)), Point((2700.0, 2700.0)), Point((0.0, 2700.0))))  # type: ignore
        )
    )
    observation_area: Interval = field(default_factory=lambda: Interval((200.0, 500.0)))
    np_random: npr.Generator = field(default_factory=lambda: npr.default_rng(0))
    obstruction_count: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7] = field(default=0)
    enforce_grid_boundaries: bool = field(default=False)
    save_gif: bool = field(default=False)
    env_ls: List[Polygon] = field(init=False)
    max_dist: float = field(init=False)
    line_segs: List[List[vis.Line_Segment]] = field(init=False)
    poly: List[Polygon] = field(init=False)
    search_area: BBox = field(
        init=False
    )  # Area Detector and Source will spawn in - each must be 1000 cm apart from the source
    walls: Polygon = field(init=False)
    world: vis.Environment = field(init=False)
    vis_graph: vis.Visibility_Graph = field(init=False)
    intensity: int = field(init=False)
    bkg_intensity: int = field(init=False)
    obs_coord: List[List[Point]] = field(init=False)

    # Agents
    agents: Dict[int, Agent] = field(init=False)
    step_size = DET_STEP
    global_min_shortest_path: float = field(init=False)

    # Source
    # TODO move into own class to easily handle multi-source
    src_coords: Point = field(init=False)
    source: vis.Point = field(init=False)

    # Values with default values which are not set in the constructor
    number_agents: int = 1
    action_space: spaces.Discrete = spaces.Discrete(A_SIZE)
    number_actions: int = A_SIZE
    detectable_directions: int = DETECTABLE_DIRECTIONS
    background_radiation_bounds: Point = Point((10, 51))
    continuous: bool = False
    done: bool = False
    epoch_cnt: int = 0
    radiation_intensity_bounds: Point = Point((1e6, 10e6))
    metadata: Metadata = field(default_factory=lambda: {"render.modes": ["human"], "video.frames_per_second": FPS})  # type: ignore
    observation_space: spaces.Box = spaces.Box(0, np.inf, shape=(11,), dtype=np.float32)
    coord_noise: bool = False
    #seed: Union[int, None] = field( default=None)  # TODO make env generation work with this
    scale: float = field(init=False)  # Used to deflate and inflate coordinates
    scaled_grid_max: Tuple = field(
        default_factory=lambda: (1, 1)
    )  # Max x and max y for grid after deflation
    epoch_end: bool = field(
        default=False
    )  # flag to reset/sample new environment parameters. This is necessary when runnning monte carlo evaluations to ensure env is standardized for all evaluation, unless indicated.

    # Step return mode
    step_data_mode: str = field(default='dict')

    # Rendering
    iter_count: int = field(
        default=0
    )  # For render function, believe it counts timesteps
    all_agent_max_count: float = field(
        init=False
    )  # Sets y limit for radiation count graph
    render_counter: int = field(default=0)
    silent: bool = field(init=False)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Stage 1
    TEST: int = field(default=0)
    DEBUG: bool = field(default=False)
    DEBUG_SOURCE_LOCATION: Point = field(default=Point((1, 1)))
    DEBUG_DETECTOR_LOCATION: Point = Point((1499.0, 1499.0))
    MIN_STARTING_DISTANCE: float = field(default=1000) # cm

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def __post_init__(self) -> None:
        # Debugging tests
        # Test 1: 15x15 grid, no obstructions, fixed start and stop points
        if self.DEBUG:
            print(f"Reward Mode - Global: {GLOBAL_REWARD}. Proportional: {PROPORTIONAL_REWARD}. Basic {BASIC_REWARD}. Original: {ORIGINAL_REWARD}")
            if BASIC_REWARD:
                print(f"Basic Reward upon success: {BASIC_SUC_AMOUNT}")    
        if self.TEST == 1:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 1 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0,0.0)),Point((1500.0,0.0)),Point((1500.0,1500.0)), Point((0.0,1500.0))))
            self.observation_area = Interval((100.0,100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.DEBUG_SOURCE_LOCATION = Point((1, 1))
            self.DEBUG_DETECTOR_LOCATION = Point((1499.0, 1499.0))

        # Test 2: 15x15 grid, no obstructions, fixed stop point
        elif self.TEST == 2:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 2 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0,0.0)),Point((1500.0,0.0)),Point((1500.0,1500.0)), Point((0.0,1500.0))))
            self.observation_area = Interval((100.0,100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.DEBUG_SOURCE_LOCATION = Point((1, 1))

        # Test 3: 15x15 grid, no obstructions, fixed start point
        elif self.TEST == 3:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 3 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0,0.0)),Point((700.0,0.0)),Point((700.0, 700.0)), Point((0.0, 700.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.DEBUG_DETECTOR_LOCATION = Point((699.0, 699.0))
            self.MIN_STARTING_DISTANCE = 350 # cm

        # Test 2: 15x15 grid, no obstructions
        elif self.TEST == 4:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 4 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0,0.0)),Point((700.0,0.0)),Point((700.0, 700.0)), Point((0.0, 700.0))))
            self.observation_area = Interval((100.0,100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.MIN_STARTING_DISTANCE = 350 # cm

        self.search_area: BBox = BBox(
            (
                Point(
                    (
                        self.bbox[0][0] + self.observation_area[0],
                        self.bbox[0][1] + self.observation_area[0],
                    )
                ),
                Point(
                    (
                        self.bbox[1][0] - self.observation_area[1],
                        self.bbox[1][1] + self.observation_area[0],
                    )
                ),
                Point(
                    (
                        self.bbox[2][0] - self.observation_area[1],
                        self.bbox[2][1] - self.observation_area[1],
                    )
                ),
                Point(
                    (
                        self.bbox[3][0] + self.observation_area[0],
                        self.bbox[3][1] - self.observation_area[1],
                    )
                ),
            )
        )
        self.epoch_end = True
        self.agents = {i: Agent(id=i) for i in range(self.number_agents)}
        self.max_dist: float = dist_p(
            self.search_area[2], self.search_area[1]
        )  # Maximum distance between two points within search area

        # Assure there is room to spawn detectors and source with proper spacing
        assert (self.max_dist > self.MIN_STARTING_DISTANCE), "Maximum distance available is too small, unable to spawn source and detector 1000 cm apart"

        self.scale = 1 / self.search_area[2][1]  # Needed for CNN network scaling
        
        # Set initial shortest path to be zero
        self.global_min_shortest_path = 0

        self.reset()

    def ping(self):
        """Test environemnt communication"""
        return "PONG"

    def step(
        self, action: Optional[Union[int, Dict, Action]] = None
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
        """
        Wrapper that captures gymAI env.step() and expands to include multiple agents for one "timestep".
        Accepts literal action for single agent, or a Dict of agent-IDs and actions.

        Returns dictionary of agent IDs and StepReturns. Agent coodinates are scaled for graph.

        Action:
        Literal single-action. Empty Action indicates agent is stalling for a timestep.

        action_list:
        A Dict of agent_IDs and their corresponding Actions. If none passed in, this will return just agents current states (often used during a environment reset).

        """

        def agent_step(
            action: Optional[Union[int, Action]],
            agent: Agent,
            proposed_coordinates: List[Point] = [],
        ) -> Tuple[npt.NDArray[np.float32], float, bool, Dict[Any, Any]]:
            """
            Method that takes an action and updates the detector position accordingly.
            Returns an observation, reward, and whether the termination criteria is met.

            Action:
            Single proposed action represented by a scalar value

            Agent:
            Agent to take the action

            Proposed Coordinates:
            A List of all resulting coordinates if all agents successfully take their actions. Used for collision prevention.
            """
            # Initial values
            agent.out_of_bounds = False
            agent.collision = False

            # Sanity check values
            measurement: Union[None, float] = None
            reward: Union[None, float] = None

            if self.take_action(
                agent=agent, action=action, proposed_coordinates=proposed_coordinates
            ):
                # Returns the length of a Polyline, which is a double
                # https://github.com/tsaoyu/PyVisiLibity/blob/80ce1356fa31c003e29467e6f08ffdfbd74db80f/visilibity.cpp#L1398
                agent.sp_dist: float = self.world.shortest_path(  # type: ignore
                    self.source, agent.detector, self.vis_graph, EPSILON
                ).length()
                agent.euc_dist = dist_p(agent.det_coords, self.src_coords)
                agent.intersect = self.is_intersect(
                    agent
                )  # checks if the line of sight is blocked by any obstructions in the environment.
                measurement = self.np_random.poisson(
                    self.bkg_intensity
                    if agent.intersect
                    else self.intensity / agent.euc_dist + self.bkg_intensity
                )

                assert measurement >= 0

                if PROPORTIONAL_REWARD:
                    if agent.sp_dist < 110:
                        reward = 0.1  # NOTE: must be the same value as a non-terminal step in correct direction, as episodes can be cut off prematurely by epoch ending.
                        self.done = True
                        agent.terminal_sto.append(True)
                    elif agent.sp_dist < agent.prev_det_dist:
                        reward = 0.1 * -1 * (agent.sp_dist - agent.prev_det_dist)
                        agent.prev_det_dist = agent.sp_dist
                        agent.terminal_sto.append(False)
                    else:
                        agent.terminal_sto.append(False)
                        if action == max(get_args(Action)):
                            reward = (
                                -1.0 * agent.sp_dist / self.max_dist
                            )  # If idle, extra penalty
                        else:
                            reward = -0.5 * agent.sp_dist / self.max_dist
                # If using global reward
                elif GLOBAL_REWARD:
                    if agent.sp_dist < 110:
                        reward = 0.1  # NOTE: must be the same value as a non-terminal step in correct direction, as episodes can be cut off prematurely by epoch ending.
                        self.done = True
                        agent.terminal_sto.append(True)
                    elif agent.sp_dist < self.global_min_shortest_path:
                        self.global_min_shortest_path = agent.sp_dist
                        reward = 0.1 
                        agent.prev_det_dist = agent.sp_dist
                        agent.terminal_sto.append(False)
                    else:
                        agent.terminal_sto.append(False)
                        # If idle, extra penalty                        
                        if action == max(get_args(Action)):
                            reward = (-1.0 * agent.sp_dist / self.max_dist)  
                        else:
                            reward = -0.5 * agent.sp_dist / self.max_dist
                elif BASIC_REWARD:
                    if agent.sp_dist < 110:
                        reward =  BASIC_SUC_AMOUNT 
                        self.done = True
                        agent.terminal_sto.append(True)
                    elif agent.sp_dist < agent.prev_det_dist:
                        reward = 0.0
                        agent.prev_det_dist = agent.sp_dist
                        agent.terminal_sto.append(False)
                    else:
                        agent.terminal_sto.append(False)
                        if action == max(get_args(Action)):
                            reward = (
                                -1.0 * agent.sp_dist / self.max_dist
                            )  # If idle, extra penalty
                        else:
                            reward = -0.5 * agent.sp_dist / self.max_dist
                elif ORIGINAL_REWARD:
                    if agent.sp_dist < 110:
                        reward = 0.1  # NOTE: must be the same value as a non-terminal step in correct direction, as episodes can be cut off prematurely by epoch ending.
                        self.done = True
                        agent.terminal_sto.append(True)
                    elif agent.sp_dist < agent.prev_det_dist:
                        reward = 0.1 
                        agent.prev_det_dist = agent.sp_dist
                        agent.terminal_sto.append(False)
                    else:
                        agent.terminal_sto.append(False)
                        if action == max(get_args(Action)):
                            reward = (
                                -1.0 * agent.sp_dist / self.max_dist
                            )  # If idle, extra penalty
                        else:
                            reward = -0.5 * agent.sp_dist / self.max_dist                                
                else:
                    raise ValueError("Reward scheme error.")

            # If take_action is false, usually due to agent being in obstacle or empty action on env reset.
            else:
                agent.terminal_sto.append(False)
                # If detector starts on obs. edge, it won't have the shortest path distance calculated
                if self.iter_count > 0:
                    # TODO remove, already calculated, Agent isnt moving
                    # agent.euc_dist = dist_p(agent.det_coords, self.src_coords)
                    # agent.sp_dist: float = self.world.shortest_path(  # type: ignore
                    #     self.source, agent.detector, self.vis_graph, EPSILON
                    # ).length()
                    agent.intersect = self.is_intersect(
                        agent
                    )  # checks if the line of sight is blocked by any obstructions in the environment.
                    measurement = self.np_random.poisson(
                        self.bkg_intensity
                        if agent.intersect  # checks if the line of sight is blocked by any obstructions in the environment.
                        else self.intensity / agent.euc_dist + self.bkg_intensity
                    )

                    # Reward logic for no action taken
                    if action == max(get_args(Action)) and not agent.collision:
                        raise ValueError(
                            "Agent should not return false if the tentative step is an idle step"
                        )
                    else:
                        reward = -0.5 * agent.sp_dist / self.max_dist
                else:
                    agent.sp_dist = (
                        agent.prev_det_dist
                    )  # Set in reset function with current coordinates
                    agent.euc_dist = dist_p(agent.det_coords, self.src_coords)
                    agent.intersect = self.is_intersect(agent)
                    measurement = self.np_random.poisson(
                        self.bkg_intensity
                        if agent.intersect  # checks if the line of sight is blocked by any obstructions in the environment.
                        else self.intensity / agent.euc_dist + self.bkg_intensity
                    )

                    if action == max(get_args(Action)) and not agent.collision:
                        raise ValueError(
                            "Take Action function returned false, but 'Idle' indicated"
                        )
                    else:
                        reward = -0.5 * agent.sp_dist / self.max_dist

            # If detector coordinate noise is desired
            noise: Point = Point(
                tuple(self.np_random.normal(scale=5, size=2))  # type: ignore
                if self.coord_noise
                else (0.0, 0.0)
            )

            # Scale detector coordinates by search area of the DRL algorithm
            det_coord_scaled: Point = scale_p(
                sum_p(agent.det_coords, noise), 1 / self.search_area[2][1]
            )

            # Observation with the radiation measurement., detector coords and detector-obstruction range measurement.

            # Sensor measurement for obstacles and boundaries directly around agent
            sensor_meas: npt.NDArray[np.float64] = (
                self.obstruction_sensors(agent=agent)
                if self.num_obs > 0 or self.enforce_grid_boundaries
                else np.zeros(DETECTABLE_DIRECTIONS)
            )
            # State is an 11-tuple ndarray
            # [intensity, x-coord, y-coord, 8 directions of obstacle detection]
            state_observation: npt.NDArray[np.float32] = np.array(
                [measurement, *det_coord_scaled, *sensor_meas]
            )

            # Sanity checks
            assert measurement is not None
            assert reward is not None

            agent.det_sto.append(agent.det_coords)
            agent.meas_sto.append(measurement)
            agent.cum_reward_sto.append(
                reward + agent.cum_reward_sto[-1]
                if len(agent.cum_reward_sto) > 0
                else reward
            )
            agent.action_sto.append(action)
            info = {
                "out_of_bounds": agent.out_of_bounds,
                "out_of_bounds_count": agent.out_of_bounds_count,
                "blocked": agent.obstacle_blocking,
                "scale": 1 / self.search_area[2][1],
            }
            return state_observation, round(reward, 2), self.done, info
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        assert (
            action is None or type(action) == int or type(action) == dict
        ), "Action not integer or a dictionary of actions."

        if type(action) is int:
            if action == -1:
                action = 8
            assert action in get_args(Action)
        elif type(action) is dict:
            for i, a in action.items():
                assert action[i] in get_args(Action)
        action_list = action if type(action) is dict else None


        aggregate_observation_result: Union[Dict, npt.NDArray]
        aggregate_reward_result: Union[Dict, npt.NDArray]
        aggregate_success_result: Union[Dict, npt.NDArray]

        if self.step_data_mode == 'list':
            aggregate_observation_result = np.zeros(combined_shape(self.number_agents, self.observation_space.shape[0]), dtype=np.float32)
            aggregate_reward_result = np.zeros((self.number_agents), dtype=np.float32)
      
        elif self.step_data_mode == 'dict':
            aggregate_observation_result = {_: None for _ in self.agents}
            aggregate_reward_result = {_: None for _ in self.agents}
        else:
            raise NotImplementedError("Unknown step data type mode")
            
        aggregate_success_result = {_: None for _ in self.agents}
        aggregate_info_result: Dict = {_: None for _ in self.agents}
        max_reward: Union[float, npt.Floating, None] = None
        min_distance: Union[float, None] = None   
        winning_id: Union[int, None] = None    

        if action_list:
            proposed_coordinates = [
                sum_p(self.agents[agent_id].det_coords, get_step(action))
                for agent_id, action in action_list.items()
            ]
            
            for agent_id, a in action_list.items():
                (
                    aggregate_observation_result[agent_id],
                    aggregate_reward_result[agent_id],
                    aggregate_success_result[agent_id],
                    aggregate_info_result[agent_id],
                ) = agent_step(
                    agent=self.agents[agent_id],
                    action=a,
                    proposed_coordinates=proposed_coordinates,
                )

            self.iter_count += 1
            # return {k: asdict(v) for k, v in aggregate_step_result.items()}
        elif not action or type(action) is int:
            # Provides backwards compatability for single actions instead of action lists for single agents.
            if type(action) == int and len(self.agents) > 1:
                print(
                    "WARNING: Passing single action to mutliple agents during step!",
                    file=sys.stderr,
                )
            # Used during reset to get initial state or during single-agent move
            for agent_id, agent in self.agents.items():
                (
                    aggregate_observation_result[agent_id],
                    aggregate_reward_result[agent_id],
                    aggregate_success_result[agent_id],
                    aggregate_info_result[agent_id],
                ) = agent_step( action=action, agent=agent)  # type: ignore

            self.iter_count += 1
        else:
            raise ValueError("Incompatible Action type")
        # Parse rewards
        if not PROPORTIONAL_REWARD: 
            # if Global shortest path was used as min shortest path distance        
            if self.step_data_mode == 'dict':
                for agent_id in self.agents:
                    # Calculate team reward
                    if not max_reward:
                        max_reward = aggregate_reward_result[agent_id]
                    elif max_reward < aggregate_reward_result[agent_id]:
                        max_reward = aggregate_reward_result[agent_id]
            elif self.step_data_mode == 'list':
                max_reward = aggregate_reward_result.max() # type: ignore
            else:
                raise ValueError("Unknown step data type mode")
                            
        if PROPORTIONAL_REWARD:
            # If rewards were calculated based on the proportional difference between last and current
            for id, agent in self.agents.items():
                if not min_distance:
                    min_distance = agent.sp_dist
                    winning_id = id                    
                elif min_distance > agent.sp_dist:
                    min_distance = agent.sp_dist
                    winning_id = id
            max_reward = np.round(aggregate_reward_result[winning_id].item(), decimals=2) if (self.step_data_mode == 'list') else aggregate_reward_result[winning_id]
            
        # Save cumulative team reward for rendering
        for agent in self.agents.values():
            if max_reward:
                agent.team_reward_sto.append(max_reward + agent.team_reward_sto[-1] if len(agent.team_reward_sto) > 0 else max_reward )            

        return (
            aggregate_observation_result,
            {"team_reward": max_reward, "individual_reward": aggregate_reward_result},
            aggregate_success_result,
            aggregate_info_result,
        )

    def reset(
        self,
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
        """
        Method to reset the environment.
        """
        for agent in self.agents.values():
            agent.reset()

        self.done = False
        self.iter_count = 0
        self.dwell_time = 1
        self.all_agent_max_count = 0.0

        if self.epoch_end:
            if self.obstruction_count == -1:
                self.num_obs = self.np_random.integers(1, 6)  # type: ignore
            elif self.obstruction_count == 0:
                self.num_obs = 0
            else:
                self.num_obs = self.obstruction_count

            self.create_obs()
            self.walls = Polygon(list(self.bbox))
            self.env_ls: List[Polygon] = [self.walls, *self.poly]

            # Create Visilibity environment
            self.world = vis.Environment(list(map(to_vis_poly, self.env_ls)))

            # Create Visilibity graph to speed up shortest path computation
            self.vis_graph = vis.Visibility_Graph(self.world, EPSILON)
            self.epoch_cnt += 1
            self.epoch_end = False

        (
            self.source,
            detector_vis_start_location,
            detector_start_location,
            self.src_coords,
        ) = self.sample_source_loc_pos()

        for agent in self.agents.values():
            agent.detector = detector_vis_start_location
            agent.det_coords = detector_start_location
            agent.prev_det_dist: float = self.world.shortest_path(  # type: ignore
                self.source, agent.detector, self.vis_graph, EPSILON
            ).length()
        
        # Set first shortest path - assumes all agents start in the same location
        self.global_min_shortest_path = self.agents[0].prev_det_dist
        
        self.intensity = self.np_random.integers(self.radiation_intensity_bounds[0], self.radiation_intensity_bounds[1])  # type: ignore
        self.bkg_intensity = self.np_random.integers(self.background_radiation_bounds[0], self.background_radiation_bounds[1])  # type: ignore
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.DEBUG:
            self.intensity = np.array(1000000, dtype=np.int32).item()
            self.bkg_intensity = np.array(0, dtype=np.int32).item()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Check if the environment is valid
        if not (self.world.is_valid(EPSILON)):
            print("Environment is not valid, retrying!")
            self.epoch_end = True
            self.reset()

        # Get current states
        step = self.step(action=None)
        
        # Reclear iteration count
        self.iter_count = 0
            
        return step

    def refresh_environment(self, env_dict: Dict, id: int, num_obs: int = 0) -> Dict:
        """
        Load saved test environment parameters from dictionary into the current instantiation of environment

        :param env_dict: (Dict) Parameters to refresh environment with
        :param id: (int) ID number of environment for dictionary
        :param num_obs: (int) Number of obstructions
        """

        # Reset counts and flags
        self.epoch_end = False
        self.done = False
        self.iter_count = 0

        key = "env_" + str(id)
        self.src_coords = env_dict[key][0]  # TODO save these in JSON format with labels instead
        self.intensity = env_dict[key][2]
        self.bkg_intensity = env_dict[key][3]
        self.source = to_vis_p(self.src_coords)

        for id, agent in self.agents.items():
            agent.reset()
            if isinstance(env_dict[key][1], tuple):
                agent.det_coords = env_dict[key][1]
            else:
                agent.det_coords = env_dict[key][1].copy()
            agent.detector = to_vis_p(agent.det_coords)

        # Get obstacles from parameters
        if num_obs > 0:
            self.num_obs = len(env_dict[key][4])
            self.poly = []
            self.line_segs = []
            obs_coord = env_dict[key][4]

            # Make compatible with latest Rad-Search env
            self.obs_coord: List[List[Point]] = [[] for _ in range(self.num_obs)]
            for i, obstruction in enumerate(obs_coord):
                self.obs_coord[i].extend(list(map(tuple, obstruction[0])))  # type: ignore

            # Create Visilibity environment
            for obs in self.obs_coord:
                poly = Polygon(obs)
                self.poly.append(poly)
                self.line_segs.append(
                    [
                        vis.Line_Segment(to_vis_p(p1), to_vis_p(p2))
                        for p1, p2 in (
                            (poly[0], poly[1]),
                            (poly[0], poly[3]),
                            (poly[2], poly[1]),
                            (poly[2], poly[3]),
                        )
                    ]
                )
            self.env_ls: List[Polygon] = [self.walls, *self.poly]
            self.world = vis.Environment(list(map(to_vis_poly, self.env_ls)))

            # Check if the environment is valid
            assert self.world.is_valid(EPSILON), "Environment is not valid"
            self.vis_graph = vis.Visibility_Graph(self.world, EPSILON)

        observation, _, _, _ = self.step(action=None)  # Take idle step

        # Erase previous results from agent's memory (for render)
        for id, agent in self.agents.items():
            if isinstance(env_dict[key][1], tuple):
                agent.det_coords = Point(env_dict[key][1])
            else:
                agent.det_sto = [env_dict[key][1].copy()]
            agent.meas_sto = [observation[id][0].copy()]
            agent.prev_det_dist = self.world.shortest_path(
                self.source, agent.detector, self.vis_graph, EPSILON
            ).length()

        # increment iteration counter
        self.iter_count = 1

        # return observation
        return observation

    def take_action(
        self,
        agent: Agent,
        action: Optional[Union[Action, int]],
        proposed_coordinates: List,
        agent_id: int = 0,
    ) -> bool:
        """
        Method that checks which direction to move the detector based on the action.
        If the action moves the detector into an obstruction, the detector position
        will be reset to the prior position.
        0: left
        1: up left
        2: up
        3: up right
        4: right
        5: down right
        6: down
        7: down left
        8: idle

        Return success of action: True if moved, false if stalled.
        """
        # Take no action
        if action is None:
            return False

        roll_back_action: bool = False
        step = get_step(action)
        tentative_coordinates = sum_p(agent.det_coords, step)

        # If proposed move will collide with another agents proposed move,
        if count_matching_p(tentative_coordinates, proposed_coordinates) > 1:
            agent.collision = True
            return False

        agent.detector = to_vis_p(tentative_coordinates)

        # If boundaries are being enforced, do not take action
        # Do not return here, to make compatible for future when agents have dimensions
        # TODO make agents have dimensions like obstacles
        if self.enforce_grid_boundaries:
            if (
                tentative_coordinates[0] < self.bbox[0][0] or tentative_coordinates[1] < self.bbox[0][1]
            ) or (
                self.bbox[2][0] <= tentative_coordinates[0] or self.bbox[2][1] <= tentative_coordinates[1]
            ):
                agent.out_of_bounds = True
                agent.out_of_bounds_count += 1
                roll_back_action = True

        # If grid boundaries are not enforced, out of bounds is the search area
        else:
            lower_b = agent.det_coords[0] < self.search_area[0][0] or agent.det_coords[1] < self.search_area[0][1]
            upper_b = self.search_area[2][0] < agent.det_coords[0] or self.search_area[2][1] < agent.det_coords[1]
            if (lower_b or upper_b):
                agent.out_of_bounds = True
                agent.out_of_bounds_count += 1

        if self.in_obstruction(agent=agent):
            roll_back_action = True
            agent.obstacle_blocking = True

        if roll_back_action:
            # If we're in an obstacle, roll back
            agent.detector = to_vis_p(agent.det_coords)
        else:
            # If we're not in an obstacle, update the detector coordinates
            agent.det_coords = from_vis_p(agent.detector)

        return False if roll_back_action else True

    def create_obs(self) -> None:
        """
        Method that randomly samples obstruction coordinates from 90% of search area dimensions.
        Obstructions are not allowed to intersect.
        """
        ii = 0
        intersect = False
        self.obs_coord: List[List[Point]] = [[] for _ in range(self.num_obs)]
        self.poly: List[Polygon] = []
        self.line_segs: List[List[vis.Line_Segment]] = []
        obs_coord: List[Point] = []

        while ii < self.num_obs:
            seed_x: float = self.np_random.integers(  # type: ignore
                self.search_area[0][0], self.search_area[2][0] * 0.9  # type: ignore
            ).astype(np.float64)
            seed_y: float = self.np_random.integers(  # type: ignore
                self.search_area[0][1], self.search_area[2][1] * 0.9  # type: ignore
            ).astype(np.float64)

            ext_x: float = self.np_random.integers(  # type: ignore
                self.observation_area[0], self.observation_area[1]  # type: ignore
            ).astype(np.float64)
            ext_y: float = self.np_random.integers(  # type: ignore
                self.observation_area[0], self.observation_area[1]  # type: ignore
            ).astype(np.float64)

            obs_coord = [
                Point(t)
                for t in (
                    (seed_x, seed_y),
                    (seed_x, seed_y + ext_y),
                    (seed_x + ext_x, seed_y + ext_y),
                    (seed_x + ext_x, seed_y),
                )
            ]

            if ii > 0:
                kk = 0
                while not intersect and kk < ii:
                    intersect = math.isclose(vis.boundary_distance(to_vis_poly(Polygon(self.obs_coord[kk])), to_vis_poly(Polygon(obs_coord))), 0.0, abs_tol=EPSILON)  # type: ignore
                    if intersect:
                        obs_coord = []
                    kk += 1

            if not intersect:
                self.obs_coord[ii].extend(obs_coord)
                poly = Polygon(obs_coord)
                self.poly.append(poly)
                self.line_segs.append(
                    [
                        vis.Line_Segment(to_vis_p(p1), to_vis_p(p2))
                        for p1, p2 in (
                            (poly[0], poly[1]),
                            (poly[0], poly[3]),
                            (poly[2], poly[1]),
                            (poly[2], poly[3]),
                        )
                    ]
                )
                ii += 1
                intersect = False
                obs_coord = []
            intersect = False

    def sample_source_loc_pos(
        self,
    ) -> Tuple[vis.Point, vis.Point, Point, Point]:
        """
        Method that randomly generate the detector and source starting locations.
        Locations can not be inside obstructions and must be at least 1000 cm apart.
        Detectors all begin at same location.
        """
        det_clear = False
        src_clear = False
        resamp = False
        obstacle_index = 0

        def rand_point() -> Point:
            """
            Generate a random point within the search area.
            """
            return Point(
                tuple(
                    self.np_random.integers(  # type: ignore
                        int(self.search_area[0][0]), int(self.search_area[1][0]), size=2
                    ).astype(np.float64)
                )
            )

        # Generate initial point values
        source: Point = rand_point()

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.DEBUG and self.TEST in [1, 2]:
            source = self.DEBUG_SOURCE_LOCATION
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        src_point = to_vis_p(source)

        detector = rand_point()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.DEBUG and self.TEST in [1, 3]:
            detector = self.DEBUG_DETECTOR_LOCATION
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        det_point = to_vis_p(detector)

        # Check if detectors starting location is in an object
        test_count = 0
        while not det_clear and test_count < MAX_CREATION_TRIES:
            while not resamp and obstacle_index < self.num_obs:
                if det_point._in(to_vis_poly(self.poly[obstacle_index]), EPSILON):  # type: ignore
                    resamp = True
                obstacle_index += 1
            if resamp:
                detector = rand_point()
                det_point = to_vis_p(detector)
                obstacle_index = 0
                resamp = False
            else:
                det_clear = True

            test_count += 1
            if test_count == MAX_CREATION_TRIES:
                raise ValueError(
                    "Creating Environment Failed - Maximum tries exceeded to clear Detector. Check bounds and observation area to ensure source and detector can spawn 10 meters apart (1000 cm)."
                )

        # Check if source starting location is in object and is far enough away from detector
        # TODO change to multi-source
        resamp = False
        inter = False
        obstacle_index = 0
        num_retry = 0

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.DEBUG and self.TEST in [1, 2]:
            pass
        else:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            test_count = 0
            while not src_clear and test_count < MAX_CREATION_TRIES:
                subtest_count = 0
                while (
                    dist_p(detector, source) < self.MIN_STARTING_DISTANCE
                    and subtest_count < MAX_CREATION_TRIES
                ):
                    source = rand_point()
                    subtest_count += 1
                    if subtest_count == MAX_CREATION_TRIES:
                        raise ValueError(
                            "Creating Environment Failed - Maximum tries exceeded to clear Detector. Check bounds and observation area to ensure source and detector can spawn 10 meters apart (1000 cm)."
                        )
                src_point = to_vis_p(source)
                L: vis.Line_Segment = vis.Line_Segment(det_point, src_point)
                while not resamp and obstacle_index < self.num_obs:
                    poly_p: vis.Polygon = to_vis_poly(self.poly[obstacle_index])
                    if src_point._in(poly_p, EPSILON):  # type: ignore
                        resamp = True
                    if not resamp and vis.boundary_distance(L, poly_p) < 0.001:  # type: ignore
                        inter = True
                    obstacle_index += 1
                if self.num_obs == 0 or (num_retry > 20 and not resamp):
                    src_clear = True
                elif resamp or not inter:
                    source = rand_point()
                    src_point = to_vis_p(source)
                    obstacle_index = 0
                    resamp = False
                    inter = False
                    num_retry += 1
                elif inter:
                    src_clear = True

                test_count += 1
                if test_count == MAX_CREATION_TRIES:
                    raise ValueError(
                        "Creating Environment Failed - Maximum tries exceeded to clear Detector. Check bounds and observation area to ensure source and detector can spawn 10 meters apart (1000 cm)."
                    )
        if self.DEBUG:
            dist = dist_p(detector, source)
            print(f"Agent: {detector}. Source: {source}. Distance {dist}")
        return src_point, det_point, detector, source

    def is_intersect(self, agent: Agent, threshold: float = 0.001) -> bool:
        """
        Method that checks if the line of sight is blocked by any obstructions in the environment.
        """
        inter = False
        kk = 0
        L = vis.Line_Segment(agent.detector, self.source)
        while not inter and kk < self.num_obs:
            if vis.boundary_distance(L, to_vis_poly(self.poly[kk])) < threshold and not math.isclose(  # type: ignore
                math.sqrt(agent.euc_dist), agent.sp_dist, abs_tol=0.1
            ):
                inter = True
            kk += 1
        return inter

    def in_obstruction(self, agent: Agent):
        """
        Method that checks if the detector position intersects or is inside an obstruction.
        """
        jj = 0
        obs_boundary = False
        while not obs_boundary and jj < self.num_obs:
            if agent.detector._in(to_vis_poly(self.poly[jj]), EPSILON):  # type: ignore
                obs_boundary = True
            jj += 1

        if obs_boundary:
            bbox: vis.Bounding_Box = to_vis_poly(self.poly[jj - 1]).bbox()
            return all(
                [  # type: ignore
                    agent.detector.y() > bbox.y_min,
                    agent.detector.y() < bbox.y_max,
                    agent.detector.x() > bbox.x_min,
                    agent.detector.x() < bbox.x_max,
                ]
            )
        else:
            return False

    def obstruction_sensors(self, agent: Agent) -> npt.NDArray[np.float64]:
        """
        Method that generates detector-obstruction range measurements with values between 0-1.
        This detects obstructions within 1.1m of itself. 0 means no obstructions were detected.
        Currently supports 8 directions
        """

        # Check for obstacles
        detector_p: Point = from_vis_p(agent.detector)

        segs: List[vis.Line_Segment] = [
            vis.Line_Segment(
                agent.detector, to_vis_p(sum_p(detector_p, get_step(action)))
            )
            for action in cast(Tuple[Directions], get_args(Directions))
        ]

        dists: List[float] = [0.0] * len(
            segs
        )  # Directions where an obstacle is detected
        obs_idx_ls: List[int] = [0] * len(
            self.poly
        )  # Keeps track of how many steps will interect with which obstacle
        inter = 0  # Intersect flag
        seg_dist: List[float] = [0.0] * 4  # Saves obstruction line segments
        if self.num_obs > 0:
            for idx, seg in enumerate(segs):  # TODO change seg to direction_segment
                for obs_idx, poly in enumerate(
                    self.line_segs
                ):  # TODO change poly to obstacle
                    for seg_idx, obs_seg in enumerate(
                        poly
                    ):  # TODO change obs_seg to obstacle_line_segment
                        if inter < 2 and vis.intersect(obs_seg, seg, EPSILON):  # type: ignore
                            # check if step dir intersects poly seg
                            obstacle_distance = vis.distance(seg.first(), obs_seg)
                            line_distance = (DIST_TH - obstacle_distance) / DIST_TH  # type: ignore
                            seg_dist[seg_idx] = line_distance
                            inter += 1
                            obs_idx_ls[obs_idx] += 1
                    if inter > 0:
                        max_distance = max(seg_dist)
                        if max_distance > dists[idx]:
                            dists[idx] = max_distance
                        seg_dist = [0.0] * 4
                inter = 0
            # If there are more than three dists equal to one, we need to correct the coordinates.
            #   Inter only triggers if inter is less than two, however there can be up to 5 intersections ... TODO
            if sum(filter(lambda x: x == 1.0, dists)) > 3:
                # Take the polygon which corresponds to the index with the maximum number of intersections.
                argmax = max(zip(obs_idx_ls, self.poly))[
                    1
                ]  # Gets the line coordinates for the obstacle that is intersecting TODO rename!
                dists = self.correct_coords(poly=argmax, agent=agent)

        assert (
            len(dists) == DETECTABLE_DIRECTIONS
        )  # Sanity check - if this is wrong it will mess up the step return shape of "state" and make training fail much later down the line

        # Check for boundaries
        if self.enforce_grid_boundaries:
            # Check left
            if agent.det_coords[0] - DIST_TH < self.bbox[0][0]:
                distance = abs(agent.det_coords[0] - self.bbox[0][0])
                line_distance = (DIST_TH - distance) / DIST_TH
                assert dists[0] == 0.0
                dists[0] = line_distance

            # Check down
            if agent.det_coords[1] - DIST_TH < self.bbox[0][1]:
                distance = abs(agent.det_coords[1] - self.bbox[0][1])
                line_distance = (DIST_TH - distance) / DIST_TH
                assert dists[6] == 0.0
                dists[6] = line_distance

            # Check right
            if self.bbox[2][0] <= agent.det_coords[0] + DIST_TH:
                distance = abs(self.bbox[2][0] - agent.det_coords[0])
                line_distance = (DIST_TH - distance) / DIST_TH
                assert dists[4] == 0.0
                dists[4] = line_distance

            # Check up
            if self.bbox[2][1] <= agent.det_coords[1] + DIST_TH:
                distance = abs(self.bbox[2][1] - agent.det_coords[1])
                line_distance = (DIST_TH - distance) / DIST_TH
                assert dists[2] == 0.0
                dists[2] = line_distance

        return np.array(dists, dtype=np.float64)

    def correct_coords(self, poly: Polygon, agent: Agent) -> List[float]:
        """
        Method that corrects the detector-obstruction range measurement if more than the correct
        number of directions are being activated due to the Visilibity implementation.
        This often happens when an agent is on the edge of an obstruction.
        """
        x_check: List[bool] = [False] * DETECTABLE_DIRECTIONS
        dist = 0.1
        length = 1
        poly_p: vis.Polygon = to_vis_poly(poly)

        qs: List[Point] = [
            from_vis_p(agent.detector)
        ] * DETECTABLE_DIRECTIONS  # Offsets agent position by 0.1 to see if actually inside obstacle
        dists: List[float] = [0.0] * DETECTABLE_DIRECTIONS
        while not any(x_check):
            for action in cast(Tuple[Directions], get_args(Directions)):
                # Gets slight offset to remove effects of being "on" an obstruction
                step = scale_p(
                    Point(
                        (
                            get_x_step_coeff(action=action),
                            get_y_step_coeff(action=action),
                        )
                    ),
                    dist * length,
                )
                qs[action] = sum_p(qs[action], step)
                if to_vis_p(qs[action])._in(poly_p, EPSILON):  # type: ignore
                    x_check[action] = True

        # i.e. if one outside the poly then
        if sum(x_check) >= 4:
            for ii in [0, 2, 4, 6]:
                if x_check[ii - 1] == True and x_check[ii + 1] == True:
                    dists[ii] = 1.0
                    dists[ii - 1] = 1.0
                    dists[ii + 1] = 1.0
                    # dists[ii - 1 : ii + 2] = [1.0, 1.0, 1.0]  # This causes there to be 11 elements when there should only be 8

        assert (
            len(dists) == DETECTABLE_DIRECTIONS
        )  # Sanity check - if this is wrong it will mess up the step return shape of "state" and make training fail
        return dists

    def render(
        self,
        save_gif: bool = True,
        path: Optional[str] = None,
        epoch_count: Optional[int] = None,
        episode_count: Optional[int] = None,
        just_env: Optional[bool] = False,
        obstacles=[],
        episode_rewards={},
        data=[],
        measurements: Optional[List[float]] = None,
        location_estimate=None,
        silent: bool = False,
    ) -> None:
        """
        Method that produces a gif of the agent interacting in the environment. Only renders one episode at a time.
        """
        reward_length = field(init=False)  # Prevent from being unbound
        # Set up global saver (for changing colors of last graph's agents)
        self.plot_saver = {i: field(init=False) for i in range(self.number_agents)}
        self.silent = silent

        # global location_estimate
        # location_estimate = None # TODO Trying to get out of global scope; this is for source prediction

        def update(
            frame_number: int,
            ax1: plt.Axes,
            ax2: plt.Axes,
            ax3: plt.Axes,
            src: Point,
            area_dim: BBox,
            flattened_rewards: List,
        ) -> None:
            """
            Renders each frame

            :param ax1: Actual grid
            :param ax2: Radiation counts
            :param ax3: Rewards
            :param src: Source coordinates
            :param area_dim:
            :param area_dim: BBox - size of grid
            :param flattened_rewards: flattened rewards between all agents
            :param silent: Indicate if print frame to render

            """
            if not silent:
                print(
                    f"Current Frame: {frame_number}", end="\r"
                )  # Acts as a progress bar

            if self.iter_count == 0:
                print("Agent must take more than one step to render")
                return

            current_index = frame_number % (self.iter_count)
            # global location_estimate # TODO Trying to get out of global scope; this is for source prediction

            # Set up graphs for first frame
            if current_index == 0:
                intensity_sci = "{:.2e}".format(self.intensity)
                ax1.cla()
                ax1.set_title(
                    "Activity: "
                    + intensity_sci
                    + " [gps] Bkg: "
                    + str(self.bkg_intensity)
                    + " [cps]"
                )

                # Plot source
                ax1.scatter(
                    src[0] / 100,
                    src[1] / 100,
                    marker_size,
                    c="red",
                    marker=MarkerStyle("*"),
                    label="Source",
                )

                for agent_id, agent in self.agents.items():
                    data = np.array(agent.det_sto[current_index]) / 100
                    data_sub = (np.array(agent.det_sto[current_index + 1]) / 100) - (
                        np.array(agent.det_sto[current_index]) / 100
                    )
                    orient = math.degrees(math.atan2(data_sub[1], data_sub[0]))
                    self.plot_saver[agent_id] = ax1.scatter(
                        data[0],
                        data[1],
                        marker_size,
                        c=[agent.marker_color],
                        marker=MarkerStyle((3, 0, orient - 90)),
                    )

                # Plot Obstacles
                ax1.grid()
                if not (obstacles == []) and obstacles != None:
                    for coord in obstacles:
                        # p_disp = PolygonPatches(coord[0] / 100, color="gray")
                        p_disp = PolygonPatches(np.array(coord) / 100, color="gray")
                        ax1.add_patch(p_disp)

                # Plot location prediction
                # TODO make multi-agent and fix
                # if not (location_estimate is None):
                #     location_estimate = ax1.scatter(
                #         location_estimate[0][current_index][1] / 100,
                #         location_estimate[0][current_index][2] / 100,
                #         marker_size,
                #         c="magenta",
                #         label="Loc. Pred.",
                #     )

                # Finish setting up grids
                ax1.set_xlim(0, area_dim[1][0] / 100)
                ax1.set_ylim(0, area_dim[2][1] / 100)
                ax1.set_xticks(np.linspace(0, area_dim[1][0] / 100 - 2, 5))
                ax1.set_yticks(np.linspace(0, area_dim[1][0] / 100 - 2, 5))
                ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax1.yaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax1.set_xlabel("X[m]")
                ax1.set_ylabel("Y[m]")
                ax1.legend(
                    loc="lower right", fontsize=8
                )  # TODO get agent labels to stay put

                # Set up radiation graph
                # TODO make this less terrible
                for agent in self.agents.values():
                    count_max: float = max(agent.meas_sto)
                    if count_max > self.all_agent_max_count:
                        self.all_agent_max_count = count_max
                # count_max: float = 0.0
                # for agent in self.agents.values():
                #     for measurement in agent.meas_sto:
                #         if count_max < measurement:
                #             count_max = measurement
                ax2.cla()
                ax2.set_xlim(0, self.iter_count)
                ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax2.set_ylim(0, (self.all_agent_max_count + 50))
                # ax2.set_ylim(0, self.intensity)
                ax2.set_xlabel("n")
                ax2.set_ylabel("Counts")
                for agent_id, agent in self.agents.items():
                    markerline, _, _ = ax2.stem(
                        [0],
                        [agent.meas_sto[0]],
                        use_line_collection=True,
                        label=f"Detector {agent_id}",
                    )
                    current_color = tuple(agent.marker_color)
                    markerline.set_markerfacecolor(current_color)
                    markerline.set_markeredgecolor(current_color)
                if self.number_agents > 5:
                    ax2.legend(
                        loc="lower center",
                        fontsize=8,
                        bbox_to_anchor=(0.5, -0.25),
                        ncol=5,
                        fancybox=True,
                        shadow=True,
                    )
                else:
                    ax2.legend(
                        loc="lower center",
                        fontsize=8,
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=5,
                        fancybox=True,
                        shadow=True,
                    )

                # Set up rewards graph
                # flattened_rewards = [x for v in episode_rewards.values() for x in v]
                ax3.cla()
                ax3.set_xlim(0, self.iter_count)
                ax3.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax3.set_ylim(min(flattened_rewards) - 0.5, max(flattened_rewards) + 0.5)
                ax3.set_xlabel("n")
                ax3.set_ylabel("Cumulative Reward")
                ax3.plot()

                # Add movement to bottom of figure
                if self.DEBUG:
                    fig.supxlabel("Start")

            else:  # If not first frame
                # TODO add this back now that multi-agent radppo is up
                # if location_estimate:
                #     location_estimate.remove()

                for agent_id, agent in self.agents.items():
                    data = np.array(agent.det_sto[current_index]) / 100
                    # If not last step, adjust orientation
                    if current_index != len(agent.det_sto) - 1:
                        data_sub = (
                            np.array(agent.det_sto[current_index + 1]) / 100
                        ) - (np.array(agent.det_sto[current_index]) / 100)
                        orient = math.degrees(math.atan2(data_sub[1], data_sub[0]))

                        # Change last detector color to lighter version of itself
                        self.plot_saver[agent_id].set_color(
                            lighten_color(agent.marker_color, factor=COLOR_FACTOR)
                        )

                        # Plot detector
                        self.plot_saver[agent_id] = ax1.scatter(
                            data[0],
                            data[1],
                            marker_size,
                            c=[agent.marker_color],
                            marker=MarkerStyle((3, 0, orient - 90)),
                        )
                    else:
                        # Change last detector color to lighter version of itself
                        self.plot_saver[agent_id].set_color(
                            lighten_color(agent.marker_color, factor=COLOR_FACTOR)
                        )
                        self.plot_saver[agent_id] = ax1.scatter(
                            data[0],
                            data[1],
                            marker_size,
                            c=[agent.marker_color],
                            marker=MarkerStyle((3, 0)),
                        )

                    # Plot detector path if not last step
                    if current_index != len(agent.det_sto) - 1:
                        data_prev: npt.NDArray[np.float64] = (
                            np.array(agent.det_sto[current_index - 1]) / 100
                        )
                        data_current: npt.NDArray[np.float64] = (
                            np.array(agent.det_sto[current_index]) / 100
                        )
                        data_next: npt.NDArray[np.float64] = (
                            np.array(agent.det_sto[current_index + 1]) / 100
                        )
                        line_data: npt.NDArray[np.float64] = np.array(
                            [data_prev, data_current, data_next]
                        )
                        ax1.plot(
                            line_data[0:2, 0],
                            line_data[0:2, 1],
                            3,
                            c=agent.marker_color,
                            alpha=0.3,
                            ls="--",
                        )

                    # Plot radiation counts - stem graph
                    assert agent.meas_sto[current_index] >= 0
                    current_color = tuple(agent.marker_color)
                    markerline, _, _ = ax2.stem(
                        [current_index],
                        [agent.meas_sto[current_index]],
                        use_line_collection=True,
                        label=f"Detector {agent_id}",
                    )
                    markerline.set_markerfacecolor(current_color)
                    markerline.set_markeredgecolor(current_color)

                    # Plot rewards graph - line graph, previous reading connects to current reading
                    # ax3.scatter(current_index, agent.team_reward_sto[current_index], marker=',', c=[agent.marker_color], s=2, label=f"{agent_id}_Detector") # Current team reward
                    # Plots individual rewards
                    # ax3.plot([current_index-1, current_index], agent.cum_reward_sto[current_index-1:current_index+1], c=agent.marker_color, label=f"Detector {agent_id}")  # Cumulative line graph
                    # Plots cumulative rewards
                    ax3.plot(
                        [current_index - 1, current_index],
                        agent.team_reward_sto[current_index - 1 : current_index + 1],
                        c=agent.marker_color,
                        label=f"Detector {agent_id}",
                    )  # Cumulative line graph

                # TODO make multi-agent and fix
                # if not (location_estimate is None):
                #     location_estimate = ax1.scatter(
                #         location_estimate[0][current_index][1] / 100,
                #         location_estimate[0][current_index][2] / 100,
                #         marker_size * 0.8,
                #         c="magenta",
                #         label="Loc. Pred.",
                #     )

                # Add movement to bottom of figure
                action_label = f"Step {current_index}:\n"
                for id, agent in self.agents.items():
                    action_label += f"A{id}: {ACTION_MAPPING[agent.action_sto[current_index]]} - {agent.det_sto[current_index]} \n"
                fig.supxlabel(action_label, fontsize=8)
                # If last step, indicate if terminal or not
                if current_index == len(agent.det_sto) - 2:
                    for agent_id, agent in self.agents.items():
                        if agent.terminal_sto[current_index]:
                            fig.supxlabel(
                                f"Success! Agent {agent_id} found the source!"
                            )

        # Initialize render environment
        if data or measurements:
            print(f"Error: Not implemented for upgraded multi-agent version. TBD")
            return
            # TODO Dont overwrite! Make local var instead
            # TODO update to work with new mulit-agent framework
            # self.intensity = params[0]
            # self.bkg_intensity = params[1]
            # self.src_coords = params[2]
            # self.iter_count = len(measurements)
            # data = np.array(data) / 100
        # else:
        # TODO change to multi-agent
        # data = np.array(agent.det_sto) / 100  # Detector stored locations in an array?
        # measurements = agent.meas_sto # Unneeded?

        # Check only rendering one episode aka data readings available match number of rewards (+1 as rewards dont include the first position).
        # if data.shape[0] != len(episode_rewards)+1:

        # TODO make multiagent
        if episode_rewards:
            print(
                f"Error: Episode rewards are deprecated. Rendering plots from existing agent storage."
            )

        cum_episode_rewards = [a.cum_reward_sto for a in self.agents.values()]
        flattened_rewards = [x for v in cum_episode_rewards for x in v]
        data_length = len(self.agents[0].det_sto)
        reward_length = (
            len(cum_episode_rewards[0]) if len(cum_episode_rewards) > 0 else 0
        )
        if data_length != reward_length and not just_env:
            print(
                f"Error: episode reward array length: {reward_length} does not match existing detector locations array length {data_length}. \
            Check: Are you trying to render more than one episode?"
            )
            return

        if obstacles == []:
            obstacles = self.obs_coord
        if just_env:
            # Setup Graph
            plt.rc("font", size=12)
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)

            # Plot source
            ax1.scatter(
                self.src_coords[0] / 100,
                self.src_coords[1] / 100,
                60,
                c="red",
                marker=MarkerStyle("*"),
                label="Source",
            )
            # Plot Agents
            for agent_id, agent in self.agents.items():
                data = np.array(agent.det_sto[0]) / 100
                ax1.scatter(
                    data[0],
                    data[1],
                    42,
                    c=[agent.marker_color],
                    # c="black",
                    marker=MarkerStyle("^"),
                    label=f"Detector {agent_id}",
                )
            # Plot Obstacles
            ax1.grid()
            if not (obstacles == []):
                for coord in obstacles:
                    p_disp = PolygonPatches((np.array(coord) / 100), color="gray")
                    ax1.add_patch(p_disp)

            # TODO Make multi-agent and fix
            # if not (location_estimate is None):
            #     ax1.scatter(
            #         # location_estimate[0][current_index][1] / 100,
            #         # location_estimate[0][current_index][2] / 100,
            #         location_estimate[0][0][1] / 100,
            #         location_estimate[0][0][2] / 100,
            #         42,
            #         c="magenta",
            #         label="Loc. Pred.",
            #     )
            # Finish Graph
            ax1.set_xlim(0, self.search_area[1][0] / 100)
            ax1.set_ylim(0, self.search_area[2][1] / 100)
            ax1.set_xlabel("X[m]")
            ax1.set_ylabel("Y[m]")
            ax1.legend(
                loc="lower right", fontsize=8
            )  # TODO get agent labels to stay put

            # Save
            if self.save_gif or save_gif:
                if os.path.isdir(str(path) + "/gifs/"):
                    fig.savefig(str(path) + f"/gifs/environment.png")
                else:
                    os.mkdir(str(path) + ".." + "/gifs/")
                    fig.savefig(str(path) + f"/gifs/environment.png")
            else:
                plt.show()
            # Figure is not reused, ok to close
            plt.close(fig)
            if not self.silent:
                print(f"Render Complete", end="\r")  # Acts as a progress bar
                print("Figures open", plt.get_fignums())
            return

        else:
            # Setup Graph for gif
            plt.rc("font", size=12)
            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(15, 5), tight_layout=True
            )
            marker_size = 25
            # fig.suptitle(
            #     'Multi-Agent Radiation Localization', fontsize=16)

            # Setup animation
            if not self.silent:
                print(f"Rendering in {str(path)}/gifs/")
                print(f"Frames to render: ", reward_length - 1)

            if data_length > 1:
                ani = animation.FuncAnimation(
                    fig,
                    update,
                    # frames=reward_length,
                    frames=data_length,
                    fargs=(
                        ax1,
                        ax2,
                        ax3,
                        self.src_coords,
                        self.bbox,
                        flattened_rewards,
                    ),
                )
                if self.save_gif or save_gif:
                    if self.DEBUG:
                        fps = 1
                    else:
                        fps = FPS
                    writer = PillowWriter(fps=fps)
                    if not os.path.isdir(str(path) + "/gifs/"):
                        os.mkdir(str(path) + "/gifs/")
                    # ani.save(str(path) + f"/gifs/test_epoch{epoch_count}.gif", writer=writer)
                    ani.save(
                        f"{str(path)}/gifs/epoch_{epoch_count}-{episode_count}({self.render_counter}).gif",
                        writer=writer,
                    )

                else:
                    plt.show()
                self.render_counter += 1
            return

    def get_agent_outOfBounds_count(self, id: int)-> int:
        return self.agents[id].out_of_bounds_count

    # TODO make multi-agent
    def FIM_step(
        self, agent: Agent, action: Action, coords: Optional[Point] = None
    ) -> Point:
        """
        Method for the information-driven controller to update detector coordinates in the environment
        without changing the actual detector positon.

        Args:
        action : action to move the detector
        coords : coordinates to move the detector from that are different from the current detector coordinates
        """

        # Make a copy of the current detector coordinates
        detector_coordinates = agent.det_coords  # TODO make multi-agent
        det_coords = detector_coordinates
        if coords:
            coords_p: vis.Point = to_vis_p(coords)
            agent.detector = coords_p
            agent.det_coords = coords  # TODO make multi-agent

        in_obs = (
            False if self.take_action(agent, action, proposed_coordinates=[]) else True
        )
        detector_coordinates = agent.det_coords  # TODO make multi-agent
        det_ret = detector_coordinates
        if coords is None and not in_obs or coords:
            # If successful movement, return new coords. Set detector back.
            det_coords_p: vis.Point = to_vis_p(det_coords)
            agent.detector = det_coords_p
            agent.det_coords = det_coords  # TODO make multi-agent

        return det_ret
