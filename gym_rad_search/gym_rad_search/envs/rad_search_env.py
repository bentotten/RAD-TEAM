import gym
import numpy as np
import math
import visilibity as vis
import os
import sys
from gym import spaces
from matplotlib.patches import Polygon

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
)

from dataclasses import dataclass, field

import numpy.typing as npt
import numpy.random as npr

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

Metadata = TypedDict("Metadata", {"render.modes": List[str], "video.frames_per_second": int})

MAX_CREATION_TRIES = 1000000000

GLOBAL_REWARD = False  # Beat the global minimum shortest path distance or get punished
PROPORTIONAL_REWARD = (
    False if GLOBAL_REWARD else False
)  # Get rewarded for improving your own shortest path, proportional to last time. Closest agent gets saved.
BASIC_REWARD = (
    False if (GLOBAL_REWARD or PROPORTIONAL_REWARD) else True
)  # 0 for every good step; prevents agent from gaining rewards by maximizing the episode length
ORIGINAL_REWARD = (
    False if (GLOBAL_REWARD or PROPORTIONAL_REWARD or BASIC_REWARD) else True
)  # +0.1 for every step that is closer than prev shortest path. Unfortunately rewards agent for extending episode

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
DIST_TH_FRAC = 78.0  # Diagonal detector-obstruction range measurement threshold in cm #TODO unused
EPSILON = 0.0000001  # Parameter for Visilibity function to check if environment is valid
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
        offset: int = (id * 22) % 255  # Create large offset for that base color, bounded by 255
        specific_color[id % 3] = (255 + specific_color[id % 3] - offset) % 255  # Perform the offset
    return Color(np.array(specific_color) / 255)


def lighten_color(color: Color, factor: float) -> Color:
    """increase tint of a color"""
    scaled_color = color * 255  # return to original scale
    return Color(np.array(list(map(lambda c: (c + (255 - c) * factor) / 255, scaled_color))))


def ping():
    return "PONG!"


class StepResult(NamedTuple):
    observation: Dict[int, npt.NDArray[np.float32]]
    reward: Dict[int, float]
    terminal: Dict[int, bool]
    info: Dict[int, Dict[Any, Any]]


### Legacy Globals ###

FPS = 50
DET_STEP = 100.0  # Detector step size at each timestep in cm/s
DET_STEP_FRAC = 71.0  # Diagonal detector step size in cm/s
DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC = 78.0  # Diagonal detector-obstruction range measurement threshold in cm
EPSILON = 0.0000001

### Legacy helper functions ###
def edges_of(vertices):
    """
    Return the vectors for the edges of the polygon p.

    p is a polygon.
    """
    edges = []
    N = len(vertices)

    for i in range(N):
        edge = vertices[i] - vertices[(i + 1) % N]
        edges.append(edge)

    return edges


def orthogonal(v):
    """
    Return a 90 degree clockwise rotation of the vector v.
    """
    return np.array([-v[1], v[0]])


def is_separating_axis(o, p1, p2):
    """
    Return True and the push vector if o is a separating axis of p1 and p2.
    Otherwise, return False and None.
    """
    min1, max1 = float("+inf"), float("-inf")
    min2, max2 = float("+inf"), float("-inf")

    for v in p1:
        projection = np.dot(v, o)

        min1 = min(min1, projection)
        max1 = max(max1, projection)

    for v in p2:
        projection = np.dot(v, o)

        min2 = min(min2, projection)
        max2 = max(max2, projection)

    if max1 >= min2 and max2 >= min1:
        return False
    else:
        return True


def collide(p1, p2):
    """
    Return True and the MPV if the shapes collide. Otherwise, return False and
    None.

    p1 and p2 are lists of ordered pairs, the vertices of the polygons in the
    clockwise direction.
    """

    edges = edges_of(p1)
    edges += edges_of(p2)
    orthogonals = [orthogonal(e) for e in edges]

    for o in orthogonals:
        separates = is_separating_axis(o, p1, p2)

        if separates:
            # they do not collide and there is no push vector
            return False
    return True


@dataclass
class Agent:
    sp_dist: float = field(init=False)  # Shortest path distance between agent and source
    euc_dist: float = field(init=False)  # Crow-Flies distance between agent and source
    det_coords: Point = field(init=False)  # Detector Coordinates
    out_of_bounds: bool = field(init=False)
    out_of_bounds_count: int = field(init=False)
    collision: bool = field(init=False)
    intersect: bool = field(default=False)  # Check if line of sight is blocked by obstacle
    obstacle_blocking: bool = field(default=False)  # For position assertions and testing
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
        self.detector = vis.Point(0, 0)
        self.det_coords = None
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

    observation_area: Interval for each obstruction area in cm. The actual search area will be the bounds box decreased by this amount. This is also
        used to offset obstacles from one another

    np_random: A random number generator

    obstruction_count: Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions,
        [1-7] -> 1 to 7 obstructions
    """

    # Environment
    #    BBox = NewType("BBox", Tuple[Point, Point, Point, Point])
    # TODO might be more efficient as an array
    # bbox: BBox = field(
    #     default_factory=lambda: BBox(tuple((Point((0.0, 0.0)), Point((2700.0, 0.0)), Point((2700.0, 2700.0)), Point((0.0, 2700.0)))))  # type: ignore
    # )
    bbox: BBox = field(
        default_factory=lambda: np.asarray([[0.0, 0.0], [2700.0, 0.0], [2700.0, 2700.0], [0.0, 2700.0]]) 
    )
    observation_area: Interval = field(default_factory=lambda: Interval((200.0, 500.0)))  # Size of obstructions
    np_random: npr.Generator = field(default_factory=lambda: npr.default_rng(0))
    obstruction_count: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7] = field(default=0)
    obstruction_max: int = field(default=7)
    enforce_grid_boundaries: bool = field(default=False)
    save_gif: bool = field(default=False)
    env_ls: List[Polygon] = field(init=False)
    max_dist: float = field(init=False)
    line_segs: List[List[vis.Line_Segment]] = field(init=False)
    poly: List[Polygon] = field(init=False)
    search_area: BBox = field(init=False)  # Area Detector and Source will spawn in - each must be 1000 cm apart from the source
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
    # seed: Union[int, None] = field( default=None)  # TODO make env generation work with this
    scale: float = field(init=False)  # Used to deflate and inflate coordinates
    scaled_grid_max: Tuple = field(default_factory=lambda: (1, 1))  # Max x and max y for grid after deflation
    # flag to reset/sample new environment parameters. This is necessary when runnning monte carlo evaluations to ensure env is standardized for all
    #   evaluation, unless indicated.
    epoch_end: bool = field(
        default=False
    )

    # Step return mode
    step_data_mode: str = field(default="dict")

    # Rendering and print
    iter_count: int = field(default=0)  # For render function, believe it counts timesteps
    all_agent_max_count: float = field(init=False)  # Sets y limit for radiation count graph
    render_counter: int = field(default=0)
    silent: bool = field(default=False)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Stage 1
    TEST: int = field(default=0)
    DEBUG: bool = field(default=False)
    DEBUG_SOURCE_LOCATION: Point = field(default=Point((1, 1)))
    DEBUG_DETECTOR_LOCATION: Point = Point((1499.0, 1499.0))
    MIN_STARTING_DISTANCE: float = field(default=1000)  # cm

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Legacy rad_ppo variables
    # metadata = {"render.modes": ["human"], "video.frames_per_second": FPS}

    continuous = False

    #bbox # Redefined above
    area_obs: Interval = field(default=False)
    obstruct: int = 0
    seed: int = 0
    coord_noise: bool = False

    def __post_init__(self) -> None:
        # Debugging tests
        # Test 1: 15x15 grid, no obstructions, fixed start and stop points
        if self.DEBUG:
            if not self.silent:
                print(f"Reward Mode - Global: {GLOBAL_REWARD}. Proportional: {PROPORTIONAL_REWARD}. Basic {BASIC_REWARD}. Original: {ORIGINAL_REWARD}")
                if BASIC_REWARD:
                    print(f"Basic Reward upon success: {BASIC_SUC_AMOUNT}")
        if self.TEST == "1":
            if not self.silent:        
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 1 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((1500.0, 0.0)), Point((1500.0, 1500.0)), Point((0.0, 1500.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.DEBUG_SOURCE_LOCATION = Point((1, 1))
            self.DEBUG_DETECTOR_LOCATION = Point((1499.0, 1499.0))

        # Test 2: 15x15 grid, no obstructions, fixed stop point
        elif self.TEST == "2":
            if not self.silent:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 2 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((1500.0, 0.0)), Point((1500.0, 1500.0)), Point((0.0, 1500.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.DEBUG_SOURCE_LOCATION = Point((1, 1))

        # Test 3: 7x7 grid, no obstructions, fixed start point
        elif self.TEST == "3":
            if not self.silent:        
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 3 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((700.0, 0.0)), Point((700.0, 700.0)), Point((0.0, 700.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.DEBUG_DETECTOR_LOCATION = Point((699.0, 699.0))
            self.MIN_STARTING_DISTANCE = 350  # cm

        # Test 4: 7x7 grid, no obstructions
        elif self.TEST == "4":
            if not self.silent:        
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 4 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((700.0, 0.0)), Point((700.0, 700.0)), Point((0.0, 700.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 0
            self.DEBUG = True
            self.MIN_STARTING_DISTANCE = 350  # cm

        # Test 5: 15x15 grid, no obstructions
        elif self.TEST == "5":
            if not self.silent: 
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 5 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((1500.0, 0.0)), Point((1500.0, 1500.0)), Point((0.0, 1500.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 0
            self.DEBUG = False
            self.MIN_STARTING_DISTANCE = 500  # cm

        # Test 6: 15x15 grid, 1 obstruction
        elif self.TEST == "6":
            if not self.silent: 
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 6 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((1500.0, 0.0)), Point((1500.0, 1500.0)), Point((0.0, 1500.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 1
            self.DEBUG = False
            self.MIN_STARTING_DISTANCE = 500  # cm

        # Test 7: 15x15 grid, 3 obstructions
        elif self.TEST == "7":
            if not self.silent: 
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TEST 7 MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((1500.0, 0.0)), Point((1500.0, 1500.0)), Point((0.0, 1500.0))))
            self.observation_area = Interval((100.0, 100.0))
            self.obstruction_count = 3
            self.DEBUG = False
            self.MIN_STARTING_DISTANCE = 500  # cm

        # FULL RUN: 15x15 grid, [1-5] obstructions
        elif self.TEST == "FULL":
            if not self.silent: 
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   FULL RUN MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((1500.0, 0.0)), Point((1500.0, 1500.0)), Point((0.0, 1500.0))))
            self.observation_area = Interval((100.0, 200.0)) # Size of obstructions
            self.obstruction_count = -1
            self.DEBUG = False
            self.MIN_STARTING_DISTANCE = 500  # cm
            self.obstruction_max = 3

        # FULL RUN: 15x15 grid, [1-5] obstructions
        elif self.TEST == "ZERO":
            if not self.silent:             
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   ZERO RUN MODE   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bbox = BBox((Point((0.0, 0.0)), Point((1500.0, 0.0)), Point((1500.0, 1500.0)), Point((0.0, 1500.0))))
            self.observation_area = Interval((100.0, 200.0))
            self.obstruction_count = 0
            self.DEBUG = False
            self.MIN_STARTING_DISTANCE = 500  # cm

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
        self.max_dist: float = dist_p(self.search_area[2], self.search_area[1])  # Maximum distance between two points within search area

        # Assure there is room to spawn detectors and source with proper spacing
        assert (
            self.max_dist > self.MIN_STARTING_DISTANCE
        ), "Maximum distance available is too small, unable to spawn source and detector 1000 cm apart"

        self.scale = 1 / self.search_area[2][1]  # Needed for CNN network scaling

        # Set initial shortest path to be zero
        self.global_min_shortest_path = 0

        # self.reset() # TODO

        # Legacy set-up
        # self.np_random = self.seed  # Need random number generator
        self.bounds = np.asarray(self.bbox)
        self.search_area = np.array(
            [
                [self.bounds[0][0] + self.observation_area[0], self.bounds[0][1] + self.observation_area[0]],
                [self.bounds[1][0] - self.observation_area[1], self.bounds[1][1] + self.observation_area[0]],
                [self.bounds[2][0] - self.observation_area[1], self.bounds[2][1] - self.observation_area[1]],
                [self.bounds[3][0] + self.observation_area[0], self.bounds[3][1] - self.observation_area[1]],
            ]
        )
        self.area_obs = self.observation_area
        self.obstruct = self.obstruction_count
        self.viewer = None
        self.intensity = None
        self.bkg_intensity = None
        self.prev_det_dist = None
        self.iter_count = 0
        self.oob_count = 0
        self.epoch_end = True
        self.epoch_cnt = 0
        self.dwell_time = 1
        self.det_sto = None
        self.meas_sto = None
        self.max_dist = math.sqrt(self.search_area[2][0] ** 2 + self.search_area[2][1] ** 2)
        self._max_episode_steps = 120
        self.coord_noise = self.coord_noise
        self.done = False
        self.int_bnd = np.array([1e6, 10e6])
        self.bkg_bnd = np.array([10, 51])
        self.a_size = 8

        self.observation_space = spaces.Box(0, np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.a_size)

        self.reset()

    def step(self, action):
        """
        Wrapper that captures gymAI env.step() and expands to include multiple agents for one "timestep".
        Accepts literal action for single agent, or a Dict of agent-IDs and actions.

        Returns dictionary of agent IDs and StepReturns. Agent coodinates are scaled for graph.

        Action:
        Literal single-action. Empty Action indicates agent is stalling for a timestep.

        action_list:
        A Dict of agent_IDs and their corresponding Actions. If none passed in, this will return just agents current states
            (often used during a environment reset).
        """

        def agent_step(action, agent, proposed_coordinates):
            """
            Method that takes an action and updates the detector position accordingly.
            Returns an observation, reward, and whether the termination criteria is met.
            """
            # Initial values
            agent.out_of_bounds = False
            agent.collision = False

            # Move detector and make sure it is not in an obstruction
            # NOTE: Agent takes action in check_action
            in_obs = self.check_action(action, agent, proposed_coordinates)
            if not in_obs:
                if np.any(agent.det_coords < (self.search_area[0])) or np.any(agent.det_coords > (self.search_area[2])):
                    agent.out_of_bounds = True
                    agent.out_of_bounds_count += 1

                agent.sp_dist = self.world.shortest_path(self.source, agent.detector, self.vis_graph, EPSILON).length()
                agent.euc_dist = ((agent.det_coords - self.src_coords) ** 2).sum()
                agent.intersect = self.is_intersect(agent)
                if agent.intersect:
                    meas = self.np_random.poisson(self.bkg_intensity)
                else:
                    meas = self.np_random.poisson(lam=(self.intensity / agent.euc_dist + self.bkg_intensity))

                # Reward logic 
                # TODO adjust so agent is not encouraged to maximize steps by going to corner first
                if agent.sp_dist < 110:
                    reward = 0.1
                    self.done = True
                elif agent.sp_dist < agent.prev_det_dist:
                    reward = 0.1
                    agent.prev_det_dist = agent.sp_dist
                else:
                    reward = -0.5 * agent.sp_dist / (self.max_dist)

            else:
                # If detector starts on obs. edge, it won't have the sp_dist calculated
                if self.iter_count > 0:
                    if agent.intersect:
                        meas = self.np_random.poisson(self.bkg_intensity)
                    else:
                        meas = self.np_random.poisson(lam=(self.intensity / agent.euc_dist + self.bkg_intensity))
                else:
                    agent.sp_dist = agent.prev_det_dist
                    agent.euc_dist = ((agent.det_coords - self.src_coords) ** 2).sum()
                    agent.intersect = self.is_intersect(agent)
                    if agent.intersect:
                        meas = self.np_random.poisson(self.bkg_intensity)
                    else:
                        meas = self.np_random.poisson(lam=(self.intensity / agent.euc_dist + self.bkg_intensity))

                reward = -0.5 * agent.sp_dist / (self.max_dist)

            # If detector coordinate noise is desired
            if self.coord_noise:
                noise = self.np_random.normal(scale=5, size=len(agent.det_coords))
            else:
                noise = np.zeros(len(agent.det_coords))

            # Scale detector coordinates by search area of the DRL algorithm
            det_coord_scaled = (agent.det_coords + noise) / self.search_area[2][1]

            # Observation with the radiation meas., detector coords and detector-obstruction range meas.
            state = np.append(meas, det_coord_scaled)
            if self.num_obs > 0:
                sensor_meas = self.dist_sensors(agent)
                state = np.append(state, sensor_meas)
            else:
                state = np.append(state, np.zeros(self.a_size))
            agent.oob = False
            agent.det_sto.append(agent.det_coords.copy())
            agent.meas_sto.append(meas)

            return state, np.round(reward, 2), self.done, {}

        ### Non-legacy compatibility for MARL ###
        assert action is None or isinstance(action, int) or isinstance(action, dict), "Action not integer or a dictionary of actions."

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

        if self.step_data_mode == "list":
            aggregate_observation_result = np.zeros(combined_shape(self.number_agents, self.observation_space.shape[0]), dtype=np.float32)
            aggregate_reward_result = np.zeros((self.number_agents), dtype=np.float32)

        elif self.step_data_mode == "dict":
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
            proposed_coordinates = [sum_p(self.agents[agent_id].det_coords, get_step(action)) for agent_id, action in action_list.items()]

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
            if isinstance(action, int) and len(self.agents) > 1:
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
                ) = agent_step(
                    action=action, agent=agent
                )  # type: ignore

            self.iter_count += 1
        else:
            raise ValueError("Incompatible Action type")
        # Parse rewards
        if not PROPORTIONAL_REWARD:
            # if Global shortest path was used as min shortest path distance
            if self.step_data_mode == "dict":
                for agent_id in self.agents:
                    # Calculate team reward
                    if not max_reward:
                        max_reward = aggregate_reward_result[agent_id]
                    elif max_reward < aggregate_reward_result[agent_id]:
                        max_reward = aggregate_reward_result[agent_id]
            elif self.step_data_mode == "list":
                max_reward = aggregate_reward_result.max()  # type: ignore
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
            max_reward = (
                np.round(aggregate_reward_result[winning_id].item(), decimals=2)
                if (self.step_data_mode == "list")
                else aggregate_reward_result[winning_id]
            )

        # Save cumulative team reward for rendering
        for agent in self.agents.values():
            if max_reward or max_reward == 0:
                agent.team_reward_sto.append(max_reward + agent.team_reward_sto[-1] if len(agent.team_reward_sto) > 0 else max_reward)

        return (
            aggregate_observation_result,
            {"team_reward": max_reward, "individual_reward": aggregate_reward_result},
            aggregate_success_result,
            aggregate_info_result,
        )

    def reset(self):
        """
        Method to reset the environment.
        If epoch_end flag is True, then all components of the environment are resampled
        If epoch_end flag is False, then only the source and detector coordinates, source activity and background
        are resampled.
        """

        ### Non-Legacy ### 
        for agent in self.agents.values():
            agent.reset()
        self.all_agent_max_count = 0.0

        # Legacy
        self.done = False
        self.oob = False
        self.iter_count = 0
        self.oob_count = 0
        self.dwell_time = 1

        if self.epoch_end:
            if self.obstruct == 0:
                self.num_obs = 0
            elif self.obstruct < 0:
                self.num_obs = self.np_random.integers(1, 6)
            else:
                self.num_obs = self.obstruct
            self.ext_ls = [[] for _ in range(self.num_obs)]
            self.create_obs()
            self.bbox = [vis.Point(self.bounds[jj][0], self.bounds[jj][1]) for jj in range(len(self.bounds))]
            self.walls = vis.Polygon(self.bbox)
            self.env_ls = [solid for solid in self.poly]
            self.env_ls.insert(0, self.walls)
            # Create Visilibity environment
            self.world = vis.Environment(self.env_ls)
            # Create Visilibity graph to speed up shortest path computation
            self.vis_graph = vis.Visibility_Graph(self.world, EPSILON)
            self.epoch_cnt += 1
            self.source, detector, det_coords, self.src_coords = self.sample_source_loc_pos()
            self.intensity = self.np_random.integers(self.int_bnd[0], self.int_bnd[1])
            self.bkg_intensity = self.np_random.integers(self.bkg_bnd[0], self.bkg_bnd[1])
            self.epoch_end = False
        else:
            self.source, detector, det_coords, self.src_coords = self.sample_source_loc_pos()
            self.intensity = self.np_random.integers(self.int_bnd[0], self.int_bnd[1])
            self.bkg_intensity = self.np_random.integers(self.bkg_bnd[0], self.bkg_bnd[1])

        ### Non-legacy ###
        for agent in self.agents.values():
            agent.det_coords = det_coords.copy()
            agent.detector.set_x(det_coords[0])
            agent.detector.set_y(det_coords[1])

            assert agent.detector.x != detector.x  # Ensure different objects
            assert agent.detector.x() == detector.x()
            assert agent.detector.y() == detector.y()

            agent.prev_det_dist = self.world.shortest_path(self.source, agent.detector, self.vis_graph, EPSILON).length()
            agent.det_sto = []  # TODO already in reset function?
            agent.meas_sto = []

        # Check if the environment is valid
        if not (self.world.is_valid(EPSILON)):
            print("Environment is not valid, retrying!")
            self.epoch_end = True
            self.reset()

        return self.step(None)

    def check_action(self, action, agent, proposed_coordinates):
        """
        Method that checks which direction to move the detector based on the action.
        If the action moves the detector into an obstruction, the detector position
        will be reset to the prior position.
        """
        in_obs = False

        ### Non-legacy ###
        step = get_step(action)
        tentative_coordinates = sum_p(agent.det_coords, step)

        if self.check_will_collide(tentative_coordinates, proposed_coordinates):
            agent.collision = True
            return False
        
        # Set tentative coordinates
        agent.detector = to_vis_p(tentative_coordinates)

        if self.check_out_of_bounds(tentative_coordinates, agent):
            agent.out_of_bounds = True
            agent.out_of_bounds_count += 1
            if self.enforce_grid_boundaries:
                agent.detector = to_vis_p(agent.det_coords)  # roll back coordinates
                return False

        if action == 0:
            # left
            self.dwell_time = 1
            agent.detector.set_x(agent.det_coords[0] - DET_STEP)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_x(agent.det_coords[0])
            else:
                agent.det_coords[0] = agent.detector.x()

        elif action == 1:
            # up left
            self.dwell_time = 1
            agent.detector.set_y(agent.det_coords[1] + DET_STEP_FRAC)
            agent.detector.set_x(agent.det_coords[0] - DET_STEP_FRAC)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_y(agent.det_coords[1])
                agent.detector.set_x(agent.det_coords[0])
            else:
                agent.det_coords[1] = agent.detector.y()
                agent.det_coords[0] = agent.detector.x()

        elif action == 2:
            # up
            self.dwell_time = 1
            agent.detector.set_y(agent.det_coords[1] + DET_STEP)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_y(agent.det_coords[1])
            else:
                agent.det_coords[1] = agent.detector.y()

        elif action == 3:
            # up right
            self.dwell_time = 1
            agent.detector.set_y(agent.det_coords[1] + DET_STEP_FRAC)
            agent.detector.set_x(agent.det_coords[0] + DET_STEP_FRAC)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_y(agent.det_coords[1])
                agent.detector.set_x(agent.det_coords[0])
            else:
                agent.det_coords[1] = agent.detector.y()
                agent.det_coords[0] = agent.detector.x()

        elif action == 4:
            # right
            self.dwell_time = 1
            agent.detector.set_x(agent.det_coords[0] + DET_STEP)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_x(agent.det_coords[0])
            else:
                agent.det_coords[0] = agent.detector.x()

        elif action == 5:
            # down right
            self.dwell_time = 1
            agent.detector.set_y(agent.det_coords[1] - DET_STEP_FRAC)
            agent.detector.set_x(agent.det_coords[0] + DET_STEP_FRAC)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_y(agent.det_coords[1])
                agent.detector.set_x(agent.det_coords[0])
            else:
                agent.det_coords[1] = agent.detector.y()
                agent.det_coords[0] = agent.detector.x()

        elif action == 6:
            # down
            self.dwell_time = 1
            agent.detector.set_y(agent.det_coords[1] - DET_STEP)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_y(agent.det_coords[1])
            else:
                agent.det_coords[1] = agent.detector.y()

        elif action == 7:
            # down left
            self.dwell_time = 1
            agent.detector.set_y(agent.det_coords[1] - DET_STEP_FRAC)
            agent.detector.set_x(agent.det_coords[0] - DET_STEP_FRAC)
            in_obs = self.in_obstruction(agent)
            if in_obs:
                agent.detector.set_y(agent.det_coords[1])
                agent.detector.set_x(agent.det_coords[0])
            else:
                agent.det_coords[1] = agent.detector.y()
                agent.det_coords[0] = agent.detector.x()
        else:
            self.dwell_time = 1
        return in_obs

    def create_obs(self):
        """
        Method that randomly samples obstruction coordinates from 90% of search area dimensions.
        Obstructions are not allowed to intersect.
        """
        seed_pt = np.zeros(2)
        ii = 0
        intersect = False
        self.obs_coord = [[] for _ in range(self.num_obs)]
        self.poly = []
        self.line_segs = []
        obs_coord = np.array([])
        while ii < self.num_obs:
            seed_pt[0] = self.np_random.integers(self.search_area[0][0], self.search_area[2][0] * 0.9, size=(1))
            seed_pt[1] = self.np_random.integers(self.search_area[0][1], self.search_area[2][1] * 0.9, size=(1))
            ext = self.np_random.integers(self.area_obs[0], self.area_obs[1], size=2)
            obs_coord = np.append(obs_coord, seed_pt)
            obs_coord = np.vstack((obs_coord, [seed_pt[0], seed_pt[1] + ext[1]]))
            obs_coord = np.vstack((obs_coord, [seed_pt[0] + ext[0], seed_pt[1] + ext[1]]))
            obs_coord = np.vstack((obs_coord, [seed_pt[0] + ext[0], seed_pt[1]]))
            if ii > 0:
                kk = 0
                while not intersect and kk < ii:
                    intersect = collide(self.obs_coord[kk][0], obs_coord)
                    if intersect:
                        obs_coord = np.array([])
                    kk += 1

            if not intersect:
                self.obs_coord[ii].append(obs_coord)
                obs_coord = list(obs_coord)
                geom = [vis.Point(float(obs_coord[jj][0]), float(obs_coord[jj][1])) for jj in range(len(obs_coord))]
                poly = vis.Polygon(geom)
                self.poly.append(poly)
                self.line_segs.append(
                    [
                        vis.Line_Segment(geom[0], geom[1]),
                        vis.Line_Segment(geom[0], geom[3]),
                        vis.Line_Segment(geom[2], geom[1]),
                        vis.Line_Segment(geom[2], geom[3]),
                    ]
                )
                ii += 1
                intersect = False
                obs_coord = np.array([])
            intersect = False

    def sample_source_loc_pos(self):
        """
        Method that randomly generate the detector and source starting locations.
        Locations can not be inside obstructions and must be at least 1000 cm apart
        """
        det_clear = False
        src_clear = False
        resamp = False
        jj = 0
        source = np.zeros(2, dtype=np.double)
        det = np.zeros(2, dtype=np.double)
        source = self.np_random.integers(self.search_area[0][0], self.search_area[1][0], size=2).astype(np.double)
        det = self.np_random.integers(self.search_area[0][0], self.search_area[1][0], size=2).astype(np.double)
        det_point = vis.Point(det[0], det[1])

        while not det_clear:
            while not resamp and jj < self.num_obs:
                if det_point._in(self.poly[jj], EPSILON):
                    resamp = True
                jj += 1
            if resamp:
                det = self.np_random.integers(self.search_area[0][0], self.search_area[1][0], size=2).astype(np.double)
                det_point.set_x(det[0])
                det_point.set_y(det[1])
                jj = 0
                resamp = False
            else:
                det_clear = True
        resamp = False
        inter = False
        jj = 0
        num_retry = 0
        while not src_clear:
            distance = np.linalg.norm(det - source)
            while distance < 1000:
                distance = np.linalg.norm(det - source)
                source = self.np_random.integers(self.search_area[0][0], self.search_area[1][0], size=2).astype(np.double)
            src_point = vis.Point(source[0], source[1])
            L = vis.Line_Segment(det_point, src_point)
            while not resamp and jj < self.num_obs:
                if src_point._in(self.poly[jj], EPSILON):
                    resamp = True
                if not resamp and vis.boundary_distance(L, self.poly[jj]) < 0.001:
                    inter = True
                jj += 1
            if self.num_obs == 0 or (num_retry > 20 and not resamp):
                src_clear = True
            elif resamp or not inter:
                source = self.np_random.integers(self.search_area[0][0], self.search_area[1][0], size=2).astype(np.double)
                src_point.set_x(source[0])
                src_point.set_y(source[1])
                jj = 0
                resamp = False
                inter = False
                num_retry += 1
            elif inter:
                src_clear = True

        return src_point, det_point, det, source

    def is_intersect(self, agent, threshold=0.001):
        """
        Method that checks if the line of sight is blocked by any obstructions in the environment.
        """
        inter = False
        kk = 0
        L = vis.Line_Segment(agent.detector, self.source)
        while not inter and kk < self.num_obs:
            if vis.boundary_distance(L, self.poly[kk]) < threshold and not math.isclose(math.sqrt(agent.euc_dist), agent.sp_dist, abs_tol=0.1):
                inter = True
            kk += 1
        return inter

    def in_obstruction(self, agent):
        """
        Method that checks if the detector position intersects or is inside an obstruction.
        """
        jj = 0
        obs_boundary = False
        while not obs_boundary and jj < self.num_obs:
            if agent.detector._in(self.poly[jj], EPSILON):
                obs_boundary = True
            jj += 1

        if obs_boundary:
            bbox = self.poly[jj - 1].bbox()
            if agent.detector.y() > bbox.y_min:
                if agent.detector.y() < bbox.y_max:
                    if agent.detector.x() > bbox.x_min:
                        if agent.detector.x() < bbox.x_max:
                            return True
            return False
        else:
            return False

    def check_out_of_bounds(self, tentative_coordinates, agent):
        if self.enforce_grid_boundaries:
            out_of_bounds = (tentative_coordinates[0] < self.bbox[0][0] or tentative_coordinates[1] < self.bbox[0][1]) or 
                            (self.bbox[2][0] <= tentative_coordinates[0] or self.bbox[2][1] <= tentative_coordinates[1])
        else:
            lower_b = agent.det_coords[0] < self.search_area[0][0] or agent.det_coords[1] < self.search_area[0][1]
            upper_b = self.search_area[2][0] < agent.det_coords[0] or self.search_area[2][1] < agent.det_coords[1]
            out_of_bounds = (lower_b or upper_b)

        return out_of_bounds

    def check_will_collide(self, tentative_coordinates, proposed_coordinates):
        return count_matching_p(tentative_coordinates, proposed_coordinates) > 1

    def dist_sensors(self, agent):
        """
        Method that generates detector-obstruction range measurements with values between 0-1.
        """
        seg_coords = [
            vis.Point(agent.detector.x() - DIST_TH, agent.detector.y()),
            vis.Point(agent.detector.x() - DIST_TH_FRAC, agent.detector.y() + DIST_TH_FRAC),
            vis.Point(agent.detector.x(), agent.detector.y() + DIST_TH),
            vis.Point(agent.detector.x() + DIST_TH_FRAC, agent.detector.y() + DIST_TH_FRAC),
            vis.Point(agent.detector.x() + DIST_TH, agent.detector.y()),
            vis.Point(agent.detector.x() + DIST_TH_FRAC, agent.detector.y() - DIST_TH_FRAC),
            vis.Point(agent.detector.x(), agent.detector.y() - DIST_TH),
            vis.Point(agent.detector.x() - DIST_TH_FRAC, agent.detector.y() - DIST_TH_FRAC),
        ]
        segs = [vis.Line_Segment(agent.detector, seg_coord) for seg_coord in seg_coords]
        dists = np.zeros(len(segs))
        obs_idx_ls = np.zeros(len(self.poly))
        inter = 0
        seg_dist = np.zeros(4)
        if self.num_obs > 0:
            for idx, seg in enumerate(segs):
                for obs_idx, poly in enumerate(self.line_segs):
                    for seg_idx, obs_seg in enumerate(poly):
                        if inter < 2 and vis.intersect(obs_seg, seg, EPSILON):
                            # check if step dir intersects poly seg
                            seg_dist[seg_idx] = (DIST_TH - vis.distance(seg.first(), obs_seg)) / DIST_TH
                            inter += 1
                            obs_idx_ls[obs_idx] += 1
                    if inter > 0:
                        dists[idx] = seg_dist.max()
                        seg_dist.fill(0)
                inter = 0
            if (dists == 1.0).sum() > 3:
                dists = self.correct_coords(self.poly[obs_idx_ls.argmax()])
        return dists

    def correct_coords(self, poly):
        """
        Method that corrects the detector-obstruction range measurement if more than the correct
        number of directions are being activated due to the Visilibity implementation.
        """
        x_check = np.zeros(self.a_size)
        dist = 0.1
        length = 1
        q0 = vis.Point(agent.detector.x(), agent.detector.y())
        q1 = vis.Point(agent.detector.x(), agent.detector.y())
        q2 = vis.Point(agent.detector.x(), agent.detector.y())
        q3 = vis.Point(agent.detector.x(), agent.detector.y())
        q4 = vis.Point(agent.detector.x(), agent.detector.y())
        q5 = vis.Point(agent.detector.x(), agent.detector.y())
        q6 = vis.Point(agent.detector.x(), agent.detector.y())
        q7 = vis.Point(agent.detector.x(), agent.detector.y())

        # qs = [vis.Point(agent.detector.x(),agent.detector.y()) for _ in range(self.a_size)]
        dists = np.zeros(self.a_size)
        while np.all(x_check == 0):  # not (xp or xn or yp or yn):
            q0.set_x(q0.x() - dist * length)

            q1.set_x(q1.x() - dist * length)
            q1.set_y(q1.y() + dist * length)

            q2.set_y(q2.y() + dist * length)

            q3.set_x(q3.x() + dist * length)
            q3.set_y(q3.y() + dist * length)

            q4.set_x(q4.x() + dist * length)

            q5.set_x(q5.x() + dist * length)
            q5.set_y(q5.y() - dist * length)

            q6.set_y(q6.y() - dist * length)

            q7.set_x(q7.x() - dist * length)
            q7.set_y(q7.y() - dist * length)
            if q0._in(poly, EPSILON):  # set to one if outside poly
                x_check[0] = True  # xn is inside poly
            if q1._in(poly, EPSILON):
                x_check[1] = True  # xnyp is inside poly
            if q2._in(poly, EPSILON):
                x_check[2] = True  # xn is inside poly
            if q3._in(poly, EPSILON):
                x_check[3] = True  # xpyp is inside poly
            if q4._in(poly, EPSILON):
                x_check[4] = True  # yp is inside poly
            if q5._in(poly, EPSILON):
                x_check[5] = True  # xpyn is inside poly
            if q6._in(poly, EPSILON):
                x_check[6] = True  # yn is inside poly
            if q7._in(poly, EPSILON):
                x_check[7] = True  # xnyn is inside poly

        if np.sum(x_check) >= 4:  # i.e. if one outside the poly then
            for ii in [0, 2, 4, 6]:
                if x_check[ii - 1] == 1 and x_check[ii + 1] == 1:
                    dists[ii] = 1.0
                    dists[ii - 1] = 1.0
                    dists[ii + 1] = 1.0
        return dists

    def render(
        self,
        save_gif=False,
        path=None,
        epoch_count=None,
        just_env=False,
        obs=None,
        ep_rew=None,
        data=None,
        meas=None,
        params=None,
        loc_est=None,
    ):
        """
        Method that produces a gif of the agent interacting in the environment.
        """
        if data and meas:
            self.intensity = params[0]
            self.bkg_intensity = params[1]
            self.src_coords = params[2]
            self.iter_count = len(meas)
            data = np.array(data) / 100
        else:
            data = np.array(self.det_sto) / 100
            meas = self.meas_sto

        if just_env:
            current_index = 0
            plt.rc("font", size=14)
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
            ax1.scatter(
                self.src_coords[0] / 100,
                self.src_coords[1] / 100,
                60,
                c="red",
                marker="*",
                label="Source",
            )
            ax1.scatter(
                data[current_index, 0],
                data[current_index, 1],
                42,
                c="black",
                marker="^",
                label="Detector",
            )
            ax1.grid()
            if not (obs == []):
                for coord in obs:
                    p_disp = Polygon(coord[0] / 100, color="gray")
                    ax1.add_patch(p_disp)
            if not (loc_est is None):
                ax1.scatter(
                    loc_est[0][current_index][1] / 100,
                    loc_est[0][current_index][2] / 100,
                    42,
                    c="magenta",
                    label="Loc. Pred.",
                )
            ax1.set_xlim(0, self.search_area[1][0] / 100)
            ax1.set_ylim(0, self.search_area[2][1] / 100)
            ax1.set_xlabel("X[m]")
            ax1.set_ylabel("Y[m]")
            ax1.legend(loc="lower left")
        else:
            plt.rc("font", size=12)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
            m_size = 25

            def update(frame_number, data, ax1, ax2, ax3, src, area_dim, meas):
                current_index = frame_number % (self.iter_count)
                global loc
                if current_index == 0:
                    intensity_sci = "{:.2e}".format(self.intensity)
                    ax1.cla()
                    ax1.set_title("Activity: " + intensity_sci + " [gps] Bkg: " + str(self.bkg_intensity) + " [cps]")
                    data_sub = data[current_index + 1] - data[current_index]
                    orient = math.degrees(math.atan2(data_sub[1], data_sub[0]))
                    ax1.scatter(
                        src[0] / 100,
                        src[1] / 100,
                        m_size,
                        c="red",
                        marker="*",
                        label="Source",
                    )
                    ax1.scatter(
                        data[current_index, 0],
                        data[current_index, 1],
                        m_size,
                        c="black",
                        marker=(3, 0, orient - 90),
                    )
                    ax1.scatter(-1000, -1000, m_size, c="black", marker="^", label="Detector")
                    ax1.grid()
                    if not (obs == []):
                        for coord in obs:
                            p_disp = Polygon(coord[0] / 100, color="gray")
                            ax1.add_patch(p_disp)
                    if not (loc_est is None):
                        loc = ax1.scatter(
                            loc_est[0][current_index][1] / 100,
                            loc_est[0][current_index][2] / 100,
                            m_size,
                            c="magenta",
                            label="Loc. Pred.",
                        )
                    ax1.set_xlim(0, area_dim[1][0] / 100)
                    ax1.set_ylim(0, area_dim[2][1] / 100)
                    ax1.set_xticks(np.linspace(0, area_dim[1][0] / 100 - 2, 5))
                    ax1.set_yticks(np.linspace(0, area_dim[1][0] / 100 - 2, 5))
                    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                    ax1.yaxis.set_major_formatter(FormatStrFormatter("%d"))
                    ax1.set_xlabel("X[m]")
                    ax1.set_ylabel("Y[m]")
                    ax2.cla()
                    ax2.set_xlim(0, self.iter_count)
                    ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                    ax2.set_ylim(0, max(meas) + 1e-6)
                    ax2.stem([current_index], [meas[current_index]], use_line_collection=True)
                    ax2.set_xlabel("n")
                    ax2.set_ylabel("Counts")
                    ax3.cla()
                    ax3.set_xlim(0, self.iter_count)
                    ax3.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                    ax3.set_ylim(min(ep_rew) - 2, max(ep_rew) + 2)
                    ax3.set_xlabel("n")
                    ax3.set_ylabel("Cumulative Reward")
                    ax1.legend(loc="lower right")
                else:
                    loc.remove()
                    data_sub = data[current_index + 1] - data[current_index]
                    orient = math.degrees(math.atan2(data_sub[1], data_sub[0]))
                    ax1.scatter(
                        data[current_index, 0],
                        data[current_index, 1],
                        m_size,
                        marker=(3, 0, orient - 90),
                        c="black",
                    )
                    ax1.plot(
                        data[current_index - 1 : current_index + 1, 0],
                        data[current_index - 1 : current_index + 1, 1],
                        3,
                        c="black",
                        alpha=0.3,
                        ls="--",
                    )
                    ax2.stem([current_index], [meas[current_index]], use_line_collection=True)
                    ax3.plot(range(current_index), ep_rew[:current_index], c="black")
                    if not (loc_est is None):
                        loc = ax1.scatter(
                            loc_est[0][current_index][1] / 100,
                            loc_est[0][current_index][2] / 100,
                            m_size * 0.8,
                            c="magenta",
                            label="Loc. Pred.",
                        )

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(ep_rew),
                fargs=(data, ax1, ax2, ax3, self.src_coords, self.search_area, meas),
            )
            if save_gif:
                writer = PillowWriter(fps=5)
                if os.path.isdir(path + "/gifs/"):
                    ani.save(path + f"/gifs/test_{epoch_count}.gif", writer=writer)
                else:
                    os.mkdir(path + "/gifs/")
                    ani.save(path + f"/gifs/test_{epoch_count}.gif", writer=writer)
            else:
                plt.show()

    def FIM_step(self, action, agent, coords=None):
        """
        Method for the information-driven controller to update detector coordinates in the environment
        without changing the actual detector positon.

        Args:
        action : action to move the detector
        coords : coordinates to move the detector from that are different from the current detector coordinates
        """
        if coords is None:
            det_coords = agent.det_coords.copy()
        else:
            det_coords = agent.det_coords.copy()
            agent.detector.set_x(coords[0])
            agent.detector.set_y(coords[1])
            agent.det_coords = coords.copy()

        in_obs = self.check_action(action, agent)

        if coords is None:
            det_ret = agent.det_coords
            if not (in_obs):  # If successful movement, return new coords. Set detector back.
                agent.det_coords = det_coords
                agent.detector.set_x(det_coords[0])
                agent.detector.set_y(det_coords[1])
        else:
            det_ret = agent.det_coords.copy()
            agent.det_coords = det_coords
            agent.detector.set_x(det_coords[0])
            agent.detector.set_y(det_coords[1])

        return det_ret
