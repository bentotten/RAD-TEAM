from dataclasses import dataclass, field
import os
import math

import numpy as np
import numpy.typing as npt
import numpy.random as npr

import gym  # type: ignore
from gym import spaces  # type: ignore

import visilibity as vis  # type: ignore

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon as PolygonPatches

from typing import Any, Literal, NewType, Optional, TypedDict, cast, get_args
from typing_extensions import TypeAlias

FPS = 50

DET_STEP = 100.0  # detector step size at each timestep in cm/s
DET_STEP_FRAC = 71.0  # diagonal detector step size in cm/s
DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC = 78.0  # Diagonal detector-obstruction range measurement threshold in cm

EPSILON = 0.0000001

Point: TypeAlias = NewType("Point", tuple[float, float])
Polygon: TypeAlias = NewType("Polygon", list[Point])
Interval: TypeAlias = NewType("Interval", tuple[float, float])
BBox: TypeAlias = NewType("BBox", tuple[Point, Point, Point, Point])


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


def dist_sq_p(p1: Point, p2: Point) -> float:
    """
    Return the squared distance between the two points.
    """
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def dist_p(p1: Point, p2: Point) -> float:
    """
    Return the distance between the two points.
    """
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


Metadata: TypeAlias = TypedDict(
    "Metadata", {"render.modes": list[str], "video.frames_per_second": int}
)

# These actions correspond to:
# 0: left
# 1: up and left
# 2: up
# 3: up and right
# 4: right
# 5: down and right
# 6: down
# 7: down and left
Action: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7]
A_SIZE = len(get_args(Action))


# 0: (-1)*DET_STEP     *x, ( 0)*DET_STEP     *y
# 1: (-1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
# 2: ( 0)*DET_STEP     *x, (+1)*DET_STEP     *y
# 3: (+1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
# 4: (+1)*DET_STEP     *x, ( 0)*DET_STEP     *y
# 5: (+1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y
# 6: ( 0)*DET_STEP     *x, (-1)*DET_STEP     *y
# 7: (-1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y

# If action is odd, then we are moving on the diagonal and so our step size is smaller.
# Otherwise, we're moving solely in a cardinal direction.
def get_step_size(action: Action) -> float:
    """
    Return the step size for the given action.
    """
    return DET_STEP if action % 2 == 0 else DET_STEP_FRAC


# The signs of the y-coeffecients follow the signs of sin(pi * (1 - action/4))
def get_y_step_coeff(action: Action) -> int:
    return round(math.sin(math.pi * (1.0 - action / 4.0)))


# The signs of the x-coefficients follow the signs of cos(pi * (1 - action/4)) = sin(pi * (1 - (action + 6)/4))
def get_x_step_coeff(action: Action) -> int:
    return get_y_step_coeff((action + 6) % 8)


def get_step(action: Action) -> Point:
    """
    Return the step for the given action.
        0: #left
        1: up left
        2: up
        3: up right             
        4: right
        5: down right
        6: down
        7: down left
    """
    return scale_p(
        Point((get_x_step_coeff(action), get_y_step_coeff(action))),
        get_step_size(action),
    )


@dataclass
class RadSearch(gym.Env):
    area_obs: Interval = np.array([200.0, 500.0])
    bbox: BBox = np.array([[0.0, 0.0], [2700.0, 0.0], [2700.0, 2700.0], [0.0, 2700.0]])
    np_random: npr.Generator = npr.default_rng(0)

    env_ls: list[Polygon] = field(init=False)
    max_dist: float = field(init=False)
    line_segs: list[list[vis.Line_Segment]] = field(init=False)
    poly: list[Polygon] = field(init=False)
    search_area: BBox = field(init=False)
    det_coords: Point = field(init=False)
    detector: vis.Point = field(init=False)
    src_coords: Point = field(init=False)
    source: vis.Point = field(init=False)
    walls: Polygon = field(init=False)
    world: vis.Environment = field(init=False)
    vis_graph: vis.Visibility_Graph = field(init=False)
    intensity: int = field(init=False)
    bkg_intensity: int = field(init=False)
    obs_coord: list[list[Point]] = field(init=False)
    det_sto: list[Point] = field(init=False)
    prev_det_dist: float = field(init=False)
    # TODO: self.sp_dist is declared and defined in step.
    sp_dist: float = field(init=False)
    # TODO: self.euc_dist is declared and defined in step.
    euc_dist: float = field(init=False)

    # Values with default values which are not set in the constructor
    action_space: spaces.Discrete = spaces.Discrete(A_SIZE)
    _max_episode_steps: int = 120
    bkg_bnd: Point = Point((10, 51))
    continuous: bool = False
    done: bool = False
    epoch_cnt: int = 0
    int_bnd: Point = Point((1e6, 10e6))
    iter_count: int = 0
    meas_sto: list[float] = field(default_factory=list)
    metadata: Metadata = field(default_factory=lambda: {"render.modes": ["human"], "video.frames_per_second": FPS})  # type: ignore
    observation_space: spaces.Box = spaces.Box(0, np.inf, shape=(11,), dtype=np.float32)
    oob_count: int = 0
    coord_noise: bool = False
    obstruct: Literal[-1, 0, 1] = 0
    oob: bool = False
    intersect: bool = False

    # bbox is the "bounding box"
    # Dimensions of radiation source search area in cm, decreased by area_obs param. to ensure visilibity graph setup is valid.
    #
    # area_obs
    # Interval for each obstruction area in cm
    #
    # seed
    # A random number generator
    #
    # obstruct
    # Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions
    def __post_init__(self):
        self.search_area: BBox = BBox(
            (
                Point(
                    (
                        self.bbox[0][0] + self.area_obs[0],
                        self.bbox[0][1] + self.area_obs[0],
                    )
                ),
                Point(
                    (
                        self.bbox[1][0] - self.area_obs[1],
                        self.bbox[1][1] + self.area_obs[0],
                    )
                ),
                Point(
                    (
                        self.bbox[2][0] - self.area_obs[1],
                        self.bbox[2][1] - self.area_obs[1],
                    )
                ),
                Point(
                    (
                        self.bbox[3][0] + self.area_obs[0],
                        self.bbox[3][1] - self.area_obs[1],
                    )
                ),
            )
        )
        self.max_dist: float = dist_p(self.search_area[2], self.search_area[1])
        self.epoch_end = True
        self.reset()

    def step(
        self, action: Optional[Action]
    ) -> tuple[npt.NDArray[np.float64], float, bool, dict[Any, Any]]:
        """
        Method that takes an action and updates the detector position accordingly.
        Returns an observation, reward, and whether the termination criteria is met.
        """
        # Move detector and make sure it is not in an obstruction
        in_obs = self.check_action(action) # TODO RENAME THIS! Actually takes the action if valid!
        if not in_obs:
            if (
                self.det_coords < self.search_area[0]
                or self.search_area[2] < self.det_coords
            ):
                self.oob = True
                self.oob_count += 1

            # Returns the length of a Polyline, which is a double
            # https://github.com/tsaoyu/PyVisiLibity/blob/80ce1356fa31c003e29467e6f08ffdfbd74db80f/visilibity.cpp#L1398
            self.sp_dist: float = self.world.shortest_path(  # type: ignore
                self.source, self.detector, self.vis_graph, EPSILON
            ).length()
            self.euc_dist: float = dist_p(self.det_coords, self.src_coords)
            self.intersect = self.is_intersect()
            meas: float = self.np_random.poisson(
                self.bkg_intensity
                if self.intersect
                else self.intensity / self.euc_dist + self.bkg_intensity
            )

            # Reward logic
            if self.sp_dist < 110:
                reward = 0.1
                self.done = True
            elif self.sp_dist < self.prev_det_dist:
                reward = 0.1
                self.prev_det_dist = self.sp_dist
            else:
                reward = -0.5 * self.sp_dist / self.max_dist

        else:
            # If detector starts on obs. edge, it won't have the sp_dist calculated
            if self.iter_count > 0:
                meas: float = self.np_random.poisson(
                    self.bkg_intensity
                    if self.intersect
                    else self.intensity / self.euc_dist + self.bkg_intensity
                )
            else:
                self.sp_dist = self.prev_det_dist
                self.euc_dist = dist_p(self.det_coords, self.src_coords)
                self.intersect = self.is_intersect()
                meas: float = self.np_random.poisson(
                    self.bkg_intensity
                    if self.intersect
                    else self.intensity / self.euc_dist + self.bkg_intensity
                )

            reward = -0.5 * self.sp_dist / self.max_dist

        # If detector coordinate noise is desired
        noise: Point = Point(
            tuple(self.np_random.normal(scale=5, size=2))
            if self.coord_noise
            else (0.0, 0.0)
        )

        # Scale detector coordinates by search area of the DRL algorithm
        det_coord_scaled: Point = scale_p(
            sum_p(self.det_coords, noise), 1 / self.search_area[2][1]
        )

        # Observation with the radiation meas., detector coords and detector-obstruction range meas.
        # TODO: State should really be better organized. If there are distinct components to it, why not make it
        # a named tuple?

        # Sensor measurement for in obstacles?
        sensor_meas: npt.NDArray[np.float64] = self.dist_sensors() if self.num_obs > 0 else np.zeros(A_SIZE)  # type: ignore
        # State is an 11-tuple ndarray
        state: npt.NDArray[np.float64] = np.array([meas, *det_coord_scaled, *sensor_meas])  # type: ignore
        self.oob = False
        self.det_sto.append(self.det_coords)
        self.meas_sto.append(meas)
        self.iter_count += 1
        return state, round(reward, 2), self.done, {}

    def reset(self) -> npt.NDArray[np.float64]:
        """
        Method to reset the environment.
        """
        self.done = False
        self.oob = False
        self.iter_count = 0
        self.oob_count = 0
        self.dwell_time = 1

        if self.epoch_end:
            if self.obstruct == -1:
                self.num_obs = self.np_random.integers(1, 6)  # type: ignore
            elif self.obstruct == 0:
                self.num_obs = 0
            else:
                self.num_obs = self.obstruct

            self.create_obs()
            self.walls = Polygon(list(self.bbox))
            self.env_ls: list[Polygon] = [self.walls, *self.poly]

            # Create Visilibity environment
            self.world = vis.Environment(list(map(to_vis_poly, self.env_ls)))

            # Create Visilibity graph to speed up shortest path computation
            self.vis_graph = vis.Visibility_Graph(self.world, EPSILON)
            self.epoch_cnt += 1
            self.epoch_end = False

        (
            self.source,
            self.detector,
            self.det_coords,
            self.src_coords,
        ) = self.sample_source_loc_pos()
        self.intensity = self.np_random.integers(self.int_bnd[0], self.int_bnd[1])  # type: ignore
        self.bkg_intensity = self.np_random.integers(self.bkg_bnd[0], self.bkg_bnd[1])  # type: ignore

        self.prev_det_dist: float = self.world.shortest_path(  # type: ignore
            self.source, self.detector, self.vis_graph, EPSILON
        ).length()
        self.det_sto = []
        self.meas_sto = []

        # Check if the environment is valid
        if not (self.world.is_valid(EPSILON)):
            print("Environment is not valid, retrying!")
            self.epoch_end = True
            self.reset()

        return self.step(None)[0]

    # TODO: Name is dishonest. If the action is valid, it actually *takes* the action!
    def check_action(self, action: Optional[Action]) -> bool:
        """
        Method that checks which direction to move the detector based on the action.
        If the action moves the detector into an obstruction, the detector position
        will be reset to the prior position.
        0: #left
        1: up left
        2: up
        3: up right             
        4: right
        5: down right
        6: down
        7: down left
        """
        in_obs: bool = False

        if action is None:
            return in_obs

        step = get_step(action)
        self.detector = to_vis_p(sum_p(self.det_coords, step))

        in_obs = self.in_obstruction()
        if in_obs:
            # If we're in an obsticle, roll back
            self.detector = to_vis_p(self.det_coords)
        else:
            # If we're not in an obsticle, update the detector coordinates
            self.det_coords = from_vis_p(self.detector)

        return in_obs

    def create_obs(self) -> None:
        """
        Method that randomly samples obstruction coordinates from 90% of search area dimensions.
        Obstructions are not allowed to intersect.
        """
        ii = 0
        intersect = False
        self.obs_coord: list[list[Point]] = [[] for _ in range(self.num_obs)]
        self.poly: list[Polygon] = []
        self.line_segs: list[list[vis.Line_Segment]] = []
        obs_coord: list[Point] = []
        while ii < self.num_obs:
            seed_x: float = self.np_random.integers(  # type: ignore
                self.search_area[0][0], self.search_area[2][0] * 0.9  # type: ignore
            ).astype(np.float64)
            seed_y: float = self.np_random.integers(  # type: ignore
                self.search_area[0][1], self.search_area[2][1] * 0.9  # type: ignore
            ).astype(np.float64)
            ext_x: float = self.np_random.integers(  # type: ignore
                self.area_obs[0], self.area_obs[1]  # type: ignore
            ).astype(np.float64)
            ext_y: float = self.np_random.integers(  # type: ignore
                self.area_obs[0], self.area_obs[1]  # type: ignore
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
                            # TODO: Why in this order?
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

    def sample_source_loc_pos(self,) -> tuple[vis.Point, vis.Point, Point, Point]:
        """
        Method that randomly generate the detector and source starting locations.
        Locations can not be inside obstructions and must be at least 1000 cm apart
        """
        det_clear = False
        src_clear = False
        resamp = False
        jj = 0

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

        source = rand_point()
        src_point = to_vis_p(source)
        detector = rand_point()
        det_point = to_vis_p(detector)

        while not det_clear:
            while not resamp and jj < self.num_obs:
                if det_point._in(to_vis_poly(self.poly[jj]), EPSILON):  # type: ignore
                    resamp = True
                jj += 1
            if resamp:
                detector = rand_point()
                det_point = to_vis_p(detector)
                jj = 0
                resamp = False
            else:
                det_clear = True
        resamp = False
        inter = False
        jj = 0
        num_retry = 0
        while not src_clear:
            while dist_p(detector, source) < 1000:
                source = rand_point()
            src_point = to_vis_p(source)
            L: vis.Line_Segment = vis.Line_Segment(det_point, src_point)
            while not resamp and jj < self.num_obs:
                poly_p: vis.Polygon = to_vis_poly(self.poly[jj])
                if src_point._in(poly_p, EPSILON):  # type: ignore
                    resamp = True
                if not resamp and vis.boundary_distance(L, poly_p) < 0.001:  # type: ignore
                    inter = True
                jj += 1
            if self.num_obs == 0 or (num_retry > 20 and not resamp):
                src_clear = True
            elif resamp or not inter:
                source = rand_point()
                src_point = to_vis_p(source)
                jj = 0
                resamp = False
                inter = False
                num_retry += 1
            elif inter:
                src_clear = True

        return src_point, det_point, detector, source

    def is_intersect(self, threshold: float = 0.001) -> bool:
        """
        Method that checks if the line of sight is blocked by any obstructions in the environment.
        """
        inter = False
        kk = 0
        L = vis.Line_Segment(self.detector, self.source)
        while not inter and kk < self.num_obs:
            if vis.boundary_distance(L, to_vis_poly(self.poly[kk])) < threshold and not math.isclose(  # type: ignore
                math.sqrt(self.euc_dist), self.sp_dist, abs_tol=0.1
            ):
                inter = True
            kk += 1
        return inter

    def in_obstruction(self):
        """
        Method that checks if the detector position intersects or is inside an obstruction.
        """
        jj = 0
        obs_boundary = False
        while not obs_boundary and jj < self.num_obs:
            if self.detector._in(to_vis_poly(self.poly[jj]), EPSILON):  # type: ignore
                obs_boundary = True
            jj += 1

        if obs_boundary:
            bbox: vis.Bounding_Box = to_vis_poly(self.poly[jj - 1]).bbox()
            return all(
                [  # type: ignore
                    self.detector.y() > bbox.y_min,
                    self.detector.y() < bbox.y_max,
                    self.detector.x() > bbox.x_min,
                    self.detector.x() < bbox.x_max,
                ]
            )
        else:
            return False

    def dist_sensors(self) -> list[float]:
        """
        Method that generates detector-obstruction range measurements with values between 0-1.
        """
        detector_p: Point = from_vis_p(self.detector)
        segs: list[vis.Line_Segment] = [
            vis.Line_Segment(
                self.detector, to_vis_p(sum_p(detector_p, get_step(action)))
            )
            for action in cast(tuple[Action], get_args(Action))
        ]
        # TODO: Currently there are only eight actions -- what happens if we change that?
        # This annotation would need to change as well.
        dists: list[float] = [0.0] * len(segs)
        obs_idx_ls: list[int] = [0] * len(self.poly)
        inter = 0
        seg_dist: list[float] = [0.0] * 4
        if self.num_obs > 0:
            for idx, seg in enumerate(segs):
                for obs_idx, poly in enumerate(self.line_segs):
                    for seg_idx, obs_seg in enumerate(poly):
                        if inter < 2 and vis.intersect(obs_seg, seg, EPSILON):  # type: ignore
                            # check if step dir intersects poly seg
                            seg_dist[seg_idx] = (  # type: ignore
                                DIST_TH - vis.distance(seg.first(), obs_seg)  # type: ignore
                            ) / DIST_TH
                            inter += 1
                            obs_idx_ls[obs_idx] += 1
                    if inter > 0:
                        dists[idx] = max(seg_dist)
                        seg_dist = [0.0] * 4
                inter = 0
            # If there are more than three dists equal to one, we need to correct the coordinates.
            if sum(filter(lambda x: x == 1.0, dists)) > 3:
                # Take the polygon which corresponds to the index with the maximum number of intersections.
                argmax = max(zip(obs_idx_ls, self.poly))[1]
                dists = self.correct_coords(argmax)
        return dists

    def correct_coords(self, poly: Polygon) -> list[float]:
        """
        Method that corrects the detector-obstruction range measurement if more than the correct
        number of directions are being activated due to the Visilibity implementation.
        """
        x_check: list[bool] = [False] * A_SIZE
        dist = 0.1
        length = 1
        poly_p: vis.Polygon = to_vis_poly(poly)

        qs: list[Point] = [from_vis_p(self.detector)] * A_SIZE
        dists: list[float] = [0.0] * A_SIZE
        while not any(x_check):
            for action in cast(tuple[Action], get_args(Action)):
                step = scale_p(
                    Point((get_x_step_coeff(action), get_y_step_coeff(action))),
                    dist * length,
                )
                qs[action] = sum_p(qs[action], step)
                if to_vis_p(qs[action])._in(poly_p, EPSILON):  # type: ignore
                    x_check[action] = True

        # i.e. if one outside the poly then
        if sum(x_check) >= 4:
            for ii in [0, 2, 4, 6]:
                if x_check[ii - 1] and x_check[ii + 1]:
                    dists[ii - 1 : ii + 2] = [1.0, 1.0, 1.0]

        return dists

    def render(
        self,
        save_gif: bool = False,
        path: Optional[str] = None,
        epoch_count: Optional[int] = None,
        just_env: Optional[bool] = False,
        obs=None,
        ep_rew=None,
        data=None,
        meas: Optional[list[float]] = None,
        params=None,
        loc_est=None,
    ):
        """
        Method that produces a gif of the agent interacting in the environment. Only renders one episode at a time.
        """

        if data and meas:
            self.intensity = params[0]
            self.bkg_intensity = params[1]
            self.src_coords = params[2]
            self.iter_count = len(meas)
            data = np.array(data) / 100
        else:
            data = np.array(self.det_sto) / 100  # Detector stored locations in an array?
            meas = self.meas_sto

        # Check only rendering one episode aka data readings available match number of rewards 
        # (+1 as rewards dont include the first position). 
        reward_length = len(ep_rew)
        if data.shape[0] != len(ep_rew)+1:
            print(f"Error: episode reward array length: {reward_length} does not match existing detector locations array length, \
            minus initial start position: {data.shape[0]}. \
            Check: Are you trying to render more than one episode?")
            return 1

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
                    p_disp = PolygonPatches(coord[0] / 100, color="gray")
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
            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(15, 5), tight_layout=True
            )
            m_size = 25

            def update(frame_number, data, ax1, ax2, ax3, src, area_dim, meas):
                print(f"Current Frame: {frame_number}", end='\r')
                current_index = frame_number % (self.iter_count)
                global loc
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
                    ax1.scatter(
                        -1000, -1000, m_size, c="black", marker="^", label="Detector"
                    )
                    ax1.grid()
                    if not (obs == []) and obs != None:
                        for coord in obs:
                            p_disp = PolygonPatches(coord[0] / 100, color="gray")
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
                    ax2.stem(
                        [current_index], [meas[current_index]], use_line_collection=True
                    )
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
                    if 'loc' in globals():
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
                    ax2.stem(
                        [current_index], [meas[current_index]], use_line_collection=True
                    )
                    ax3.plot(range(current_index), ep_rew[:current_index], c="black")
                    if not (loc_est is None):
                        loc = ax1.scatter(
                            loc_est[0][current_index][1] / 100,
                            loc_est[0][current_index][2] / 100,
                            m_size * 0.8,
                            c="magenta",
                            label="Loc. Pred.",
                        )

            print("Frames to render ", len(ep_rew))
            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(ep_rew),
                fargs=(data, ax1, ax2, ax3, self.src_coords, self.search_area, meas),
            )
            if save_gif:
                writer = PillowWriter(fps=5)
                if os.path.isdir(str(path) + "/gifs/"):
                    ani.save(str(path) + f"/gifs/test_{epoch_count}.gif", writer=writer)
                    print("")
                else:
                    os.mkdir(str(path) + "/gifs/")
                    ani.save(str(path) + f"/gifs/test_{epoch_count}.gif", writer=writer)
            else:
                plt.show()

            return 0

    def FIM_step(self, action: Action, coords: Optional[Point] = None) -> Point:
        """
        Method for the information-driven controller to update detector coordinates in the environment
        without changing the actual detector positon.

        Args:
        action : action to move the detector
        coords : coordinates to move the detector from that are different from the current detector coordinates
        """

        # Make a copy of the current detector coordinates
        det_coords = self.det_coords
        if coords:
            coords_p: vis.Point = to_vis_p(coords)
            self.detector = coords_p
            self.det_coords = coords

        in_obs = self.check_action(action)
        det_ret = self.det_coords
        if coords is None and not in_obs or coords:
            # If successful movement, return new coords. Set detector back.
            det_coords_p: vis.Point = to_vis_p(det_coords)
            self.detector = det_coords_p
            self.det_coords = det_coords

        return det_ret
