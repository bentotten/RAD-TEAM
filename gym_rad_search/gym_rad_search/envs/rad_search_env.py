import gym
import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
import visilibity as vis
import os
from gym import spaces
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon
from typing import Any, Callable, Literal, Optional, TypedDict
from typing_extensions import TypeAlias

FPS = 50

DET_STEP = 100.0  # detector step size at each timestep in cm/s
DET_STEP_FRAC = 71.0  # diagonal detector step size in cm/s
DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC = 78.0  # Diagonal detector-obstruction range measurement threshold in cm

EPSILON = 0.0000001

Metadata: TypeAlias = TypedDict(
    "Metadata", {"render.modes": list[str], "video.frames_per_second": int}
)
Interval: TypeAlias = tuple[float, float]
Dimensions: TypeAlias = tuple[Interval, Interval, Interval, Interval]
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


class RadSearch(gym.Env):
    # Values set in the constructor
    action_space: spaces.Discrete
    area_obs: Interval
    bounds: npt.NDArray[np.float64]
    bbox: list[vis.Point]
    coord_noise: bool
    env_ls: list[vis.Polygon]
    max_dist: float
    line_segs: list[list[vis.Line_Segment]]
    poly: list[vis.Polygon]
    np_random: np.random.Generator
    obstruct: Literal[-1, 0, 1]
    search_area: npt.NDArray[np.float64]
    # TODO: self.det_coords isn't initialized until reset() is called.
    det_coords: npt.NDArray[np.float64]
    # TODO: self.src_coords isn't initialized until reset() is called.
    src_coords: npt.NDArray[np.float64]
    walls: vis.Polygon
    world: vis.Environment
    vis_graph: vis.Visibility_Graph
    # TODO: self.intensity isn't initialized until reset() is called but is used in step.
    intensity: int
    # TODO: self.bkg_intensity isn't initialized until reset() is called but is used in step.
    bkg_intensity: int
    obs_coord: list[list[npt.NDArray[np.float64]]]

    # Values with default values which are not set in the constructor
    _max_episode_steps: int = 120
    a_size: int = 8
    bkg_bnd: npt.NDArray[np.float64] = np.array([10, 51])
    continuous: bool = False
    det_sto: list[npt.NDArray[np.float64]] = []
    done: bool = False
    dwell_time: int = 1
    epoch_cnt: int = 0
    epoch_end: bool = True
    int_bnd: npt.NDArray[np.float64] = np.array([1e6, 10e6])
    iter_count: int = 0
    meas_sto: list[float] = []
    metadata: Metadata = {"render.modes": ["human"], "video.frames_per_second": FPS}
    observation_space: spaces.Box = spaces.Box(0, np.inf, shape=(11,), dtype=np.float32)
    oob_count: int = 0
    oob: bool = False
    # TODO: self.prev_det_dist isn't initialized until reset() is called but is used in step.
    prev_det_dist: float = None
    viewer: None = None
    # TODO: self.sp_dist is declared and defined in step.
    sp_dist: float = None
    # TODO: self.euc_dist is declared and defined in step.
    euc_dist: float = None
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
    def __init__(
        self,
        bbox: Dimensions,
        area_obs: Interval,
        seed: np.random.Generator,
        obstruct: Literal[-1, 0, 1] = 0,
        coord_noise: bool = False,
    ):
        self.np_random = seed
        self.bounds = np.asarray(bbox)
        self.search_area = np.array(
            [
                [self.bounds[0][0] + area_obs[0], self.bounds[0][1] + area_obs[0]],
                [self.bounds[1][0] - area_obs[1], self.bounds[1][1] + area_obs[0]],
                [self.bounds[2][0] - area_obs[1], self.bounds[2][1] - area_obs[1]],
                [self.bounds[3][0] + area_obs[0], self.bounds[3][1] - area_obs[1]],
            ]
        )
        self.area_obs = area_obs
        self.obstruct = obstruct
        self.max_dist = math.sqrt(
            self.search_area[2][0] ** 2 + self.search_area[2][1] ** 2
        )
        self.coord_noise = coord_noise
        self.action_space: spaces.Discrete = spaces.Discrete(self.a_size)

    def step(
        self, action: Optional[Action]
    ) -> tuple[npt.NDArray[np.float64], float, bool, dict[Any, Any]]:
        """
        Method that takes an action and updates the detector position accordingly.
        Returns an observation, reward, and whether the termination criteria is met.
        """
        # Move detector and make sure it is not in an obstruction
        in_obs = self.check_action(action)
        if not in_obs:
            # TODO: self.det_coords isn't initialized until reset() is called.
            if np.any(self.det_coords < (self.search_area[0])) or np.any(
                self.det_coords > (self.search_area[2])
            ):
                self.oob = True
                self.oob_count += 1

            # Returns the length of a Polyline, which is a double
            # https://github.com/tsaoyu/PyVisiLibity/blob/80ce1356fa31c003e29467e6f08ffdfbd74db80f/visilibity.cpp#L1398
            self.sp_dist: float = self.world.shortest_path(
                self.source, self.detector, self.vis_graph, EPSILON
            ).length()
            # TODO: self.src_coords isn't initialized until reset() is called.
            self.euc_dist: float = ((self.det_coords - self.src_coords) ** 2).sum()
            self.intersect = self.is_intersect()
            # TODO: self.bkg_intensity isn't initialized until reset() is called.
            # TODO: self.intensity isn't initialized until reset() is called.
            meas: float = self.np_random.poisson(
                self.bkg_intensity
                if self.intersect
                else self.intensity / self.euc_dist + self.bkg_intensity
            )

            # Reward logic
            if self.sp_dist < 110:
                reward = 0.1
                self.done = True
            # TODO: self.prev_det_dist isn't initialized until reset() is called.
            elif self.sp_dist < self.prev_det_dist:
                reward = 0.1
                self.prev_det_dist = self.sp_dist
            else:
                reward = -0.5 * self.sp_dist / (self.max_dist)

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
                self.euc_dist = ((self.det_coords - self.src_coords) ** 2).sum()
                self.intersect = self.is_intersect()
                meas: float = self.np_random.poisson(
                    self.bkg_intensity
                    if self.intersect
                    else self.intensity / self.euc_dist + self.bkg_intensity
                )

            reward = -0.5 * self.sp_dist / (self.max_dist)

        # If detector coordinate noise is desired
        if self.coord_noise:
            noise: npt.NDArray[np.float64] = self.np_random.normal(
                scale=5, size=len(self.det_coords)
            )
        else:
            noise: npt.NDArray[np.float64] = np.zeros(len(self.det_coords))

        # Scale detector coordinates by search area of the DRL algorithm
        det_coord_scaled: npt.NDArray[np.float64] = (
            self.det_coords + noise
        ) / self.search_area[2][1]

        # Observation with the radiation meas., detector coords and detector-obstruction range meas.
        state: npt.NDArray[np.float64] = np.append(meas, det_coord_scaled)
        if self.num_obs > 0:
            sensor_meas: npt.NDArray[np.float64] = self.dist_sensors()
            state = np.append(state, sensor_meas)
        else:
            state = np.append(state, np.zeros(self.a_size))
        self.oob = False
        self.det_sto.append(self.det_coords.copy())
        self.meas_sto.append(meas)
        self.iter_count += 1
        return state, round(reward, 2), self.done, {}

    def reset(self) -> npt.NDArray[np.float64]:
        """
        Method to reset the environment.
        If epoch_end flag is True, then all components of the environment are resampled
        If epoch_end flag is False, then only the source and detector coordinates, source activity and background
        are resampled.
        """
        self.done = False
        self.oob = False
        self.iter_count = 0
        self.oob_count = 0
        self.dwell_time = 1

        if self.epoch_end:
            if self.obstruct == -1:
                self.num_obs = self.np_random.integers(1, 6)
            elif self.obstruct == 0:
                self.num_obs = 0
            else:
                self.num_obs = self.obstruct

            self.create_obs()
            self.bbox: list[vis.Point] = list(
                map(lambda tuple: vis.Point(*tuple), self.bounds)
            )
            self.walls = vis.Polygon(self.bbox)

            # TODO: self.poly isn't initialized yet
            self.env_ls: list[vis.Polygon] = [self.walls, *self.poly]

            # Create Visilibity environment
            self.world = vis.Environment(self.env_ls)

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
        self.intensity = self.np_random.integers(self.int_bnd[0], self.int_bnd[1])
        self.bkg_intensity = self.np_random.integers(self.bkg_bnd[0], self.bkg_bnd[1])

        self.prev_det_dist: float = self.world.shortest_path(
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

    def check_action(self, action: Optional[Action]) -> bool:
        """
        Method that checks which direction to move the detector based on the action.
        If the action moves the detector into an obstruction, the detector position
        will be reset to the prior position.
        """
        in_obs: bool = False

        if action is None:
            return in_obs

        # 0: (-1)*DET_STEP     *x, ( 0)*DET_STEP     *y
        # 1: (-1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
        # 2: ( 0)*DET_STEP     *x, (+1)*DET_STEP     *y
        # 3: (+1)*DET_STEP_FRAC*x, (+1)*DET_STEP_FRAC*y
        # 4: (+1)*DET_STEP     *x, ( 0)*DET_STEP     *y
        # 5: (+1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y
        # 6: ( 0)*DET_STEP     *x, (-1)*DET_STEP     *y
        # 7: (-1)*DET_STEP_FRAC*x, (-1)*DET_STEP_FRAC*y

        # The signs of the y-coeffecients follow the signs of sin(pi * (1 - action/4))
        # The signs of the x-coefficients follow the signs of cos(pi * (1 - action/4)) = sin(pi * (1 - (action - 2)/4))
        y_step_coeffs: Callable[[int], int] = lambda y: round(
            math.sin(math.pi * (1.0 - y / 4.0))
        )
        y_step_coeff = y_step_coeffs(action)
        x_step_coeff = y_step_coeffs((action - 2) % 8)

        # If action is odd, then we are moving on the diagonal and so our step size is smaller.
        # Otherwise, we're moving solely in a cardinal direction.
        step_size: float = DET_STEP if action % 2 == 0 else DET_STEP_FRAC

        self.detector.set_y(self.det_coords[1] + y_step_coeff * step_size)
        self.detector.set_x(self.det_coords[0] + x_step_coeff * step_size)
        in_obs = self.in_obstruction()
        if in_obs:
            self.detector.set_y(self.det_coords[1])
            self.detector.set_x(self.det_coords[0])
        else:
            self.det_coords[1] = self.detector.y()
            self.det_coords[0] = self.detector.x()

        return in_obs

    def create_obs(self) -> None:
        """
        Method that randomly samples obstruction coordinates from 90% of search area dimensions.
        Obstructions are not allowed to intersect.
        """
        seed_pt: npt.NDArray[np.float64] = np.zeros(2)
        ii = 0
        intersect = False
        # TODO: Though the two of these should be equivalent, using the former causes the following warning
        # to be thrown when running ppo:
        #   The boundary of hole 1 intersects the boundary of hole 4.
        #   Environment is not valid, retrying!
        # self.obs_coord: list[list[npt.NDArray[np.float64]]] = self.num_obs * [[]]
        self.obs_coord: list[list[npt.NDArray[np.float64]]] = [
            [] for _ in range(self.num_obs)
        ]
        self.poly = []
        self.line_segs: list[list[vis.Line_Segment]] = []
        obs_coord: npt.NDArray[np.float64] = np.array([])
        while ii < self.num_obs:
            seed_pt[0] = self.np_random.integers(
                self.search_area[0][0], self.search_area[2][0] * 0.9, size=(1)
            )
            seed_pt[1] = self.np_random.integers(
                self.search_area[0][1], self.search_area[2][1] * 0.9, size=(1)
            )
            ext: npt.NDArray[np.float64] = self.np_random.integers(
                self.area_obs[0], self.area_obs[1], size=2
            )
            obs_coord = np.append(obs_coord, seed_pt)
            obs_coord = np.vstack((obs_coord, [seed_pt[0], seed_pt[1] + ext[1]]))
            obs_coord = np.vstack(
                (obs_coord, [seed_pt[0] + ext[0], seed_pt[1] + ext[1]])
            )
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
                obs_coord_list: list[npt.NDArray[np.float64]] = list(obs_coord)
                geom: list[vis.Point] = list(
                    map(lambda tuple: vis.Point(*tuple), obs_coord_list)
                )
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
        source = self.np_random.integers(
            self.search_area[0][0], self.search_area[1][0], size=2
        ).astype(np.double)
        det = self.np_random.integers(
            self.search_area[0][0], self.search_area[1][0], size=2
        ).astype(np.double)
        det_point = vis.Point(det[0], det[1])

        while not det_clear:
            while not resamp and jj < self.num_obs:
                if det_point._in(self.poly[jj], EPSILON):
                    resamp = True
                jj += 1
            if resamp:
                det = self.np_random.integers(
                    self.search_area[0][0], self.search_area[1][0], size=2
                ).astype(np.double)
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
            while np.linalg.norm(det - source) < 1000:
                source = self.np_random.integers(
                    self.search_area[0][0], self.search_area[1][0], size=2
                ).astype(np.double)
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
                source = self.np_random.integers(
                    self.search_area[0][0], self.search_area[1][0], size=2
                ).astype(np.double)
                src_point.set_x(source[0])
                src_point.set_y(source[1])
                jj = 0
                resamp = False
                inter = False
                num_retry += 1
            elif inter:
                src_clear = True

        return src_point, det_point, det, source

    def is_intersect(self, threshold=0.001):
        """
        Method that checks if the line of sight is blocked by any obstructions in the environment.
        """
        inter = False
        kk = 0
        L = vis.Line_Segment(self.detector, self.source)
        while not inter and kk < self.num_obs:
            if vis.boundary_distance(L, self.poly[kk]) < threshold and not math.isclose(
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
            if self.detector._in(self.poly[jj], EPSILON):
                obs_boundary = True
            jj += 1

        if obs_boundary:
            bbox = self.poly[jj - 1].bbox()
            if self.detector.y() > bbox.y_min:
                if self.detector.y() < bbox.y_max:
                    if self.detector.x() > bbox.x_min:
                        if self.detector.x() < bbox.x_max:
                            return True
            return False
        else:
            return False

    def dist_sensors(self):
        """
        Method that generates detector-obstruction range measurements with values between 0-1.
        """
        seg_coords = [
            vis.Point(self.detector.x() - DIST_TH, self.detector.y()),
            vis.Point(
                self.detector.x() - DIST_TH_FRAC, self.detector.y() + DIST_TH_FRAC
            ),
            vis.Point(self.detector.x(), self.detector.y() + DIST_TH),
            vis.Point(
                self.detector.x() + DIST_TH_FRAC, self.detector.y() + DIST_TH_FRAC
            ),
            vis.Point(self.detector.x() + DIST_TH, self.detector.y()),
            vis.Point(
                self.detector.x() + DIST_TH_FRAC, self.detector.y() - DIST_TH_FRAC
            ),
            vis.Point(self.detector.x(), self.detector.y() - DIST_TH),
            vis.Point(
                self.detector.x() - DIST_TH_FRAC, self.detector.y() - DIST_TH_FRAC
            ),
        ]
        segs = [vis.Line_Segment(self.detector, seg_coord) for seg_coord in seg_coords]
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
                            seg_dist[seg_idx] = (
                                DIST_TH - vis.distance(seg.first(), obs_seg)
                            ) / DIST_TH
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
        q0 = vis.Point(self.detector.x(), self.detector.y())
        q1 = vis.Point(self.detector.x(), self.detector.y())
        q2 = vis.Point(self.detector.x(), self.detector.y())
        q3 = vis.Point(self.detector.x(), self.detector.y())
        q4 = vis.Point(self.detector.x(), self.detector.y())
        q5 = vis.Point(self.detector.x(), self.detector.y())
        q6 = vis.Point(self.detector.x(), self.detector.y())
        q7 = vis.Point(self.detector.x(), self.detector.y())

        # qs = [vis.Point(self.detector.x(),self.detector.y()) for _ in range(self.a_size)]
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
            fig, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=(15, 5), tight_layout=True
            )
            m_size = 25

            def update(frame_number, data, ax1, ax2, ax3, src, area_dim, meas):
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

    def FIM_step(self, action, coords=None):
        """
        Method for the information-driven controller to update detector coordinates in the environment
        without changing the actual detector positon.

        Args:
        action : action to move the detector
        coords : coordinates to move the detector from that are different from the current detector coordinates
        """
        if coords is None:
            det_coords = self.det_coords.copy()
        else:
            det_coords = self.det_coords.copy()
            self.detector.set_x(coords[0])
            self.detector.set_y(coords[1])
            self.det_coords = coords.copy()

        in_obs = self.check_action(action)

        if coords is None:
            det_ret = self.det_coords
            if not (
                in_obs
            ):  # If successful movement, return new coords. Set detector back.
                self.det_coords = det_coords
                self.detector.set_x(det_coords[0])
                self.detector.set_y(det_coords[1])
        else:
            det_ret = self.det_coords.copy()
            self.det_coords = det_coords
            self.detector.set_x(det_coords[0])
            self.detector.set_y(det_coords[1])

        return det_ret


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
