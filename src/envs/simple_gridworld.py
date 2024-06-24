from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pygame
from gymnasium import Env
from gymnasium import spaces as Spaces

# From gymnasium core, for type checker
RenderFrame = TypeVar("RenderFrame")


#: Valid actions
class Action(IntEnum):
    UP: int = 1
    RIGHT: int = 2
    LEFT: int = 3
    DOWN: int = 4


#: Mapping of actions to the directional movement.
#:  The following dictionary maps abstract actions from `self.action_space` to
#:  the direction we will walk in if that action is taken.
#:  I.e. 0 corresponds to "right", 1 to "up" etc.
ActionDirectionMap: Dict[int, npt.NDArray[np.int32]] = {
    Action.UP: np.array([0, 1], dtype=np.int32),
    Action.RIGHT: np.array([1, 0], dtype=np.int32),
    Action.LEFT: np.array([-1, 0], dtype=np.int32),
    Action.DOWN: np.array([0, -1], dtype=np.int32),
}


class SimpleGrid(Env):
    """
    Simple single-agent 2D gridworld with fixed start and a single fixed, static terminal position.

    Action space: There are four valid actions (up, down, left, right).

    Observation space: the acting agent's (x,y) coordinates. Note that an "Oracle" observation is available, which shows both the agent coordinates and the terminal coordinates

    Reward space:
        - Step towards terminal position: 0
        - Step away from terminal position: -1
        - Step within 1 grid mark of terminal position: +1

    :param start: npt.NDArray[np.float32], (x, y) list of coordinates to start at.
    :param terminal: npt.NDArray[np.float32], (x, y) coordinates of terminal positions.
    :param size: int, size of the square grid (upper boundary)

    (https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation)
    """

    env_id = "SimpleGrid-v0"

    # See Gymnasium documentation for more details:
    #   - Environment API: https://gymnasium.farama.org/api/env/
    #   - Spaces Datatypes: https://gymnasium.farama.org/api/spaces/fundamental/

    start: npt.NDArray[np.int32]
    terminal: npt.NDArray[np.int32]
    size: int

    #: The agent that will be moving around this environment. This is represented by its (x, y) coordinates
    agent: npt.NDArray[np.int32]
    #: The previous distance between the agent and the terminal position
    agent_previous_dist: np.float32
    #: The Space object corresponding to valid actions, all valid actions should be contained within the space.
    action_space: Spaces.MultiDiscrete
    #: The Space object corresponding to valid observations, all valid observations should be contained within the space.
    observation_space: Spaces.Box
    #: A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode. The default reward range is set to .
    reward_range: Tuple

    #: The random number generator for the environment.
    np_random: np.random.Generator
    #: Instead of using a passed-in random generator, can choose to instead use a seed and make an internal rng
    seed: Optional[int] = None
    #: An environment spec that contains the information used to initialize the environment from gymnasium.make()
    spec: None = None
    #: The metadata of the environment, i.e. render modes, render fps
    metadata: dict = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    #: Size of the render window
    window_size: int = 512
    window: Optional[pygame.Surface] = None
    clock: Optional[pygame.time.Clock] = None

    def __init__(
        self,
        start: npt.NDArray[np.int32] = np.array((0,0)),
        terminal: npt.NDArray[np.int32] = np.array((0,0)),
        size: int = 10,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.start = start
        self.terminal = terminal
        self.size = size

        if not self.np_random:
            self.np_random = np.random.default_rng(seed=self.seed)

        # Setup spaces
        self.action_space = Spaces.MultiDiscrete([act.value for act in Action], seed=self.np_random)
        self.observation_space = Spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=np.int32, seed=self.np_random)
        # Unbounded reward range
        self.reward_range = (np.NINF, np.inf)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Note: Intentionally not calling reset() here to avoid rendering an unnecessary frame
        self.agent = self.start

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> tuple[npt.NDArray[np.int32], dict[str, Any]]:
        """Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info."""
        # Note: Environment already has a prng, no need to reset with a seed
        super().reset(seed=seed, options=options)
        self.agent = self.start
        self.agent_previous_dist = self._get_distance()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: int) -> Tuple[npt.NDArray[np.int32], int, bool, bool, Dict[str, Any]]:
        """
        Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
        :param action: int, an int that cooresponds to a specific movement in the gridworld

        :returns observation, reward, done, Truncated, info
        """
        action = Action(action)  # Convert to internal action

        # Use np.clip to make sure agent doesnt leave the grid
        self.agent = np.clip(self.agent + ActionDirectionMap[action], 0, self.size - 1)

        # Calculate rewards and determine if "done" state has been reached
        distance: np.float32 = self._get_distance()

        done = False
        if distance <= 1.0:
            done = True
            reward = 1
        # Note: previous distance for first step is set in the reset function
        elif distance < self.agent_previous_dist:
            reward = 0
        elif distance >= self.agent_previous_dist:
            reward = -1
        else:
            raise Exception("Something has gone wrong in the step function")

        # Avoid recalculating distance by saving here
        self.agent_previous_dist = distance

        observation = self._get_obs()
        info = self._get_info(distance=distance)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, False, info

    def render(self) -> Optional[RenderFrame | list[RenderFrame]]:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self) -> None:
        """Closes the environment, important when external software is used, i.e. pygame for rendering, databases"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_distance(self) -> np.float32:
        """Get the manhattan distance between the agent and the target"""
        return np.linalg.norm(self.agent - self.terminal, ord=1)

    def _get_oracle_obs(self) -> Dict[str, npt.NDArray[np.int32]]:
        """Get an "Oracle" observation of the environment state. Includes agent and terminal positions."""
        return {"agent": self.agent, "target": self.terminal}

    def _get_obs(self) -> npt.NDArray[np.int32]:
        """Get an the agents observation of the environment state."""
        return self.agent

    def _get_info(self, distance: Optional[np.float32] = None) -> Dict[str, Any]:
        """Get the Oracles view of the env and the distance between the agent and the target"""
        distance = distance if distance else self._get_distance()
        return {**self._get_oracle_obs(), "distance": distance}

    def _render_frame(self) -> Optional[RenderFrame | list[RenderFrame]]:
        """Renders using pygame. See: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#rendering"""

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels

        # First we draw the target
        # TODO: sort out type signature issues
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.terminal,  # type: ignore
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        # TODO: sort out type signature issues
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent + 0.5) * pix_square_size,  # type: ignore
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # Use human mode
        if self.window and self.clock and self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            return None
        # Return the rgb_array
        elif self.render_mode == "rgb_array":
            # TODO sort out type signature issues
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))  # type: ignore
        else:
            raise Exception("Unsupported environment render mode")
