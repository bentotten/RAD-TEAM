from typing import List, Union

import gymnasium
import numpy as np
import numpy.typing as npt
import pytest

from src.envs.simple_gridworld import Action, ActionDirectionMap, SimpleGrid


def arrays_match(array1: Union[List, npt.NDArray], array2: Union[List, npt.NDArray]) -> bool:
    return bool(np.all(array1 == array2))


def get_manhattan_distance(start: npt.NDArray, stop: npt.NDArray) -> np.float32:
    return np.linalg.norm(start - stop, ord=1)


class Test_SimpleGrid:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup the env"""
        self.start: npt.NDArray = np.array([5, 5], dtype=np.int32)
        self.terminal: npt.NDArray = np.array([0, 0], dtype=np.int32)
        self.env: SimpleGrid = SimpleGrid(self.start, self.terminal, render_mode="rgb_array")
        self.original_distance = get_manhattan_distance(self.start, self.terminal)
        self.env.reset()

    @pytest.fixture(autouse=True)
    def teardown(self) -> None:
        self.env.close()

    def test_gym_make(self) -> None:
        """Ensure Gymnasium successfully makes environment and functionality matches manual initialization"""
        gym_env = gymnasium.make("SimpleGrid", start=self.start, terminal=self.terminal, render_mode="rgb_array")
        assert self.env is not gym_env

        # Check API functionality matches explicitly declared class
        obs1, info1 = self.env.reset()
        gym_obs1, gym_info1 = gym_env.reset()
        assert arrays_match(obs1, gym_obs1)
        for (key, val), (gym_key, gym_val) in zip(info1.items(), gym_info1.items()):
            assert key == gym_key
            if type(val) is np.ndarray:
                assert arrays_match(val, gym_val)
            else:
                assert val == gym_val

        obs2, rew2, done2, trunc2, info2 = self.env.step(1)
        gym_obs2, gym_rew2, gym_done2, gym_trunc2, gym_info2 = gym_env.step(1)
        assert arrays_match(obs2, gym_obs2)
        assert rew2 == gym_rew2
        assert done2 == gym_done2
        assert trunc2 == gym_trunc2

        for (key, val), (gym_key, gym_val) in zip(info2.items(), gym_info2.items()):
            assert key == gym_key
            if type(val) is np.ndarray:
                assert arrays_match(val, gym_val)
            else:
                assert val == gym_val

        # TODO Make  typechecking work with render function and gym wrapper
        render = self.env.render()  # type: ignore
        gym_render = gym_env.render()  # type: ignore
        assert arrays_match(render, gym_render)  # type: ignore

    def test_reset(self) -> None:
        """Ensure reset function works before and after stepping"""
        # Test initial state
        obs, info = self.env.reset()
        assert arrays_match(obs, self.start)

        info_keys = info.keys()
        assert len(info_keys) == 3
        assert "agent" in info_keys
        assert "target" in info_keys
        assert "distance" in info_keys
        assert arrays_match(info["agent"], self.start)
        assert arrays_match(info["target"], self.terminal)
        assert info["distance"] == 10
        assert self.env.agent_previous_dist == self.original_distance

        # Test state after a step and a reset
        _ = self.env.step(1)
        obs, info = self.env.reset()

        assert arrays_match(obs, self.start)

        info_keys = info.keys()
        assert len(info_keys) == 3
        assert "agent" in info_keys
        assert "target" in info_keys
        assert "distance" in info_keys
        assert arrays_match(info["agent"], self.start)
        assert arrays_match(info["target"], self.terminal)
        assert info["distance"] == 10

    def test_step(self) -> None:
        """Ensure step function works as expected"""
        _ = self.env.reset()

        # Test an invalid step
        with pytest.raises(ValueError, match="0 is not a valid Action"):
            _ = self.env.step(0)

        # Test action mappings
        for act in Action:
            position_check = self.start + ActionDirectionMap[act]
            distance_check = get_manhattan_distance(position_check, self.terminal)
            obs, rew, done, trunc, info = self.env.step(act.value)

            # Check return values
            assert not done
            assert not trunc
            assert arrays_match(obs, position_check)

            if distance_check > self.original_distance:
                assert rew == -1
            else:
                assert rew == 0

            info_keys = info.keys()
            assert len(info_keys) == 3
            assert "agent" in info_keys
            assert "target" in info_keys
            assert "distance" in info_keys
            assert arrays_match(info["agent"], position_check)
            assert arrays_match(info["target"], self.terminal)
            assert info["distance"] == distance_check

            # Check internal updates
            assert arrays_match(self.env.agent, position_check)
            # Note: Previous distance gets updated after the reward is calculated within step
            assert self.env.agent_previous_dist == distance_check

            _ = self.env.reset()

        # Test terminal step
        temp_terminal: npt.NDArray = self.start + 1
        terminal_env: SimpleGrid = SimpleGrid(self.start, temp_terminal)
        _ = terminal_env.reset()

        position_check = self.start + ActionDirectionMap[1]
        distance_check = get_manhattan_distance(position_check, temp_terminal)
        obs, rew, done, trunc, info = terminal_env.step(1)

        # Check return values
        assert done
        assert not trunc
        assert arrays_match(obs, position_check)

        assert rew == 1

        info_keys = info.keys()
        assert len(info_keys) == 3
        assert "agent" in info_keys
        assert "target" in info_keys
        assert "distance" in info_keys
        assert arrays_match(info["agent"], position_check)
        assert arrays_match(info["target"], temp_terminal)
        assert info["distance"] == distance_check

    def test_get_distance(self) -> None:
        """Ensure the _get_distance function returns the manhattan"""
        assert self.env._get_distance() == get_manhattan_distance(self.start, self.terminal)

    def test_get_oracle_obs(self) -> None:
        """Ensure the oracle observation returns accurate agent and target locations"""
        oracle_obs = self.env._get_oracle_obs()
        oracle_obs_keys = oracle_obs.keys()
        assert len(oracle_obs_keys) == 2
        assert "agent" in oracle_obs_keys
        assert "target" in oracle_obs_keys
        assert arrays_match(oracle_obs["agent"], self.start)
        assert arrays_match(oracle_obs["target"], self.terminal)

    def test_get_obs(self) -> None:
        """Ensure the get observation function returns the accurate coordinates for the agent"""
        assert arrays_match(self.env._get_obs(), self.start)

    def test_get_info(self) -> None:
        """Ensure the info function returns both the oracle observation and accurate distance information"""
        info = self.env._get_info()

        info_keys = info.keys()
        assert len(info_keys) == 3
        assert "agent" in info_keys
        assert "target" in info_keys
        assert "distance" in info_keys
        assert arrays_match(info["agent"], self.start)
        assert arrays_match(info["target"], self.terminal)
        assert info["distance"] == get_manhattan_distance(self.start, self.terminal)

    def test_render(self) -> None:
        # TODO
        pass

    def test_render_frame(self) -> None:
        # TODO
        pass

    def test_close(self) -> None:
        # TODO
        pass
