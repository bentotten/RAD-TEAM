import warnings
from dataclasses import dataclass, field
from math import log
from typing import Optional, Union


@dataclass
class Normalizer:
    """
    Mix-max normalization. Method to do min-max normalization to the range [0,1]. If min is below zero, the data will be shifted by the absolute value of the minimum


    (B. Totten, "Multi-Agent Deep Reinforcement Learning", Section 3.4.1: Obstacle Detection Heatmap)
    """

    # Private
    # First base value that is used for class. The max value represents the maximum possible value (steps per episode multiplied by the number
    #   of agents).
    _base_max: Union[float, int, None] = field(default=None)

    # First increment value that is used for class. The increment value represents the amount that the existing value from shadow table is expected to
    #   increment by this amount every time.
    _base_step_size: Optional[int] = field(default=None)

    def normalize(self, current_value: float, max: float, min: Optional[float] = None) -> float:
        """
        :param current_value: (Any) value to be normalized
        :param max: (Any) Maximum possible
        :returns: (float) Normalized value for current_value input via min-max method.
        """

        # Check for edge cases and invalid inputs
        if current_value == 0:
            return 0.0
        assert max >= current_value, "Value error - Current value is less than max"

        # Process min (if current is negative, that is ok as it will be offset by min's absolute value)
        offset: Union[float, int]
        if min:
            assert current_value >= min, "Value error - current is more than max."
            offset = abs(min) if min < 0 else 0

        # If no min provided, assume min == 0 and adjust negative current values
        elif not min:
            min = 0.0
            # If current value is less than 0, it will always be offset to equal 0
            if current_value < 0:
                return 0
            offset = 0

        assert max + offset > 0, "Value error - Max is 0 but current value is not."

        result = ((current_value + offset) - (min + offset)) / ((max + offset) - (min + offset))
        assert result >= 0 and result <= 1, "Normalization error"
        return result


@dataclass
class BensLogNormalizer:
    """
    Method to normalize on a logarithmic scale. This is specifically for a value that increases incrementally every time.
    For RAD-TEAM, every time an agent accesses a grid coordinate, a visits count shadow table is incremented by 1.
    That value is multiplied by the step_size (here using 2 due to log(1) == 0) and the log is taken. This value
    is then multiplied by 1/ the increment value multiplied by the max in order to put it between 0 and 1. The max is
    the maximum number of possible steps in an episode multiplied by the number of agents.

    (B. Totten, "Multi-Agent Deep Reinforcement Learning", Section 3.4.1: Exploration Map)
    """

    # Private
    # First base value that is used for class. The max value represents the maximum possible value (steps per episode multiplied by the number
    #   of agents).
    _base_max: Union[float, int, None] = field(default=None)

    # First increment value that is used for class. The increment value represents the amount that the existing value from shadow table is expected to
    #   increment by this amount every time.
    _base_step_size: Optional[int] = field(default=None)

    def normalize_incremental_logscale(
        self, current_value: Union[float, int], max: Union[float, int], step_size: int = 2
    ) -> float:
        """
        :param current_value: (Any) value to be normalized
        :param max: (Any) Maximum possible value (steps per episode multiplied by the number of agents)
        :param step_size (int): Value from shadow table is expected to increment by this amount every time

        :returns: (float) Normalized value for current_value input via log method.
        """

        assert current_value >= 0 and max > 0 and step_size > 0, "Value error - input was negative that should not be"

        # Warnings for different scales
        if not self._base_max:
            self._base_max = max
        elif self._base_max != max:
            warnings.warn(
                UserWarning(
                    "mismatch, Max mismatch from first use of normalize_incremental_logscale function! Ensure this was intentional! "
                )
            )
        if not self._base_step_size:
            self._base_step_size = step_size
        if self._base_step_size != step_size:
            warnings.warn(
                UserWarning(
                    "mismatch, Step size mismatch from first use of normalize_incremental_logscale function! Ensure this was intentional"
                )
            )

        result = (log(step_size + current_value, max)) * 1 / log(step_size * max, max)  # Put in range [0, 1]
        assert (
            result >= 0 and result <= 1
        ), f"Normalization error for Result: {result}, step_size: {step_size}, Current value: {current_value}, Max: {max}"

        return result
