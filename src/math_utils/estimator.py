from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Tuple


@dataclass
class IntensityResamplingEstimator:
    """
    Hash table that stores radiation intensity levels as seen at each unscaled coordinate into a buffer. Because radiation intensity readings are
    drawn from a poisson distribution, the more samples that are available, the more accurate the reading. This can be used before standardizing
    the input for processing in order to get the most accurate radiation reading possible.

    (B. Totten, "Multi-Agent Deep Reinforcement Learning", Section 3.4.1: Radiation Intensity Heatmap)
    """

    #: Hash table containing explored coordinates (keys) and radiation readings detected there (list of values)
    readings: Dict[Tuple[int, int], List[float]] = field(default_factory=lambda: dict())

    # Private
    # Minimum radiation reading estimate
    _min: float = field(default=0.0)
    # Maximum radiation reading estimate. This is used for normalization in simple normalization mode.
    _max: float = field(default=0.0)

    def reset(self) -> None:
        """Method to reset class members to defaults"""
        self.readings = dict()
        self._min = 0.0
        self._max = 0.0

    def update(self, key: Tuple[int, int], value: float) -> None:
        """
        Method to add value to radiation hashtable. If key does not exist, creates key and new buffer with value. Also updates running max/min
        estimate, if applicable. Note that the max/min is the ESTIMATE of the true value, not the observed value.

        :param key: (Tuple[int, int]) Inflated coordinates where radiation intensity (value) was sampled
        :param value: (float) Sampled radiation intensity value
        """
        if self.check_key(key=key):
            self.readings[key].append(value)
        else:
            self.readings[key] = [value]

        estimate = self.get_estimate(key)
        if estimate > self._max or self._max == 0:
            self._set_max(estimate)
        if estimate < self._min or self._min == 0:
            self._set_min(estimate)

    def get_buffer(self, key: Tuple[int, int]) -> List:
        """
        Method to return existing buffer for key. Raises exception if key does not exist.

        :param key: (Point) Coordinates where radiation intensity (value) was sampled
        :returns: (List) Buffer containing all radiation observations at this hash key (typically grid coordinates)
        """
        if not self.check_key(key=key):
            raise ValueError("Key does not exist")
        return self.readings[key]

    def get_estimate(self, key: Tuple[int, int]) -> float:
        """
        Method to returns radiation estimate for current coordinates. Raises exception if key does not exist.

        :param key: (Point) Coordinates for desired radiation intensity estimate
        :returns: (float) Estimation of true radiation reading at this hash key (typically grid coordinates)
        """
        if not self.check_key(key=key):
            raise ValueError("Key does not exist")
        return median(self.readings[key])

    def get_max(self) -> float:
        """
        Method to return the maximum radiation reading estimated thus far. This can be used for normalization in simple normalization mode.
        NOTE: the max/min is the ESTIMATE of the true value, not the observed value.

        :returns: (float) Maximum radiation reading estimated thus far.
        """
        return self._max

    def get_min(self) -> float:
        """
        Method to return the minimum radiation reading estimated thus far.
        NOTE: the max/min is the ESTIMATE of the true value, not the observed value.

        :returns: (float) Minimum radiation reading estimated thus far.
        """
        return self._min

    def check_key(self, key: Tuple[int, int]) -> bool:
        """
        Method to check if coordinates (key) exist in hashtable.
        :returns: (Bool) Indication of key's existance in hashtable.
        """
        return True if key in self.readings else False

    def _set_max(self, value: float) -> None:
        """
        Method to set the maximum radiation reading estimated thus far.
        :param value: (float) Sampled radiation intensity value
        """
        self._max = value

    def _set_min(self, value: float) -> None:
        """
        Method to set the minimum radiation reading estimated thus far.
        :param value: (float) Sampled radiation intensity value
        """
        self._min = value
