""" Collections of tools used to standardize data """

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class WelfordsOnlineStandardization:
    """
    Calculates mean and variance from an estimated running sample. Used for standardizing readings when an Agent collects observations online and does not know the range of intensity values it will encounter
    beforehand.

    (B. Welford, "Note on a method for calculating corrected sums of squares and products")
    """

    #: Running mean of entire dataset, represented by mu
    mean: float = 0.0
    #: Aggregated squared distance from the mean, represented by M_2
    square_dist_mean: float = 0.0
    #: Sample variance, represented by s^2. This is used instead of the running variance to reduce bias.
    sample_variance: float = 0.0
    #: Standard-deviation, represented by sigma
    std: float = 1.0
    #: Count of how many samples have been seen so far
    count: int = 0

    # Private
    # Maximum radiation reading estimate. This is used for normalization in simple normalization mode.
    _max: float = field(init=False)
    # Minimum radiation reading estimate. This is used for shifting normalization data in the case of a negative.
    _min: float = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Method to reset class members to defaults"""
        self.mean = 0.0
        self.square_dist_mean = 0.0
        self.sample_variance = 0.0
        self.std = 1.0
        self.count = 0
        self._max = 0.0
        self._min = 0.0

    def update(self, reading: float) -> None:
        """Method to update estimate running mean and sample variance for standardizing radiation intensity readings. Also updates max standardized
         value for normalization, if applicable.

         #. The existing mean is subtracted from the new reading to get the initial delta.
         #. This delta is then divided by the number of samples seen so far and added to the existing mean to create a new mean.
         #. This new mean is then subtracted from the reading to get new delta.
         #. This new delta is multiplied by the old delta and added to the existing squared distance from the mean.
         #. To get the sample variance, the new existing squared distance from the mean is divided by the number of samples seen so far minus 1.
         #. To get the sample standard deviation, the square root of this value is taken.

        B. Welford, "Note on a method for calculating corrected sums of squares and products"
        `Wiki - Algorithms for calculating variance <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#cite_ref-5>`_
        `NZMaths - Sample Variance <https://nzmaths.co.nz/category/glossary/sample-variance>`_

         :param reading: (float) radiation intensity reading
        """
        assert reading >= 0

        self.count += 1
        if self.count == 1:
            self.mean = reading  # For first reading, mean is equal to that reading
        else:
            mean_new = self.mean + (reading - self.mean) / (self.count)
            square_dist_mean_new = self.square_dist_mean + (reading - self.mean) * (reading - mean_new)
            self.mean = mean_new
            self.square_dist_mean = square_dist_mean_new
            self.sample_variance = square_dist_mean_new / (self.count - 1)
            self.std = max(sqrt(self.sample_variance), 1)

        new_standard = self.standardize(reading=reading)
        if new_standard > self._max:
            self._max = new_standard
        if new_standard < self._min:
            self._min = new_standard

    def standardize(self, reading: float) -> float:
        """
        Method to standardize input data using the Z-score method by by subtracting the mean and dividing by the standard deviation.
        Standardizing input data increases training stability and speed. Once the standardization is done, all the features will have a mean of zero
        and a standard deviation of one, and thus, the same scale. NOTE: Because the first reading will always be standardized to zero, it is
        important to standardize after a reset before the first step to ensure first steps reading is not wasted.

        :param reading: (float) radiation intensity reading
        :return: (float) Standardized radiation reading (z-score) where all existing samples have a std of 1
        """
        assert reading >= 0
        return (reading - self.mean) / self.std

    def get_max(self) -> float:
        """
        Method to return the current maximum standardized sample (updated during update function).
        :return: Current maximum standardized sample.
        """
        return self._max

    def get_min(self) -> float:
        """
        Method to return the current minimum standardized sample (updated during update function).
        :return: Current minimum standardized sample.
        """
        return self._min
