import pytest

from src.math_utils.estimator import IntensityResamplingEstimator


class Test_IntensityResamplingEstimator:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup the estimator"""
        self.estimator = IntensityResamplingEstimator()

    def test_Update(self) -> None:
        """
        Test update function.
        Should add values to buffer according to the coordinate key
        """
        self.estimator.update(key=(1, 2), value=1000)
        assert (1, 2) in self.estimator.readings.keys()
        assert [1000] in self.estimator.readings.values()

    def test_GetBuffer(self) -> None:
        """
        Test get buffer function
        Should pull values into a list from a buffer
        """
        # Non-existant key
        with pytest.raises(ValueError):
            self.estimator.get_buffer(key=(1, 2))

        # Get buffer
        self.estimator.update(key=(1, 2), value=1000)
        test_buffer: list = self.estimator.get_buffer(key=(1, 2))
        assert len(test_buffer) == 1
        assert test_buffer[0] == 1000

        # Add another
        self.estimator.update(key=(1, 2), value=2000)
        test_buffer2: list = self.estimator.get_buffer(key=(1, 2))
        assert len(test_buffer2) == 2
        assert test_buffer2[0] == 1000
        assert test_buffer2[1] == 2000

        # Add different coordinate
        self.estimator.update(key=(3, 3), value=350)
        test_buffer2_2: list = self.estimator.get_buffer(key=(1, 2))
        assert len(test_buffer2_2) == 2
        assert test_buffer2_2[0] == 1000
        assert test_buffer2_2[1] == 2000
        test_buffer3: list = self.estimator.get_buffer(key=(3, 3))
        assert len(test_buffer3) == 1
        assert test_buffer3[0] == 350

    def test_GetEstimate(self) -> None:
        """
        Test get median function.
        Should take the median of the existing values stored in a single buffers location
        """
        # Non-existant key
        with pytest.raises(ValueError):
            self.estimator.get_estimate(key=(1, 2))

        # Test median
        self.estimator.update(key=(1, 2), value=1000)
        self.estimator.update(key=(1, 2), value=2000)
        median: float = self.estimator.get_estimate(key=(1, 2))
        assert median == 1500

        # Add another value
        self.estimator.update(key=(1, 2), value=500)
        median2: float = self.estimator.get_estimate(key=(1, 2))
        assert median2 == 1000

    def test_GetMinMax(self) -> None:
        """
        Test get max and get min functions. Should update with latest estimate of true radiation value at that location
        NOTE: the max/min is the ESTIMATE of the true value, not the observed value.
        Should properly update values as more observations are added to the buffers
        """
        # Test initial values
        assert self.estimator.get_max() == 0.0
        assert self.estimator.get_min() == 0.0

        # Test first update
        self.estimator.update(key=(1, 2), value=1000)
        assert self.estimator.get_max() == 1000
        assert self.estimator.get_min() == 1000

        # Test new max update for same location
        self.estimator.update(key=(1, 2), value=2000)
        assert self.estimator.get_max() == 1500
        assert self.estimator.get_min() == 1000

        # Test new min update for same location
        self.estimator.update(key=(1, 2), value=300)
        self.estimator.update(key=(1, 2), value=300)
        assert self.estimator.get_max() == 1500
        assert self.estimator.get_min() == 650

        # Test min update for new location
        self.estimator.update(key=(3, 3), value=50)
        assert self.estimator.get_max() == 1500
        assert self.estimator.get_min() == 50

        # Test max update for new location
        self.estimator.update(key=(4, 4), value=3000)
        assert self.estimator.get_max() == 3000
        assert self.estimator.get_min() == 50

    def test_CheckKey(self) -> None:
        """
        Test check key function. Should return true if key exists and false if key does not
        """
        assert self.estimator.check_key((1, 1)) is False
        self.estimator.update(key=(4, 4), value=3000)
        assert self.estimator.check_key((1, 1)) is False
        assert self.estimator.check_key((4, 4)) is True

    def test_reset(self) -> None:
        """
        Test reset function. Should reset to a new class object
        """
        baseline = IntensityResamplingEstimator()
        assert self.estimator is not baseline

        baseline_list = [a for a in dir(baseline) if not a.startswith("__") and not callable(getattr(baseline, a))]

        # Add values
        self.estimator.update(key=(1, 2), value=300)
        self.estimator.reset()

        for baseline_att, estimator_att in zip(
            baseline_list,
            [a for a in dir(self.estimator) if not a.startswith("__") and not callable(getattr(self.estimator, a))],
        ):
            assert getattr(self.estimator, estimator_att) == getattr(baseline, baseline_att)
