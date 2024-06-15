import pytest

from src.math_utils.standardize import WelfordsOnlineStandardization


class Test_Standardization:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup the estimator"""
        self.stats = WelfordsOnlineStandardization()

    def test_Update(self) -> None:
        """Test the update function. Should update the running statistics correctly"""

        # Invalid reading
        with pytest.raises(AssertionError):
            self.stats.update(reading=-1.0)

        # Set initial mean
        self.stats.update(reading=1000.0)
        assert self.stats.mean == 1000.0
        assert self.stats.count == 1
        assert self.stats._max == 0
        assert self.stats._min == 0

        # Set next parameter that sets new max
        self.stats.update(reading=2000.0)
        assert self.stats.count == 2
        assert self.stats.mean == 1500.0
        assert self.stats.square_dist_mean == 500000
        assert self.stats.sample_variance == 500000
        assert self.stats.std == pytest.approx(707.10678)
        assert self.stats._max == pytest.approx(0.70710678)
        assert self.stats._min == 0

        # Set next parameter that sets new min
        self.stats.update(reading=100.0)
        assert self.stats.count == 3
        assert self.stats.mean == pytest.approx(1033.33333)
        assert self.stats.square_dist_mean == pytest.approx(1806666.66666)
        assert self.stats.sample_variance == pytest.approx(903333.33333)
        assert self.stats.std == pytest.approx(950.43849)
        assert self.stats._max == pytest.approx(0.70710678)
        assert self.stats._min == pytest.approx(-0.9820028733646521)

    def test_Standardize(self) -> None:
        """Test the standardize function. Should standardize with running statistics correctly"""

        # Invalid reading
        with pytest.raises(AssertionError):
            self.stats.standardize(reading=-1.0)

        # Set initial mean
        self.stats.update(reading=1000.0)
        assert self.stats.standardize(1) == -999
        assert self.stats.standardize(1000) == 0
        assert self.stats.standardize(10000) == 9000

        # Set next parameter that sets new max
        self.stats.update(reading=2000.0)
        assert self.stats.standardize(1) == pytest.approx(-2.11990612999)
        assert self.stats.standardize(1000) == pytest.approx(-0.7071067811865475)
        assert self.stats.standardize(10000) == pytest.approx(12.020815280171307)

        # Make sure min and max are not updated during standardize function
        assert self.stats._max == pytest.approx(0.70710678)
        assert self.stats._min == 0

    def test_GetMaxMin(self) -> None:
        """Test the get max and min functions. Should get the correct max/min"""
        # Set initial mean
        self.stats.update(reading=1000.0)
        assert self.stats.get_max() == 0
        assert self.stats.get_min() == 0

        # Set next parameter that sets new max
        self.stats.update(reading=2000.0)
        assert self.stats.get_max() == pytest.approx(0.70710678)
        assert self.stats.get_min() == 0

        # Set next parameter that sets new min
        self.stats.update(reading=100.0)
        assert self.stats.get_max() == pytest.approx(0.70710678)
        assert self.stats.get_min() == pytest.approx(-0.9820028733646521)

    def test_Reset(self) -> None:
        """Test the reset function. Should reset correctly to default"""
        baseline = WelfordsOnlineStandardization()
        baseline_list = [a for a in dir(baseline) if not a.startswith("__") and not callable(getattr(baseline, a))]

        # Add values
        self.stats.update(reading=1000.0)
        self.stats.update(reading=2000.0)
        self.stats.update(reading=100)

        # Set next parameter that sets new min
        self.stats.reset()

        for baseline_att, stats_att in zip(
            baseline_list, [a for a in dir(self.stats) if not a.startswith("__") and not callable(getattr(self.stats, a))]
        ):
            assert getattr(self.stats, stats_att) == getattr(baseline, baseline_att)
