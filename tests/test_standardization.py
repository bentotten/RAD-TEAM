import pytest

from src.utils.standardize import WelfordsOnlineStandardization


class Test_Standardization:
    def test_Update(self) -> None:
        """Test the update function. Should update the running statistics correctly"""
        stats = WelfordsOnlineStandardization()

        # Invalid reading
        with pytest.raises(AssertionError):
            stats.update(reading=-1.0)

        # Set initial mean
        stats.update(reading=1000.0)
        assert stats.mean == 1000.0
        assert stats.count == 1
        assert stats._max == 0
        assert stats._min == 0

        # Set next parameter that sets new max
        stats.update(reading=2000.0)
        assert stats.count == 2
        assert stats.mean == 1500.0
        assert stats.square_dist_mean == 500000
        assert stats.sample_variance == 500000
        assert stats.std == pytest.approx(707.10678)
        assert stats._max == pytest.approx(0.70710678)
        assert stats._min == 0

        # Set next parameter that sets new min
        stats.update(reading=100.0)
        assert stats.count == 3
        assert stats.mean == pytest.approx(1033.33333)
        assert stats.square_dist_mean == pytest.approx(1806666.66666)
        assert stats.sample_variance == pytest.approx(903333.33333)
        assert stats.std == pytest.approx(950.43849)
        assert stats._max == pytest.approx(0.70710678)
        assert stats._min == pytest.approx(-0.9820028733646521)

    def test_Standardize(self) -> None:
        """Test the standardize function. Should standardize with running statistics correctly"""
        stats = WelfordsOnlineStandardization()

        # Invalid reading
        with pytest.raises(AssertionError):
            stats.standardize(reading=-1.0)

        # Set initial mean
        stats.update(reading=1000.0)
        assert stats.standardize(1) == -999
        assert stats.standardize(1000) == 0
        assert stats.standardize(10000) == 9000

        # Set next parameter that sets new max
        stats.update(reading=2000.0)
        assert stats.standardize(1) == pytest.approx(-2.11990612999)
        assert stats.standardize(1000) == pytest.approx(-0.7071067811865475)
        assert stats.standardize(10000) == pytest.approx(12.020815280171307)

        # Make sure min and max are not updated during standardize function
        assert stats._max == pytest.approx(0.70710678)
        assert stats._min == 0

    def test_GetMaxMin(self) -> None:
        """Test the get max and min functions. Should get the correct max/min"""
        stats = WelfordsOnlineStandardization()
        # Set initial mean
        stats.update(reading=1000.0)
        assert stats.get_max() == 0
        assert stats.get_min() == 0

        # Set next parameter that sets new max
        stats.update(reading=2000.0)
        assert stats.get_max() == pytest.approx(0.70710678)
        assert stats.get_min() == 0

        # Set next parameter that sets new min
        stats.update(reading=100.0)
        assert stats.get_max() == pytest.approx(0.70710678)
        assert stats.get_min() == pytest.approx(-0.9820028733646521)

    def test_Reset(self) -> None:
        """Test the reset function. Should reset correctly to default"""
        stats = WelfordsOnlineStandardization()
        baseline = WelfordsOnlineStandardization()
        baseline_list = [a for a in dir(baseline) if not a.startswith("__") and not callable(getattr(baseline, a))]

        # Add values
        stats.update(reading=1000.0)
        stats.update(reading=2000.0)
        stats.update(reading=100)

        # Set next parameter that sets new min
        stats.reset()

        for baseline_att, stats_att in zip(
            baseline_list, [a for a in dir(stats) if not a.startswith("__") and not callable(getattr(stats, a))]
        ):
            assert getattr(stats, stats_att) == getattr(baseline, baseline_att)
