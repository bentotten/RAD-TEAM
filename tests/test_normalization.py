import pytest

from src.math_utils.normalize import BensLogNormalizer, Normalizer


class Test_Normalizers:
    def test_Normalize(self) -> None:
        """Test the normalization function. Should put between range of [0,1]"""
        normalizer = Normalizer()

        # Negative max, 0 max, or max that is smaller than current
        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=1.0, max=0.0)

        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=1.0, max=-1.0)

        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=100, max=1.0)

        # Test negative current with zero max
        assert normalizer.normalize(current_value=-50.0, max=0.0) == 0

        # Min greater than current
        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=10, max=100, min=11)

        # Processing without min, regular, zero, and negative values for current
        assert normalizer.normalize(current_value=50, max=100) == 0.5
        assert normalizer.normalize(current_value=-501.0, max=100) == 0.0
        assert normalizer.normalize(current_value=0, max=100) == 0.0

        # Process with min, regular, zero, and negative values for min
        assert normalizer.normalize(current_value=50, max=100, min=10) == pytest.approx(0.4444444444444444)
        assert normalizer.normalize(current_value=50, max=100, min=-10) == pytest.approx(0.5454545454545454)
        assert normalizer.normalize(current_value=0, max=100, min=-10) == 0.0
        assert normalizer.normalize(current_value=50, max=100, min=0) == pytest.approx(0.5)

    # @pytest.mark.filterwarnings("ignore:mismatch")
    def test_LogNormalize(self) -> None:
        """Test the logrithmic normalization function. Should put between range of [0,1]"""
        normalizer = BensLogNormalizer()

        # Test invalid inputs
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=-1.0, max=10)
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=1.0, max=-10)
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=1.0, max=0)
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=1.0, max=10, step_size=0)
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=1.0, max=10, step_size=-1)

        # Test normal
        assert normalizer.normalize_incremental_logscale(current_value=4.0, max=10, step_size=2) == pytest.approx(0.598104004)
        assert normalizer.normalize_incremental_logscale(current_value=4.0, max=10, step_size=2) == (
            pytest.approx(normalizer.normalize_incremental_logscale(current_value=4.0, max=10))
        )
        # Test Max
        assert normalizer.normalize_incremental_logscale(current_value=18.0, max=10, step_size=2) == 1

        # Test realistic min
        assert normalizer.normalize_incremental_logscale(current_value=1.0, max=10, step_size=2) == pytest.approx(0.366725791)

        # Test assert fail for out of boundaries
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=30.0, max=10, step_size=2)

        # Test warning for change of max or increment value
        with pytest.warns(UserWarning):
            normalizer.normalize_incremental_logscale(current_value=10.0, max=100, step_size=2)

        with pytest.warns(UserWarning):
            normalizer.normalize_incremental_logscale(current_value=10.0, max=10, step_size=10)
