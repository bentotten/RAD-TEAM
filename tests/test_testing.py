import pytest


@pytest.mark.happy
class Test_Happycase:
    def test_typing(self) -> None:
        assert 1 == 1

    def test_failed_test(self) -> None:
        assert 1 == 0
