import pytest


@pytest.mark.happy
class Test_Happycase:
    def test_typing(self) -> None:
        assert 1 == 1
