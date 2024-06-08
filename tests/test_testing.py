import pytest


@pytest.mark.happy
class Test_Happycase:
    def missing_type_defs(self, test) -> int:
        return test

    def missing_return_defs(self, test: int):
        return test

    def fetch_an_int(self, im_an_int: int) -> int:
        return im_an_int

    def test_typing(self) -> None:
        im_an_int = self.fetch_an_int("clearly not an int")
        assert im_an_int == im_an_int

    def test_failed_test(self) -> None:
        assert 1 == 0