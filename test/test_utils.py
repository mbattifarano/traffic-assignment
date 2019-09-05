from typing import Optional

from hypothesis import given, infer
from traffic_assignment import utils


@given(a=infer, b=infer)
def test_value_or_default(a: Optional[int], b: int):
    actual = utils.value_or_default(a, b)
    if a is None:
        assert actual == b
    else:
        assert actual == a
