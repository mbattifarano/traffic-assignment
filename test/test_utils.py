from typing import Optional
import time
from hypothesis import given, infer
from traffic_assignment import utils


@given(a=infer, b=infer)
def test_value_or_default(a: Optional[int], b: int):
    actual = utils.value_or_default(a, b)
    if a is None:
        assert actual == b
    else:
        assert actual == a


def test_timer():
    eps = 1e-2
    timer = utils.Timer()
    assert timer.t0 is None
    timer.start()
    time.sleep(0.5)
    assert abs(timer.time_elapsed() - 0.5) <= eps
    time.sleep(0.5)
    assert abs(timer.time_elapsed() - 1.0) <= eps
    timer.start()
    time.sleep(0.5)
    assert abs(timer.time_elapsed() - 0.5) <= eps
