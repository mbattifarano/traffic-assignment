from hypothesis import given, example
from hypothesis.strategies import integers

from traffic_assignment.frank_wolfe import step_size


@given(integers(min_value=0))
@example(0)
@example(2)
def test_monotone_decreasing_step_size(k):
    step = step_size.MonotoneDecreasingStepSize()
    actual_step = step.step(k)
    expected_step = 2 / (k+2)
    assert actual_step == expected_step
    # test some known values
    if k == 0:
        assert expected_step == 1.0
    if k == 2:
        assert expected_step == 0.5
