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
    assert 0 < actual_step <= 1
    # test some known values
    if k == 0:
        assert expected_step == 1.0
    if k == 2:
        assert expected_step == 0.5


@given(integers(min_value=0), integers(min_value=0))
def test_monotone_decreasing_step_size_pairs(j, k):
    j, k = sorted([j, k])
    step = step_size.MonotoneDecreasingStepSize()
    step_j = step.step(j)
    assert 0 < step_j <= 1
    step_k = step.step(k)
    assert 0 < step_k <= 1
    assert step_j >= step_k
    if j != k:
        assert step_j > step_k
