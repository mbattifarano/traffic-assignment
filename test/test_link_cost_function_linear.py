from traffic_assignment.link_cost_function.linear import LinearLinkCostFunction

import numpy as np

from hypothesis import given
from hypothesis.strategies import integers, floats, tuples, builds
from hypothesis.extra.numpy import arrays

non_negatives = floats(min_value=0.0, max_value=2**16, allow_infinity=False)


def non_negative_vectors(n):
    return arrays(np.dtype('float64'), n, non_negatives)


number_of_links = integers(min_value=0, max_value=2**16)

link_cost_link_flow_pairs = number_of_links.flatmap(
    lambda n: tuples(
        builds(LinearLinkCostFunction,
               non_negative_vectors(n),
               non_negative_vectors(n)),
        non_negative_vectors(n)
    )
)


@given(link_cost_link_flow_pairs)
def test_linear_link_cost_function(link_cost_link_flow_pair):
    cost_fn, link_flow = link_cost_link_flow_pair
    actual_cost = cost_fn.link_cost(link_flow)
    expected_cost = cost_fn.weights * link_flow + cost_fn.biases
    assert actual_cost.shape == link_flow.shape
    assert np.allclose(actual_cost, expected_cost)
