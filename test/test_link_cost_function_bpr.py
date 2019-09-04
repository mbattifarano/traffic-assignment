from traffic_assignment.link_cost_function.bpr import BPRLinkCostFunction

import numpy as np

from hypothesis import given
from hypothesis.strategies import builds, tuples, floats, integers
from hypothesis.extra.numpy import arrays


number_of_links = integers(min_value=1, max_value=2**16)
non_negative_floats = floats(min_value=0.0, max_value=2**16,
                             allow_infinity=False, allow_nan=False)
positive_floats = floats(min_value=0.1, max_value=2**16,
                         allow_infinity=False, allow_nan=False)


def link_vector_of(shape, elements):
    return arrays(np.dtype('float64'), shape, elements=elements)


link_cost_link_flow_pairs = number_of_links.flatmap(
    lambda n: tuples(
        builds(BPRLinkCostFunction,
               link_vector_of(n, non_negative_floats),
               link_vector_of(n, positive_floats)),
        link_vector_of(n, non_negative_floats)
    )
)


@given(link_cost_link_flow_pairs)
def test_bpr_link_cost_function(link_cost_link_flow_pair):
    bpr, link_flow = link_cost_link_flow_pair
    actual_cost = bpr.link_cost(link_flow)
    assert actual_cost.shape == link_flow.shape
    assert (actual_cost >= 0).all()
    occupancy = link_flow / bpr.capacity
    adjusted_occupancy = 0.15 * (occupancy ** 4)
    expected_cost = bpr.free_flow_travel_time * (1 + adjusted_occupancy)
    assert np.allclose(actual_cost, expected_cost)
