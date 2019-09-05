import networkx as nx
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (composite, sampled_from, builds, lists,
                                   floats, integers)
from traffic_assignment.frank_wolfe.search_direction import (
    ShortestPathSearchDirection)
from traffic_assignment.link_cost_function.bpr import BPRLinkCostFunction
from traffic_assignment.network.road_network import RoadNetwork, Demand

non_negative_floats = floats(min_value=0.0, max_value=2**16,
                             allow_infinity=False, allow_nan=False)
positive_floats = floats(min_value=0.1, max_value=2**16,
                         allow_infinity=False, allow_nan=False)

random_graph = builds(nx.generators.gn_graph,
                      integers(min_value=10, max_value=1000)
                      ).map(lambda g: g.to_undirected().to_directed())
random_network = builds(RoadNetwork, random_graph)


def link_vector_of(shape, elements):
    return arrays(np.dtype('float64'), shape, elements=elements)


def bpr_link_cost_function_of(shape):
    return builds(
        BPRLinkCostFunction,
        link_vector_of(shape, non_negative_floats),
        link_vector_of(shape, positive_floats)
    )


def demand_on(nodes):
    return builds(Demand, nodes, nodes, non_negative_floats)


@composite
def shortest_path_search_directions(draw):
    network = draw(random_network)
    n = network.number_of_links()
    nodes = sampled_from(network.nodes)
    travel_cost = draw(link_vector_of(n, non_negative_floats))
    demand = draw(lists(
        demand_on(nodes).filter(
            lambda d: network.has_path(d.origin, d.destination)
        ),
        min_size=1,
        max_size=10,
    ))
    return ShortestPathSearchDirection(network, travel_cost, demand)


@composite
def search_link_flow_pairs(draw):
    search = draw(shortest_path_search_directions())
    link_flow = draw(link_vector_of(search.network.number_of_links(),
                                    non_negative_floats))
    return search, link_flow


@given(search_link_flow_pairs())
def test_shortest_path_search_direction(data):
    search, x = data
    actual_search_direction = search.search_direction(x)
    assert actual_search_direction.shape == x.shape
    # actual_search_direction = y - x, recover y, the shortest path assignment
    y = actual_search_direction + x
    assert (y >= 0).all()
    total_demand = sum(d.volume for d in search.demand)
    assert (y <= total_demand).all()
