import networkx as nx
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (composite, sampled_from, builds, lists,
                                   integers)
from traffic_assignment.frank_wolfe.search_direction import (
    ShortestPathSearchDirection)
from traffic_assignment.network.road_network import RoadNetwork
from traffic_assignment.network.demand import Demand, TravelDemand

# Use integers for all numeric types to avoid rouding errors in comparisons
non_negatives = integers(min_value=0, max_value=2**8)
positives = integers(min_value=1, max_value=2**8)

random_graph = builds(nx.generators.gn_graph,
                      integers(min_value=10, max_value=1000)
                      ).map(lambda g: g.to_undirected().to_directed())
random_network = builds(RoadNetwork, random_graph)


def link_vector_of(shape, elements):
    return arrays(np.dtype('int64'), shape, elements=elements)


def demand_on(nodes):
    return builds(Demand, nodes, nodes, non_negatives)


@composite
def shortest_path_search_directions(draw):
    network = draw(random_network)
    nodes = sampled_from(network.nodes)
    demand = draw(
        builds(
            TravelDemand,
            lists(
                demand_on(nodes).filter(
                    lambda d: network.has_path(d.origin, d.destination)
                ),
                min_size=1,
                max_size=10,
            )
        )
    )
    return ShortestPathSearchDirection(network, demand)


@composite
def search_link_flow_pairs(draw):
    search = draw(shortest_path_search_directions())
    link_flow = draw(link_vector_of(search.network.number_of_links(),
                                    non_negatives))
    travel_cost = draw(link_vector_of(search.network.number_of_links(),
                                      non_negatives))
    return search, travel_cost, link_flow


@given(search_link_flow_pairs())
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_shortest_path_search_direction(data):
    search, cost, x = data
    actual_search_direction, used_paths = search.search_direction(cost, x)
    network = search.network
    # TODO: test used path set

    assert actual_search_direction.shape == x.shape
    # actual_search_direction = y - x, recover y, the shortest path assignment
    y = actual_search_direction + x
    assert y.min() >= 0
    total_demand = search.demand.to_array().sum()
    assert y.max() <= total_demand
