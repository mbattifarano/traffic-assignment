from traffic_assignment.network.road_network import RoadNetwork
from traffic_assignment.network.demand import Demand
from traffic_assignment.network.path import Path

import numpy as np
import networkx as nx

import pytest
from hypothesis import given
from hypothesis.strategies import (builds, composite, sampled_from, lists,
                                   integers, floats)
from hypothesis.extra.numpy import arrays

paths = builds(Path, lists(integers()))
random_graph = builds(nx.generators.gn_graph, integers(min_value=1,
                                                       max_value=100))
random_network = builds(RoadNetwork, random_graph)


@composite
def random_network_with_demand(draw):
    network = draw(random_network)
    volume = floats(min_value=1.0)
    nodes = sampled_from(network.nodes)
    demand = draw(builds(Demand, nodes, nodes, volume))
    return network, demand


@composite
def random_network_with_demand_and_travel_cost(draw):
    network, demand = draw(random_network_with_demand())
    link_costs = floats(min_value=0, max_value=10000,
                        allow_nan=False, allow_infinity=False)
    travel_cost = draw(arrays(np.dtype('float64'), network.number_of_links(),
                              elements=link_costs))
    return network, demand, travel_cost


@given(paths)
def test_path(path):
    n_nodes = len(path.nodes)
    n_edges = len(path.edges)
    if n_nodes == 0:
        assert n_edges == 0
    else:
        assert n_edges == n_nodes - 1
    for i, (u, v) in enumerate(path.edges):
        assert u == path.nodes[i]
        assert v == path.nodes[i+1]


@given(random_network)
def test_road_network_number_of_links(road_network: RoadNetwork):
    assert (road_network.number_of_links() ==
            road_network.graph.number_of_edges())
    assert (road_network.number_of_nodes() ==
            road_network.graph.number_of_nodes())


@given(random_network)
def test_road_network_links(road_network: RoadNetwork):
    for link in road_network.links:
        u, v = link.edge
        assert road_network.graph.edges[u, v]['link'] is link


def find_next(links, node):
    for link in links:
        if link.origin == node:
            return link


def has_path(network, link_flow, demand):
    link_indices = np.argwhere(link_flow > 0.0).flatten()
    links = set(network.links[i] for i in link_indices)
    node = demand.origin
    while links and node != demand.destination:
        link = find_next(links, node)
        links.remove(link)
        node = link.destination
    return node == demand.destination


@given(random_network_with_demand_and_travel_cost())
def test_shortest_path_assignment(data):
    network, demand, travel_cost = data
    try:
        path = nx.shortest_path(network.graph,
                                demand.origin.name, demand.destination.name)
    except nx.NetworkXNoPath:
        path = None

    if path is not None:  # if the O-D is connected
        link_flow = network.shortest_path_assignment(demand, travel_cost)
        # assert that the weights have been assigned to each network edge
        for i, cost in enumerate(travel_cost):
            assert network.graph.edges[network.links[i].edge]['weight'] == cost
        if network.number_of_links() == 0:
            # if there are no links, the link flow vector should be empty
            assert len(link_flow) == 0
        elif demand.origin is demand.destination:
            # if the origin and destination are the same, there is no link flow
            assert set(link_flow) == {0}
        elif len(path) - 1 == network.number_of_links():
            # if the path is the entire network, then all links should have
            # the same flow
            assert set(link_flow) == {demand.volume}
            assert has_path(network, link_flow, demand)
        else:
            # the normal scenario: each link has 0 flow or flow equal to volume
            assert set(link_flow) == {0, demand.volume}
            assert has_path(network, link_flow, demand)
    else:
        # If there is no path, an exception will be raised.
        with pytest.raises(nx.NetworkXNoPath):
            network.shortest_path_assignment(demand, travel_cost)


def test_braess_network_edges(braess_network):
    edges = [(l.origin.name, l.destination.name) for l in braess_network.links]
    expected_edges = [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
        (2, 1),
    ]
    assert edges == expected_edges


def test_path_incidences(braess_network, braess_demand_augmented):
    link_path, path_od, path_index = braess_network.path_incidences(braess_demand_augmented)
    assert len(path_index) == 5
    assert link_path.shape == (5, 5)
    assert path_od.shape == (5, 2)
    link_index = {(link.origin.name, link.destination.name): i
                  for i, link in enumerate(braess_network.links)}
    od_index = {(d.origin.name, d.destination.name): i
                for i, d in enumerate(braess_demand_augmented.demand)}
    for path, i in path_index.items():
        for (u, v) in path.edges:
            assert link_path[link_index[u, v], i] == 1
        orgn = path.nodes[0]
        dest = path.nodes[-1]
        assert path_od[i, od_index[(orgn, dest)]] == 1
