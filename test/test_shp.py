from traffic_assignment.mac_shp import network as net, features
from traffic_assignment.mac_shp.link import INF_CAPACITY
from traffic_assignment.network.graph import is_graph_indexed
from traffic_assignment.network.road_network import RoadNetwork
from traffic_assignment.link_cost_function.bpr import (BPRLinkCostFunction, BPRMarginalLinkCostFunction)

import numpy as np


def test_graph_from_shp(pittsburgh_shp):
    graph = net.graph_from_shp(pittsburgh_shp)
    n_nodes = 9299
    n_links = 12991  # this counts two ways as a single link
    assert graph.number_of_nodes() == n_nodes
    assert n_links <= graph.number_of_edges() <= 2 * n_links
    for u, v, data in graph.edges(data=True):
        assert set(data.keys()) == {
            'id',
            'length',
            'speed',
            'capacity',
            'zone',
            'lanes',
            'from_point',
            'to_point',
            'geometry',
            'is_link',
            'is_virtual',
            'free_flow_travel_time',
        }
    assert is_graph_indexed(graph)


def test_road_network_from_shp(pittsburgh_graph):
    graph = pittsburgh_graph
    n_nodes = graph.number_of_nodes()
    n_links = graph.number_of_edges()
    network = RoadNetwork(graph)
    assert network.number_of_nodes() == n_nodes
    assert network.number_of_links() == n_links
    for graph_node, network_node in zip(sorted(graph.nodes), network.nodes):
        assert network_node.name == graph_node
        assert network_node.id == network_node.name  # the nodes should be ordered/labeled correctly.
    for edge, link in zip(net.edges(graph), network.links):
        assert edge == (link.origin.name, link.destination.name)



def test_capacity_array(pittsburgh_graph):
    capacity = net.to_capacity(pittsburgh_graph)
    assert len(capacity) == pittsburgh_graph.number_of_edges()
    assert (capacity > 0).all()
    assert (capacity <= INF_CAPACITY).all()


def test_free_flow_array(pittsburgh_graph):
    free_flow = net.to_free_flow_travel_time(pittsburgh_graph)
    assert len(free_flow) == pittsburgh_graph.number_of_edges()
    assert (free_flow >= 0).all()


def test_bpr(pittsburgh_graph):
    capacity = net.to_capacity(pittsburgh_graph)
    assert not np.isnan(capacity).any()
    assert (capacity >= 0).all()
    assert np.isfinite(capacity).all()
    free_flow = net.to_free_flow_travel_time(pittsburgh_graph)
    t = BPRLinkCostFunction(
        capacity=capacity,
        free_flow_travel_time=free_flow,
    )
    cost = t.link_cost(0.0)
    idx, = np.isnan(cost).nonzero()
    for i in idx:
        assert free_flow[i] * (1.0 + t.alpha * (0.0 / capacity[i])**t.beta) == cost[i]
    assert not np.isnan(cost).any()
    assert (cost >= 0).all()
    assert np.isfinite(cost).all()
    mt = BPRMarginalLinkCostFunction(
        capacity=capacity,
        free_flow_travel_time=free_flow,
    )
    mcost = mt.link_cost(0.0)
    assert not np.isnan(mcost).any()
    assert (mcost >= 0).all()
    assert np.isfinite(mcost).all()

