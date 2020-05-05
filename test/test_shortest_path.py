from hypothesis import given, settings, note
from hypothesis.strategies import builds, integers, lists, composite
import networkx as nx
import numpy as np

from traffic_assignment.network.shortest_path import all_paths_shorter_than, _all_paths_shorter_than, _igraph_to_numba_adjdict, _igraph_to_numba_weights, get_all_shortish_paths
import igraph as ig
import pytest
from traffic_assignment.utils import Timer

WEIGHT_KEY = 'weight'


def _generate_path(n, k):
    """Generate a path of length k over n nodes"""
    return np.random.permutation(n)[:k]


def _edges(path):
    return zip(path, path[1:])


def _to_graph(paths):
    g = nx.DiGraph()
    for path in paths:
        g.add_edges_from(_edges(path))
    return g


def _add_weights(draw, graph, weight=integers(min_value=1, max_value=10)):
    for e in graph.edges():
        graph.edges[e][WEIGHT_KEY] = draw(weight)


number_of_nodes = integers(min_value=4, max_value=50)


paths = integers(min_value=4, max_value=20).flatmap(
    lambda n: lists(integers(min_value=2, max_value=n)
                    .map(lambda k: _generate_path(n, k)),
                    min_size=1)
)


@composite
def random_graph(draw):
    graph = draw(paths.map(_to_graph))
    _add_weights(draw, graph)
    return graph


@given(random_graph())
@settings(deadline=None)
def test_shortest_paths(graph):
    assert 2 <= graph.number_of_nodes() <= 100
    assert 1 <= graph.number_of_edges()
    least_cost_paths = nx.shortest_path_length(graph, weight=WEIGHT_KEY)
    od_costs = (
        (source, target, cost)
        for source, targets in least_cost_paths
        for target, cost in targets.items()
        if source != target
    )
    n_ods = 0
    for source, target, cost in od_costs:
        n_ods += 1
        for tolerance in [0, 0.5]:
            cutoff = cost * (1 + tolerance)
            assert cutoff >= cost
            shortish_paths = list(all_paths_shorter_than(
                graph,
                source,
                target,
                WEIGHT_KEY,
                cutoff
            ))
            note(f"G = {graph.edges(data=True)}")
            note(f"{source}->{target}; cost = {cost}; cutoff = {cutoff}; n paths = {len(shortish_paths)}")
            assert shortish_paths
            for path in shortish_paths:
                path_cost = sum(graph.edges[e][WEIGHT_KEY] for e in _edges(path))
                assert path_cost <= cutoff
    assert n_ods >= 1


def display_path(graph, path, weight_key):
    edge_format = "-({})->{}"
    path_strings = [f"{path[0]}"]
    for u, v in _edges(path):
        w = graph.edges[u,v][weight_key]
        path_strings.append(edge_format.format(w, v))
    return ''.join(path_strings)


def test_igraph_shortest_paths_less_than():
    g = ig.Graph(directed=True)
    g.add_vertices(4)
    #g.add_edges([
    #    (0, 1),
    #    (0, 2),
    #    (1, 3),
    #    (2, 3),
    #    (1, 2),
    #])
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(0, 2, weight=3.0)
    g.add_edge(1, 3, weight=1.0)
    g.add_edge(2, 3, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    s = 0
    t = 3
    st_pairs = np.array([[s, t]])
    weight_key = 'weight'
    index_node = np.arange(4)
    cost_matrix = np.array(g.shortest_paths(weights=weight_key))
    #g.es[weight_key] = [1.0, 3.0, 1.0, 1.0, 1.0]
    adjdict = _igraph_to_numba_adjdict(g)
    weights = _igraph_to_numba_weights(g, weight_key)
    cost = g.shortest_paths(s, t, weights=weight_key)[0][0]
    assert cost == cost_matrix[s, t]
    path = g.get_shortest_paths(s, t, weights=weight_key)[0]
    assert cost == pytest.approx(2.0)
    assert path == [0, 1, 3]
    timer = Timer().start()
    paths = [tuple(p)
             for p in _all_paths_shorter_than(adjdict, s, t, weights, 2.0, index_node)]
    print(timer.time_elapsed())
    timer.start()
    _paths = [tuple(p)
              for p, i in get_all_shortish_paths(st_pairs, cost_matrix, adjdict,
                                                 weights, 0.0, index_node)]
    print(timer.time_elapsed())
    assert len(paths) == 1
    assert (0, 1, 3) in paths
    assert paths == _paths

    timer.start()
    paths = [tuple(p)
             for p in _all_paths_shorter_than(adjdict, s, t, weights, 3.0, index_node)]
    print(timer.time_elapsed())
    timer.start()
    _paths = [tuple(p)
              for p, i in get_all_shortish_paths(st_pairs, cost_matrix, adjdict,
                                                 weights, 0.5, index_node)]
    print(timer.time_elapsed())
    assert len(paths) == 2
    assert (0, 1, 3) in paths
    assert (0, 1, 2, 3) in paths
    assert paths == _paths

    timer.start()
    paths = [tuple(p)
             for p in _all_paths_shorter_than(adjdict, s, t, weights, 4.0, index_node)]
    print(timer.time_elapsed())
    timer.start()
    _paths = [tuple(p)
              for p, i in get_all_shortish_paths(st_pairs, cost_matrix, adjdict,
                                                 weights, 1.0, index_node)]

    print(timer.time_elapsed())
    assert len(paths) == 3
    assert (0, 1, 3) in paths
    assert (0, 1, 2, 3) in paths
    assert (0, 2, 3) in paths
    assert paths == _paths

