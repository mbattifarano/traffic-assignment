import pytest
import time

import numba
import numpy as np
import cProfile


from traffic_assignment.graph.numba_graph import (
    DiGraph,
    add_edges_randomly,
    InvalidNodeException,
    InvalidEdgeException,
)
from traffic_assignment.graph.shortest_path import (
    single_node_uniform_cost_search,
    single_source_uniform_cost_search,
    label_correcting_single_source_uniform_cost_search,
    all_pairs_label_correcting_search,
    all_pairs_uniform_cost_search,
)
from traffic_assignment.floyd_warshall import floyd_warshall_parallelized
from traffic_assignment.cgraph import DiGraph as cDiGraph


def test_numba_graph():
    g = DiGraph()
    assert g.n_nodes == 0
    assert g.n_edges == 0
    assert g.edges == {}
    assert g.edge_features.shape == (0, 0)

    n1 = g.add_vertices(2)
    n2 = g.add_vertices(3)
    assert n1 == 2
    assert n2 == 5
    assert g.n_nodes == 5
    assert g.n_edges == 0
    assert g.edges == {
        0: {},
        1: {},
        2: {},
        3: {},
        4: {}
    }
    assert g.edge_features.shape == (0, 0)

    eid = g.add_edge(0, 2)
    assert eid == 0
    assert g.n_nodes == 5
    assert g.n_edges == 1
    assert g.edges == {
        0: {2: 0},
        1: {},
        2: {},
        3: {},
        4: {}
    }
    assert g.edge_features.shape == (1, 0)

    with pytest.raises(InvalidEdgeException):
        eid = g.add_edge(0, 2)
    assert g.n_edges == 1

    with pytest.raises(InvalidNodeException):
        eid = g.add_edge(0, 5)
    assert g.n_edges == 1

    with pytest.raises(InvalidNodeException):
        eid = g.add_edge(5, 0)
    assert g.n_edges == 1

    eids = g.add_edges([(0, 1), (1, 2)])
    assert list(eids) == [1, 2]
    assert g.n_nodes == 5
    assert g.n_edges == 3
    assert g.edges == {
        0: {1: 1, 2: 0},
        1: {2: 2},
        2: {},
        3: {},
        4: {},
    }
    assert g.edge_features.shape == (3, 0)

    shape = g.set_edge_feature("weight", 5.0)
    assert shape == (3, 1)
    assert g.edge_features.shape == (3, 1)
    assert np.allclose(g.get_edge_feature("weight"), 5.0 * np.ones(3))

    shape = g.set_edge_feature("length", np.arange(3))
    assert shape == (3, 2)
    assert g.edge_features.shape == (3, 2)
    assert np.allclose(g.get_edge_feature("length"), np.arange(3))

    value = np.array([3., 4., 5.])
    shape = g.set_edge_feature("weight", value)
    assert shape == (3, 2)
    assert g.edge_features.shape == (3, 2)
    assert np.allclose(g.get_edge_feature("weight"), value)


def test_shortest_path_correctness():
    g = DiGraph()
    g.add_vertices(4)
    g.add_edge(0, 1)
    g.add_edge(1, 3)
    g.add_edge(0, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 2)
    edge_weights = (np.arange(g.n_edges) + 1).astype(np.float64)
    g.set_edge_feature("weight", edge_weights)

    cg = cDiGraph()
    cg.add_nodes(4)
    cg.add_edge(0, 1)
    cg.add_edge(1, 3)
    cg.add_edge(0, 2)
    cg.add_edge(2, 3)
    cg.add_edge(1, 2)

    expected_costs = {
        0: 0.,
        1: 1.,
        2: 3.,
        3: 3.,
    }
    c = single_node_uniform_cost_search(g, "weight", 0, 3)
    assert c == 3.0

    actual_costs = dict(single_source_uniform_cost_search(g, "weight", 0))
    assert expected_costs == actual_costs
    actual_costs_lca = label_correcting_single_source_uniform_cost_search(
        g, "weight", 0
    )
    assert np.allclose(actual_costs_lca, np.array([0., 1., 3., 3.]))

    actual_costs_cg = cg.single_source_shortest_path(edge_weights, 0)
    assert np.allclose(actual_costs_cg, np.array([0., 1., 3., 3.]))

    adj = g.to_adjacency_matrix("weight")
    #actual_cost_matrix = floyd_warshall(g, "weight")
    actual_cost_matrix = floyd_warshall_parallelized(adj)
    expected_cost_matrix = np.array([
        [0, 1, 3, 3],
        [np.inf, 0, 5, 2],
        [np.inf, np.inf, 0, 4],
        [np.inf, np.inf, np.inf, 0]
    ]).astype(np.float64)
    assert np.allclose(actual_cost_matrix, expected_cost_matrix)

    actual_cost_matrix_ucs = all_pairs_uniform_cost_search(g, "weight")
    assert np.allclose(actual_cost_matrix_ucs, expected_cost_matrix)

    actual_cost_matrix_lca = all_pairs_label_correcting_search(g, "weight")
    assert np.allclose(actual_cost_matrix_lca, expected_cost_matrix)


def test_single_od_ucs():
    print()
    g = DiGraph()
    cg = cDiGraph()
    _digraph_t = numba.typeof(g)
    s = 0
    t = 9
    t0 = time.time()
    g.add_vertices(2e4)
    cg.add_nodes(2e4)
    print(f"added {g.n_nodes} nodes in {time.time()-t0}s")
    t0 = time.time()
    elist = add_edges_randomly(g, 0.00015)  # avg edge density in PGH == 0.00015
    for u, v in elist:
        cg.add_edge(u, v)
    print(f"added {len(elist)} edges in {time.time()-t0}s")
    weights = np.random.rand(g.n_edges)
    g.set_edge_feature("weight", weights)
    t0 = time.time()
    c_s_t = single_node_uniform_cost_search(g, "weight", s, t)
    print(f"shortest path in {time.time() - t0}")
    assert c_s_t
    #t0 = time.time()
    #_c_s_t = single_node_uniform_cost_search(g, "weight", s, t)
    #print(f"shortest path in {time.time() - t0}")
    #assert _c_s_t == c_s_t
    total = []
    for _ in range(100):
        t0 = time.time()
        costs_lca = dict(single_source_uniform_cost_search(g, "weight", s))
        total.append(time.time() - t0)
        assert costs_lca.get(t, np.inf) == c_s_t
    print(f"found single source via ucs in {np.mean(total)}s +/- {np.std(total)}")

    total = []
    for _ in range(100):
        t0 = time.time()
        costs_lca = label_correcting_single_source_uniform_cost_search(g, "weight", s)
        total.append(time.time() - t0)
        assert costs_lca[t] == c_s_t
    print(f"found single source via lca in {np.mean(total)}s +/- {np.std(total)}")

    costs_lca = cg.single_source_shortest_path(weights, s)

    total = []
    for _ in range(100):
        t0 = time.time()
        costs_lca = cg.single_source_shortest_path(weights, s)
        total.append(time.time() - t0)
    print(f"found single source via lca (cython) in {np.mean(total)}s +/- {np.std(total)}")
    assert costs_lca[t] == c_s_t



    t0 = time.time()
    costs_lca = all_pairs_label_correcting_search(g, "weight")
    print(f"found all pairs via lca in {time.time() - t0}s")
    assert costs_lca[s, t] == c_s_t

    #t0 = time.time()
    #costs_ucs = all_pairs_uniform_cost_search(g, "weight")
    #print(f"found all pairs via ucs in {time.time()-t0}")
    #assert costs_ucs[s, t] == c_s_t



    #adj = g.to_adjacency_matrix("weight")
    #t0 = time.time()
    #cost_matrix_fw = floyd_warshall_parallelized(adj)
    #costs = floyd_warshall(g, "weight")
    #print(f"found all shortest paths via floyd-warshall in {time.time() - t0}s")
    #assert cost_matrix_fw[s, t] == c_s_t



