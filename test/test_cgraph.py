from traffic_assignment.cgraph import DiGraph
import numpy as np

import cProfile, pstats


def test_digraph():
    g = DiGraph()
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0

    g.add_nodes(2)
    assert g.number_of_nodes() == 2
    g.add_nodes(3)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 0

    g.add_edge(0, 1)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 1
    assert g.get_eid(0, 1) == 0

    g.add_edge(0, 2)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 2
    assert g.get_eid(0, 1) == 0
    assert g.get_eid(0, 2) == 1

    g.add_edge(1, 2)
    assert g.get_eid(1, 2) == 2

    weights = np.array([1., 3., 1.])
    expected = np.array([0.0, 1.0, 2.0, np.inf, np.inf])
    actual = g.single_source_shortest_path(weights, 0)
    assert np.allclose(expected, actual)

    cProfile.runctx("g.single_source_shortest_path(weights, 0)",
                    globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()


