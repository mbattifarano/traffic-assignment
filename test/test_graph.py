from traffic_assignment.network import graph
import numpy as np
from scipy.sparse.csgraph import shortest_path
import time
import igraph as ig
import networkx as nx

from hypothesis import given, infer, settings
from hypothesis.strategies import integers, lists, tuples
from hypothesis.extra.numpy import arrays
from hypothesis import HealthCheck


WEIGHT = 'weight'
LETTERS = 'abcdefghijklmnopqrstuvwxyz'

weighted_digraph = (
    arrays(
        dtype=int,
        shape=integers(3, 10).map(lambda n: (n, n)),
        elements=integers(0, 10),
    ).map(lambda a: a * (1 - np.identity(a.shape[0]))
    ).map(lambda a: nx.from_numpy_array(a, create_using=nx.DiGraph)
    ).map(lambda g: nx.relabel_nodes(g, dict(zip(range(g.number_of_nodes()),
                                                       LETTERS)))
          )
)

LARGE = int(150)

huge_weighted_adjacency = (
    arrays(
        dtype=int,
        shape=(LARGE, LARGE),
        elements=integers(0, 10),
    ).map(lambda a: a * (1 - np.identity(a.shape[0]))
    )
)

@given(g=weighted_digraph)
def test_scipy_shortest_path(g):
    nx_paths = nx.shortest_path(g, weight=WEIGHT)
    idx = list(g.nodes())
    nodes = {n: i for i, n in enumerate(g.nodes())}
    cs_graph = nx.to_scipy_sparse_matrix(g, weight=WEIGHT)
    od_pairs = []
    st_pairs = []
    for o, ds in nx_paths.items():
        for d in ds:
            od_pairs.append((o, d))
            st_pairs.append((nodes[o], nodes[d]))
    sp_paths = graph.shortest_paths_scipy(cs_graph, st_pairs)
    nxs_paths = graph.shortest_paths_nx_via_scipy(g, WEIGHT)
    for o, d in od_pairs:
        if o == d:
            continue
        nxp = nx_paths[o][d]
        nx_cost = sum(g.edges[u, v][WEIGHT] for u, v in zip(nxp, nxp[1:]))
        spp = list(sp_paths[nodes[o], nodes[d]])
        sp_cost = sum(g.edges[idx[i], idx[j]][WEIGHT] for i, j in zip(spp, spp[1:]))
        nxsp = nxs_paths[o][d]
        nxs_cost = sum(g.edges[u, v][WEIGHT] for u, v in zip(nxsp, nxsp[1:]))
        assert nx_cost == sp_cost
        assert nx_cost == nxs_cost


@given(weights=huge_weighted_adjacency)
@settings(suppress_health_check=HealthCheck.all(), deadline=None)
def test_shortest_path(weights):
    WEIGHT_KEY = 'weight'
    idx = weights.nonzero()
    w = weights[idx]
    edges = list(zip(*idx))
    print(f"Creating graphs with {len(edges)} edges")

    gn = nx.from_numpy_array(weights, create_using=nx.DiGraph)
    assert (len(edges) == gn.number_of_edges())
    gi = ig.Graph(edges, edge_attrs={WEIGHT_KEY: w})

    times = []

    t0 = time.time()
    nx_shortest_paths = list(nx.shortest_path(gn, weight=WEIGHT_KEY))
    times.append((time.time() - t0, 'networkx'))

    t1 = time.time()
    for i in range(gi.vcount()):
        ig_shortest_paths = gi.get_shortest_paths(i, weights=WEIGHT_KEY)
    times.append((time.time() - t1, 'igraph'))

    t2 = time.time()
    paths = graph.shortest_paths_nx_via_scipy(gn, WEIGHT_KEY)
    times.append((time.time() - t2, 'scipy'))

    print(" ".join([f"{t:0.4f} ({name})" for t, name in sorted(times)]))

    #graph.profile.print_stats()

    assert True


@given(g=weighted_digraph)
def test_to_igraph(g):
    assert g.is_directed()
    all_paths = nx.shortest_path(g, weight=WEIGHT)
    ig = graph.to_igraph(g, WEIGHT)
    assert ig.is_directed()
    assert ig.is_weighted()
    assert ig.vcount() == g.number_of_nodes()
    assert ig.ecount() == g.number_of_edges()
    for u, v, data in g.edges(data=True):
        assert ig.es[ig.get_eid(str(u), str(v))]['data'] == data
    for s, target_paths in all_paths.items():
        for t, path in target_paths.items():
            ig_path = ig.get_shortest_paths(str(s), str(t), weights=WEIGHT)[0]
            if s == t:
                assert path == [s]
                assert ig_path == []
            else:
                assert ig.vs[ig_path[0]]["name"] == str(s), f"{s}->{t}: {ig_path}"
                assert ig.vs[ig_path[-1]]["name"] == str(t)
                ig_cost = sum(ig.es(ig.get_eids(path=ig_path))[WEIGHT])
                nx_cost = sum(g.edges[e][WEIGHT] for e in zip(path, path[1:]))
                assert ig_cost == nx_cost


@given(g=weighted_digraph)
def test_reindex_graph(g):
    h = graph.reindex_graph(g)
    assert graph.NODE_ID_MAPS in h.graph
    assert h.number_of_nodes() == g.number_of_nodes()
    assert h.number_of_edges() == g.number_of_edges()
    for u, v, data in g.edges(data=True):
        u_new = h.graph[graph.NODE_ID_MAPS][0][u]
        v_new = h.graph[graph.NODE_ID_MAPS][0][v]
        assert h.edges[u_new, v_new] == data


invertible_map = (
    integers(
        min_value=0,
        max_value=20,
    ).flatmap(
        lambda n: tuples(
            lists(integers(), min_size=n, max_size=n, unique=True),
            lists(integers(), min_size=n, max_size=n, unique=True),
        )
    ).map(lambda ts: dict(zip(*ts)))
)


@given(m=invertible_map)
def test_invert_map(m):
    assert m == graph.invert_node_map(graph.invert_node_map(m))


@given(m=invertible_map)
def test_reduce_map(m):
    ms = [m, graph.invert_node_map(m)]
    m0 = graph.reduce_maps(ms)
    assert all(
        k == v
        for k, v in m0.items()
    )
    ms.append(m)
    m1 = graph.reduce_maps(ms)
    assert m1 == m
