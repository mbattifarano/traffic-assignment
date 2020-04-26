import networkx as nx
import igraph as ig
from scipy.sparse import csgraph
from collections import deque, defaultdict
from itertools import product
import numba
import numpy as np


NODE_ID_MAPS = 'node_id_maps'
NX_ID = '_nx_vid'
NAME = 'name'  # name has a special meaning to igraph


def reindex_graph(graph: nx.Graph) -> nx.Graph:
    """Re-index the nodes of a graph to use auto-incrementing integer ids"""
    old = sorted(graph.nodes)
    new = range(graph.number_of_nodes())
    node_id_map = dict(zip(old, new))
    out = nx.relabel_nodes(graph, node_id_map)
    out.graph.setdefault(NODE_ID_MAPS, []).append(node_id_map)
    return out


def is_graph_indexed(graph: nx.Graph) -> bool:
    return list(graph.nodes) == list(range(graph.number_of_nodes()))


def to_igraph(graph: nx.Graph, weight_key: str) -> ig.Graph:
    kwargs = dict(
        directed=graph.is_directed(),
        edge_attrs={weight_key: 0.0},
    )
    out = ig.Graph(**kwargs)
    for k, v in graph.graph.items():
        if isinstance(k, str):
            out[k] = v
    for n, data in sorted(graph.nodes(data=True)):
        data[NX_ID] = n  # preserve the networkx id (before stringifying)
        if NAME in data:
            data['_name'] = data.pop(NAME)
        out.add_vertex(str(n), **data)
    for u, v, data in sorted(graph.edges(data=True)):
        weight = data.get(weight_key, 0.0)
        d = {
            weight_key: weight,
            'data': data
        }
        out.add_edge(str(u), str(v), **d)
    return out


def invert_node_maps(ms):
    return invert_node_map(reduce_maps(ms))


def invert_node_map(m):
    return {
        new: old
        for old, new in m.items()
    }


def reduce_maps(ms):
    it = iter(ms)
    m0 = next(it)
    for m in it:
        m0 = compose_maps(m0, m)
    return m0


def compose_maps(m1, m2):
    return {
        k: m2[v]
        for k, v in m1.items()
    }

from line_profiler import LineProfiler

profile = LineProfiler()


def shortest_paths_nx_via_scipy(graph: nx.DiGraph, weight_key: str, od_pairs=None):
    index_node = np.array(graph.nodes())
    node_index = {n: i for i, n in enumerate(index_node)}
    csg = nx.to_scipy_sparse_matrix(graph, weight=weight_key)
    od_pairs = od_pairs or product(index_node, index_node)
    st_pairs = [(node_index[o], node_index[d]) for o, d in od_pairs]
    return list(shortest_paths_scipy(csg, st_pairs, index_node))


def shortest_paths_igraph(graph: ig.Graph, source, targets, weight_key, names=None):
    paths = {}
    names = names or graph.vs["name"]
    for path in graph.get_shortest_paths(source, targets, weights=weight_key):
        if path:
            p = [names[v] for v in path]
            t = p[-1]
            paths[t] = p
    return paths


#@profile
def shortest_paths_scipy(graph, od_pairs, index_node):
    costs, predescessors = csgraph.shortest_path(graph, directed=True, return_predecessors=True)
    for s, t in od_pairs:
        yield unwind_path(predescessors, s, t, index_node)


_numba_int_tuple_t = numba.typeof((0, 0))
_numba_path_t = numba.core.types.int64[:]


#@profile
def unwind_path_with_cache(predecessors, s, t, cache):
    path = [t]
    prev = t
    while True:
        prev = predecessors[s, prev]
        if prev == -9999:  # there is no path from s to prev
            return None, cache
        rest = cache.get((s, prev), None)
        if rest:
            # if the cache has a path s->prev we are done. return it
            p = rest + path
            cache[(s, t)] = p
            return p, cache
        path = [prev] + path
        # update the cache with the subpath
        cache[(prev, t)] = path
        if prev == s:
            return path, cache




@numba.jit(nopython=True, parallel=True)
def unwind_path_to_link_flow(st_array, volume_array, n_links,
                             link_index, predecessors):
    n = len(volume_array)
    link_flow = np.zeros(n_links)
    for i in numba.prange(n):
        s, t = st_array[i]
        volume = volume_array[i]
        path = unwind_path(predecessors, s, t)
        link_flow += assign_to_links(link_index, n_links, path, volume)
    return link_flow


@numba.jit(nopython=True)
def unwind_path(predecessors, s, t):
    empty = np.empty(0, dtype=np.uint16)
    path = np.array([t])
    prev = t
    while prev != s:
        prev = predecessors[s, prev]
        if prev < 0:
            return empty
        path = np.append(path, prev)
    return np.flip(path)


@numba.jit(nopython=True, parallel=True)
def assign_to_links(link_index, n_links, path, volume):
    link_flow = np.zeros(n_links)
    for ui in numba.prange(len(path) - 1):
        u = path[ui]
        v = path[ui+1]
        link_i = link_index[u, v]
        link_flow[link_i] += volume
    return link_flow
