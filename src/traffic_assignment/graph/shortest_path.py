import heapq
import numpy as np

from numba import jit, prange, int64


@jit(nopython=True)
def single_node_uniform_cost_search(graph, weight, s, t):
    frontier = []
    frontier.append((0.0, s))
    heapq.heapify(frontier)
    explored = set()
    edge_costs = graph.get_edge_feature(weight)

    while frontier:
        cost, node = heapq.heappop(frontier)
        if node == t:
            return cost
        else:
            explored.add(node)
            assert node in graph.edges
            for (v, eid) in graph.edges[node].items():
                if v not in explored:
                    heapq.heappush(frontier, (cost + edge_costs[eid], v))
    return np.inf


@jit(nopython=True, nogil=True, fastmath=True)
def single_source_uniform_cost_search(graph, weight, s):
    frontier = []
    frontier.append((0.0, s))
    heapq.heapify(frontier)
    explored = set()
    edge_costs = graph.get_edge_feature(weight)
    nodes_found = set()

    while frontier:
        cost, node = heapq.heappop(frontier)
        if node not in nodes_found:
            nodes_found.add(node)
            yield node, cost
        explored.add(node)
        for (v, eid) in graph.edges[node].items():
            if v not in explored:
                heapq.heappush(frontier, (cost + edge_costs[eid], v))

# TODO: implement SLF strategy (Bertsekas 1993)


@jit(nopython=True, nogil=True)
def label_correcting_single_source_uniform_cost_search(graph, weight, s):
    edge_costs = graph.get_edge_feature(weight)
    frontier = [s]
    bottom = frontier.copy()  # type the list
    bottom.clear()  # empty it
    d = np.empty(graph.n_nodes)
    d[:] = np.inf
    d[s] = 0.0
    parents = -np.ones(graph.n_nodes)

    while frontier or bottom:
        if not frontier:
            frontier = bottom[::-1]
            bottom.clear()
        parent = frontier.pop()
        for (child, eid) in graph.edges[parent].items():
            cost_via_parent = d[parent] + edge_costs[eid]
            if cost_via_parent < d[child]:
                d[child] = cost_via_parent
                parents[child] = parent
                if frontier and d[child] <= d[frontier[-1]]:
                    frontier.append(child)
                else:
                    bottom.append(child)
                    #frontier.insert(0, child)
    return d


@jit(nopython=True, nogil=True, parallel=True)
def all_pairs_uniform_cost_search(graph, weight):
    n = graph.n_nodes
    d = np.empty((n, n), dtype=np.float64)
    d[:] = np.inf
    for i in prange(n):
        s = int64(i)
        for t, cost in single_source_uniform_cost_search(graph, weight, s):
            d[s, t] = cost
    return d


@jit(nopython=True, nogil=True, parallel=True)
def all_pairs_label_correcting_search(graph, weight):
    n = graph.n_nodes
    d = np.empty((n, n), dtype=np.float64)
    d[:] = np.inf
    for i in prange(n):
        s = int64(i)
        d[s, :] = label_correcting_single_source_uniform_cost_search(graph, weight, s)
    return d

MAX_FLOAT = np.finfo(np.float64).max


@jit(nopython=True, nogil=True, parallel=True)
def floyd_warshall(graph, weight):
    edge_costs = graph.get_edge_feature(weight)
    n = graph.n_nodes
    d = np.empty((n, n), dtype=np.float64)
    d[:] = np.inf
    for u, vs in graph.edges.items():
        d[u, u] = 0.0
        for v, eid in vs.items():
            d[u, v] = edge_costs[eid]
    for s in range(n):
        for t in range(n):
            min_cost_thru_v = (d[s, :] + d[:, t]).min()
            if (min_cost_thru_v < MAX_FLOAT) and (min_cost_thru_v < d[s, t]):
                d[s, t] = min_cost_thru_v
    return d
