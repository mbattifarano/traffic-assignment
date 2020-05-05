"""Port of the networkx shortest path implementation.

Implement a small change to improve performance by allowing single-source
dijkstra to accept multiple target nodes.
"""
from heapq import heappush, heappop
from itertools import count
import numba
from numba.typed import List, Dict
import numpy as np


def _parent_children(graph, node):
    return node, iter(graph[node])


def all_paths_shorter_than(graph, source, target, weight, cutoff):
    visited = [source]
    # Replace _parent_children with list of lists/arrays
    stack = [_parent_children(graph, source)]

    # replace with dict
    def cost_of(node):
        path = visited + [node]
        path_cost = sum(graph.edges[e][weight] for e in zip(path, path[1:]))
        return path_cost
    while stack:
        parent, children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        elif cost_of(child) <= cutoff:
            if child == target:
                yield visited + [target]
            elif child not in visited:
                visited.append(child)
                stack.append(_parent_children(graph, child))

_index_t = numba.uint16


def _igraph_to_numba_adjdict(graph):
    d = numba.typed.Dict.empty(
        key_type=_index_t,
        value_type=numba.types.ListType(_index_t),
    )
    for idx in range(graph.vcount()):
        children = numba.typed.List.empty_list(_index_t)
        for n in graph.neighbors(idx, mode='OUT'):
            children.append(n)
        d[idx] = children
    return d


def _igraph_to_numba_weights(graph, weight_key):
    d = numba.typed.Dict.empty(
        key_type=numba.types.UniTuple(_index_t, 2),
        value_type=numba.float64,
    )
    for e in list(graph.es):
        d[(e.source, e.target)] = e[weight_key]
    return d


@numba.jit(nopython=True)
def _cost_of(weights, visited, node):
    cost = 0.0
    for u, v in zip(visited, visited[1:]):
        cost += weights[(u, v)]
    cost += weights[(visited[-1], node)]
    return cost


@numba.jit(nopython=True)
def _numba_parent_children(adjdict, parent):
    return parent, iter(adjdict[parent])


@numba.jit(nopython=True, nogil=True, debug=True)
def _all_paths_shorter_than(adjdict, source, target, weights, cutoff, index_node,
                            debug=False):
    if debug:
        print(source, target, cutoff)
    visited = List()
    visited_set = set()
    costs = List()
    stack = List()

    visited.append(source)
    visited_set.add(source)
    costs.append(0.0)
    stack.append((source, 0))

    n_pruned = 0
    n_explored = 0

    iteration = 0
    while stack:
        iteration += 1
        parent, child_i = stack.pop()
        children = adjdict[parent]
        if child_i >= len(children):
            # we are done exploring this parent
            n_explored += 1
            visited.pop()
            visited_set.remove(parent)
            costs.pop()
            print("explored node", parent)
            print(iteration, len(visited), len(stack), n_explored, n_pruned)
        else:
            child = children[child_i]
            # put the parent back on the stack and advance the iterator
            stack.append((parent, child_i+1))
            cost_so_far = costs[-1] + weights[(parent, child)]
            if cost_so_far <= cutoff:
                if child == target:
                    out = List()
                    for v in visited:
                        out.append(index_node[v])
                    out.append(index_node[target])
                    print(iteration, "found path")
                    yield out
                elif child not in visited_set:
                    visited.append(child)
                    visited_set.add(child)
                    costs.append(cost_so_far)
                    stack.append((child, 0))
            else:
                n_pruned += 1


#        if children:
#            child = children.pop(0)
#            edge_cost = weights[(parent, child)]
#            cost_so_far = costs[-1] + edge_cost
#            if cost_so_far <= cutoff:
#                if child == target:
#                    out = List()
#                    for v in visited:
#                        out.append(index_node[v])
#                    out.append(index_node[target])
#                    if debug:
#                        print("found a path", source, target, cost_so_far)
#                    yield out
#                elif child not in visited:
#                    visited.append(child)
#                    costs.append(cost_so_far)
#                    stack.append(child)
#                    children_of[child] = adjdict[child].copy()
#        else:  # no more children
#            visited.pop()
#            costs.pop()
#            stack.pop()
#            #if debug:
#            # print("visited", visited)


@numba.jit(nopython=True, nogil=True, parallel=True)
def get_all_shortish_paths(st_pairs, cost_matrix, adjdict, weights, tolerance, index_node):
    out = List()
    is_open = np.zeros(len(st_pairs), dtype=np.uint8)
    for i in numba.prange(len(st_pairs)):
        is_open[i] = 1
        s, t = st_pairs[i]
        min_path_cost = cost_matrix[s, t]
        cutoff = min_path_cost * (1.0 + tolerance)
        paths = _all_paths_shorter_than(
            adjdict,
            s, t,
            weights,
            cutoff,
            index_node,
            False,  # debug if i = 9
        )
        for p in paths:
            out.append((p, i))
        is_open[i] = 0
        open = is_open.nonzero()[0]
        print("end", i, "open", len(open), '\n\t', open)
    return out


def single_source_dijkstra(G, source, targets=None, weight='weight'):
    """Compute shortest paths and lengths in a weighted graph G.

    Uses Dijkstra's algorithm for shortest paths.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    targets : node label, optional
       Ending node for path

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    Returns
    -------
    distance,path : dictionaries
       Returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from the source.
       The second stores the path from the source to that node.


    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length,path=nx.single_source_dijkstra(G,0)
    >>> print(length[4])
    4
    >>> print(length)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    ---------
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path()
    single_source_dijkstra_path_length()
    """
    get_weight = get_weight_factory(weight, G.is_multigraph())
    paths = {source: [source]}  # dictionary of paths
    return _dijkstra(G, source, get_weight, paths=paths,
                     targets=targets)


def get_weight_factory(weight, is_multigraph):
    def get_weight_multigraph(u, v, data):
        return min(eattr.get(weight, 1) for eattr in data.values())

    def get_weight(u, v, data):
        return data.get(weight, 1)

    return get_weight_multigraph if is_multigraph else get_weight


def _dijkstra(G, source, get_weight, paths=None, cutoff=None,
              targets=None):
    """Implementation of Dijkstra's algorithm

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    get_weight: function
        Function for getting edge weight

    paths: dict, optional (default=None)
        Path from the source to a target node.

    targets : list of node labels, optional
       Ending node(s) for path

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    distance,path : dictionaries
       Returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from the source.
       The second stores the path from the source to that node.

    pred,distance : dictionaries
       Returns two dictionaries representing a list of predecessors
       of a node and the distance to each node.

    distance : dictionary
       Dictionary of shortest lengths keyed by target.
    """
    targets = set(targets or G.nodes)
    G_succ = G.succ if G.is_directed() else G.adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {source: 0}
    c = count()
    fringe = []  # use heapq with (distance,label) tuples
    push(fringe, (0, next(c), source))
    # continue while there is a fringe AND there are targets remaining
    while fringe and targets:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        # remove v from target set once it has been found
        if v in targets:
            targets.remove(v)

        for u, e in G_succ[v].items():
            cost = get_weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + get_weight(v, u, e)
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]

    if paths is not None:
        return (dist, paths)
    return dist
