"""Port of the networkx shortest path implementation.

Implement a small change to improve performance by allowing single-source
dijkstra to accept multiple target nodes.
"""
from heapq import heappush, heappop
from itertools import count


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
