import numpy as np

from numba import njit, types, typed
from numba.experimental import jitclass

_node_t = types.int64
_edge_t = types.int64

_edge_dict_t = types.DictType(
    _node_t,
    types.DictType(
        _node_t,
        _edge_t
    ),
)

_feature_index_t = types.DictType(
    types.unicode_type,
    types.int64
)

_graph_t = [
    ('n_nodes', _node_t),
    ('n_edges', types.int64),
    ('edges', _edge_dict_t),
    ('edge_features', types.float64[:, :]),
    ('feature_index', _feature_index_t)
]


class NumbaGraphException(Exception):
    pass


class InvalidNodeException(NumbaGraphException):
    pass


class InvalidEdgeException(NumbaGraphException):
    pass


@jitclass(_graph_t)
class DiGraph:
    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        # initialize the adjacency dict so numba can infer the type
        self.edges = {0: {0: 0}}
        self.edges.clear()
        self.edge_features = np.zeros((0, 0), dtype=np.float64)
        self.feature_index = {'': 0}
        self.feature_index.clear()

    def add_vertices(self, n):
        self.n_nodes += n
        new = {0: 0}
        new.clear()
        for i in range(self.n_nodes):
            if i not in self.edges:
                self.edges[i] = new.copy()
        return self.n_nodes

    def _is_valid_node(self, u):
        return 0 <= u < self.n_nodes

    def add_edge(self, u, v):
        if not (self._is_valid_node(u) and self._is_valid_node(v)):
            raise InvalidNodeException("Invalid node id")
        e_id = self.n_edges
        if u not in self.edges:
            self.edges[u] = {v: e_id}
        else:
            if v in self.edges[u]:
                raise InvalidEdgeException("Edge already exists")
            self.edges[u][v] = e_id
        self.n_edges += 1
        _, d = self.edge_features.shape
        self.edge_features = np.append(
            self.edge_features,
            np.zeros((1, d), dtype=np.float64),
            0
        )
        return e_id

    def add_edges(self, v_pairs):
        e_ids = typed.List()
        for u, v in v_pairs:
            e_id = self.add_edge(u, v)
            e_ids.append(e_id)
        return e_ids

    def set_edge_feature(self, name, value):
        if name not in self.feature_index:
            self._append_edge_feature(name, value)
        else:
            self.edge_features[:, self.feature_index[name]] = value
        return self.edge_features.shape

    def get_edge_feature(self, name):
        return self.edge_features[:, self.feature_index[name]]

    def _append_edge_feature(self, name, value):
        n, d = self.edge_features.shape
        self.feature_index[name] = d
        self.edge_features = np.append(
            self.edge_features,
            np.zeros((n, 1), dtype=np.float64),
            1
        )
        self.edge_features[:, d] = value
        return self.edge_features.shape

    def to_adjacency_matrix(self, weights):
        a = np.inf + np.empty((self.n_nodes, self.n_nodes))
        w = self.get_edge_feature(weights)
        for u, vs in self.edges.items():
            a[u, u] = 0.0
            for v, eid in vs.items():
                a[u, v] = w[eid]
        return a

_di_graph_t = DiGraph.class_type.instance_type

@njit
def add_edges_randomly(graph, p):
    n = graph.n_nodes
    us, vs = (np.random.random((n, n)) < p).nonzero()
    elist = list(zip(us, vs))
    for u, v in zip(us, vs):
        if u != v:
            graph.add_edge(u, v)
    return elist
