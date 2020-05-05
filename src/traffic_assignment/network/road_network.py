from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from collections import defaultdict
from typing import Iterable, NamedTuple, Set, Tuple, Optional, List
from toolz import memoize

import networkx as nx
import time
import numpy as np
from scipy.sparse import dok_matrix as sparse_matrix

import igraph as ig

from .demand import TravelDemand
from .link import Link
from .node import Node
from .path import Path
from .shortest_path import single_source_dijkstra, all_paths_shorter_than, get_all_shortish_paths, _igraph_to_numba_weights, _igraph_to_numba_adjdict
from .graph import shortest_paths_igraph, NX_ID, assign_all_to_links
import numba


class Network(ABC):

    @abstractmethod
    def number_of_nodes(self) -> int:
        pass

    @abstractmethod
    def number_of_links(self) -> int:
        pass

    @abstractmethod
    def number_of_paths(self, demand: TravelDemand) -> int:
        pass

    @abstractmethod
    def path_incidences(self, demand: TravelDemand):
        pass

    @abstractmethod
    def path_set_incidences(self, demand: TravelDemand, path_set: Iterable[Path]):
        pass

    @abstractmethod
    def least_cost_paths(self, demand: TravelDemand,
                         travel_costs: np.ndarray,
                         tolerance: float) -> Iterable[Tuple[Path, int]]:
        pass

    @abstractmethod
    def least_cost_path_indices(self, demand: TravelDemand,
                                travel_costs: np.ndarray):
        pass

    @abstractmethod
    def get_node(self, name) -> Node:
        pass

    @abstractmethod
    def has_path(self, origin: Node, destination: Node) -> bool:
        pass

    @abstractmethod
    def shortest_path_assignment(self, demand: TravelDemand,
                                 travel_costs: np.ndarray) -> PathAssignment:
        pass

    @abstractmethod
    def set_link_costs(self, travel_costs: np.ndarray) -> ():
        pass


def path_incidences_key(args, kwargs):
    road_network, travel_demand = args
    return str(hash((
        tuple(road_network.nodes),
        tuple(road_network.links),
        travel_demand,
    )))


class PathAssignment(NamedTuple):
    link_flow: np.ndarray
    used_paths: frozenset


def _to_igraph(nodes: List[Node], links: List[Link]):
    N = len(nodes)
    g = ig.Graph(directed=True)
    g['index_node'] = np.array([n.name for n in nodes], dtype=np.uint16)
    g['node_index'] = {n: i for i, n in enumerate(g['index_node'])}
    link_matrix = -np.ones((N, N), dtype=np.uint16)
    g.add_vertices(N)
    for link in links:
        i = link.origin.id
        j = link.destination.id
        g.add_edges([(i, j)])
        link_matrix[i, j] = link.id
    g['link_matrix'] = link_matrix
    return g


class RoadNetwork(Network):
    NODE_KEY = 'node'
    LINK_KEY = 'link'
    WEIGHT_KEY = 'weight'

    def __init__(self, graph: nx.DiGraph):
        self.graph = nx.freeze(graph)
        self.nodes = list(self._build_nodes())
        self.links = list(self._build_links())
        self._link_index = {
            link: i
            for i, link in enumerate(self.links)
        }
        self._igraph = _to_igraph(self.nodes, self.links)

    def number_of_links(self) -> int:
        return len(self.links)

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    @memoize
    def get_all_paths(self, demand: TravelDemand):
        all_paths = {}
        for orgn, dest, _ in demand.demand:
            all_paths[(orgn, dest)] = list(
                self._get_all_paths_between(orgn, dest)
            )
        return all_paths

    def number_of_paths(self, demand: TravelDemand) -> int:
        all_paths = self.get_all_paths(demand)
        return sum(map(len, all_paths.values()))

    def path_incidences(self, demand: TravelDemand):
        """Create link-path and path-od incidence matrices"""
        all_paths = self.get_all_paths(demand)
        n_paths = self.number_of_paths(demand)
        link_path = np.zeros((self.number_of_links(), n_paths),
                             dtype=np.uint8)
        path_od = np.zeros((n_paths, demand.number_of_od_pairs),
                           dtype=np.uint8)
        link_index = self._link_index
        od_index = {(d.origin, d.destination): i
                    for i, d in enumerate(demand.demand)}
        path_counter = count()
        path_index = {}
        for (o, d), paths in all_paths.items():
            for path in paths:
                i = next(path_counter)
                path_index[path] = i
                j = od_index[(o, d)]
                path_od[i, j] = 1
                for (u, v) in path.edges:
                    link = self._get_link(u, v)
                    k = link_index[link]
                    link_path[k, i] = 1
        return link_path, path_od, path_index

    def path_set_incidences(self, demand: TravelDemand, path_set: Iterable[Path]):
        n_paths = len(path_set)
        n_links = self.number_of_links()
        link_index = self._link_index
        od_index = {(d.origin.name, d.destination.name): i
                    for i, d in enumerate(demand.demand)}
        link_path = sparse_matrix((n_links, n_paths))
        path_od = sparse_matrix((n_paths, len(od_index)))
        for j, p in enumerate(path_set):
            k = od_index[(p.origin, p.destination)]
            path_od[j, k] = 1
            for (u, v) in p.edges:
                i = link_index[self._get_link(u, v)]
                link_path[i, j] = 1
        return link_path, path_od

    def least_cost_paths(self, demand: TravelDemand,
                         travel_costs: np.ndarray,
                         tolerance: float = 1e-3) -> Iterable[Tuple[Path, int]]:
        #self.set_link_costs(travel_costs)
        self._igraph.es[self.WEIGHT_KEY] = travel_costs
        _index_node = self._igraph['index_node']
        _node_index = self._igraph['node_index']
        # high cost step
        cost_matrix = np.array(self._igraph.shortest_paths(weights=self.WEIGHT_KEY))
        st_pairs = np.array([
            [_node_index[d.origin.name], _node_index[d.destination.name]]
            for d in demand
        ], dtype=np.uint16)
        adjdict = _igraph_to_numba_adjdict(self._igraph)
        weights = _igraph_to_numba_weights(self._igraph, self.WEIGHT_KEY)
        for path, i in get_all_shortish_paths(st_pairs, cost_matrix, adjdict,
                                              weights, tolerance, _index_node):
            yield Path(path), i

        #for i, (orgn, dest, _) in enumerate(demand):
        #    #min_path_cost = nx.shortest_path_length(
        #    #    self.graph,
        #    #    source=orgn.name,
        #    #    target=dest.name,
        #     #    weight=self.WEIGHT_KEY,
        #     #)
        #     s = _index_node[orgn.name]
        #     t = _index_node[dest.name]
        #     min_path_cost = cost_matrix[s, t]
        #     shortest_paths = all_paths_shorter_than(
        #         self.graph,
        #         source=orgn.name,
        #         target=dest.name,
        #         weight=self.WEIGHT_KEY,
        #         cutoff=min_path_cost * (1 + tolerance),
        #     )
        #     for path in shortest_paths:
        #         yield Path(path), i

    def least_cost_path_indices(self, demand: TravelDemand,
                                travel_costs: np.ndarray):
        self.set_link_costs(travel_costs)
        n_links = self.number_of_links()
        n_ods = len(demand)
        n_paths_guess = n_links
        link_path_incidence = sparse_matrix((n_links, n_paths_guess))
        trip_path_incidence = sparse_matrix((n_ods, n_paths_guess))
        n_paths = 0
        for path, trip_index in self.least_cost_path_indices(demand, travel_costs):
            if n_paths >= n_paths_guess:
                n_paths_guess *= 2
                link_path_incidence.resize(n_links, n_paths_guess)
                trip_path_incidence.resize(n_ods, n_paths_guess)
            trip_path_incidence[trip_index, n_paths] = 1
            for u, v in path.edges:
                link = self._get_link(u, v)
                k = self._link_index[link]
                link_path_incidence[k, n_paths] = 1
            n_paths += 1
        link_path_incidence.resize(n_links, n_paths)
        trip_path_incidence.resize(n_ods, n_paths)
        return link_path_incidence, trip_path_incidence

    def _get_all_paths_between(self, orgn: Node, dest: Node) -> Iterable[Path]:
        return map(Path, nx.all_simple_paths(self.graph, orgn.name, dest.name))

    def shortest_path_assignment(self, demand: TravelDemand,
                                 travel_costs: np.ndarray) -> PathAssignment:
        self._igraph.es[self.WEIGHT_KEY] = travel_costs
        n_links = self.number_of_links()
        link_flow = np.zeros(n_links)
        _node_index = self._igraph['node_index']
        _link_matrix = self._igraph['link_matrix']
        for orgn, dest_volumes in demand.origin_based_index.items():
            s = _node_index[orgn]
            ts, vs = zip(*dest_volumes.items())
            ts = [_node_index[t] for t in ts]
            vs = np.array(vs)
            _paths = self._igraph.get_shortest_paths(
                s, ts, weights=self.WEIGHT_KEY
            )
            paths = numba.typed.List(
                np.array(p, dtype=np.uint16)
                for p in _paths
            )
            assign_all_to_links(
                _link_matrix,
                paths,
                vs,
                link_flow,
            )
        return PathAssignment(link_flow, frozenset())
        #self.set_link_costs(travel_costs)
        #link_flow = np.zeros(self.number_of_links())
        #od_pairs = {(d.origin.name,  d.destination.name): d.volume
        #            for d in demand}
        #paths = shortest_paths_nx_via_scipy(self.graph,
        #                                    self.WEIGHT_KEY,
        #                                    od_pairs.keys())
        #for u, v, p in paths:
        #    path = Path(p)
        #    link_flow = self._assign_path_flow_to_links(path, od_pairs[u, v],
        #                                                link_flow)
        #return PathAssignment(link_flow, frozenset())
        #if self._use_igraph:
        #    self._igraph.es[self.WEIGHT_KEY] = travel_costs
        #    _vertex_names = self._igraph.vs[NX_ID]
        #else:
        #    self.set_link_costs(travel_costs)
        #link_flow = np.zeros(self.number_of_links())
        #used_paths = set()
        #_igraph_time = 0
        #for orgn, dest_volumes in demand.origin_based_index.items():
        #    targets = list(dest_volumes.keys())
        #    if self._use_igraph:
        #        t0 = time.time()
        #         paths = shortest_paths_igraph(self._igraph, str(orgn),
        #                                       list(map(str, targets)),
        #                                       self.WEIGHT_KEY,
        #                                       names=_vertex_names)
        #         _igraph_time += (time.time() - t0)
        #     else:
        #         _, paths = single_source_dijkstra(
        #             self.graph, orgn, targets, weight=self.WEIGHT_KEY
        #         )
        #     for dest, volume in dest_volumes.items():
        #         try:
        #             shortest_path = Path(paths[dest])
        #             #used_paths.add(shortest_path)
        #         except KeyError:
        #             raise nx.NetworkXNoPath(f"No path from {orgn} to {dest}.")
        #         link_flow = self._assign_path_flow_to_links(
        #             shortest_path,
        #             volume,
        #             link_flow
        #         )
        # print(f"igraph time: {_igraph_time:0.4f}s")
        # return PathAssignment(link_flow, frozenset(used_paths))

    def _shortest_path(self, origin: Node, destination: Node) -> Path:
        return Path(nx.shortest_path(
            self.graph,
            origin.name,
            destination.name,
            weight=self.WEIGHT_KEY,
        ))

    def has_path(self, origin: Node, destination: Node) -> bool:
        try:
            self._shortest_path(origin, destination)
        except nx.NetworkXNoPath:
            return False
        else:
            return True

    def get_node(self, name) -> Node:
        return self._get_node(name)

    def _get_node(self, name: str) -> Node:
        return self.graph.nodes[name][self.NODE_KEY]

    def _set_node(self, name: str, node: Node):
        self.graph.nodes[name][self.NODE_KEY] = node

    def _get_link(self, u: str, v: str) -> Link:
        return self.graph.edges[u, v][self.LINK_KEY]

    def _set_link(self, u: str, v: str, link: Link):
        self.graph.edges[u, v][self.LINK_KEY] = link

    def _build_nodes(self) -> Iterable[Node]:
        for i, u in enumerate(sorted(self.graph.nodes)):
            node = Node(i, u)
            self._set_node(u, node)
            yield node

    def _build_links(self) -> Iterable[Link]:
        index = count()
        for from_node in self.nodes:
            u = from_node.name
            for v in sorted(self.graph.successors(u)):
                to_node = self._get_node(v)
                link = Link(next(index), from_node, to_node)
                self._set_link(u, v, link)
                yield link

    def set_link_costs(self, travel_costs: np.ndarray) -> ():
        for link, cost in zip(self.links, travel_costs):
            u, v = link.edge
            self.graph.edges[u, v][self.WEIGHT_KEY] = cost

    def _assign_path_flow_to_links(self, path: Path, volume: float,
                                   link_flow: np.ndarray) -> np.ndarray:
        for (u, v) in path.edges:
            link_flow[self._get_link(u, v).id] += volume
        return link_flow
