from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from typing import Iterable

import networkx as nx
import numpy as np

from .demand import TravelDemand
from .link import Link
from .node import Node
from .path import Path
from .shortest_path import single_source_dijkstra


class Network(ABC):

    @abstractmethod
    def number_of_nodes(self) -> int:
        pass

    @abstractmethod
    def number_of_links(self) -> int:
        pass

    @abstractmethod
    def get_node(self, name) -> Node:
        pass

    @abstractmethod
    def has_path(self, origin: Node, destination: Node) -> bool:
        pass

    @abstractmethod
    def shortest_path_assignment(self, demand: TravelDemand,
                                 travel_costs: np.ndarray) -> np.ndarray:
        pass


class RoadNetwork(Network):
    NODE_KEY = 'node'
    LINK_KEY = 'link'
    WEIGHT_KEY = 'weight'

    def __init__(self, graph: nx.DiGraph):
        self.graph = nx.freeze(graph)
        self.nodes = list(self._build_nodes())
        self.links = list(self._build_links())

    def number_of_links(self) -> int:
        return len(self.links)

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def shortest_path_assignment(self, demand: TravelDemand,
                                 travel_costs: np.ndarray) -> np.ndarray:
        self._set_link_costs(travel_costs)
        link_flow = np.zeros(self.number_of_links())
        for orgn, dest_volumes in demand.origin_based_index.items():
            targets = list(dest_volumes.keys())
            _, paths = single_source_dijkstra(
                self.graph,
                orgn,
                targets,
                weight=self.WEIGHT_KEY,
            )
            for dest, volume in dest_volumes.items():
                shortest_path = Path(paths[dest])
                link_flow = self._assign_path_flow_to_links(
                    shortest_path,
                    volume,
                    link_flow
                )
        return link_flow

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

    def _set_link_costs(self, travel_costs: np.ndarray):
        for link, cost in zip(self.links, travel_costs):
            u, v = link.edge
            self.graph.edges[u, v][self.WEIGHT_KEY] = cost

    def _assign_path_flow_to_links(self, path: Path, volume: float,
                                   link_flow: np.ndarray) -> np.ndarray:
        for (u, v) in path.edges:
            link_flow[self._get_link(u, v).id] += volume
        return link_flow
