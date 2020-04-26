import fiona
from collections import defaultdict
from toolz import valmap, curry
import networkx as nx
import os
import numpy as np
from typing import List, Tuple, Iterable, Optional
from .node import Coordinate, Node
from .schemas import LinkAttributes, Feature
from .features import to_node, to_links
from .demand import read_csv, to_demand
from traffic_assignment.spatial_index import SpatialIndex, insert_point, nearest_neighbor
from traffic_assignment.network.road_network import RoadNetwork, Node as NetworkNode
from traffic_assignment.network.demand import Demand, TravelDemand
from traffic_assignment.network.graph import reindex_graph
import warnings

DEFAULT_NODE_FILE = 'node.shp'
DEFAULT_LINK_FILE = 'link.shp'
ZONES = 'zones'
LINK_KEY = 'link'


class MalformedNetworkException(Exception):
    pass


class NoNodeFoundException(MalformedNetworkException):
    pass


def shp_items(shp_file: str):
    return fiona.open(shp_file).values()


def nearest_coordinate(idx: SpatialIndex, pt: Coordinate) -> Node:
    x, y = pt
    result = nearest_neighbor(idx, x, y)
    if result is None:
        raise NoNodeFoundException
    else:
        return result.object


def graph_from_shp(directory: str,
                   node_file: str = DEFAULT_NODE_FILE,
                   link_file: str = DEFAULT_LINK_FILE,
                   ) -> nx.DiGraph:
    graph = nx.DiGraph(name=directory)
    idx = SpatialIndex()
    zones = defaultdict(list)
    for i, node in enumerate(shp_items(os.path.join(directory, node_file))):
        n = to_node(i, node[Feature.properties])
        insert_point(idx, n.id, *n.coordinate, obj=n)
        graph.add_node(n.id, **n._asdict())
    for edge_pair in shp_items(os.path.join(directory, link_file)):
        for link in to_links(edge_pair[Feature.properties],
                             edge_pair[Feature.geometry]):
            if link.is_link:
                _from_node = nearest_coordinate(idx, link.from_point)
                _to_node = nearest_coordinate(idx, link.to_point)
                e = (_from_node.id, _to_node.id)
                if link.is_virtual:
                    zones[link.zone].append(e)
                graph.add_edge(*e, **link.to_dict())
    graph.graph['zones'] = valmap(extract_virtual_node(graph), zones)
    return graph


def _csv_files(directory: str) -> Iterable[str]:
    for fname in os.listdir(directory):
        if fname.endswith('.csv'):
            yield os.path.join(directory, fname)


def _get_virtual_node(network: RoadNetwork,
                      zone_id: int) -> Optional[NetworkNode]:
    node_id = network.graph.graph[ZONES].get(zone_id)
    if node_id is None:
        return None
    else:
        return network.get_node(node_id)


def travel_demand(network: RoadNetwork, directory: str):
    od_matrix = sum(map(read_csv, _csv_files(directory)))
    for d in to_demand(od_matrix):
        origin = _get_virtual_node(network, d.from_zone)
        if origin is None:
            continue
        destination = _get_virtual_node(network, d.to_zone)
        if destination is None:
            continue
        if origin != destination:
            yield Demand(
                origin,
                destination,
                d.volume,
            )


@curry
def extract_virtual_node(graph: nx.DiGraph,
                         virtual_links: List[Tuple[int, int]]) -> Optional[int]:
    ns = set()
    for l in virtual_links:
        ns.update(l)
    it = iter(virtual_links)
    s = set(next(it))
    for l in it:
        s.intersection_update(l)
    n_candidates = len(s)
    if n_candidates == 1:
        return s.pop()
    elif n_candidates == 0:
        warnings.warn("Could not resolve a virtual node, returning None")
        return None
    else:
        _, n = min(
            (graph.degree(n), n) for n in s
        )
        return n


def edges(graph):
    return sorted(graph.edges)


def array_of(graph: nx.DiGraph, key: str) -> np.ndarray:
    return np.array([
        graph.edges[e][key]
        for e in edges(graph)
    ])


def to_free_flow_travel_time(graph: nx.DiGraph):
    return array_of(graph, LinkAttributes.free_flow)


def to_capacity(graph: nx.DiGraph):
    return array_of(graph, LinkAttributes.capacity)
