from __future__ import annotations

from dataclasses import dataclass
from itertools import dropwhile
from typing import NamedTuple, List, Tuple, Union
from io import FileIO

import networkx as nx
import numpy as np
from marshmallow import Schema, fields, post_load
from traffic_assignment.link_cost_function.base import LinkCostFunction
from traffic_assignment.link_cost_function.bpr import (
    BPRLinkCostFunction, BPRMarginalLinkCostFunction
)
from traffic_assignment.network.road_network import RoadNetwork

from . import common

from traffic_assignment.utils import ArrayOrFloat, condense_array


@dataclass
class TNTPNetwork:
    meta_data: MetaData
    links: List[Link]

    @classmethod
    def read_text(cls, contents: str) -> TNTPNetwork:
        lines = contents.splitlines()
        meta_data = MetaData.from_lines(lines)
        items = filter(common.is_nonempty, dropwhile(common.is_header, lines))
        links = sorted(map(Link.from_line, items),
                       key=lambda l: l.id)
        return TNTPNetwork(meta_data, links)

    @classmethod
    def read_file(cls, fp: FileIO) -> TNTPNetwork:
        return cls.read_text(fp.read())

    def _links_as_columns(self):
        """Turn a list of Links into a Link with array attributes.
        Basically a dataframe.
        """
        return Link(*map(np.array, zip(*self.links)))

    def to_link_cost_function(self) -> LinkCostFunction:
        links = self._links_as_columns()
        return BPRLinkCostFunction(
            links.free_flow_time,
            links.capacity,
            condense_array(links.b),
            condense_array(links.power),
        )

    def to_marginal_link_cost_function(self, fleet_link_flow=None) -> LinkCostFunction:
        links = self._links_as_columns()
        return BPRMarginalLinkCostFunction(
            links.free_flow_time,
            links.capacity,
            condense_array(links.b),
            condense_array(links.power),
            fleet_link_flow
        )

    def to_networkx_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for link in self.links:
            graph.add_edge(link.from_node, link.to_node)
        return graph

    def to_road_network(self) -> RoadNetwork:
        return RoadNetwork(self.to_networkx_graph())


class MetaData(NamedTuple):
    n_zones: int
    n_nodes: int
    n_links: int
    first_node: int

    @classmethod
    def from_lines(cls, lines: List[str]) -> MetaData:
        data = common.metadata(lines)

        def get_int(key: str) -> int:
            v = data.get(key)
            return int(v) if v is not None else None

        return MetaData(
            *map(get_int,
                 [common.metadata_tags.number_of_zones.key,
                  common.metadata_tags.number_of_nodes.key,
                  common.metadata_tags.number_of_links.key,
                  common.metadata_tags.first_thru_node.key,
                  ]
                 )
        )


class Link(NamedTuple):
    from_node: int
    to_node: int
    capacity: float
    length: float
    free_flow_time: float
    b: float
    power: float
    speed_limit: float
    toll: float
    link_type: int

    @classmethod
    def from_line(cls, line: str) -> Link:
        items = line.strip(common.END_OF_LINE).strip().split(common.DELIMITER)
        schema = LinkSchema()
        data = dict(zip(schema.declared_fields.keys(), items))
        return schema.load(data)

    @property
    def id(self) -> Tuple[int, int]:
        return self.from_node, self.to_node


class LinkSchema(Schema):
    class Meta:
        ordered = True

    from_node = fields.Integer()
    to_node = fields.Integer()
    capacity = fields.Float()
    length = fields.Float()
    free_flow_time = fields.Float()
    b = fields.Float()
    power = fields.Float()
    speed_limit = fields.Float()
    toll = fields.Float()
    link_type = fields.Integer()

    @post_load
    def to_link(self, data: dict, many, **kw) -> Link:
        return Link(**data)
