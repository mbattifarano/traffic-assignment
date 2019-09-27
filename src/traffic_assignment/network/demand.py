from __future__ import annotations
from typing import NamedTuple, List, Mapping
from dataclasses import dataclass
from collections import defaultdict

from .node import Node

OriginBasedIndex = Mapping[int, Mapping[int, float]]


@dataclass
class TravelDemand:
    demand: List[Demand]
    origin_based_index: OriginBasedIndex = None

    def __post_init__(self):
        self.origin_based_index = _reindex_demand(self.demand)


class Demand(NamedTuple):
    origin: Node
    destination: Node
    volume: float


def _reindex_demand(demand: List[Demand]) -> OriginBasedIndex:
    index = defaultdict(dict)
    for d in demand:
        index[d.origin.name][d.destination.name] = d.volume
    return dict(index)
