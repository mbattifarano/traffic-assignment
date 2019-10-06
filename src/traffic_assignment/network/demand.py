from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, List, Mapping

import numpy as np

from .node import Node

OriginBasedIndex = Mapping[int, Mapping[int, float]]


@dataclass
class TravelDemand:
    demand: List[Demand]
    origin_based_index: OriginBasedIndex = None

    def __post_init__(self):
        self.origin_based_index = _reindex_demand(self.demand)
        self.demand = tuple(sorted(self.demand))

    def __hash__(self):
        return hash(tuple(self.demand))

    @property
    def number_of_od_pairs(self) -> int:
        return len(self.demand)

    def to_array(self) -> np.ndarray:
        return np.array([d.volume for d in self.demand])


class Demand(NamedTuple):
    origin: Node
    destination: Node
    volume: float


def _reindex_demand(demand: List[Demand]) -> OriginBasedIndex:
    index = defaultdict(dict)
    for d in demand:
        index[d.origin.name][d.destination.name] = d.volume
    return dict(index)
