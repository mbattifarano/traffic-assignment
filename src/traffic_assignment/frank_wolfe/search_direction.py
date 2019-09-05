from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from traffic_assignment.network.demand import Demand
from traffic_assignment.network.road_network import Network


class SearchDirection(ABC):

    @abstractmethod
    def search_direction(self, x: np.ndarray) -> np.ndarray:
        pass


@dataclass(frozen=True)
class ShortestPathSearchDirection(SearchDirection):
    network: Network
    travel_cost: np.ndarray
    demand: List[Demand]

    def search_direction(self, x: np.ndarray) -> np.ndarray:
        y = sum(self.network.shortest_path_assignment(d, self.travel_cost)
                for d in self.demand)
        return y - x

