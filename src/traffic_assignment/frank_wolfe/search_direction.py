from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from traffic_assignment.network.demand import Demand
from traffic_assignment.network.road_network import Network
from traffic_assignment.link_cost_function.base import LinkCostFunction


class SearchDirection(ABC):

    def search_direction(self, travel_cost: np.ndarray, link_flow: np.ndarray)\
            -> np.ndarray:
        return self.minimum_point(travel_cost, link_flow) - link_flow

    @abstractmethod
    def minimum_point(self, travel_cost: np.ndarray, link_flow: np.ndarray)\
            -> np.ndarray:
        pass


@dataclass(frozen=True)
class ShortestPathSearchDirection(SearchDirection):
    network: Network
    demand: List[Demand]

    def minimum_point(self, travel_cost: np.ndarray, link_flow: np.ndarray)\
            -> np.ndarray:
        return sum(
            self.network.shortest_path_assignment(d, travel_cost)
            for d in self.demand
        )
