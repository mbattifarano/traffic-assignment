from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from traffic_assignment.network.demand import TravelDemand
from traffic_assignment.network.road_network import Network


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
    demand: TravelDemand

    def minimum_point(self, travel_cost: np.ndarray, link_flow: np.ndarray)\
            -> np.ndarray:
        return self.network.shortest_path_assignment(self.demand, travel_cost)
