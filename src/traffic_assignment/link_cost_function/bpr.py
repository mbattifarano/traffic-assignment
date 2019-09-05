from dataclasses import dataclass

from typing import Optional
import numpy as np

from .base import LinkCostFunction
from ..utils import value_or_default


@dataclass(frozen=True)
class BPRLinkCostFunction(LinkCostFunction):
    free_flow_travel_time: np.ndarray
    capacity: np.ndarray
    alpha: float = 0.15
    beta: float = 4

    def link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        return _bpr(self.alpha, self.beta, self.free_flow_travel_time,
                    self.capacity, link_flow)


@dataclass(frozen=True)
class BPRMarginalLinkCostFunction(BPRLinkCostFunction):
    fleet_link_flow: Optional[np.ndarray] = None

    def link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        fleet_link_flow = value_or_default(self.fleet_link_flow, link_flow)
        travel_time = _bpr(self.alpha, self.beta, self.free_flow_travel_time,
                           self.capacity, link_flow)
        travel_time_gradient = _d_bpr(self.alpha, self.beta,
                                      self.free_flow_travel_time,
                                      self.capacity, link_flow)
        return travel_time + fleet_link_flow * travel_time_gradient


def _bpr(alpha: float, beta: float, free_flow: np.ndarray, capacity: np.ndarray,
         link_flow: np.ndarray) -> np.ndarray:
    """The Bureau of Public Road (BPR) link cost function."""
    return (
        free_flow * (1.0 + alpha * (link_flow / capacity)**beta)
    )


def _d_bpr(alpha: float, beta: float, free_flow: np.ndarray,
           capacity: np.ndarray, link_flow: np.ndarray) -> np.ndarray:
    """The gradient of the Bureau of Public Road (BPR) link cost function."""
    return (
        free_flow * alpha * beta * link_flow**(beta - 1) / capacity**beta
    )
