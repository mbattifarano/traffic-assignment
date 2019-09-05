from dataclasses import dataclass

import numpy as np

from .base import LinkCostFunction


@dataclass(frozen=True)
class BPRLinkCostFunction(LinkCostFunction):
    free_flow_travel_time: np.ndarray
    capacity: np.ndarray
    alpha: float = 0.15
    beta: float = 4

    def link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        return (
            self.free_flow_travel_time
            * (1.0 + self.alpha * (link_flow / self.capacity)**self.beta)
        )
