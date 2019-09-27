from dataclasses import dataclass

import numpy as np

from .base import LinkCostFunction


@dataclass
class LinearLinkCostFunction(LinkCostFunction):
    weights: np.ndarray
    biases: np.ndarray

    def link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        return link_flow * self.weights + self.biases

    def integral_link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        return 0.5 * self.weights * link_flow**2 + self.biases * link_flow

