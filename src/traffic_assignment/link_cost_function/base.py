from abc import ABC, abstractmethod

import numpy as np
from traffic_assignment.utils import ArrayOrFloat


class LinkCostFunction(ABC):

    @abstractmethod
    def link_cost(self, link_flow: ArrayOrFloat) -> np.ndarray:
        pass

    @abstractmethod
    def integral_link_cost(self, link_flow: ArrayOrFloat) -> np.ndarray:
        pass

    @abstractmethod
    def derivative_link_cost(self, link_flow: ArrayOrFloat) -> np.ndarray:
        pass
