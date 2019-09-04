from abc import ABC, abstractmethod

import numpy as np


class LinkCostFunction(ABC):

    @abstractmethod
    def link_cost(self, link_flow: np.ndarray) -> np.ndarray:
        pass
