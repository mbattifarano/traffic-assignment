from abc import ABC, abstractmethod
from itertools import count

from dataclasses import dataclass
import numpy as np
from traffic_assignment.link_cost_function.base import LinkCostFunction


class StepSize(ABC):

    @abstractmethod
    def step(self, k: int, x: np.ndarray, d: np.ndarray) -> float:
        pass


class MonotoneDecreasingStepSize(StepSize):

    def step(self, k: int, x=None, d=None) -> float:
            return 2 / (k + 2)


@dataclass(frozen=True)
class LineSearchStepSize(StepSize):
    cost: LinkCostFunction

    def step(self, k: int, x: np.ndarray, d: np.ndarray) -> float:
        return LineSearcher(self.cost, x, d).find_step_size()


@dataclass
class LineSearcher:
    cost: LinkCostFunction
    link_flow: np.ndarray
    search_direction: np.ndarray
    max_iterations: int = 100
    tolerance: float = 1e-5
    a: float = 0
    b: float = 1

    @property
    def alpha(self) -> float:
        return 0.5 * (self.a + self.b)

    @property
    def x(self) -> np.ndarray:
        return self.link_flow + self.alpha * self.search_direction

    @property
    def t(self) -> np.ndarray:
        return self.cost.link_cost(self.x)

    @property
    def sigma(self) -> float:
        return float(np.dot(self.t, self.search_direction))

    def _search(self) -> ():
        alpha = self.alpha
        if self.sigma < 0:
            self.a = alpha
        else:
            self.b = alpha

    def _continue_search(self, k: int) -> bool:
        return (
            (k < self.max_iterations
                and (self.b - self.a > self.tolerance))
            or self.sigma > 0
        )

    def find_step_size(self) -> float:
        iteration = count()
        while self._continue_search(next(iteration)):
            self._search()
        return self.alpha
