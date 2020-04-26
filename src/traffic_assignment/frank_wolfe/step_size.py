from abc import ABC, abstractmethod
from itertools import count
from typing import Tuple

from dataclasses import dataclass
import numpy as np
from traffic_assignment.link_cost_function.base import LinkCostFunction
import numba

class StepSize(ABC):

    @abstractmethod
    def step(self, k: int, x: np.ndarray, d: np.ndarray) -> float:
        pass


@dataclass(frozen=True)
class MonotoneDecreasingStepSize(StepSize):
    offset: int = 2

    def step(self, k: int, x=None, d=None) -> float:
            return 2 / (k + self.offset)


@dataclass(frozen=True)
class LineSearchStepSize(StepSize):
    cost: LinkCostFunction
    large_initial_step: bool = True

    def step(self, k: int, x: np.ndarray, d: np.ndarray) -> float:
        """Take a large first step. Then find optimal step size."""
        if self.large_initial_step and (k == 0):
            return 1.0
        else:
            return LineSearcher(self.cost, x, d).find_step_size()


@dataclass
class LineSearcher:
    cost: LinkCostFunction
    link_flow: np.ndarray
    search_direction: np.ndarray
    max_iterations: int = 100
    tolerance: np.float = 1e-12
    a: np.float = 0.0
    b: np.float = 1.0

    #def __post_init__(self):
    #    self.search_direction = self.search_direction

    @property
    def alpha(self) -> float:
        return 0.5 * (self.a + self.b)
        #return _alpha(self.a, self.b)

    @property
    def x(self) -> np.ndarray:
        return _x(self.link_flow, self.alpha, self.search_direction)
        # return self.link_flow + self.alpha * self.search_direction

    @property
    def t(self) -> np.ndarray:
        return self.cost.link_cost(self.x)

    @property
    def sigma(self):
        return _sigma(self.t, self.search_direction)
        #return np.dot(self.t, self.search_direction)

    def _search(self) -> ():
        alpha = self.alpha
        if self.sigma < 0:
            self.a = alpha
        else:
            self.b = alpha

    def _continue_search(self, k: int) -> Tuple[bool, str]:
        return _continue_search(k, self.max_iterations,
                                self.a, self.b, self.tolerance,
                                self.sigma)
        #under_iterations = k < self.max_iterations
        #above_threshold = (self.b - self.a > self.tolerance)
        #positive_sigma = self.sigma > 0
        #reason = ("iteration limit exceeded;" * (not under_iterations)
        #          + "tolerance met;" * (not above_threshold))
        #return (
        #    (under_iterations and above_threshold)
        #    or positive_sigma
        #), reason

    def find_step_size(self) -> float:
        #fp = open('line-search.debug', 'a')
        #fp.write("Starting line search.\n")
        iteration = count()
        k = next(iteration)
        criteria = self._continue_search(k)
        while criteria:
            self._search()
            #fp.write(f"{k}: b-a = {self.b - self.a}; sigma = {self.sigma}\n")
            k = next(iteration)
            criteria = self._continue_search(k)
        #fp.write(f"Returning alpha = {self.b} with reason {reason}.\n")
        return self.b


@numba.jit(nopython=True)
def _sigma(cost, direction):
    return np.dot(cost, direction)


@numba.jit(nopython=True)
def _x(x, alpha, direction):
    return x + alpha * direction


@numba.jit(nopython=True)
def _alpha(a, b):
    return 0.5 * (a + b)


@numba.jit(nopython=True)
def _continue_search(k, max_iter, a, b, tolerance, sigma):
    under_iterations = k < max_iter
    above_threshold = (b - a) > tolerance
    positive_sigma = sigma > 0
    return (
        (under_iterations and above_threshold)
        or positive_sigma
    )
    #reason = ""
    #if not under_iterations:
    #    reason += "iteration limit exceeded;"
    #if not above_threshold:
    #    reason += "tolerance met;"
    #return (
    #               (under_iterations and above_threshold)
    #               or positive_sigma
    #       ), reason

