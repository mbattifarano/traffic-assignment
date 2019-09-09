
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from typing import List

from .search_direction import SearchDirection
from .step_size import StepSize
from traffic_assignment.link_cost_function.base import LinkCostFunction


@dataclass
class Solver:
    step_size: StepSize
    search_direction: SearchDirection
    link_cost_function: LinkCostFunction
    iterations: List[Iteration] = field(default_factory=list)
    max_iterations: int = 100
    tolerance: float = 1e-5

    def __post_init__(self):
        self.iterations.append(self.initial_iteration())

    def initial_iteration(self) -> np.ndarray:
        cost = self.link_cost_function.link_cost(0.0)
        link_flow = self.search_direction.minimum_point(cost, 0.0)
        return Iteration(0, cost, link_flow, np.inf)

    @property
    def iteration(self) -> Iteration:
        return self.iterations[-1]

    def update(self, iteration: Iteration) -> Iteration:
        cost = self.link_cost_function.link_cost(iteration.link_flow)
        direction = self.search_direction.search_direction(cost,
                                                           iteration.link_flow)
        step = self.step_size.step(self.iteration.iteration)
        link_flow = iteration.link_flow + step * direction
        gap = -np.dot(cost, direction)
        return Iteration(self.iteration.iteration + 1, cost, link_flow, gap)

    def _continue(self) -> bool:
        return (
            (self.iteration.gap > self.tolerance)
            and (self.iteration.iteration < self.max_iterations)
        )

    def solve(self) -> Iteration:
        while self._continue():
            next_iteration = self.update(self.iteration)
            self.iterations.append(next_iteration)
        return self.iteration


@dataclass(frozen=True)
class Iteration:
    iteration: int
    cost: np.ndarray
    link_flow: np.ndarray
    gap: float
