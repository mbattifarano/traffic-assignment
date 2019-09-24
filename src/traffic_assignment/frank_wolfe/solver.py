
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from traffic_assignment.utils import Timer
from traffic_assignment.link_cost_function.base import LinkCostFunction

from .search_direction import SearchDirection
from .step_size import StepSize


@dataclass
class Solver:
    step_size: StepSize
    search_direction: SearchDirection
    link_cost_function: LinkCostFunction
    iterations: List[Iteration] = field(default_factory=list)
    max_iterations: int = 100
    tolerance: float = 1e-12
    timer: Timer = Timer()

    def __post_init__(self):
        self.iterations.append(self.initial_iteration())

    def initial_iteration(self) -> Iteration:
        self.timer.start()
        cost = self.link_cost_function.link_cost(0.0)
        link_flow = self.search_direction.minimum_point(cost, 0.0)
        return Iteration(0, cost, link_flow, np.zeros(len(link_flow)), np.inf,
                         self.timer.time_elapsed())

    @property
    def iteration(self) -> Iteration:
        return self.iterations[-1]

    @property
    def gaps(self):
        return np.array([i.gap for i in self.iterations])

    def update(self, iteration: Iteration) -> Iteration:
        self.timer.start()
        i = iteration.iteration
        link_flow = iteration.link_flow
        cost = self.link_cost_function.link_cost(link_flow)
        direction = self.search_direction.search_direction(cost, link_flow)
        step = self.step_size.step(i, link_flow, direction)
        new_link_flow = iteration.link_flow + step * direction
        gap = -np.dot(cost, direction)
        return Iteration(i + 1, cost, new_link_flow, direction, gap,
                         self.timer.time_elapsed())

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
    search_direction: np.ndarray
    gap: float
    duration: float
