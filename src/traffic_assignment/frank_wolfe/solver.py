
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
    tolerance: float = 1e-15
    timer: Timer = Timer()

    def __post_init__(self):
        self.iterations.append(self.initial_iteration())

    def initial_iteration(self) -> Iteration:
        self.timer.start()
        cost = self.link_cost_function.link_cost(0.0)
        link_flow = self.search_direction.minimum_point(cost, 0.0)
        return Iteration(
            iteration=0,
            cost=cost,
            link_flow=link_flow,
            step=np.NaN,
            search_direction=np.zeros(len(link_flow)),
            best_lower_bound=0.0,
            gap=np.inf,
            duration=self.timer.time_elapsed())

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
        gap, best_lower_bound = self._relative_gap(cost, direction,
                                                   new_link_flow)
        return Iteration(
            iteration=i + 1,
            cost=cost,
            link_flow=new_link_flow,
            step=step,
            search_direction=direction,
            best_lower_bound=best_lower_bound,
            gap=gap,
            duration=self.timer.time_elapsed()
        )

    def objective_value(self, link_flow: np.ndarray) -> float:
        return self.link_cost_function.integral_link_cost(link_flow).sum()

    def _relative_gap(self, cost, direction, link_flow):
        gap = np.dot(cost, direction)
        best_lower_bound = max(
            self.objective_value(link_flow) + gap,
            self.iteration.best_lower_bound
        )
        relative_gap = -gap / best_lower_bound
        return relative_gap, best_lower_bound

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
    step: float
    search_direction: np.ndarray
    best_lower_bound: float
    gap: float
    duration: float
