
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
    report_interval: int = 1000
    initial_point: np.ndarray = None

    def __post_init__(self):
        self.iterations.append(self.initial_iteration())

    def initial_iteration(self) -> Iteration:
        self.timer.start()
        if self.initial_point is None:
            print("Starting from free flow travel cost.")
            cost = self.link_cost_function.link_cost(0.0)
            link_flow = self.search_direction.minimum_point(cost, 0.0)
        else:
            print("Warm start.")
            cost = self.link_cost_function.link_cost(self.initial_point)
            link_flow = self.initial_point
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

    def log_progress(self):
        i = self.iteration.iteration
        if (i == 1) or ((i > 0) and (i % self.report_interval == 0)):
            print(f"iteration {i}: relative gap = {self.iteration.gap:g} (absolute gap = {self.iteration.absolute_gap:g})")

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
            self.log_progress()
            next_iteration = self.update(self.iteration)
            self.iterations.append(next_iteration)
        return self.iteration

    def best_iteration(self):
        return min(self.iterations, key=lambda it: it.gap)


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

    @property
    def absolute_gap(self):
        return -np.dot(self.cost, self.search_direction)
