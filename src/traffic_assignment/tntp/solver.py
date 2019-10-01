from __future__ import annotations

from dataclasses import dataclass

from traffic_assignment.frank_wolfe.search_direction import (
    ShortestPathSearchDirection)
from traffic_assignment.frank_wolfe.solver import Solver
from traffic_assignment.frank_wolfe.step_size import LineSearchStepSize
from traffic_assignment.link_cost_function.base import LinkCostFunction
from traffic_assignment.network.demand import TravelDemand
from traffic_assignment.network.road_network import Network

from .common import TNTPDirectory
from .network import TNTPNetwork
from .solution import TNTPSolution
from .trips import TNTPTrips


@dataclass
class TNTPProblem:
    network: TNTPNetwork
    trips: TNTPTrips
    solution: TNTPSolution
    name: str

    @classmethod
    def from_directory(cls, path: str) -> TNTPProblem:
        tntp_directory = TNTPDirectory(path)
        name = tntp_directory.name()
        with tntp_directory.network_file() as fp:
            network = TNTPNetwork.read_text(fp.read())
        with tntp_directory.trips_file() as fp:
            trips = TNTPTrips.read_text(fp.read())
        with tntp_directory.solution_file() as fp:
            solution = TNTPSolution.read_text(fp.read())
        return TNTPProblem(
            network,
            trips,
            solution,
            name
        )

    def _solver(self, link_cost_function: LinkCostFunction) -> Solver:
        road_network = self.network.to_road_network()
        return _create_solver(
            road_network,
            self.trips.to_demand(road_network),
            link_cost_function
        )

    def ue_solver(self) -> Solver:
        return self._solver(self.network.to_link_cost_function())

    def so_solver(self) -> Solver:
        return self._solver(self.network.to_marginal_link_cost_function())


def _create_solver(network: Network, demand: TravelDemand,
                   link_cost_function: LinkCostFunction) -> Solver:
    return Solver(
        LineSearchStepSize(
            link_cost_function
        ),
        ShortestPathSearchDirection(
            network,
            demand,
        ),
        link_cost_function,
        tolerance=1e-6,
        max_iterations=50000
    )
