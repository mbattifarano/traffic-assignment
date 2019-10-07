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

from traffic_assignment.control_ratio_range.utils import (NetworkParameters,
                                                          Variables,
                                                          Constants)
from traffic_assignment.control_ratio_range.lp import (UpperControlRatio,
                                                       LowerControlRatio)


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

    def _solver(self, link_cost_function: LinkCostFunction,
                tolerance, max_iterations) -> Solver:
        road_network = self.network.to_road_network()
        return _create_solver(
            road_network,
            self.trips.to_demand(road_network),
            link_cost_function,
            tolerance,
            max_iterations,
        )

    def ue_solver(self, tolerance=1e-6, max_iterations=100000) -> Solver:
        return self._solver(self.network.to_link_cost_function(),
                            tolerance, max_iterations)

    def so_solver(self, tolerance=1e-6, max_iterations=100000) -> Solver:
        return self._solver(self.network.to_marginal_link_cost_function(),
                            tolerance, max_iterations)

    def _prepare_control_ratio(self, target_link_flow):
        road_network = self.network.to_road_network()
        link_cost = self.network.to_link_cost_function()
        marginal_link_cost = self.network.to_marginal_link_cost_function()
        demand = self.trips.to_demand(road_network)
        params = NetworkParameters.from_network(road_network, demand)
        variables = Variables.from_network_parameters(params)
        constants = Constants.from_network(
            road_network,
            demand,
            link_cost,
            marginal_link_cost,
            target_link_flow
        )
        return constants, variables

    def lower_control_ratio(self, user_equilibrium_link_flow):
        constants, variables = self._prepare_control_ratio(
            user_equilibrium_link_flow
        )
        return LowerControlRatio(constants, variables)

    def upper_control_ratio(self, system_optimal_link_flow):
        constants, variables = self._prepare_control_ratio(
            system_optimal_link_flow
        )
        return UpperControlRatio(constants, variables)


def _create_solver(network: Network, demand: TravelDemand,
                   link_cost_function: LinkCostFunction,
                   tolerance, max_iterations) -> Solver:
    return Solver(
        LineSearchStepSize(
            link_cost_function
        ),
        ShortestPathSearchDirection(
            network,
            demand,
        ),
        link_cost_function,
        tolerance=tolerance,
        max_iterations=max_iterations
    )
