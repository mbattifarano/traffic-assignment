from __future__ import annotations

from dataclasses import dataclass

from typing import NamedTuple
from traffic_assignment.utils import Timer
from warnings import warn
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
                                                          HeuristicVariables,
                                                          Constants,
                                                          HeuristicConstants,
                                                          ControlRatioSchema,
                                                          ProblemData)
from traffic_assignment.control_ratio_range.lp import (UpperControlRatio,
                                                       LowerControlRatio,
                                                       MinimumFleetControlRatio,
                                                       RestrictedLowerControlRatio,
                                                       HeuristicStatus)
from traffic_assignment.utils import FileCache
from toolz import memoize


def problem_cache_key(args, kwargs):
    problem, target_link_flow, tol = args
    return f"{problem.name}_{hash(tuple(target_link_flow.ravel()))}_{tol:g}"


problem_data_cache = FileCache(ControlRatioSchema(), '__problem_data_cache__')


@dataclass
class TNTPProblem:
    network: TNTPNetwork
    trips: TNTPTrips
    solution: TNTPSolution
    name: str

    @classmethod
    def from_directory(cls, path: str) -> TNTPProblem:
        timer = Timer()
        print("Reading tntp problem.")
        timer.start()
        tntp_directory = TNTPDirectory(path)
        name = tntp_directory.name()
        with tntp_directory.network_file() as fp:
            network = TNTPNetwork.read_text(fp.read())
        with tntp_directory.trips_file() as fp:
            trips = TNTPTrips.read_text(fp.read())
        with tntp_directory.solution_file() as fp:
            solution = TNTPSolution.read_text(fp.read())
        print(f"Read tntp problem in {timer.time_elapsed():0.2f} (s).")
        return TNTPProblem(
            network,
            trips,
            solution,
            name
        )

    def road_network(self):
        return self.network.to_road_network()

    def travel_demand(self):
        return self.trips.to_demand(self.road_network())

    def _solver(self, link_cost_function: LinkCostFunction,
                tolerance, max_iterations, **kwargs) -> Solver:
        return _create_solver(
            self.road_network(),
            self.travel_demand(),
            link_cost_function,
            tolerance,
            max_iterations,
            **kwargs
        )

    def ue_solver(self, tolerance=1e-6, max_iterations=100000, **kwargs) -> Solver:
        return self._solver(self.network.to_link_cost_function(),
                            tolerance, max_iterations, **kwargs)

    def so_solver(self, tolerance=1e-6, max_iterations=100000, **kwargs) -> Solver:
        return self._solver(self.network.to_marginal_link_cost_function(),
                            tolerance, max_iterations, **kwargs)

    #@memoize(cache=problem_data_cache, key=problem_cache_key)
    def _prepare_control_ratio(self, target_link_flow, tolerance=1e-8):
        road_network = self.road_network()
        link_cost = self.network.to_link_cost_function()
        marginal_link_cost = self.network.to_marginal_link_cost_function()
        demand = self.travel_demand()
        params = NetworkParameters.from_network(road_network, demand)
        variables = Variables.from_network_parameters(params)
        constants = Constants.from_network(
            road_network,
            demand,
            link_cost,
            marginal_link_cost,
            target_link_flow,
            tolerance
        )
        return ProblemData(constants, variables)

    def lower_control_ratio(self, user_equilibrium_link_flow, tolerance=1e-8):
        timer = Timer()
        print("Creating problem data.")
        timer.start()
        constants, variables = self._prepare_control_ratio(
            user_equilibrium_link_flow,
            tolerance
        )
        print(f"Prepared problem data in {timer.time_elapsed():0.2f} (s).")
        return LowerControlRatio(constants, variables)

    def restricted_lower_control_ratio(self, ue_link_flow, mcr_fleet_demand, tolerance=1e-8):
        timer = Timer()
        print("Creating problem data.")
        timer.start()
        constants, variables = self._prepare_control_ratio(
            ue_link_flow,
            tolerance
        )
        print(f"Prepared problem data in {timer.time_elapsed():0.2f} (s).")
        return RestrictedLowerControlRatio(
            constants,
            variables,
            mcr_fleet_demand=mcr_fleet_demand,
        )

    def upper_control_ratio(self, system_optimal_link_flow, tolerance=1e-4):
        constants, variables = self._prepare_control_ratio(
            system_optimal_link_flow,
            tolerance
        )
        return UpperControlRatio(constants, variables)

    def _prepare_fomcr_constants(self, system_optimal_link_flow,
                                 fleet_path_set):
        net = self.road_network()
        link_cost = self.network.to_link_cost_function()
        demand = self.travel_demand()
        constants = HeuristicConstants.from_network(
            net,
            demand,
            link_cost,
            fleet_path_set,
            system_optimal_link_flow,
        )
        return constants

    def minimum_fleet_control_ratio(self, system_optimal_link_flow,
                                    fleet_path_set):
        constants = self._prepare_fomcr_constants(system_optimal_link_flow,
                                                  fleet_path_set)
        variables = HeuristicVariables.from_constants(constants)
        return MinimumFleetControlRatio(
            self.road_network(),
            self.travel_demand(),
            constants,
            variables,
        )


def _create_solver(network: Network, demand: TravelDemand,
                   link_cost_function: LinkCostFunction,
                   tolerance, max_iterations, **kwargs) -> Solver:
    try:
        large_initial_step = kwargs.pop('large_initial_step')
    except KeyError:
        large_initial_step = True
    return Solver(
        LineSearchStepSize(
            link_cost_function,
            large_initial_step
        ),
        ShortestPathSearchDirection(
            network,
            demand,
        ),
        link_cost_function,
        tolerance=tolerance,
        max_iterations=max_iterations,
        **kwargs
    )
