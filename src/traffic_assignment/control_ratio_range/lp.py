from __future__ import annotations
from dataclasses import dataclass

import cvxpy as cp
import gurobipy as grb
import time

from datetime import timedelta
from enum import Enum, auto
from typing import NamedTuple

from .utils import Constants, Variables, HeuristicConstants, HeuristicVariables, path_indices, ConstraintTolerance
from traffic_assignment.utils import Timer
from traffic_assignment.link_cost_function.base import LinkCostFunction
from traffic_assignment.network.road_network import Network
from traffic_assignment.network.demand import TravelDemand
from warnings import warn
import numpy as np

from scipy.sparse import diags


@dataclass
class UpperControlRatio:
    constants: Constants
    variables: Variables
    tolerance: float = 1e-5

    def objective(self):
        return cp.sum(self.variables.fleet_demand)

    @property
    def sense(self):
        return cp.Minimize

    def constraints(self):
        v = self.variables
        c = self.constants
        return [
            v.fleet_path_flow >= 0.0,
            v.user_path_flow >= 0.0,
            v.user_link_flow + v.fleet_link_flow == c.target_link_flow,
            c.fleet_link_path_incidence() @ v.fleet_path_flow == v.fleet_link_flow,
            c.user_link_path_incidence() @ v.user_path_flow == v.user_link_flow,
            c.fleet_path_od_incidence() @ v.fleet_path_flow == v.fleet_demand,
            c.user_path_od_incidence() @ v.user_path_flow == v.user_demand,
            v.user_demand + v.fleet_demand == c.total_demand,
        ]

    def problem(self):
        timer = Timer()
        print("Creating problem.")
        timer.start()
        p = cp.Problem(
            self.sense(self.objective()),
            self.constraints()
        )
        print(f"Problem created in {timer.time_elapsed():0.2f} (s).")
        return p


class LowerControlRatio(UpperControlRatio):
    @property
    def sense(self):
        return cp.Maximize


@dataclass
class RestrictedLowerControlRatio(LowerControlRatio):
    mcr_fleet_demand: np.ndarray = None

    @property
    def sense(self):
        return cp.Minimize

    def objective(self):
        return cp.sum(cp.abs(self.mcr_fleet_demand - self.variables.fleet_demand))

    def constraints(self):
        cs = super().constraints()
        if self.mcr_fleet_demand is None:
            warn("RestrictedLowerControlRatio is being run with out a constraining fleet demand.")
        return cs


@dataclass
class MinimumFleetControlRatio:
    network: Network
    demand: TravelDemand
    constants: HeuristicConstants
    variables: HeuristicVariables
    tolerances: ConstraintTolerance = None

    def objective(self):
        return cp.sum(self.variables.fleet_path_flow) / self.constants.total_demand.sum()

    @property
    def user_demand(self):
        return (self.constants.trip_path_incidence
                @ self.variables.user_path_flow)

    @property
    def fleet_demand(self):
        return (self.constants.trip_path_incidence
                @ self.variables.fleet_path_flow)

    @property
    def user_link_flow(self):
        return (self.constants.link_path_incidence
                @ self.variables.user_path_flow)

    @property
    def fleet_link_flow(self):
        return (self.constants.link_path_incidence
                @ self.variables.fleet_path_flow)

    @property
    def user_impact(self):
        return cp.multiply(self.user_link_flow,
                           self.constants.link_cost_gradient)

    @property
    def total_marginal_path_cost(self):
        return (self.constants.link_path_incidence.T
                @ self.constants.marginal_link_cost)

    @property
    def path_cost(self):
        return (self.constants.link_path_incidence.T
                @ self.constants.link_cost)

    @property
    def fleet_marginal_path_cost(self):
        return (self.constants.link_path_incidence.T
                @ (self.constants.marginal_link_cost - self.user_impact))

    @property
    def min_fleet_path_costs(self):
        return (self.constants.trip_path_incidence.T
                @ self.variables.min_path_costs)

    @property
    def link_flow_error(self):
        return (self.user_link_flow + self.fleet_link_flow
                - self.constants.target_link_flow)

    @property
    def demand_error(self):
        return (self.user_demand + self.fleet_demand
                - self.constants.total_demand)

    @property
    def sense(self):
        return cp.Minimize

    def constraints(self):
        v = self.variables
        c = self.constants

        if self.tolerances is None:
            self.tolerances = c.check_feasibility(c.fleet_paths)
            print(f"tolerances: {self.tolerances}")
        return [
            v.user_path_flow >= 0.0,
            v.user_path_flow[~c.user_paths] == 0.0,
            v.fleet_path_flow >= 0.0,
            v.fleet_path_flow[~c.fleet_paths] == 0.0,
            self.link_flow_error <= self.tolerances.link_flow_tolerance,
            self.link_flow_error >= -self.tolerances.link_flow_tolerance,
            self.demand_error <= self.tolerances.demand_tolerance,
            self.demand_error >= -self.tolerances.demand_tolerance,
            self.fleet_marginal_path_cost[c.fleet_paths] == self.min_fleet_path_costs[c.fleet_paths],
            self.fleet_marginal_path_cost[~c.fleet_paths] >= self.min_fleet_path_costs[~c.fleet_paths],
        ]

    def problem(self):
        timer = Timer()
        timer.start()
        p = cp.Problem(
            self.sense(self.objective()),
            self.constraints()
        )
        # print(f"Created linear program in {timer.time_elapsed():0.2f} seconds.")
        return p

    def paths_to_add(self, fleet_link_flow=None):
        """Return the paths to add to the problem as fleet unusable
        """
        if fleet_link_flow is None:
            fleet_link_flow = self.fleet_link_flow.value
        fleet_link_cost = self.constants.fleet_link_cost(fleet_link_flow)
        least_cost_paths = {
            path
            for path, _
            in self.network.least_cost_paths(self.demand, fleet_link_cost,
                                             self.constants.path_cost_tolerance)
        }
        known_path_set = set(self.constants.known_paths)
        bad_count = 0
        for path, cost in self.network.least_cost_paths(self.demand,
                                                        self.constants.link_cost,
                                                        self.constants.path_cost_tolerance):
            if path not in known_path_set:
                bad_count += 1
                print(f"unkown user path! {path}")
        assert bad_count == 0, "user paths should have been already discovered!"
        # Return the set of least cost paths that are not known paths
        return list(least_cost_paths - known_path_set)

    def paths_to_remove(self):
        fleet_path_flow = self.variables.fleet_path_flow.value
        unused_paths = fleet_path_flow == 0.0
        feasible_paths = self.constants.fleet_paths
        # Return a bool vector of unused feasible paths
        return unused_paths & feasible_paths

    def _trip_score(self):
        """Return a score for each trip as the sum of the gradient
           of the user impact for each trip
           this prioritizes placing users where they can have the most impact
        """
        user_paths = diags(self.constants.user_paths.astype(int))
        trip_link_incidence = (self.constants.trip_path_incidence
                               @ user_paths
                               @ self.constants.link_path_incidence.T)
        od_pair_score = trip_link_incidence @ self.constants.link_cost_gradient
        paths_per_trip = np.asarray(self.constants.trip_path_incidence.sum(1)).squeeze()
        return od_pair_score / paths_per_trip

    def _path_score(self):
        user_paths = diags(self.constants.user_paths.astype(int))
        user_path_link_incidence = user_paths @ self.constants.link_path_incidence.T
        path_score = user_path_link_incidence @ self.constants.link_cost_gradient
        return path_score

    def _subset_paths_to_abandon(self, paths):
        """Return a random subset of the paths to remove"""
        # count candidate paths to remove by trip, candidate trips have at least
        # one path
        trip_based = True
        if trip_based:
            candidate_trips = (self.constants.trip_path_incidence @ paths) > 0
            # set scores of trips with no candidate paths to something
            score = self._trip_score().astype(float)
            score[~candidate_trips] = 0
            #i = np.random.multinomial(1, score / score.sum()).nonzero()[0][0]
            i = score.argmax()
            paths_for_trip = self.constants.trip_path_incidence[i, :].toarray().squeeze()
            # abandon all unused paths on this trip
            paths = np.multiply(paths_for_trip, paths).astype(bool)
            print(f"Abandoning {paths.sum()} unused feasible paths from trip {i}")
        else:
            score = self._path_score()
            score[~paths] = -np.inf
            i = score.argmax()
            paths = np.zeros_like(paths, dtype=bool)
            paths[i] = True
            print(f"Abandoning path at index {i}")

        sample_fraction = 1
        if sample_fraction >= 1:
            return paths
        else:
            k = int(np.ceil(sample_fraction * paths.sum()))
            index = np.random.choice(paths.nonzero()[0], k, replace=False)
            paths_to_remove = np.zeros_like(paths)
            paths_to_remove[index] = True
            return paths_to_remove

    def update_lp(self, status, paths):
        if status is HeuristicStatus.ADD_UNUSABLE_PATHS:
            paths_to_add = paths
            print(f"Adding {len(paths_to_add)} paths.")
            self.constants = self.constants.add_paths(self.network, self.demand,
                                                      paths_to_add)
            self.variables = HeuristicVariables.from_constants(self.constants)
        elif status is HeuristicStatus.ABANDON_PATHS:
            paths_to_remove = self._subset_paths_to_abandon(paths)
            print(f"Abandoning {paths_to_remove.sum()} unused feasible paths.")
            self.constants = self.constants.abandon_paths(self.network,
                                                          self.demand,
                                                          paths_to_remove
                                                          )
            self.variables = HeuristicVariables.from_constants(self.constants)
        return self

    def _next_heuristic(self):
        paths_to_add = self.paths_to_add()
        if paths_to_add:
            return HeuristicStatus.ADD_UNUSABLE_PATHS, paths_to_add
        paths_to_remove = self.paths_to_remove()
        if paths_to_remove.sum():
            return HeuristicStatus.ABANDON_PATHS, paths_to_remove
        return HeuristicStatus.DONE, None

    def heuristic(self, solver=cp.GUROBI):
        done = False
        #initial_add_paths = self.paths_to_add(self.constants.target_link_flow)
        #if initial_add_paths and False:
        #    print(f"Adding initial paths.")
        #    self.update_lp(HeuristicStatus.ADD_UNUSABLE_PATHS,
        #                   initial_add_paths)
        while not done:
            lp = self.problem()
            result = lp.solve(solver=solver)
            status, paths = self._next_heuristic()
            yield status, result
            self.update_lp(status, paths)
            done = status is HeuristicStatus.DONE

    def to_mip(self):
        return MinimumFleetControlRatioMIP(
            self.network,
            self.demand,
            self.constants,
            self.variables
        )


class MinimumFleetControlRatioMIP(MinimumFleetControlRatio):

    @property
    def solver(self):
        return cp.GUROBI

    def constraints(self):
        m1 = self.constants.total_demand.max()
        m2 = self.total_marginal_path_cost.max() - self.path_cost.min()
        if self.tolerances is None:
            self.tolerances = self.constants.check_feasibility(self.constants.fleet_paths)
            print(f"tolerances: {self.tolerances}")
        fleet_forbidden = 1 - self.variables.fleet_paths
        return [
            self.variables.user_path_flow >= 0.0,
            self.variables.user_path_flow[~self.constants.user_paths] == 0.0,
            self.variables.fleet_path_flow >= 0.0,
            self.variables.fleet_path_flow <= m1 * self.variables.fleet_paths,
            self.variables.fleet_paths[~self.constants.fleet_paths] == False,

            self.link_flow_error <= self.tolerances.link_flow_tolerance,
            self.link_flow_error >= -self.tolerances.link_flow_tolerance,
            self.demand_error <= self.tolerances.demand_tolerance,
            self.demand_error >= -self.tolerances.demand_tolerance,

            self.fleet_marginal_path_cost >= self.min_fleet_path_costs,
            self.fleet_marginal_path_cost <= self.min_fleet_path_costs + m2 * fleet_forbidden,
        ]

    def make_callback(self, mip):
        _, chain, inverse_data = mip.get_problem_data(solver=self.solver)
        cb = grb.GRB.Callback
        paths_to_add = []

        def callback(model, where):
            if where == cb.MIPSOL:
                time_elapsed = int(model.cbGet(cb.RUNTIME))
                print(f"{time_elapsed:d}s: Found new feasible solution!")
                best_bound = model.cbGet(cb.MIPSOL_OBJBND)
                best_objective = model.cbGet(cb.MIPSOL_OBJBST)
                x_grb = model.cbGetSolution(model.getVars())
                solution = {
                    'model': model,
                    'x': x_grb,
                    'objVal': best_objective,
                }
                mip.unpack_results(solution, chain, inverse_data)
                paths = self.paths_to_add()
                if paths:
                    print(f"{time_elapsed:d}s: [{best_bound:0.4f}, ?] {len(paths)} unusable paths found!")
                    paths_to_add.append(paths)
                    model.terminate()
                else:
                    n_solutions = model.cbGet(cb.MIPSOL_SOLCNT)
                    print(f"{time_elapsed:d}s: [{best_bound:0.4f}, {best_objective:0.4f}] New incumbent found ({n_solutions} found so far)!")

        return paths_to_add, callback

    def heuristic(self, verbose=True):
        mip = self.problem()
        done = False
        i = 0
        while not done:
            print(f"iteration {i}")
            paths_to_add, func = self.make_callback(mip)
            kwargs = {
                'MIPFocus': MIPFocus.bound,
                'Presolve': 2,
                'Presparsify': 1,
                'cuts': 3,
                'callback': func,
                'DisplayInterval': timedelta(minutes=5).total_seconds(),
                'MIPGapAbs': 0.01,
            }
            result = mip.solve(solver=self.solver, verbose=verbose, **kwargs)
            if paths_to_add:
                paths = paths_to_add.pop()
                print(f"Adding {len(paths)} paths.")
                status = HeuristicStatus.ADD_UNUSABLE_PATHS
                yield status, result
                mip = self.update_lp(status, paths).problem()
            else:
                status = HeuristicStatus.DONE
                print(f"done")
                yield status, result
            done = status is HeuristicStatus.DONE
            i += 1


class MIPFocus:
    feasibility = 1
    optimality = 2
    bound = 3


class HeuristicStatus(Enum):
    DONE = auto()
    ADD_UNUSABLE_PATHS = auto()
    ABANDON_PATHS = auto()
    CONTINUE = auto()


class HeuristicLog(NamedTuple):
    iteration: int
    time_elapsed: float
    status: HeuristicStatus
    objective_value: float
    paths_added: int
