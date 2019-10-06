from __future__ import annotations

from enum import Enum
from typing import NamedTuple
from traffic_assignment.network.road_network import Network
from traffic_assignment.network.demand import TravelDemand
from traffic_assignment.link_cost_function.base import LinkCostFunction

import cvxpy as cp
import numpy as np


class Solvers(Enum):
    """The linear programming solvers available to cvxopt"""
    glpk = 'glpk'
    mosek = 'mosek'
    none = None


class NetworkParameters(NamedTuple):
    number_of_links: int
    number_of_paths: int
    number_of_od_pairs: int

    @classmethod
    def from_network(cls, network: Network, demand: TravelDemand) -> NetworkParameters:
        return NetworkParameters(
            network.number_of_links(),
            network.number_of_paths(demand),
            demand.number_of_od_pairs,
        )


class Variables(NamedTuple):
    user_link_flow: cp.Variable
    user_path_flow: cp.Variable
    user_demand: cp.Variable
    fleet_link_flow: cp.Variable
    fleet_path_flow: cp.Variable
    fleet_demand: cp.Variable

    @classmethod
    def from_network_parameters(cls, params: NetworkParameters) -> Variables:
        n_links, n_paths, n_ods = params
        return Variables(
            cp.Variable(n_links, name="user_link_flow"),
            cp.Variable(n_paths, name="user_path_flow"),
            cp.Variable(n_ods, name="user_demand"),
            cp.Variable(n_links, name="fleet_link_flow"),
            cp.Variable(n_paths, name="fleet_path_flow"),
            cp.Variable(n_ods, name="fleet_demand"),
        )


class Constants(NamedTuple):
    target_link_flow: np.ndarray
    link_path_incidence: np.ndarray
    path_od_incidence: np.ndarray
    user_paths: np.ndarray
    fleet_paths: np.ndarray
    total_demand: np.ndarray

    @classmethod
    def from_network(cls, network: Network, demand: TravelDemand,
                     link_cost: LinkCostFunction,
                     marginal_link_cost: LinkCostFunction,
                     target_link_flow: np.ndarray):
        link_path, path_od, _ = network.path_incidences(demand)
        total_demand = demand.to_array()
        user_paths = get_minimal_paths(link_path, path_od, link_cost,
                                       target_link_flow)
        fleet_paths = get_minimal_paths(link_path, path_od, marginal_link_cost,
                                        target_link_flow)
        return Constants(
            target_link_flow,
            link_path,
            path_od,
            user_paths,
            fleet_paths,
            total_demand
        )

    def user_link_path_incidence(self):
        return self.link_path_incidence * self.user_paths

    def fleet_link_path_incidence(self):
        return self.link_path_incidence * self.fleet_paths

    def user_path_od_incidence(self):
        return self.path_od_incidence.T * self.user_paths

    def fleet_path_od_incidence(self):
        return self.path_od_incidence.T * self.fleet_paths


def path_cost(link_path_incidence: np.ndarray,
              link_cost: LinkCostFunction, link_flow: np.ndarray):
    return link_path_incidence.T @ link_cost.link_cost(link_flow)


def minimal_paths(path_od_incidence: np.ndarray, path_cost: np.ndarray):
    minimal_paths = np.zeros_like(path_cost)
    _, n_ods = path_od_incidence.shape
    for j in range(n_ods):
        path_selector = path_od_incidence[:, j].astype(bool)
        od_path_costs = path_cost[path_selector]
        minimal_paths[path_selector] = (od_path_costs == min(od_path_costs))
    return minimal_paths.astype(float)


def get_minimal_paths(link_path_incidence: np.ndarray,
                      path_od_incidence: np.ndarray,
                      link_cost: LinkCostFunction, link_flow: np.ndarray):
    return minimal_paths(
        path_od_incidence,
        path_cost(link_path_incidence, link_cost, link_flow),
    )
