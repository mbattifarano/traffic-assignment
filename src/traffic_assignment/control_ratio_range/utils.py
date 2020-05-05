from __future__ import annotations

import os
from enum import Enum
from typing import NamedTuple, Set, Iterable, List
from itertools import chain
import pickle

from traffic_assignment.network.road_network import Network
from traffic_assignment.network.demand import TravelDemand
from traffic_assignment.network.path import Path
from traffic_assignment.link_cost_function.base import LinkCostFunction
from marshmallow import Schema, fields, post_load, pre_dump
from io import BytesIO

import cvxpy as cp
import numpy as np
from scipy.sparse import dok_matrix, save_npz, hstack as sparse_hstack

sparse_matrix = dok_matrix


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
    min_path_costs: cp.Variable

    @classmethod
    def from_network_parameters(cls, params: NetworkParameters) -> Variables:
        n_links, n_paths, n_ods = params
        return Variables(
            cp.Variable(n_links, name="user_link_flow"),
            cp.Variable(n_paths, name="user_path_flow", nonneg=True),
            cp.Variable(n_ods, name="user_demand"),
            cp.Variable(n_links, name="fleet_link_flow"),
            cp.Variable(n_paths, name="fleet_path_flow", nonneg=True),
            cp.Variable(n_ods, name="fleet_demand"),
            cp.Variable(n_ods, name="min_path_costs"),
        )


FLOAT = np.double
BINARY = np.uint8  # Not really but as small as we are gonna get


class HeuristicVariables(NamedTuple):
    user_path_flow: cp.Variable
    fleet_path_flow: cp.Variable
    min_path_costs: cp.Variable
    fleet_paths: cp.Variable

    @classmethod
    def from_constants(cls, constants: HeuristicConstants):
        n_trips, n_paths = constants.trip_path_incidence.shape
        return cls(
            cp.Variable(n_paths, name="user_path_flow", nonneg=True),
            cp.Variable(n_paths, name="fleet_path_flow", nonneg=True),
            cp.Variable(n_trips, name="min_path_cost"),
            cp.Variable(n_paths, name="fleet_paths", boolean=True)
           )

    def save(self, filename):
        np.savez(filename,
                 user_path_flow=self.user_path_flow.value,
                 fleet_path_flow=self.fleet_path_flow.value,
                 min_path_costs=self.min_path_costs.value,
                 fleet_paths=self.fleet_paths.value,
                 )

def first(it):
    return next(iter(it))

def _mape(estimated, actual):
    return np.nanmean(abs(estimated - actual)/actual)


def _mae(estimated, actual):
    return abs(estimated - actual).mean()


class ConstraintTolerance(NamedTuple):
    link_flow_tolerance: float
    demand_tolerance: float


class HeuristicConstants(NamedTuple):
    target_link_flow: np.ndarray
    total_demand: np.ndarray
    link_path_incidence: sparse_matrix
    trip_path_incidence: sparse_matrix
    known_paths: List[Path]
    user_paths: np.ndarray
    fleet_paths: np.ndarray
    link_cost: np.ndarray
    link_cost_gradient: np.ndarray
    marginal_link_cost: np.ndarray
    path_cost_tolerance: float = 1e-3

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.savez(os.path.join(directory, 'arrays.npz'),
                 target_link_flow=self.target_link_flow,
                 total_demand=self.total_demand,
                 user_paths=self.user_paths,
                 fleet_paths=self.fleet_paths,
                 link_cost=self.link_cost,
                 link_cost_gradient=self.link_cost,
                 marginal_link_cost=self.marginal_link_cost,
                 )
        save_npz(os.path.join(directory, 'link-path-incidence.npz'),
                 self.link_path_incidence.tocoo()
                 )
        save_npz(os.path.join(directory, 'trip-path-incidence.npz'),
                 self.trip_path_incidence.tocoo()
                 )
        with open(os.path.join(directory, 'paths.pkl'), 'wb') as fp:
            pickle.dump(self.known_paths, fp)

    def check_feasibility(self, useable_paths: np.ndarray) -> ConstraintTolerance:
        """Return the tolerance required to make the problem feasible"""
        _, n_paths = self.link_path_incidence.shape
        path_flow = cp.Variable(n_paths)
        link_flow_error = (self.link_path_incidence @ path_flow
                           - self.target_link_flow)
        demand_error = (self.trip_path_incidence @ path_flow
                        - self.total_demand)
        objective = cp.sum_squares(cp.hstack([
            link_flow_error,
            demand_error
        ]))
        constraints = [
            path_flow >= 0.0,  # all path flow must be non-negative
            path_flow[~useable_paths] == 0.0,  # no flow is allowed on un-useable paths
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        assert problem.status == cp.OPTIMAL
        return ConstraintTolerance(
            link_flow_tolerance=abs(link_flow_error.value).max(),
            demand_tolerance=abs(demand_error.value).max()
        )

    @staticmethod
    def _useable_paths(known_paths, cost, link_path_incidence, trip_path_incidence, path_cost_tolerance):
        print(f"Finding usable paths within tolerance {path_cost_tolerance}.")
        path_cost = link_path_incidence.T @ cost
        path_cost_ratio = np.zeros(len(known_paths))
        for i, trip_paths in enumerate(trip_path_incidence):
            trip_path_mask = trip_paths.toarray().squeeze().astype(np.bool)
            _path_c = path_cost[trip_path_mask]
            path_cost_ratio[trip_path_mask] = _path_c / _path_c.min()
        return path_cost_ratio <= (1.0 + path_cost_tolerance)

    @classmethod
    def from_network(cls, network: Network, demand: TravelDemand,
                     link_cost: LinkCostFunction,
                     known_paths: List[Path],
                     target_link_flow: np.ndarray,
                     path_cost_tolerance: float = 1e-5):
        cost = link_cost.link_cost(target_link_flow)
        cost_gradient = link_cost.derivative_link_cost(target_link_flow)
        marginal_cost = cost + np.multiply(target_link_flow, cost_gradient)

        print("building path set")
        if known_paths is None:
            known_paths = list(
                map(first,
                    chain(
                        network.least_cost_paths(demand, cost, path_cost_tolerance),
                        network.least_cost_paths(demand, marginal_cost, path_cost_tolerance)
                    )
                )
            )

        print("building path indcidence matrices")
        link_path_incidence, path_od_incidence = network.path_set_incidences(demand, known_paths)
        trip_path_incidence = path_od_incidence.T

        print("finding usable paths")
        user_paths = cls._useable_paths(known_paths, cost,
                                        link_path_incidence,
                                        trip_path_incidence,
                                        path_cost_tolerance)
        fleet_paths = cls._useable_paths(known_paths, marginal_cost,
                                         link_path_incidence,
                                         trip_path_incidence,
                                         path_cost_tolerance)
        print("returning constants")
        return HeuristicConstants(
            target_link_flow=target_link_flow,
            total_demand=demand.to_array(),
            link_path_incidence=link_path_incidence,
            trip_path_incidence=trip_path_incidence,
            known_paths=known_paths,
            user_paths=user_paths,
            fleet_paths=fleet_paths,
            link_cost=cost,
            link_cost_gradient=cost_gradient,
            marginal_link_cost=marginal_cost,
            path_cost_tolerance=path_cost_tolerance
        )

    @property
    def fleet_path_set(self):
        return {
            self.known_paths[i]
            for i in self.fleet_paths.nonzero()[0]
        }

    def fleet_link_cost(self, fleet_link_flow: np.ndarray) -> np.ndarray:
        return self.link_cost + np.multiply(fleet_link_flow,
                                            self.link_cost_gradient)

    def abandon_paths(self, network: Network, demand: TravelDemand,
                      unused_feasible_paths: np.ndarray):
        fleet_paths = self.fleet_paths.copy()
        fleet_paths[unused_feasible_paths] = False
        return self._replace(
            fleet_paths=fleet_paths
        )

    def add_paths(self, network: Network, demand: TravelDemand, paths: List[Path]):
        known_paths = self.known_paths + paths
        link_path_incidence, path_od_incidence = network.path_set_incidences(demand, known_paths)
        trip_path_incidence = path_od_incidence.T
        user_paths = self._useable_paths(known_paths, self.link_cost,
                                         link_path_incidence,
                                         trip_path_incidence,
                                         self.path_cost_tolerance)
        fleet_paths = self._useable_paths(known_paths, self.marginal_link_cost,
                                          link_path_incidence,
                                          trip_path_incidence,
                                          self.path_cost_tolerance)
        return self._replace(
            known_paths=known_paths,
            fleet_paths=fleet_paths,
            user_paths=user_paths,
            link_path_incidence=link_path_incidence,
            trip_path_incidence=trip_path_incidence,
        )


def _normalize_columns(A):
    return A / np.linalg.norm(A, axis=0)


def _column_similarity(A, B):
    return _normalize_columns(A).T.dot(_normalize_columns(B))


def _approx_equals(a, b, tol=1e-6):
    return abs(a-b) < tol


def path_indices(old_link_path, new_link_path):
    similarity = _column_similarity(old_link_path, new_link_path)
    index_map = {}
    for old_path_i, new_path_i in zip(*_approx_equals(similarity, 1).nonzero()):
        if new_path_i in index_map:
            raise Exception("Duplicate path found")
        else:
            index_map[new_path_i] = old_path_i
    return index_map


class Constants(NamedTuple):
    target_link_flow: np.ndarray
    link_path_incidence: sparse_matrix
    path_od_incidence: sparse_matrix
    user_paths: sparse_matrix
    fleet_paths: sparse_matrix
    total_demand: np.ndarray
    link_cost: np.ndarray
    link_cost_gradient: np.ndarray
    marginal_link_cost: np.ndarray
    tolerance: float = 1e-4

    @classmethod
    def from_network(cls, network: Network, demand: TravelDemand,
                     link_cost: LinkCostFunction,
                     marginal_link_cost: LinkCostFunction,
                     target_link_flow: np.ndarray, tolerance: float = 1e-4):
        link_path, path_od, _ = network.path_incidences(demand)
        total_demand = demand.to_array()
        user_paths = get_minimal_paths(link_path, path_od, link_cost,
                                       target_link_flow, tolerance)
        fleet_paths = get_minimal_paths(link_path, path_od, marginal_link_cost,
                                        target_link_flow, tolerance)
        cost = link_cost.link_cost(target_link_flow)
        cost_gradient = link_cost.derivative_link_cost(target_link_flow)
        marginal_cost = marginal_link_cost.link_cost(target_link_flow)

        return Constants(
            target_link_flow.astype(FLOAT),
            sparse_matrix(link_path.astype(FLOAT)),
            sparse_matrix(path_od.astype(FLOAT)),
            sparse_matrix(user_paths.astype(FLOAT).reshape(-1, 1)),
            sparse_matrix(fleet_paths.astype(FLOAT).reshape(-1, 1)),
            total_demand.astype(FLOAT),
            cost.astype(FLOAT),
            cost_gradient.astype(FLOAT),
            marginal_cost.astype(FLOAT),
        )

    def user_link_path_incidence(self):
        return self.link_path_incidence.T.multiply(self.user_paths).T

    def fleet_link_path_incidence(self):
        return self.link_path_incidence.T.multiply(self.fleet_paths).T

    def user_path_od_incidence(self):
        return self.path_od_incidence.multiply(self.user_paths)

    def fleet_path_od_incidence(self):
        return self.path_od_incidence.multiply(self.fleet_paths)

    @property
    def large_constant(self):
        return self.total_demand.max() / self.tolerance


class NumpyNamedTupleField(fields.Field):
    """Efficient serialization of a named tuple of numpy arrays"""
    encoding = 'latin-1'
    factory = None

    def _serialize(self, value, attr, obj, **kwargs):
        with BytesIO() as b:
            np.savez_compressed(b, **value._asdict())
            return b.getvalue().decode(self.encoding)

    def _deserialize(self, value, attr, data, **kwargs):
        data = dict(np.load(BytesIO(value.encode(self.encoding))).items())
        if self.factory is None:
            return data
        else:
            return self.factory(**data)

    def using_factory(self, cls):
        self.factory = cls
        return self


class VariableSchema(Schema):
    shape = fields.List(fields.Integer())
    name = fields.String()

    @pre_dump
    def extract_data(self, obj, many, **kwargs):
        """The default marshmallow behavior fails to extract relevant data.
        Do it manually."""
        return {
            'shape': list(obj.shape),
            'name': obj.name(),
        }

    @post_load
    def to_variable(self, data, **kwargs):
        shape = tuple(data['shape'])
        name = data['name']
        return cp.Variable(shape, name=name)


class VariablesSchema(Schema):
    user_link_flow = fields.Nested(VariableSchema)
    user_path_flow = fields.Nested(VariableSchema)
    user_demand = fields.Nested(VariableSchema)
    fleet_link_flow = fields.Nested(VariableSchema)
    fleet_path_flow = fields.Nested(VariableSchema)
    fleet_demand = fields.Nested(VariableSchema)
    min_path_costs = fields.Nested(VariableSchema)

    @post_load
    def to_variables(self, data, **kwargs):
        return Variables(**data)


class ProblemData(NamedTuple):
    constants: Constants
    variables: Variables


class ControlRatioSchema(Schema):
    constants = NumpyNamedTupleField().using_factory(Constants)
    variables = fields.Nested(VariablesSchema)

    @post_load
    def make_object(self, data, **kwargs):
        return ProblemData(**data)


def path_cost(link_path_incidence: np.ndarray,
              link_cost: LinkCostFunction, link_flow: np.ndarray):
    return link_path_incidence.T @ link_cost.link_cost(link_flow)


def minimal_paths(path_od_incidence: np.ndarray, path_cost: np.ndarray,
                  tolerance: float):
    minimal_paths = np.zeros_like(path_cost)
    _, n_ods = path_od_incidence.shape
    for j in range(n_ods):
        path_selector = path_od_incidence[:, j].astype(bool)
        od_path_costs = path_cost[path_selector]
        # robust shortest paths
        min_cost_bound = min(od_path_costs)*(1.0 + tolerance)
        minimal_paths[path_selector] = (od_path_costs <= min_cost_bound)
    return minimal_paths


def get_minimal_paths(link_path_incidence: np.ndarray,
                      path_od_incidence: np.ndarray,
                      link_cost: LinkCostFunction, link_flow: np.ndarray,
                      tolerance: float = 1e-8):
    return minimal_paths(
        path_od_incidence,
        path_cost(link_path_incidence, link_cost, link_flow),
        tolerance
    )
