from __future__ import annotations

from enum import Enum
from typing import NamedTuple
from traffic_assignment.network.road_network import Network
from traffic_assignment.network.demand import TravelDemand
from traffic_assignment.link_cost_function.base import LinkCostFunction
from marshmallow import Schema, fields, post_load, pre_dump
from io import BytesIO
import json

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


FLOAT = np.float32
BINARY = np.uint8  # Not really but as small as we are gonna get


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
                     target_link_flow: np.ndarray, tolerance: float = 1e-4):
        link_path, path_od, _ = network.path_incidences(demand)
        total_demand = demand.to_array()
        user_paths = get_minimal_paths(link_path, path_od, link_cost,
                                       target_link_flow, tolerance)
        fleet_paths = get_minimal_paths(link_path, path_od, marginal_link_cost,
                                        target_link_flow, tolerance)
        return Constants(
            target_link_flow.astype(FLOAT),
            link_path.astype(BINARY),
            path_od.astype(BINARY),
            user_paths.astype(BINARY),
            fleet_paths.astype(BINARY),
            total_demand.astype(FLOAT),
        )

    def user_link_path_incidence(self):
        return self.link_path_incidence * self.user_paths

    def fleet_link_path_incidence(self):
        return self.link_path_incidence * self.fleet_paths

    def user_path_od_incidence(self):
        return self.path_od_incidence.T * self.user_paths

    def fleet_path_od_incidence(self):
        return self.path_od_incidence.T * self.fleet_paths


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
    return minimal_paths.astype(np.uint8)


def get_minimal_paths(link_path_incidence: np.ndarray,
                      path_od_incidence: np.ndarray,
                      link_cost: LinkCostFunction, link_flow: np.ndarray,
                      tolerance: float = 1e-8):
    return minimal_paths(
        path_od_incidence,
        path_cost(link_path_incidence, link_cost, link_flow),
        tolerance
    )
