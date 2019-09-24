import networkx as nx
import numpy as np
import pytest

from collections import defaultdict
from traffic_assignment.frank_wolfe import (Solver,
                                            LineSearchStepSize,
                                            ShortestPathSearchDirection)
from traffic_assignment.link_cost_function.linear import LinearLinkCostFunction
from traffic_assignment.network.demand import Demand
from traffic_assignment.network.road_network import RoadNetwork


@pytest.fixture
def braess_network():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_edges_from([
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
        (2, 1),
    ])
    return RoadNetwork(g)


@pytest.fixture
def braess_cost_function():
    return LinearLinkCostFunction(
        np.array([1.0, 10.0, 10.0, 1.0, 1.0]),
        np.array([50.0, 0.0, 0.0, 50.0, 10.0])
    )


@pytest.fixture
def braess_demand(braess_network):
    return [
        Demand(
            braess_network.nodes[0],
            braess_network.nodes[3],
            6.0
        )
    ]


@pytest.fixture
def tolerance():
    return 1e-12


@pytest.fixture
def braess_solver(braess_network, braess_cost_function, braess_demand,
                  tolerance):
    return Solver(
        LineSearchStepSize(braess_cost_function),
        ShortestPathSearchDirection(
            braess_network,
            braess_demand,
        ),
        braess_cost_function,
        tolerance=tolerance,
        max_iterations=np.inf,
    )


@pytest.fixture
def braess_solution():
    return np.array([2.0, 4.0, 4.0, 2.0, 2.0])


@pytest.fixture(scope="session")
def data_store():
    results = defaultdict(list)
    return results
