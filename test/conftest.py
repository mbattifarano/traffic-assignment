import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pytest
from traffic_assignment.frank_wolfe import (Solver,
                                            LineSearchStepSize,
                                            ShortestPathSearchDirection)
from traffic_assignment.link_cost_function.linear import LinearLinkCostFunction
from traffic_assignment.network.demand import Demand, TravelDemand
from traffic_assignment.network.road_network import RoadNetwork
from traffic_assignment.tntp.solver import TNTPProblem

FIXTURES_DIR = os.path.join('test', 'fixtures')


class TransportationNetworksDirectories:
    parent_directory = os.path.join(FIXTURES_DIR, 'TransportationNetworks')
    sioux_falls = os.path.join(parent_directory, 'SiouxFalls')


@pytest.fixture
def braess_network():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    # The edge order is not relevant here but it is sorted lexographically later
    g.add_edges_from([
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 1),
        (2, 3),
    ])
    return RoadNetwork(g)


@pytest.fixture
def braess_cost_function():
    return LinearLinkCostFunction(
        np.array([ 1.0, 10.0, 10.0,  1.0,  1.0]),
        np.array([50.0,  0.0,  0.0, 10.0, 50.0])
    )


@pytest.fixture
def braess_so_cost_function():
    return LinearLinkCostFunction(
        np.array([ 2.0, 20.0, 20.0,  2.0,  2.0]),
        np.array([50.0,  0.0,  0.0, 10.0, 50.0])
    )


@pytest.fixture
def braess_demand(braess_network):
    return TravelDemand([
        Demand(
            braess_network.nodes[0],
            braess_network.nodes[3],
            6.0
        )
    ])


@pytest.fixture
def braess_demand_augmented(braess_network):
    return TravelDemand([
        Demand(
            braess_network.nodes[0],
            braess_network.nodes[3],
            6.0
        ),
        Demand(
            braess_network.nodes[0],
            braess_network.nodes[1],
            2.0
        ),
    ])


@pytest.fixture
def tolerance():
    return 1e-15


@pytest.fixture
def braess_ue_solver(braess_network, braess_cost_function, braess_demand,
                     tolerance):
    return Solver(
        LineSearchStepSize(braess_cost_function),
        ShortestPathSearchDirection(
            braess_network,
            braess_demand,
        ),
        braess_cost_function,
        tolerance=tolerance,
        max_iterations=100,
    )


@pytest.fixture
def braess_so_solver(braess_network, braess_so_cost_function, braess_demand,
                     tolerance):
    return Solver(
        LineSearchStepSize(braess_so_cost_function),
        ShortestPathSearchDirection(
            braess_network,
            braess_demand,
        ),
        braess_so_cost_function,
        tolerance=tolerance,
        max_iterations=np.inf,
    )


@pytest.fixture
def braess_ue_solution():
    return np.array([2.0, 4.0, 4.0, 2.0, 2.0])


@pytest.fixture
def braess_so_solution():
    return np.array([3.0, 3.0, 3.0, 0.0, 3.0])


@pytest.fixture(scope="session")
def data_store():
    results = defaultdict(list)
    return results


@pytest.fixture
def sioux_falls_network():
    network_file = os.path.join(
        TransportationNetworksDirectories.sioux_falls,
        'SiouxFalls_net.tntp'
    )
    with open(network_file) as fp:
        contents = fp.read()
    return contents


@pytest.fixture
def sioux_falls_trips():
    trips_file = os.path.join(
        TransportationNetworksDirectories.sioux_falls,
        'SiouxFalls_trips.tntp'
    )
    with open(trips_file) as fp:
        contents = fp.read()
    return contents


@pytest.fixture
def sioux_falls_solution():
    solution_file = os.path.join(
        TransportationNetworksDirectories.sioux_falls,
        'SiouxFalls_flow.tntp'
    )
    with open(solution_file) as fp:
        contents = fp.read()
    return contents


names = [
    'SiouxFalls',
    'Anaheim',
    'Barcelona',
    'Winnipeg',
]


@pytest.fixture(params=names)
def transportation_network(request):
    directory = TransportationNetworksDirectories.parent_directory
    name = request.param
    return TNTPProblem.from_directory(os.path.join(directory, name))
