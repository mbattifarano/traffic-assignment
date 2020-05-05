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
from traffic_assignment.mac_shp.network import graph_from_shp, travel_demand as shp_travel_demand
from dataclasses import dataclass
from typing import MutableMapping, Iterable
import pickle
from warnings import warn
import time

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
    #'Anaheim',
    #'Barcelona',
    #'Winnipeg',
]


@pytest.fixture(params=names)
def transportation_network(request):
    directory = TransportationNetworksDirectories.parent_directory
    name = request.param
    print(f"Reading the {name} network.")
    return TNTPProblem.from_directory(
        os.path.join(directory, name),
    )


@pytest.fixture(scope='session')
def pittsburgh_shp():
    return os.path.join(
        FIXTURES_DIR,
        'pittsburgh-network',
    )


@dataclass
class NumpyFileCache(MutableMapping):
    directory: str

    def __post_init__(self):
        try:
            os.mkdir(self.directory)
        except FileExistsError:
            pass

    def _extension(self):
        return "npy"

    def _file_of(self, k):
        return f"{self.directory}/{k}.{self._extension()}"

    def __setitem__(self, k, v) -> None:
        np.save(self._file_of(k), v)

    def __delitem__(self, k) -> None:
        os.remove(self._file_of(k))

    def __getitem__(self, k):
        try:
            return np.load(self._file_of(k))
        except FileNotFoundError:
            raise KeyError(k)

    def __len__(self) -> int:
        return len(os.listdir(self.directory))

    def __iter__(self) -> Iterable:
        return iter(os.listdir(self.directory))


@dataclass
class PickleFileCache(NumpyFileCache):

    def _extension(self):
        return "pkl"

    def __setitem__(self, key, value):
        with open(self._file_of(key), 'wb') as fp:
            pickle.dump(value, fp)

    def __getitem__(self, key):
        try:
            with open(self._file_of(key), 'rb') as fp:
                return pickle.load(fp)
        except FileNotFoundError:
            raise KeyError(key)



@pytest.fixture(scope='session')
def numpy_cache():
    return NumpyFileCache('test/artifacts/numpy_cache')


@pytest.fixture(scope='session')
def pickle_cache():
    return PickleFileCache('test/artifacts/path_set_cache')


@pytest.fixture(scope="session")
def pittsburgh_graph(pittsburgh_shp, pickle_cache):
    key = 'pittsburgh_graph'
    try:
        t0 = time.time()
        graph = pickle_cache[key]
        print(f"loaded graph from cache in {time.time()-t0:0.4f}s")
    except KeyError:
        graph = graph_from_shp(pittsburgh_shp)
        pickle_cache[key] = graph
    return graph


@pytest.fixture(scope="session")
def pittsburgh_road_network(pittsburgh_graph, pickle_cache):
    key = 'pittsburgh_road_network'
    try:
        t0 = time.time()
        net = pickle_cache[key]
        print(f"loaded network from cache in {time.time()-t0:0.4f}s")
        return net
    except KeyError:
        net = RoadNetwork(pittsburgh_graph)
        pickle_cache[key] = net
    return net


@pytest.fixture(scope="session")
def pittsburgh_demand(pittsburgh_shp, pittsburgh_road_network, pickle_cache):
    key = 'pittsburgh_demand'
    try:
        t0 = time.time()
        demand = pickle_cache[key]
        print(f"loaded demand from cache in {time.time()-t0:0.4f}s")
    except KeyError:
        od_data_dir = os.path.join(pittsburgh_shp, 'ODmatrix')
        demand = TravelDemand(list(shp_travel_demand(pittsburgh_road_network, od_data_dir)))
        pickle_cache[key] = demand
    return demand

