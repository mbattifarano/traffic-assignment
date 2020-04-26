import os
import time
from traffic_assignment.mac_shp.network import (graph_from_shp,
                                                to_free_flow_travel_time,
                                                to_capacity,
                                                travel_demand)
from traffic_assignment.network.road_network import RoadNetwork
from traffic_assignment.network.demand import TravelDemand
from traffic_assignment.link_cost_function.bpr import (BPRLinkCostFunction,
                                                       BPRMarginalLinkCostFunction)
from traffic_assignment.frank_wolfe.search_direction import ShortestPathSearchDirection
from traffic_assignment.frank_wolfe.step_size import LineSearchStepSize
from traffic_assignment.frank_wolfe.solver import Solver
from traffic_assignment.network.graph import shortest_paths_scipy, unwind_path_to_link_flow
from scipy.sparse import csgraph

import networkx as nx
import numpy as np


def test_pittsburgh_so(pittsburgh_shp, numpy_cache):
    so_link_flow_key = f"pittsburgh-link_flow-so"
    initial_point = numpy_cache.get(so_link_flow_key).astype(np.float64)
    print()
    od_data_dir = os.path.join(pittsburgh_shp, 'ODmatrix')
    print(f"building graph")
    graph = graph_from_shp(pittsburgh_shp)
    print(f"building road network")
    network = RoadNetwork(graph)
    print(f"building travel demand")
    demand = TravelDemand(list(travel_demand(network, od_data_dir)))
    print(f"building link cost function")
    capacity = to_capacity(graph)
    assert np.isfinite(capacity).all()
    assert not np.isnan(capacity).any()
    free_flow = to_free_flow_travel_time(graph)
    assert np.isfinite(free_flow).all()
    assert not np.isnan(free_flow).any()
    so_link_cost_function = BPRMarginalLinkCostFunction(
        capacity=capacity,
        free_flow_travel_time=free_flow,
    )
    print("building step size")
    step_size = LineSearchStepSize(
        so_link_cost_function,
        large_initial_step=False
    )
    print(f"building search direction")
    search_direction = ShortestPathSearchDirection(
        network,
        demand
    )
    print("building solver")
    so_solver = Solver(
        step_size=step_size,
        search_direction=search_direction,
        link_cost_function=so_link_cost_function,
        initial_point=initial_point,
        max_iterations=25,
        report_interval=1,
    )
    print("done")
    assert so_solver
    iteration = so_solver.solve()
    print(f"s/it: {np.mean([i.duration for i in so_solver.iterations[1:]])}")
    numpy_cache[so_link_flow_key] = iteration.link_flow


def test_scipy_shortest_paths(pittsburgh_road_network, pittsburgh_demand):
    print("starting setup")
    t0 = time.time()
    network = pittsburgh_road_network
    nxg = network.graph
    index_node = np.array([node.name for node in network.nodes])
    N = nxg.number_of_nodes()
    L = nxg.number_of_edges()
    node_index = {n: i for i, n in enumerate(index_node)}
    # link_matrix[i,j] = index of link from index_node[i], index_node[j]
    link_matrix = -np.ones((N, N), dtype=np.uint16)
    for link in network.links:  # or network.links?
        i = node_index[link.origin.name]
        j = node_index[link.destination.name]
        link_matrix[i, j] = link.id
    csg = nx.to_scipy_sparse_matrix(nxg, nodelist=index_node)
    # ^ can all be done ONCE
    print(f"setup took {time.time()-t0:0.4f}s")
    # v must be one each shortest paths call
    t0 = time.time()
    st_array = np.zeros((pittsburgh_demand.number_of_od_pairs, 2), dtype=np.uint16)
    volume_array = np.zeros(pittsburgh_demand.number_of_od_pairs)
    # ^ could attach to demand?
    for i, d in enumerate(pittsburgh_demand):
        st_array[i] = [
            node_index[d.origin.name],
            node_index[d.destination.name],
        ]
        volume_array[i] = d.volume
    cost = 1 + np.arange(nxg.number_of_edges())
    csg.data = cost
    cost_matrix, predecessors = csgraph.shortest_path(csg,
                                                      directed=True,
                                                      return_predecessors=True)
    link_flow = unwind_path_to_link_flow(
        st_array,
        volume_array,
        L,
        link_matrix,
        predecessors.astype(np.uint16)
    )
    print(f"scipy + numba shortest path assignment {time.time()-t0:0.4f}s.")
    assert len(link_flow) == L
    t0 = time.time()
    assignment = pittsburgh_road_network.shortest_path_assignment(
        pittsburgh_demand,
        cost
    )
    print(f"igraph {time.time() - t0:0.4f}s.")
    assert np.allclose(link_flow, assignment.link_flow)


def test_shortest_paths(pittsburgh_road_network, pittsburgh_demand):
    costs = np.arange(pittsburgh_road_network.number_of_links())

    gn = pittsburgh_road_network.graph
    gi = pittsburgh_road_network._igraph

    pittsburgh_road_network._use_igraph = False
    t0 = time.time()
    nx_assignment = pittsburgh_road_network.shortest_path_assignment(
        pittsburgh_demand,
        costs
    )
    print()
    print(f"shortest paths (nx) done in {time.time()-t0:0.4f}s")

    pittsburgh_road_network._use_igraph = True
    t0 = time.time()
    ig_assignment = pittsburgh_road_network.shortest_path_assignment(
        pittsburgh_demand,
        costs
    )
    print(f"shortest paths (ig) done in {time.time()-t0:0.4f}s")
    pittsburgh_road_network._use_igraph = False

    #for p in nx_assignment.used_paths:
    #    nx_cost = sum(gn.edges[e]['weight'] for e in p.edges)
    #    ig_cost = sum(gi.es(gi.get_eids(p.edges))['weight'])
    #    assert nx_cost == ig_cost

    #assert nx_assignment.used_paths == ig_assignment.used_paths
    assert np.allclose(nx_assignment.link_flow,
                       ig_assignment.link_flow)


def test_braess_shortest_path(braess_network, braess_demand):
    costs = np.arange(braess_network.number_of_links())
    gn = braess_network.graph
    gi = braess_network._igraph

    braess_network._use_igraph = False
    t0 = time.time()
    nx_assignment = braess_network.shortest_path_assignment(
        braess_demand,
        costs
    )
    print()
    print(f"shortest paths (nx) done in {time.time()-t0:0.4f}s")

    braess_network._use_igraph = True
    t0 = time.time()
    ig_assignment = braess_network.shortest_path_assignment(
        braess_demand,
        costs
    )
    print(f"shortest paths (ig) done in {time.time()-t0:0.4f}s")
    braess_network._use_igraph = False

    print(f"igraph edges:")
    for e in gi.es:
        print(f"{e.source}->{e.target}: {e.attributes()}")
    print(f"networkx edges:")
    for u, v, data in gn.edges(data=True):
        print(f"{u}->{v}: {data}")
    print()

    for p in nx_assignment.used_paths:
        print(f"cost of path {p} (nx): {sum(gn.edges[e]['weight'] for e in p.edges)}")
    for p in ig_assignment.used_paths:
        ws = gi.es(gi.get_eids(p.edges))['weight']
        print(f"cost of path {p} ig: {sum(ws)}")

    assert nx_assignment.used_paths == ig_assignment.used_paths
    assert np.allclose(nx_assignment.link_flow,
                       ig_assignment.link_flow)


def test_tn_shortest_path(transportation_network):
    network = transportation_network.road_network()
    demand = transportation_network.travel_demand()
    costs = np.arange(network.number_of_links())
    gn = network.graph
    gi = network._igraph

    network._use_igraph = False
    t0 = time.time()
    nx_assignment = network.shortest_path_assignment(
        demand,
        costs
    )
    print()
    print(f"shortest paths (nx) done in {time.time()-t0:0.4f}s")

    network._use_igraph = True
    t0 = time.time()
    ig_assignment = network.shortest_path_assignment(
        demand,
        costs
    )
    print(f"shortest paths (ig) done in {time.time()-t0:0.4f}s")
    network._use_igraph = False

    print(f"igraph edges:")
    for e in gi.es:
        print(f"{e.source}->{e.target}: {e.attributes()}")
    print(f"networkx edges:")
    for u, v, data in gn.edges(data=True):
        print(f"{u}->{v}: {data}")
    print()

    for p in nx_assignment.used_paths:
        print(f"cost of path {p} (nx): {sum(gn.edges[e]['weight'] for e in p.edges)}")
    for p in ig_assignment.used_paths:
        ws = gi.es(gi.get_eids(p.edges))['weight']
        print(f"cost of path {p} ig: {sum(ws)}")

    assert nx_assignment.used_paths == ig_assignment.used_paths
    assert np.allclose(nx_assignment.link_flow,
                       ig_assignment.link_flow)
