import os
import time
from traffic_assignment.mac_shp.network import (graph_from_shp,
                                                to_free_flow_travel_time,
                                                to_capacity,
                                                travel_demand)
from traffic_assignment.network.road_network import RoadNetwork, _to_igraph
from traffic_assignment.network.demand import TravelDemand
from traffic_assignment.link_cost_function.bpr import (BPRLinkCostFunction,
                                                       BPRMarginalLinkCostFunction)
from traffic_assignment.frank_wolfe.search_direction import ShortestPathSearchDirection
from traffic_assignment.frank_wolfe.step_size import LineSearchStepSize
from traffic_assignment.frank_wolfe.solver import Solver
from traffic_assignment.network.graph import shortest_paths_scipy, unwind_path_to_link_flow, assign_to_links, assign_all_to_links
from traffic_assignment.network.shortest_path import _igraph_to_numba_weights, _igraph_to_numba_adjdict, _all_paths_shorter_than
from scipy.sparse import csgraph
from traffic_assignment.utils import Timer

from traffic_assignment.control_ratio_range.lp import MinimumFleetControlRatio, HeuristicStatus
from traffic_assignment.control_ratio_range.utils import HeuristicConstants, HeuristicVariables
import cvxpy as cp

import networkx as nx
import numpy as np
import numba


def test_pittsburgh_fo_mcr(pittsburgh_graph, pittsburgh_road_network, pittsburgh_demand, numpy_cache):
    so_link_flow_key = f"pittsburgh-link_flow-so"
    so_link_flow = numpy_cache.get(so_link_flow_key).astype(np.float64)
    graph = pittsburgh_graph
    network = pittsburgh_road_network
    demand = pittsburgh_demand
    print("building link cost function")
    link_cost = BPRLinkCostFunction(
        capacity=to_capacity(graph),
        free_flow_travel_time=to_free_flow_travel_time(graph),
    )
    print("building heuristic constants")
    constants = HeuristicConstants.from_network(
        network=network,
        demand=demand,
        link_cost=link_cost,
        known_paths=None,
        target_link_flow=so_link_flow,
    )
    print("saving constants")
    constants.save('test/artifacts/pittsburgh-network-fomcr-constants')
    print("creating variables")
    variables = HeuristicVariables.from_constants(constants)
    print("creating problem")
    mfcr = MinimumFleetControlRatio(
        network=network,
        demand=demand,
        constants=constants,
        variables=variables,
    )
    print(f"Fleet paths: {mfcr.constants.fleet_paths.sum()} usable; {(~mfcr.constants.fleet_paths).sum()} un-usable")
    timer = Timer().start()
    _bin_edges = np.linspace(0.0, 1.0, 11)
    _bin_labels = [f"[{a:0.1f}, {b:0.1f}]" for a, b in zip(_bin_edges, _bin_edges[1:])]
    upper_bounds = []
    lower_bounds = []
    for i, (status, value) in enumerate(mfcr.heuristic()):
        if status is HeuristicStatus.ADD_UNUSABLE_PATHS:
            upper_bounds = []
            lower_bounds.append((value, i))
        else:
            upper_bounds.append((value, i))
        fleet_od_volume = mfcr.fleet_demand.value
        user_od_volume = mfcr.user_demand.value
        _control_ratio = 100 * mfcr.variables.fleet_path_flow.value.sum() / constants.total_demand.sum()
        print(f"{i+1} ({timer.time_elapsed():0.2f} seconds): LP terminated with {status} and objective value {100 *value:0.2f}% and control ratio {(_control_ratio):0.4f}%")
        fleet_marginal_path_cost = (mfcr.constants.link_path_incidence.T @ (
                mfcr.constants.link_cost + cp.multiply(mfcr.fleet_link_flow, mfcr.constants.link_cost_gradient)
        )).value

        assert np.allclose(fleet_marginal_path_cost, mfcr.fleet_marginal_path_cost.value)
        assert (abs((fleet_od_volume + user_od_volume - constants.total_demand)) <= 0.05).all()
        assert (abs((mfcr.fleet_link_flow + mfcr.user_link_flow - mfcr.constants.target_link_flow).value) <= 0.05).all()
        vi = cp.multiply(
            mfcr.variables.fleet_path_flow,
            mfcr.fleet_marginal_path_cost - mfcr.min_fleet_path_costs
        ).value
        print(f"vi in range [{vi.min()}, {vi.max()}]")
        assert (abs(vi) <= 1e-3).all(),  f"vi in range [{vi.min()}, {vi.max()}]"

        fleet_od_ratio = fleet_od_volume / constants.total_demand
        counts, _ = np.histogram(fleet_od_ratio, _bin_edges)
        _print_hist = "\n".join(
            f"{_bin}: {_count:d}" for _bin, _count in zip(_bin_labels, counts))
        # print(f"fleet ratios:\n{_print_hist}")

    print(f"Fleet paths: {mfcr.constants.fleet_paths.sum()} usable; {(~mfcr.constants.fleet_paths).sum()} un-usable; total = {len( mfcr.constants.known_paths)}")
    best_value, best_i = min(upper_bounds)
    print(f"Best objective value {best_value} on iteration {best_i}")
    print(f"Lower bounds {lower_bounds}")


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
        max_iterations=10,  # approx 26 s/it
        report_interval=10,
    )
    print("done")
    assert so_solver
    iteration = so_solver.solve()
    print(f"s/it: {np.mean([i.duration for i in so_solver.iterations[1:]])}")
    numpy_cache[so_link_flow_key] = iteration.link_flow


def test_least_cost_paths(pittsburgh_road_network, pittsburgh_demand):
    cost = np.arange(pittsburgh_road_network.number_of_links()) + 1
    ig = pittsburgh_road_network._igraph
    _weight = pittsburgh_road_network.WEIGHT_KEY
    ig.es[_weight] = cost
    i = 70
    _demand = pittsburgh_demand.demand[i]
    s = ig['node_index'][_demand.origin.name]
    t = ig['node_index'][_demand.destination.name]
    c = ig.shortest_paths(s, t, weights=_weight)[0][0]
    print(f"cost {s}->{t} = {c}")

    adjdict = _igraph_to_numba_adjdict(ig)
    weights = _igraph_to_numba_weights(ig, _weight)

    paths = list(
        _all_paths_shorter_than(
            adjdict, s, t, weights, c, ig['index_node'], debug=True
        ))

    #max_i = 100
    #demand = TravelDemand(pittsburgh_demand.demand[:max_i])
    #demand = pittsburgh_demand
    #n = len(demand)
    #print(f"Running shortish paths with {n} of {len(pittsburgh_demand)} OD pairs.")
    #t0 = time.time()
    #paths = list(pittsburgh_road_network.least_cost_paths(demand, cost))
    #t = time.time()-t0
    #print(f"took {t:0.4f}s ({t/n:0.4f}s/od, {t/len(paths):0.4f}s/path).")


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
    ig = _to_igraph(network.nodes, network.links)
    # ^ can all be done ONCE
    print(f"setup took {time.time()-t0:0.4f}s")
    # v must be one each shortest paths call
    #t0 = time.time()
    #st_array = np.zeros((pittsburgh_demand.number_of_od_pairs, 2), dtype=np.uint16)
    #volume_array = np.zeros(pittsburgh_demand.number_of_od_pairs)
    ## ^ could attach to demand?
    #for i, d in enumerate(pittsburgh_demand):
    #    st_array[i] = [
    #        node_index[d.origin.name],
    #        node_index[d.destination.name],
    #    ]
    #    volume_array[i] = d.volume
    cost = 1 + np.arange(nxg.number_of_edges())
    #csg.data = cost
    #t1 = time.time()
    #cost_matrix, predecessors = csgraph.shortest_path(csg,
    #                                                  directed=True,
    #                                                  return_predecessors=True)
    #print(f"scipy took {time.time()-t1:0.4f}s")
    #t2 = time.time()
    #link_flow = unwind_path_to_link_flow(
    #    st_array,
    #    volume_array,
    #    L,
    #    link_matrix,
    #    predecessors.astype(np.uint16)
    #)
    #print(f"unwinding took {time.time()-t2:0.4f}s")
    #print(f"scipy + numba shortest path assignment {time.time()-t0:0.4f}s.")
    #assert len(link_flow) == L

    t0 = time.time()
    ig.es['weight'] = cost
    link_flow_2 = np.zeros(L)
    _ig_time = 0
    _assign_time = 0
    for origin, dest_volumes in pittsburgh_demand.origin_based_index.items():
        s = ig['node_index'][origin]
        ts = [ig['node_index'][dest] for dest in dest_volumes]
        vs = np.array(list(dest_volumes.values()))
        t1 = time.time()
        paths = numba.typed.List(
            np.array(p, dtype=np.uint16)
            for p in ig.get_shortest_paths(s, ts, weights='weight')
        )
        _ig_time += time.time()-t1
        t2 = time.time()
        assign_all_to_links(
            ig['link_matrix'],
            paths,
            vs,
            link_flow_2,
        )
        _assign_time += time.time() - t2
    print(f"igraph ({_ig_time:0.4f}s) + numba assign ({_assign_time:0.4f}s): {time.time()-t0:0.4f}s")

    t0 = time.time()

    assignment = pittsburgh_road_network.shortest_path_assignment(
        pittsburgh_demand,
        cost
    )
    print(f"igraph {time.time() - t0:0.4f}s.")
    # assert np.allclose(link_flow, assignment.link_flow)
    assert np.allclose(link_flow_2, assignment.link_flow)


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
    print("starting test")
    network = transportation_network.road_network()
    demand = transportation_network.trips.to_demand(network)
    costs = np.arange(network.number_of_links()) + 1
    gn = network.graph
    gi = network._igraph

    print("Running shortest paths")
    t0 = time.time()
    nx_assignment = network.shortest_path_assignment(
        demand,
        costs
    )
    print()
    print(f"shortest paths done in {time.time()-t0:0.4f}s")

    t0 = time.time()
    ig_assignment = network.shortest_path_assignment(
        demand,
        costs
    )
    print(f"shortest paths done in {time.time()-t0:0.4f}s")

    assert np.allclose(nx_assignment.link_flow,
                       ig_assignment.link_flow)
