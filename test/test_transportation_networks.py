import os
import time
from dataclasses import dataclass
from typing import MutableMapping, Iterable
import pickle

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from traffic_assignment.utils import Timer
from textwrap import dedent
import pandas as pd
from traffic_assignment.control_ratio_range.lp import HeuristicStatus


def test_networks(transportation_network):
    tol = 5e-6
    _g = transportation_network.network.to_networkx_graph()
    # assert sorted(_g.nodes) == list(range(_g.number_of_nodes()))
    ue_solver = transportation_network.ue_solver(tolerance=tol)
    ue_solver.report_interval = 500
    result = ue_solver.solve()
    actual_flow = result.link_flow
    expected_flow = transportation_network.solution.link_flow()

    actual_score = ue_solver.objective_value(actual_flow)
    expected_score = ue_solver.objective_value(expected_flow)

    total_error = [np.linalg.norm(it.link_flow - expected_flow)
                   for it in ue_solver.iterations]

    plt.figure()
    plt.semilogy(ue_solver.gaps)
    plt.savefig(f"test/artifacts/{transportation_network.name}.gaps.{time.time()}.png")
    plt.close()

    plt.figure()
    plt.semilogy(total_error)
    plt.savefig(f"test/artifacts/{transportation_network.name}.errors.{time.time()}.png")
    plt.close()

    print(f"""
    {transportation_network.name}
    {'='*len(transportation_network.name)}
    objective value: {actual_score}
    best known objective value: {expected_score} (difference = {100*(actual_score - expected_score)/expected_score}%)
    relative gap: {result.gap}
    iterations: {result.iteration}
    it/s: {result.iteration / sum(it.duration for it in ue_solver.iterations)}
    max error: {abs(expected_flow - actual_flow).max()}
    average error: {abs(expected_flow - actual_flow).mean()}
    """)
    assert abs((expected_flow - actual_flow) / expected_flow).max() <= 1e-3


def test_consistency(transportation_network):
    ue_link_flow = transportation_network.solution.link_flow()
    link_cost_fn = transportation_network.network.to_link_cost_function()
    link_cost = link_cost_fn.link_cost(ue_link_flow)

    road_network = transportation_network.road_network()
    travel_demand = transportation_network.travel_demand()
    network_links = road_network.links
    solution_links = transportation_network.solution.links
    assert len(network_links) == len(solution_links)

    for network_link, solution_link in zip(network_links, solution_links):
        network_edge = network_link.origin.name, network_link.destination.name
        solution_edge = solution_link.origin, solution_link.destination
        assert network_edge == solution_edge

    all_paths = road_network.get_all_paths(travel_demand)
    link_path, path_od, path_index = road_network.path_incidences(travel_demand)
    link_index = {(l.origin.name, l.destination.name): i
                  for i, l in enumerate(road_network.links)}
    path_cost = link_path.T @ link_cost
    reverse_path_index = {i: p for p, i in path_index.items()}
    for j, d in enumerate(travel_demand.demand):
        orgn = d.origin
        dest = d.destination
        expected_paths = set(all_paths[(orgn, dest)])
        for p in expected_paths:
            assert p.nodes[0] == orgn.name
            assert p.nodes[-1] == dest.name
        od_paths = path_od[:, j]
        path_indices, = np.where(od_paths == 1)
        assert len(expected_paths) == len(path_indices)
        for i in path_indices:
            p = reverse_path_index[i]
            assert path_cost[i] == pytest.approx(sum(
                link_cost[link_index[(u, v)]]
                for (u, v) in p.edges
            ))
            assert p in expected_paths
            assert p.nodes[0] == orgn.name
            assert p.nodes[-1] == dest.name


def test_ue_path_cost_condition(transportation_network, numpy_cache, pickle_cache):
    path_incidence_cache = numpy_cache
    path_set_cache = pickle_cache
    best_ue_link_flow = transportation_network.solution.link_flow()
    ue_link_flow_key = f"{transportation_network.name}-ue_link_flow"
    warm_start_key = f"{ue_link_flow_key}-final"
    path_set_key = f"{transportation_network.name}-ue_path_set"
    print(f"Finding UE link flow for {warm_start_key}")
    initial_point = path_incidence_cache.get(warm_start_key)
    path_set = set([]) if initial_point is None else path_set_cache[path_set_key]
    ue_solver = transportation_network.ue_solver(
        tolerance=1e-15,
        max_iterations=5001,
        initial_point=path_incidence_cache.get(warm_start_key),
        path_set=path_set,
        large_initial_step=False,
    )
    print(f"Solving UE link flow")
    with timer():
        final_iteration = ue_solver.solve()
        ue_solution = ue_solver.best_iteration()
    print(f"Solved UE link flow (gap={ue_solution.absolute_gap}, relative gap={ue_solution.gap})")
    ue_link_flow = ue_solution.link_flow
    path_incidence_cache[ue_link_flow_key] = ue_link_flow
    path_incidence_cache[warm_start_key] = final_iteration.link_flow
    path_set_cache[path_set_key] = ue_solver.path_set

    link_cost_fn = transportation_network.network.to_link_cost_function()
    road_network = transportation_network.road_network()
    travel_demand = transportation_network.travel_demand()
    demand_vec = travel_demand.to_array()

    link_path_incidence, path_od_incidence = road_network.path_set_incidences(
        travel_demand,
        ue_solver.path_set,
    )
    trip_path_incidence = path_od_incidence.T

    n_links, n_paths = link_path_incidence.shape
    n_trips, _ = trip_path_incidence.shape
    print(f"Found {n_paths} paths connected {n_trips} origin destination pairs.")
    link_cost = link_cost_fn.link_cost(ue_link_flow)
    path_cost = link_path_incidence.T.dot(link_cost)

    path_cost_ratio = np.zeros(n_paths)
    for i, trips_paths in enumerate(trip_path_incidence):
        trip_path_mask = trips_paths.toarray().squeeze().astype(bool)
        trip_path_cost = path_cost[trip_path_mask]
        min_trip_cost = trip_path_cost.min()
        path_cost_ratio[trip_path_mask] = trip_path_cost / min_trip_cost

    usable_paths = path_cost_ratio <= (1 + 1e-3)

    ue_path_flow = cp.Variable(n_paths, name="ue_path_flow")
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(cp.hstack([
            link_path_incidence @ ue_path_flow - ue_link_flow,
            trip_path_incidence @ ue_path_flow - demand_vec,
        ]))),
        [
            ue_path_flow >= 0.0,
            ue_path_flow[~usable_paths] == 0.0,
        ]
    )
    problem.solve(solver=cp.GUROBI)
    assert problem.status == cp.OPTIMAL

    link_flow_est = link_path_incidence @ ue_path_flow.value
    demand_est = trip_path_incidence @ ue_path_flow.value
    link_flow_error = abs(link_flow_est - ue_link_flow)
    demand_error = abs(demand_est - demand_vec)

    def _mape(expected, actual):
        return abs((expected - actual)/actual).mean()

    ps = [0.25, 0.5, 0.75, 0.95, 1.0]
    print(f"link flow error ({ps} percentiles): {[np.quantile(100*link_flow_error/ue_link_flow, p) for p in ps]}")
    print(f"demand error ({ps} percentiles): {[np.quantile(100*demand_error/demand_vec, p) for p in ps]}")
    print(f"link flow mape (wrt best known): {100*_mape(best_ue_link_flow, link_flow_est)}%, {100*_mape(best_ue_link_flow, ue_link_flow)}%")
    print(f"Our usable path set based estimate is better than FW: {_mape(best_ue_link_flow, link_flow_est) <= _mape(best_ue_link_flow, ue_link_flow)}")

    assert (link_flow_error/ue_link_flow).max() <= 1e-3
    assert (demand_error/demand_vec).max() <= 1e-3
    assert _mape(best_ue_link_flow, ue_link_flow) <= 1e-5


def test_so_path_cost_condition(transportation_network, numpy_cache, pickle_cache):
    path_incidence_cache = numpy_cache
    path_set_cache = pickle_cache
    so_link_flow_key = f"{transportation_network.name}-so_link_flow"
    warm_start_key = f"{so_link_flow_key}-final"
    path_set_key = f"{transportation_network.name}-so_path_set"
    print(f"Finding SO link for for {warm_start_key}")
    so_solver = transportation_network.so_solver(
        tolerance=1e-15,
        max_iterations=5001,
    )
    so_solver.report_interval = 100
    #    initial_point=path_incidence_cache.get(warm_start_key),
    #    path_set=path_set_cache.get(path_set_key, set([])),
    #    large_initial_step=False
    #)
    print("Solving SO link flow")
    with timer():
        final_iteration = so_solver.solve()
        so_solution = so_solver.best_iteration()
    print(f"Solved SO link flow (gap={so_solution.absolute_gap}, relative gap = {so_solution.gap})")
    so_link_flow = so_solution.link_flow
    path_incidence_cache[so_link_flow_key] = so_link_flow
    path_incidence_cache[warm_start_key] = final_iteration.link_flow
    path_set_cache[path_set_key] = so_solver.path_set
    link_cost_fn = transportation_network.network.to_marginal_link_cost_function()
    road_network = transportation_network.road_network()
    travel_demand = transportation_network.travel_demand()
    link_path_incidence, path_od_incidence = road_network.path_set_incidences(travel_demand, so_solver.path_set)
    trip_path_incidence = path_od_incidence.T
    demand_array = travel_demand.to_array()
    _, n_paths = link_path_incidence.shape
    _, n_ods = path_od_incidence.shape
    print(f"Found {n_paths} paths connecting {n_ods} origin destination pairs.")
    so_path_flow = cp.Variable(n_paths, name="so_path_flow")
    so_path_cost = link_path_incidence.T @ link_cost_fn.link_cost(so_link_flow)
    problem = cp.Problem(
        cp.Minimize(so_path_cost @ so_path_flow),
        [
            so_path_flow >= 0,
            link_path_incidence @ so_path_flow == so_link_flow,
            trip_path_incidence @ so_path_flow == demand_array,
        ]
    )
    problem.solve(solver=cp.GUROBI)
    path_demand = trip_path_incidence.T @ demand_array
    path_flow = so_path_flow.value.copy()

    link_flow_est = link_path_incidence @ path_flow
    demand_est = trip_path_incidence @ path_flow

    cost = link_cost_fn.link_cost(so_link_flow)
    path_costs = link_path_incidence.T @ cost

    min_od_path_cost = np.zeros(n_ods)
    max_od_path_cost_diff = np.zeros(n_ods)
    path_cost_ratio = np.zeros(n_paths)
    path_flow_fraction = np.zeros(n_paths)
    trip_ids = np.zeros(n_paths)
    for i, trip_paths in enumerate(trip_path_incidence):
        trip_path_mask = trip_paths.toarray().squeeze().astype(bool)
        trip_path_cost = path_costs[trip_path_mask]
        min_od_path_cost[i] = trip_path_cost.min()
        trip_ids[trip_path_mask] = i
        path_cost_ratio[trip_path_mask] = trip_path_cost / min_od_path_cost[i]
        path_flow_fraction[trip_path_mask] = path_flow[trip_path_mask] / path_flow[trip_path_mask].sum()
        trip_path_cost_diff = trip_path_cost - trip_path_cost.min()
        max_od_path_cost_diff[i] = trip_path_cost_diff.max()

    df = pd.DataFrame({
        'trip_id': trip_ids,
        'path_flow_fraction': path_flow_fraction,
        'path_cost_ratio': path_cost_ratio
    })
    print(df[(df.path_flow_fraction > 0.0) & (df.path_cost_ratio > 1)].sort_values(by='path_cost_ratio', ascending=False).head(10))

    print(df[df.path_flow_fraction > 0.0].groupby(by='trip_id').path_cost_ratio.describe().sort_values(by='max', ascending=False).head(10))

    usable_paths = path_cost_ratio <= (1 + 1e-3)
    augmented_problem = cp.Problem(
        cp.Minimize(cp.sum_squares(cp.hstack([
            link_path_incidence @ so_path_flow - so_link_flow,
            trip_path_incidence @ so_path_flow - demand_array
            ]
        ))),
            [
            so_path_flow >= 0,
            so_path_flow[~usable_paths] == 0.0
            ]
    )
    augmented_problem.solve(solver=cp.GUROBI)
    assert augmented_problem.status == cp.OPTIMAL

    link_flow_est = link_path_incidence @ so_path_flow.value
    demand_est = trip_path_incidence @ so_path_flow.value
    link_flow_error = abs(link_flow_est - so_link_flow)
    demand_error = abs(demand_est - demand_array)
    ps = [0.25, 0.5, 0.75, 0.95, 1.0]
    print(f"link flow error ({ps} percentiles): {[np.quantile(100*link_flow_error/so_link_flow, p) for p in ps]}")
    print(f"demand error ({ps} percentiles): {[np.quantile(100*demand_error/demand_array, p) for p in ps]}")

    assert abs((link_flow_est - so_link_flow)/so_link_flow).max() <= 1e-3
    assert abs((demand_est - demand_array)/demand_array).max() <= 1e-3


def test_lower_control_ratio(transportation_network, numpy_cache):
    path_incidence_cache = numpy_cache
    print("testing lower control ratio")
    ue_link_flow = transportation_network.solution.link_flow()
    so_link_flow_key = f"{transportation_network.name}-so_link_flow"
    so_link_flow = path_incidence_cache[so_link_flow_key]
    link_cost_fn = transportation_network.network.to_link_cost_function()
    ue_total_cost = link_cost_fn.link_cost(ue_link_flow).dot(ue_link_flow)
    so_total_cost = link_cost_fn.link_cost(so_link_flow).dot(so_link_flow)
    print(f"Total cost at UE = {ue_total_cost:0.2f}, total cost at SO = {so_total_cost:0.2f}")
    lower_control_ratio = transportation_network.lower_control_ratio(ue_link_flow)
    lcr_lp = lower_control_ratio.problem()
    variables = lower_control_ratio.variables
    constants = lower_control_ratio.constants
    print("Solving problem.")
    timer = Timer()
    timer.start()
    solver_tolerance = 1e-4
    lcr_lp.solve(solver='SCS', eps=solver_tolerance, verbose=True)
    print(f"Terminated in {timer.time_elapsed():0.2f} (s).")
    stats = lcr_lp.solver_stats
    total_demand = transportation_network.travel_demand().to_array().sum()
    print(f"""
    {transportation_network.name}
    {'='*len(transportation_network.name)}
    solver terminated with status '{lcr_lp.status}' in {stats.solve_time} (s).
    setup time: {stats.setup_time} (s)
    eps: {solver_tolerance}
    number of iterations: {stats.num_iters}
     
    lower control ratio: {lcr_lp.value / total_demand:0.4f} ({lcr_lp.value:0.2f} total).
    
    fleet path flow is between [{min(variables.fleet_link_flow.value)}, {max(variables.fleet_link_flow.value)}]
    total fleet flow is {variables.fleet_demand.value.sum()} (== {variables.fleet_path_flow.value.sum()})
    
    user path flow is between [{min(variables.user_link_flow.value)}, {max(variables.user_link_flow.value)}]
    total user flow is {variables.user_demand.value.sum()} (== {variables.user_path_flow.value.sum()}
    
    maximum link flow error: {(variables.fleet_link_flow + variables.user_link_flow - constants.target_link_flow).value.max()}
    maximum demand error: {(variables.fleet_demand + variables.user_demand - constants.total_demand).value.max()}
    """)
    path_incidence_cache[f"{transportation_network.name}-maximum_ue_ratio-fleet_demand"] = variables.fleet_demand.value
    assert lcr_lp.status == 'optimal'


def test_restricted_lower_control_ratio(transportation_network, numpy_cache):
    path_incidence_cache = numpy_cache
    print("testing restricted lower control ratio")
    ue_link_flow = transportation_network.solution.link_flow()
    mcr_fleet_demand = path_incidence_cache[f"{transportation_network.name}-MCR-fleet_demand"]
    lower_control_ratio = transportation_network.restricted_lower_control_ratio(
        ue_link_flow,
        mcr_fleet_demand,
    )
    lcr_lp = lower_control_ratio.problem()
    variables = lower_control_ratio.variables
    constants = lower_control_ratio.constants
    print("Solving problem.")
    timer = Timer()
    timer.start()
    solver_tolerance = 1e-4
    lcr_lp.solve(solver='SCS', eps=solver_tolerance, verbose=True)
    print(f"Terminated in {timer.time_elapsed():0.2f} (s).")
    assert lcr_lp.status.startswith('optimal')
    stats = lcr_lp.solver_stats
    total_demand = constants.total_demand.sum()
    net_difference = (
        variables.fleet_demand.value - mcr_fleet_demand
    )
    absolute_difference = abs(net_difference)
    print(f"""
    {transportation_network.name}
    {'=' * len(transportation_network.name)}
    solver terminated with status '{lcr_lp.status}' in {stats.solve_time} (s).
    setup time: {stats.setup_time} (s)
    eps: {solver_tolerance}
    number of iterations: {stats.num_iters}

    objective value: {lcr_lp.value} ({100 * lcr_lp.value / total_demand:0.2f} %)
    total difference: {absolute_difference.sum()} ({100* absolute_difference.sum() / total_demand: 0.2f} %)
    net difference: {net_difference.sum()} ({100 * net_difference.sum() / total_demand}
    total fleet demand: {variables.fleet_demand.value.sum()}
    MCR total fleet demand: {mcr_fleet_demand.sum()}
    how many od pairs have reduced fleet demand: {(variables.fleet_demand.value <= mcr_fleet_demand).sum()} (out of {len(mcr_fleet_demand)})
    total demand: {total_demand}

    fleet path flow is between [{min(variables.fleet_link_flow.value)}, {max(
        variables.fleet_link_flow.value)}]
    total fleet flow is {variables.fleet_demand.value.sum()} (== {variables.fleet_path_flow.value.sum()})

    user path flow is between [{min(variables.user_link_flow.value)}, {max(
        variables.user_link_flow.value)}]
    total user flow is {variables.user_demand.value.sum()} (== {variables.user_path_flow.value.sum()}

    maximum link flow error: {(
                variables.fleet_link_flow + variables.user_link_flow - constants.target_link_flow).value.max()}
    maximum demand error: {(
                variables.fleet_demand + variables.user_demand - constants.total_demand).value.max()}
    """)
    #path_incidence_cache[
    #    f"{transportation_network.name}-maximum_ue_ratio-fleet_demand"] = variables.fleet_demand.value


def test_upper_control_ratio(transportation_network, numpy_cache):
    path_incidence_cache = numpy_cache
    so_link_flow_key = f"{transportation_network.name}-so_link_flow"
    try:
        so_link_flow = path_incidence_cache[so_link_flow_key]
        print("cache hit. loaded SO link flow")
    except KeyError:
        so_solver = transportation_network.so_solver(1e-7)
        print("Solving SO link flow")
        with timer():
            so_solution = so_solver.solve()
        print(f"Solved SO link flow (gap={so_solution.gap})")
        so_link_flow = so_solution.link_flow
        path_incidence_cache[so_link_flow_key] = so_link_flow
    upper_control_ratio = transportation_network.upper_control_ratio(so_link_flow,
                                                                     1e-5)
    total_demand = transportation_network.travel_demand().total_demand()
    ucr_lp = upper_control_ratio.problem()
    print("Solving upper control ratio.")
    solver_tolerance = 1e-4
    with timer():
        ucr_lp.solve(solver='SCS', eps=solver_tolerance, verbose=True)
    stats = ucr_lp.solver_stats
    variables = upper_control_ratio.variables
    constants = upper_control_ratio.constants

    print(f"""
       {transportation_network.name}
       {'=' * len(transportation_network.name)}
       solver terminated with status '{ucr_lp.status}' in {stats.solve_time} (s).
       setup time: {stats.setup_time} (s)
       eps: {solver_tolerance}
       number of iterations: {stats.num_iters}

       upper control ratio: {ucr_lp.value / total_demand:0.4f} ({ucr_lp.value:0.2f} total).

       fleet path flow is between [{min(
        variables.fleet_link_flow.value)}, [{max(
        variables.fleet_link_flow.value)}]
       user path flow is between [{min(variables.user_link_flow.value)}, {max(
        variables.user_link_flow.value)}]
       maximum link flow error: {(
                variables.fleet_link_flow + variables.user_link_flow - constants.target_link_flow).value.max()}
       maximum demand error: {(
                variables.fleet_demand + variables.user_demand - constants.total_demand).value.max()}
       """)
    assert ucr_lp.status == 'optimal'
    path_incidence_cache[f"{transportation_network.name}-MCR-fleet_demand"] = variables.fleet_demand.value


from contextlib import contextmanager

@contextmanager
def timer():
    t = Timer()
    t.start()
    yield
    print(f"{t.time_elapsed():0.2f} seconds elapsed.")


def test_marginal_cost_expression(transportation_network):
    n_links = len(transportation_network.network.links)
    fleet_link_flow = cp.Variable(n_links, name="fleet_link_flow")
    ue_link_flow = transportation_network.solution.link_flow()
    link_cost_fn = transportation_network.network.to_marginal_link_cost_function(fleet_link_flow)
    fleet_link_cost = link_cost_fn.link_cost(ue_link_flow)
    assert fleet_link_cost.shape == (n_links,)


# TODO: implement path removal:
#   each iteration remove paths that are unused by the fleet from fleet paths
# TODO: implement path generation:
#   each iteration add paths that are min fleet marginal cost (as unusable by
#   the fleet)
def test_minimum_fleet_control_ratio(transportation_network, numpy_cache, pickle_cache):
    path_incidence_cache = numpy_cache
    path_set_cache = pickle_cache
    #user_path_set_key = f"{transportation_network.name}-ue_path_set"
    so_path_set_key = f"{transportation_network.name}-so_path_set"
    fleet_paths = list(path_set_cache[so_path_set_key])
    so_link_flow_key = f"{transportation_network.name}-so_link_flow"
    so_link_flow = path_incidence_cache[so_link_flow_key]
    print("Creating problem data")
    timer = Timer().start()
    mfcr = transportation_network.minimum_fleet_control_ratio(so_link_flow,
                                                              fleet_paths)
    print(f"Created problem data in {timer.time_elapsed():0.2f} seconds")
    variables = mfcr.variables
    constants = mfcr.constants
    total_demand = mfcr.constants.total_demand.sum()

    timer.start()
    mfcr_lp = mfcr.problem()
    timer.start()
    print("Solving problem")
    mfcr_lp.solve(solver=cp.GUROBI, verbose=True)
    print(f"Solved problem in {timer.time_elapsed():0.2f} seconds.")
    assert mfcr_lp.status == cp.OPTIMAL
    stats = mfcr_lp.solver_stats
    user_demand = (constants.trip_path_incidence @ variables.user_path_flow).value
    fleet_demand = (constants.trip_path_incidence @ variables.fleet_path_flow).value

    user_link_flow = (constants.link_path_incidence @ variables.user_path_flow).value
    fleet_link_flow = (constants.link_path_incidence @ variables.fleet_path_flow).value
    unused_feasible_paths = variables.fleet_path_flow.value[constants.fleet_paths] == 0.0
    fleet_link_cost = constants.fleet_link_cost(fleet_link_flow)
    # TODO: compute shortest paths based on fleet link cost
    #least_cost_paths = transportation_network.network.least_cost_path_indices()
    print(f"""
       {transportation_network.name}
       {'=' * len(transportation_network.name)}
       solver terminated with status '{mfcr_lp.status}' in {stats.solve_time} (s).
       setup time: {stats.setup_time} (s)
       number of iterations: {stats.num_iters}

       upper control ratio: {100* mfcr_lp.value / total_demand:0.4f}% ({mfcr_lp.value:0.2f} total).

       There are {unused_feasible_paths.sum()} unused feasible fleet paths (out of {constants.fleet_paths.sum()})
       fleet path flow is between [{fleet_link_flow.min()}, {fleet_link_flow.max()}]
       user path flow is between [{user_link_flow.min()}, {user_link_flow.max()}]
       maximum link flow error: {abs(
            fleet_link_flow + user_link_flow - constants.target_link_flow).max()}
       maximum demand error: {abs(
            fleet_demand + user_demand - constants.total_demand).max()}
       """)
    assert mfcr_lp.status == 'optimal'


def test_heuristic_fo_mcr(transportation_network, numpy_cache, pickle_cache):
    path_incidence_cache = numpy_cache
    path_set_cache = pickle_cache
    so_path_set_key = f"{transportation_network.name}-so_path_set"
    fleet_paths = list(path_set_cache[so_path_set_key])
    so_link_flow_key = f"{transportation_network.name}-so_link_flow"
    so_link_flow = path_incidence_cache[so_link_flow_key]
    timer = Timer().start()
    mfcr = transportation_network.minimum_fleet_control_ratio(so_link_flow,
                                                              None)
    variables = mfcr.variables
    constants = mfcr.constants
    total_demand = mfcr.constants.total_demand.sum()
    _bin_edges = np.linspace(0.0, 1.0, 11)
    _bin_labels = [f"[{a:0.1f}, {b:0.1f}]" for a, b in zip(_bin_edges, _bin_edges[1:])]

    print(f"Fleet paths: {mfcr.constants.fleet_paths.sum()} usable; {(~mfcr.constants.fleet_paths).sum()} un-usable")

    timer.start()
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

        #mfcr.variables.save(f'mfcr-variables-{i:03d}')
        #mfcr.constants.save(f"mfcr-constants-{i:03d}")

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
        _print_hist = "\n".join(f"{_bin}: {_count:d}" for _bin, _count in zip(_bin_labels, counts))
        #print(f"fleet ratios:\n{_print_hist}")

    print(f"Fleet paths: {mfcr.constants.fleet_paths.sum()} usable; {(~mfcr.constants.fleet_paths).sum()} un-usable; total = {len(mfcr.constants.known_paths)}")
    best_value, best_i = min(upper_bounds)
    print(f"Best objective value {best_value} on iteration {best_i}")
    print(f"Lower bounds {lower_bounds}")

    return
    known_paths = mfcr.constants.known_paths

    mip = transportation_network.minimum_fleet_control_ratio(so_link_flow,
                                                             known_paths).to_mip()
    print("="*80)
    print(f"Running MIP with {len(mip.constants.known_paths)} paths")
    for i, (status, value) in enumerate(mip.heuristic()):
        _control_ratio = 100 * mip.variables.fleet_path_flow.value.sum() / mip.constants.total_demand.sum()
        print(f"{i+1} ({timer.time_elapsed():0.2f} seconds): LP terminated with {status} and objective value {100 *value:0.2f}% and control ratio {(_control_ratio):0.4f}%")
    now = int(time.time())
    mip.variables.save(f"fo-mcr-variables-{transportation_network.name}.{now}.npz")
    mip.constaints.save(f"fo-mcr-constants-{transportation_network.name}.{now}.npz")
