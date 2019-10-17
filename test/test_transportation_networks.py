import os
import time
from dataclasses import dataclass
from typing import MutableMapping, Iterable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from traffic_assignment.utils import Timer


def test_networks(transportation_network):
    tol = 2e-5
    ue_solver = transportation_network.ue_solver(tolerance=tol)
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


@dataclass
class NumpyFileCache(MutableMapping):
    directory = 'test/artifacts/numpy_cache'

    def __post_init__(self):
        try:
            os.mkdir(self.directory)
        except FileExistsError:
            pass

    def _file_of(self, k):
        return f"{self.directory}/{k}.npy"

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


path_incidence_cache = NumpyFileCache()


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


def test_path_cost_condition(transportation_network):
    ue_link_flow = transportation_network.solution.link_flow()
    link_cost_fn = transportation_network.network.to_link_cost_function()
    road_network = transportation_network.road_network()
    travel_demand = transportation_network.travel_demand()
    _link_path_key = f"{transportation_network.name}-link_path"
    _path_od_key = f"{transportation_network.name}-path_od"
    try:
        timer1 = Timer()
        timer1.start()
        link_path_incidence = path_incidence_cache[_link_path_key]
        path_od_incidence = path_incidence_cache[_path_od_key]
        print(f"cache hit: loaded arrays in {timer1.time_elapsed()} (s)")
    except KeyError:
        link_path_incidence, path_od_incidence, _ = road_network.path_incidences(
            travel_demand
        )
        path_incidence_cache[_link_path_key] = link_path_incidence
        path_incidence_cache[_path_od_key] = path_od_incidence
    n_links, n_paths = link_path_incidence.shape
    link_cost = link_cost_fn.link_cost(ue_link_flow)
    path_cost = link_path_incidence.T.dot(link_cost)
    demand_vec = travel_demand.to_array()

    min_paths = np.zeros(n_paths, dtype=np.uint8)

    cost_tolerance = 1e-8
    for j, _ in enumerate(travel_demand):
        path_selector = path_od_incidence[:, j].astype(bool)
        od_path_costs = path_cost[path_selector]
        min_paths[path_selector] = (od_path_costs <= (min(od_path_costs) * (1.0 + cost_tolerance))).astype(np.uint8)

    print("Creating problem.")
    path_flow_v = cp.Variable((n_paths,), name="path_flow")
    problem = cp.Problem(
        cp.Minimize(1),
        [
            path_flow_v >= 0,
            (path_od_incidence.T * min_paths) @ path_flow_v == demand_vec,
            (link_path_incidence * min_paths) @ path_flow_v == ue_link_flow,
        ]
    )
    print("Solving problem.")
    problem.solve(solver="SCS", verbose=True, eps=1e-8)
    computed_link_flow = (link_path_incidence @ path_flow_v).value
    computed_demand = (path_od_incidence.T @ path_flow_v).value
    path_flow = path_flow_v.value
    print("max link flow error: ", abs(computed_link_flow - ue_link_flow).max())
    print("max demand error: ", abs(computed_demand - demand_vec).max())
    assert np.allclose(computed_link_flow, ue_link_flow)
    assert np.allclose(path_flow[~min_paths.astype(bool)], 0)
    assert np.allclose(computed_demand, demand_vec)


def test_so_path_cost_condition(transportation_network):
    so_link_flow_key = f"{transportation_network.name}-so_link_flow"
    warm_start_key = f"{so_link_flow_key}-final"
    so_solver = transportation_network.so_solver(
        tolerance=1e-15,
        max_iterations=10000,
        initial_point=path_incidence_cache.get(warm_start_key),
        large_initial_step=False
    )
    print("Solving SO link flow")
    with timer():
        final_iteration = so_solver.solve()
        so_solution = so_solver.best_iteration()
    print(f"Solved SO link flow (gap={so_solution.absolute_gap}, relative gap = {so_solution.gap})")
    so_link_flow = so_solution.link_flow
    path_incidence_cache[so_link_flow_key] = so_link_flow
    path_incidence_cache[warm_start_key] = final_iteration.link_flow
    link_cost_fn = transportation_network.network.to_marginal_link_cost_function()
    road_network = transportation_network.road_network()
    travel_demand = transportation_network.travel_demand()
    _link_path_key = f"{transportation_network.name}-link_path"
    _path_od_key = f"{transportation_network.name}-path_od"
    try:
        timer1 = Timer()
        timer1.start()
        link_path_incidence = path_incidence_cache[_link_path_key]
        path_od_incidence = path_incidence_cache[_path_od_key]
        print(f"cache hit: loaded arrays in {timer1.time_elapsed()} (s)")
    except KeyError:
        link_path_incidence, path_od_incidence, _ = road_network.path_incidences(
            travel_demand
        )
        path_incidence_cache[_link_path_key] = link_path_incidence
        path_incidence_cache[_path_od_key] = path_od_incidence
    n_links, n_paths = link_path_incidence.shape
    link_cost = link_cost_fn.link_cost(so_link_flow)
    path_cost = link_path_incidence.T.dot(link_cost)
    demand_vec = travel_demand.to_array()

    min_paths = np.zeros(n_paths, dtype=np.uint8)

    cost_tolerance = 1e-5
    for j, _ in enumerate(travel_demand):
        path_selector = path_od_incidence[:, j].astype(bool)
        od_path_costs = path_cost[path_selector]
        min_paths[path_selector] = (od_path_costs <= (
                    min(od_path_costs) * (1.0 + cost_tolerance))).astype(
            np.uint8)

    print("Creating problem.")
    path_flow_v = cp.Variable((n_paths,), name="path_flow")
    problem = cp.Problem(
        cp.Minimize(1),
        [
            path_flow_v >= 0,
            (path_od_incidence.T * min_paths) @ path_flow_v == demand_vec,
            (link_path_incidence * min_paths) @ path_flow_v == so_link_flow,
        ]
    )
    print("Solving problem.")
    problem.solve(solver="SCS", verbose=True, eps=1e-6)
    assert problem.status == 'optimal'
    computed_link_flow = (link_path_incidence @ path_flow_v).value
    computed_demand = (path_od_incidence.T @ path_flow_v).value
    path_flow = path_flow_v.value
    print("max link flow error: ", abs(computed_link_flow - so_link_flow).max())
    print("max demand error: ", abs(computed_demand - demand_vec).max())
    assert np.allclose(computed_link_flow, so_link_flow)
    assert np.allclose(path_flow[~min_paths.astype(bool)], 0)
    assert np.allclose(computed_demand, demand_vec)


def test_lower_control_ratio(transportation_network):
    print("testing lower control ratio")
    ue_link_flow = transportation_network.solution.link_flow()
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


def test_upper_control_ratio(transportation_network):
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
