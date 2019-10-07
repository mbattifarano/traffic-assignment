import random
import time
import warnings
from itertools import permutations, count

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (builds, integers, composite, floats, lists,
                                   sampled_from)
from traffic_assignment.frank_wolfe.search_direction import (
    ShortestPathSearchDirection
)
from traffic_assignment.frank_wolfe.solver import Solver
from traffic_assignment.frank_wolfe.step_size import LineSearchStepSize
from traffic_assignment.link_cost_function.bpr import BPRLinkCostFunction
from traffic_assignment.network.demand import Demand, TravelDemand
from traffic_assignment.network.path import Path
from traffic_assignment.network.road_network import RoadNetwork


def make_twoway(g: nx.DiGraph):
    return g.to_undirected().to_directed()


def add_random_edges(g: nx.DiGraph, p=0.05):
    for e in permutations(range(len(g)), 2):
        if (e not in g.edges) and (random.random() < p):
            g.add_edge(*e)
    return g


"""Random graphs that are strongly connected and have low node degree"""
random_graphs = builds(
    nx.generators.gn_graph,
    integers(min_value=10, max_value=20),
).map(make_twoway).map(add_random_edges)

random_networks = builds(RoadNetwork, random_graphs)

non_negatives = floats(min_value=0.0, max_value=2 ** 8,
                       allow_nan=False, allow_infinity=False)
positives = non_negatives.filter(lambda x: x > 0.0)


def link_vector_of(shape, elements):
    return arrays(np.dtype('float64'), shape, elements=elements)


def link_cost_function_of(shape):
    return builds(
        BPRLinkCostFunction,
        link_vector_of(shape, positives),
        link_vector_of(shape, positives),
    )


@composite
def demands(draw, nodes):
    nodes = list(nodes)
    origin = draw(sampled_from(nodes))
    nodes.remove(origin)
    destination = draw(sampled_from(nodes))
    volume = draw(positives)
    return Demand(origin, destination, volume)


@composite
def solvers(draw):
    network = draw(random_networks)
    n = network.number_of_links()
    link_cost_function = draw(link_cost_function_of(n))
    step_size = LineSearchStepSize(link_cost_function)
    demand = draw(
        builds(
            TravelDemand,
            lists(
                demands(network.nodes),
                min_size=1,
                max_size=5,
                unique_by=lambda d: d.trip(),
            )
        )
    )
    search = ShortestPathSearchDirection(network, demand)
    return Solver(
        step_size,
        search,
        link_cost_function,
        max_iterations=10000,
        tolerance=1e-5,
    )


@given(solvers())
def test_solver_initial_iteration(solver):
    network = solver.search_direction.network
    iteration = solver.initial_iteration()
    assert iteration.iteration == 0
    assert len(iteration.cost) == network.number_of_links()
    assert iteration.gap >= np.inf
    link_flow = iteration.link_flow
    assert len(link_flow) == network.number_of_links()
    assert link_flow.min() >= 0
    assert link_flow.max() <= solver.search_direction.demand.to_array().sum() + 1e-8
    link_flows = {(link.origin.name, link.destination.name): link_flow[i]
                  for i, link in enumerate(network.links)}
    for demand in solver.search_direction.demand:
        paths = nx.all_simple_paths(network.graph,
                                    demand.origin.name,
                                    demand.destination.name)
        assert sum(min(link_flows[e] for e in Path(path).edges)
                   for path in paths) >= demand.volume


@given(solvers())
def test_solver_update(solver):
    first_iteration = solver.initial_iteration()
    second_iteration = solver.update(first_iteration)
    # TODO: test feasible
    # the initial gap in inf, the updated gap should be smaller
    assert ((second_iteration.best_lower_bound == 0.0)
            or (second_iteration.gap < first_iteration.gap))


solver_counter = count()


@given(solvers())
@settings(max_examples=10, deadline=None)
def test_solver_solve(data_store, solver):
    example_number = next(solver_counter)
    t0 = time.time()
    final_iteration = solver.solve()
    t = time.time() - t0
    n = final_iteration.iteration
    data_store['solvers'].append((solver, t))
    assert n >= 1
    assert n <= solver.max_iterations
    gaps = solver.gaps
    best_lower_bounds = np.array([i.best_lower_bound for i in solver.iterations])
    assert gaps.min() >= 0
    assert gaps[0] >= np.inf
    assert best_lower_bounds.min() == 0.0 or gaps[1:].max() < np.inf
    assert not np.isnan(gaps).any()
    # TODO: test feasible
    if (gaps > 0.0).all():
        plt.figure()
        plt.semilogy(gaps)
        plt.savefig(f"test/artifacts/gaps.{example_number}.png")
        plt.close()
    assert ((final_iteration.gap <= solver.tolerance)
            or (final_iteration.iteration == solver.max_iterations))


def test_solver_braess_ue(braess_ue_solver, braess_ue_solution, tolerance):
    braess_solver = braess_ue_solver
    solution_iteration = braess_solver.solve()
    actual_link_flow = solution_iteration.link_flow
    errors = [
        np.linalg.norm(
            braess_solver.iterations[i].link_flow
            - braess_solver.iterations[i-1].link_flow
        )
        for i in range(1, len(braess_solver.iterations))
    ]
    plt.figure()
    plt.semilogy(errors)
    plt.savefig(f"test/artifacts/braess.ue.errors.{time.time()}.png")
    plt.close()

    plt.figure()
    plt.semilogy(braess_solver.gaps)
    plt.savefig(f"test/artifacts/braess.ue.gaps.{time.time()}.png")
    plt.close()
    assert 0 <= solution_iteration.gap < tolerance
    assert np.allclose(actual_link_flow, braess_ue_solution)


def test_solver_braess_so(braess_so_solver, braess_so_solution, tolerance):
    braess_solver = braess_so_solver
    solution_iteration = braess_solver.solve()
    actual_link_flow = solution_iteration.link_flow
    errors = [
        np.linalg.norm(
            braess_solver.iterations[i].link_flow
            - braess_solver.iterations[i-1].link_flow
        )
        for i in range(1, len(braess_solver.iterations))
    ]
    plt.figure()
    plt.semilogy(errors)
    plt.savefig(f"test/artifacts/braess.so.errors.{time.time()}.png")
    plt.close()

    plt.figure()
    plt.semilogy(braess_solver.gaps)
    plt.savefig(f"test/artifacts/braess.so.gaps.{time.time()}.png")
    plt.close()

    assert 0 <= solution_iteration.gap < tolerance
    assert np.allclose(actual_link_flow, braess_so_solution)


def test_report(data_store):
    results = data_store['solvers']
    data = []
    total_solution_time = 0
    for i, (solver, t) in enumerate(results):
        total_solution_time += t
        n_iter = len(solver.iterations)
        n_links = len(solver.iteration.link_flow)
        total_demand = sum(d.volume for d in solver.search_direction.demand)
        final_gap = solver.iteration.gap
        error = np.linalg.norm(
            solver.iterations[-1].link_flow
            - solver.iterations[-2].link_flow,
            1
        )/ solver.iteration.link_flow.sum()  # absolute percentage error
        direction_mass = np.linalg.norm(solver.iteration.search_direction)
        free_flow_cost = solver.link_cost_function.link_cost(0.0).max()
        x = solver.iteration.link_flow
        total_delay = solver.link_cost_function.link_cost(x).dot(x)
        it_per_sec = 1 / np.mean([it.duration for it in solver.iterations[1:]])
        data.append((i, n_iter, n_links, total_demand, free_flow_cost,
                     total_delay, direction_mass, final_gap, error, it_per_sec,
                     t))
    df = pd.DataFrame(
        data,
        columns=['solver', 'iterations', 'links', 'demand',
                 'free flow travel time (max)', 'total delay',
                 'direction magnitude',
                 'gap', 'link flow error', 'it/sec', 'total duration']
    ).set_index('solver')
    df.to_csv('test/artifacts/solver_report.csv')
    warnings.warn("\n"+df.to_string(float_format='{:0.3f}'.format))
    warnings.warn(f"total time to solve: {total_solution_time:0.2f}s; ({total_solution_time / len(results):0.2f}s per solution).")
