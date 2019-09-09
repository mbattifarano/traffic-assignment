import random
from itertools import permutations

import time
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from hypothesis import given, HealthCheck, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (builds, integers, composite, floats, lists,
                                   sampled_from)
from traffic_assignment.frank_wolfe.search_direction import (
    ShortestPathSearchDirection
)
from traffic_assignment.frank_wolfe.solver import Solver
from traffic_assignment.frank_wolfe.step_size import MonotoneDecreasingStepSize
from traffic_assignment.link_cost_function.bpr import BPRLinkCostFunction
from traffic_assignment.network.demand import Demand
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
    integers(min_value=5, max_value=20),
).map(make_twoway).map(add_random_edges)

random_networks = builds(RoadNetwork, random_graphs)

non_negatives = floats(min_value=0.0, max_value=2 ** 8,
                       allow_nan=False, allow_infinity=False)
positives = non_negatives.map(lambda x: x + 0.01)


def link_vector_of(shape, elements):
    return arrays(np.dtype('float64'), shape, elements=elements)


def link_cost_function_of(shape):
    return builds(
        BPRLinkCostFunction,
        link_vector_of(shape, non_negatives),
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
    step_size = MonotoneDecreasingStepSize()
    link_cost_function = draw(link_cost_function_of(n))
    demand = draw(lists(
        demands(network.nodes),
        min_size=1,
        max_size=5,
    ))
    search = ShortestPathSearchDirection(network, demand)
    return Solver(step_size, search, link_cost_function)


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
    assert link_flow.max() > 0.0
    link_flows = {(link.origin.name, link.destination.name): link_flow[link.id]
                  for link in network.links}
    for demand in solver.search_direction.demand:
        paths = nx.all_simple_paths(network.graph,
                                    demand.origin.name,
                                    demand.destination.name)
        # there should be at least one path for which all links in that path
        # have flow greater than or equal to the demand volume
        # this is a necessary but not sufficient condition.
        assert any(min(link_flows[e] for e in Path(path).edges) >= demand.volume
                   for path in paths)


@given(solvers())
def test_solver_update(solver):
    first_iteration = solver.initial_iteration()
    second_iteration = solver.update(first_iteration)
    # TODO: test feasible
    # the initial gap in inf, the updated gap should be smaller
    assert second_iteration.gap < first_iteration.gap


@given(solvers())
@settings(max_examples=10, deadline=None)
def test_solver_solve(solver):
    final_iteration = solver.solve()
    assert final_iteration.iteration >= 1
    assert final_iteration.iteration <= 100
    gaps = np.array([iteration.gap for iteration in solver.iterations])
    assert gaps.min() >= 0
    assert gaps[0] >= np.inf
    assert gaps[1:].max() < np.inf
    assert not np.isnan(gaps).any()
    # TODO: test feasible
    plt.figure()
    plt.semilogy(gaps)
    plt.savefig(f"test/artifacts/gaps.{time.time()}.png")
    plt.close()
    # assert final_iteration.gap <= gaps.min() + solver.tolerance
    assert ((final_iteration.gap <= solver.tolerance)
            or (final_iteration.iteration == solver.max_iterations))



