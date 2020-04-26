import pytest
import numpy as np
import cvxpy as cp
from traffic_assignment.control_ratio_range import lp
from traffic_assignment.control_ratio_range.utils import (NetworkParameters,
                                                          Variables,
                                                          HeuristicVariables,
                                                          Constants,
                                                          HeuristicConstants,
                                                          )
from warnings import warn


def test_path_index():
    # four links three paths
    link_path_incidence = np.array([
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]).T
    idx = 1
    path_links = link_path_incidence[:, idx]
    assert path_index(link_path_incidence, path_links) == idx

def test_upper_control_ratio(braess_so_solution, braess_demand, braess_network,
                             braess_cost_function, braess_so_cost_function):
    params = NetworkParameters.from_network(braess_network, braess_demand)
    variables = Variables.from_network_parameters(params)
    constants = Constants.from_network(
        braess_network,
        braess_demand,
        braess_cost_function,
        braess_so_cost_function,
        braess_so_solution,
    )
    mcr = lp.UpperControlRatio(constants, variables)
    problem = mcr.problem()
    problem.solve()
    assert problem.value == pytest.approx(6.0)
    assert np.allclose(variables.fleet_link_flow.value, braess_so_solution)
    assert np.allclose(variables.user_link_flow.value,
                       np.zeros_like(braess_so_solution))
    stats = problem.solver_stats
    warn(f"""
    solver terminated with status '{problem.status}' in {stats.solve_time} (s).
    setup time: {stats.setup_time} (s)
    number of iterations: {stats.num_iters}
    """)


def test_lower_control_ratio(braess_ue_solution, braess_demand, braess_network,
                             braess_cost_function, braess_so_cost_function):
    params = NetworkParameters.from_network(braess_network, braess_demand)
    variables = Variables.from_network_parameters(params)
    constants = Constants.from_network(
        braess_network,
        braess_demand,
        braess_cost_function,
        braess_so_cost_function,
        braess_ue_solution,
    )
    mcr = lp.LowerControlRatio(constants, variables)
    problem = mcr.problem()
    problem.solve()
    assert problem.value == pytest.approx(4.0)
    assert np.allclose(variables.fleet_path_flow.value,
                       np.array([2.0, 0.0, 2.0]))
    assert np.allclose(variables.user_path_flow.value,
                       np.array([0.0, 2.0, 0.0]))


def test_fleet_upper_control_ratio(braess_so_solution, braess_demand, braess_network,
                             braess_cost_function, braess_so_cost_function):
    params = NetworkParameters.from_network(braess_network, braess_demand)

    constants = HeuristicConstants.from_network(
        braess_network,
        braess_demand,
        braess_cost_function,
        None,
        braess_so_solution,
    )
    variables = HeuristicVariables.from_constants(constants)
    mcr = lp.MinimumFleetControlRatio(braess_network, braess_demand,
                                      constants, variables)
    for i, (status, value) in enumerate(mcr.heuristic()):
        print(f"iteration {i}: {status}, {value}")
        tol = 1e-5
        fleet_link_flow = mcr.fleet_link_flow.value
        user_link_flow = mcr.user_link_flow.value
        assert value == pytest.approx(1.0)
        assert cp.sum(mcr.variables.fleet_path_flow).value == pytest.approx(6.0)
        print(f"fleet link flow: {fleet_link_flow}")
        print(f"fleet path flow: {variables.fleet_path_flow.value}")
        print(f"fleet demand: {cp.sum(mcr.variables.fleet_path_flow).value}")
        print(f"user link flow: {user_link_flow}")
        assert np.allclose(fleet_link_flow, braess_so_solution,
                           atol=tol)
        assert np.allclose(user_link_flow,
                           np.zeros_like(braess_so_solution), atol=tol)

    mip = lp.MinimumFleetControlRatio(braess_network,
                                      braess_demand,
                                      constants, variables)
    for i, (status, value) in enumerate(mip.heuristic()):
        print(f"iteration {i}: {status}, {value}")
