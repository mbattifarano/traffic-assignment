import pytest
import numpy as np
from traffic_assignment.control_ratio_range import lp
from traffic_assignment.control_ratio_range.utils import (NetworkParameters,
                                                          Variables,
                                                          Constants)


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
