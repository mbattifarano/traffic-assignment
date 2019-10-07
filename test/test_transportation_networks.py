import numpy as np
import matplotlib.pyplot as plt
import time


def test_networks(transportation_network):
    tol = 1e-4
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


def test_lower_control_ratio(transportation_network):
    total_demand = transportation_network.trips.total_demand()
    ue_link_flow = transportation_network.solution.link_flow()
    lower_control_ratio = transportation_network.lower_control_ratio(ue_link_flow)
    lcr_lp = lower_control_ratio.problem()
    lcr_lp.solve()
    print(f"""
    {transportation_network.name}
    {'='*len(transportation_network.name)}
    lower control ratio: {lcr_lp.value/total_demand} ({lcr_lp.value} total).
    """)
    assert False


def test_upper_control_ratio(transportation_network):
    tol = 1e-4
    total_demand = transportation_network.trips.total_demand()
    so_solver = transportation_network.so_solver(tol)
    so_link_flow = so_solver.solve()
    upper_control_ratio = transportation_network.upper_control_ratio(so_link_flow)
    ucr_lp = upper_control_ratio.problem()
    ucr_lp.solve()
    print(f"""
    {transportation_network.name}
    {'=' * len(transportation_network.name)}
    lower control ratio: {ucr_lp.value / total_demand} ({ucr_lp.value} total).
    """)
    assert False

