import numpy as np
import matplotlib.pyplot as plt
import time


def test_networks(transportation_network):
    tol = 1e-5
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
