from traffic_assignment.tntp import solution


def test_read_tsv(sioux_falls_solution):
    sol = solution.TNTPSolution.read_text(sioux_falls_solution)
    assert len(sol.links) == 76
    for link in sol.links:
        assert link.volume >= 0
        assert link.cost >= 0


