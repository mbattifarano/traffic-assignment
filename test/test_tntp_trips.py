from traffic_assignment.tntp.trips import TNTPTrips


def test_read_text(sioux_falls_trips):
    trips = TNTPTrips.read_text(sioux_falls_trips)
    actual_total_flow = sum(t.volume for t in trips.trips)
    assert trips.meta_data.total_flow == actual_total_flow
