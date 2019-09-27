from traffic_assignment.tntp.network import TNTPNetwork


def test_tntp_network(sioux_falls_network):
    network = TNTPNetwork.read_text(sioux_falls_network)
    assert network.meta_data.n_zones == 24
    assert network.meta_data.n_nodes == 24
    assert network.meta_data.n_links == 76
    assert network.meta_data.first_node == 1
    assert len(network.links) == 76
    nodes = set()
    for link in network.links:
        nodes.add(link.from_node)
        nodes.add(link.to_node)
        assert link.b == 0.15
        assert link.power == 4
        assert link.speed_limit == 0
        assert link.toll == 0
        assert link.link_type == 1
    assert len(nodes) == 24
