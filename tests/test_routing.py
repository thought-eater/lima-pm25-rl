from dataclasses import replace

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from lima_pm25_rl.routing import DIRECTIONS, RoadRouter


def test_north_is_positive_row_and_paths_keep_original_graph(domain):
    original = domain.road_graph.copy()
    router = RoadRouter(domain)
    assert DIRECTIONS[0] == (1, 0)
    assert router.plan((0, 0), 0).cell == (1, 0)
    assert router.plan((0, 0), 6) is None
    assert nx.utils.graphs_equal(original, domain.road_graph)


def test_cell_without_osm_nodes_is_reachable_without_ignoring_one_way(domain):
    graph = nx.MultiDiGraph(crs=domain.grid.crs)
    graph.add_node("west", x=100, y=100)
    graph.add_node("east", x=700, y=100)
    graph.add_edge(
        "west", "east", length=600, geometry=LineString(((100, 100), (700, 100)))
    )
    router = RoadRouter(replace(domain, road_graph=graph))
    route = router.plan("west", 2)
    assert route.cell == (0, 1)
    assert domain.grid.xy_to_cell(*route.coordinates[-1]) == (0, 1)
    assert 200 < route.coordinates[-1][0] < 400
    assert router.plan("east", 6) is None
    assert router.plan(route.destination, 6) is None


def test_diagonal_uses_streets_not_a_straight_line(domain):
    router = RoadRouter(domain)
    route = router.plan((0, 0), 1)
    assert route.cell == (1, 1)
    assert len(route.coordinates) > 2
    coordinates = np.asarray(route.coordinates)
    differences = np.diff(coordinates, axis=0)
    assert np.all((differences[:, 0] == 0) | (differences[:, 1] == 0))
    assert route.distance_m == np.linalg.norm(differences, axis=1).sum()


def test_parallel_edges_and_reversed_stored_geometry_do_not_change_the_path(domain):
    graph = nx.MultiDiGraph(crs=domain.grid.crs)
    graph.add_node(1, x=100, y=100)
    graph.add_node(2, x=300, y=100)
    graph.add_edge(
        1, 2, length=1000, geometry=LineString(((100, 100), (100, 500), (300, 100)))
    )
    graph.add_edge(1, 2, length=200, geometry=LineString(((300, 100), (100, 100))))
    router = RoadRouter(replace(domain, road_graph=graph))
    route = router.plan(1, 2)
    assert route.distance_m == 150
    assert route.coordinates[0] == (100, 100)
    assert route.coordinates[-1] == (250, 100)


def test_routes_cannot_escape_the_permitted_road_buffer(domain):
    graph = nx.MultiDiGraph(crs=domain.grid.crs)
    graph.add_node(1, x=100, y=100)
    graph.add_node(2, x=100, y=300)
    line = LineString(((100, 100), (-200, 100), (-200, 300), (100, 300)))
    graph.add_edge(1, 2, geometry=line, length=line.length)
    router = RoadRouter(replace(domain, road_graph=graph))
    assert router.plan(1, 0) is None
