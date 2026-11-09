"""Traducción de ocho intenciones en grilla a recorridos viales dirigidos."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from math import ceil, floor, hypot

import networkx as nx
import numpy as np
from lima_pm25.domain import SpatialDomain
from shapely.geometry import LineString, Point
from shapely.ops import substring

# La fila de core aumenta hacia el norte, no hacia abajo en una imagen.
DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
DIRECTION_NAMES = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")


@dataclass(frozen=True)
class Route:
    destination: object
    cell: tuple[int, int]
    coordinates: tuple[tuple[float, float], ...]
    distance_m: float


class RoadRouter:
    """Busca con A* sobre una copia del grafo, sin modificar el dominio.

    Añade paradas en los intervalos de vía interiores a cada celda. Así una
    celda atravesada por una arista larga no necesita una intersección OSM.
    Las paradas son discretas y no garantizan la primera entrada exacta a la
    celda ni la mayor proximidad a su centro. No se modelan restricciones de
    giro adicionales a la conectividad del grafo de core.
    """

    def __init__(self, domain: SpatialDomain) -> None:
        self.grid = domain.grid
        self._permitted = domain.boundary.buffer(domain.config.road_network_buffer_m)
        self.graph = nx.DiGraph()
        self.original_nodes = tuple(domain.road_graph.nodes)
        self._goals: dict[tuple[int, int], object] = {}
        for node, data in domain.road_graph.nodes(data=True):
            self.graph.add_node(node, xy=(float(data["x"]), float(data["y"])))
        for index, (u, v, _key, data) in enumerate(
            domain.road_graph.edges(keys=True, data=True)
        ):
            start, end = self.position(u), self.position(v)
            line = data.get("geometry", LineString((start, end)))
            if not isinstance(line, LineString):
                raise TypeError("Cada vía debe tener geometría LineString")
            if Point(start).distance(Point(line.coords[-1])) < Point(start).distance(
                Point(line.coords[0])
            ):
                line = LineString(tuple(reversed(line.coords)))
            # Conservar también la conexión exacta con las intersecciones.
            coordinates = list(line.coords)
            if coordinates[0] != start:
                coordinates.insert(0, start)
            if coordinates[-1] != end:
                coordinates.append(end)
            line = LineString(coordinates)
            stops = [(0.0, u)]
            cuts = self._cell_crossings(line)
            for part, (left, right) in enumerate(pairwise(cuts)):
                distance = (left + right) / 2
                point = line.interpolate(distance)
                if self.grid.is_active_xy(point.x, point.y) and self._permitted.covers(
                    point
                ):
                    node = ("rl-stop", index, part)
                    self.graph.add_node(node, xy=(point.x, point.y))
                    stops.append((distance, node))
            stops.append((line.length, v))
            for (left, a), (right, b) in pairwise(stops):
                geometry = substring(line, left, right)
                if not self._permitted.covers(geometry):
                    continue
                points = (
                    tuple(geometry.coords)
                    if isinstance(geometry, LineString)
                    else (self.position(a), self.position(b))
                )
                length = right - left
                previous = self.graph.get_edge_data(a, b)
                if previous is None or length < previous["length"]:
                    self.graph.add_edge(a, b, length=length, coordinates=points)

        self.navigable_mask = np.zeros(self.grid.shape, dtype=bool)
        for node in tuple(self.graph):
            cell = self.cell(node)
            if cell is not None:
                goal = self._goals.setdefault(cell, ("rl-goal", *cell))
                self.graph.add_node(goal, xy=self.grid.cell_center(*cell))
                self.graph.add_edge(node, goal, length=0.0)
                self.navigable_mask[cell] = True
        self._cached_plan = lru_cache(maxsize=4096)(self._search)

    def position(self, node: object) -> tuple[float, float]:
        return self.graph.nodes[node]["xy"]

    def cell(self, node: object) -> tuple[int, int] | None:
        x, y = self.position(node)
        return (
            self.grid.xy_to_cell(x, y)
            if self.grid.is_active_xy(x, y) and self._permitted.covers(Point(x, y))
            else None
        )

    def target(self, node: object, action: int) -> tuple[int, int]:
        cell = self.cell(node)
        if cell is None:
            raise ValueError("El vehículo debe estar dentro de una celda activa")
        dr, dc = DIRECTIONS[action]
        return cell[0] + dr, cell[1] + dc

    def plan(self, node: object, action: int) -> Route | None:
        return self._cached_plan(node, action)

    def _search(self, node: object, action: int) -> Route | None:
        cell = self.target(node, action)
        if cell not in self._goals:
            return None
        row, column = cell
        x0, y0 = self.grid.grid_bounds_m[:2]
        size = self.grid.cell_size_m
        left, bottom = x0 + column * size, y0 + row * size

        def heuristic(current: object, _goal: object) -> float:
            x, y = self.position(current)
            return hypot(
                max(left - x, 0, x - left - size), max(bottom - y, 0, y - bottom - size)
            )

        try:
            path = nx.astar_path(
                self.graph,
                node,
                self._goals[cell],
                heuristic=heuristic,
                weight="length",
            )[:-1]
        except nx.NetworkXNoPath:
            return None
        points = [self.position(node)]
        distance = 0.0
        for a, b in pairwise(path):
            edge = self.graph.edges[a, b]
            points.extend(edge["coordinates"][1:])
            distance += edge["length"]
        return Route(path[-1], cell, tuple(points), distance)

    def _cell_crossings(self, line: LineString) -> list[float]:
        cuts = {0.0, line.length}
        origin = self.grid.grid_bounds_m[:2]
        size = self.grid.cell_size_m
        offset = 0.0
        for start, end in pairwise(line.coords):
            length = hypot(end[0] - start[0], end[1] - start[1])
            for axis in (0, 1):
                delta = end[axis] - start[axis]
                if delta == 0:
                    continue
                low, high = sorted((start[axis], end[axis]))
                for index in range(
                    floor((low - origin[axis]) / size) + 1,
                    ceil((high - origin[axis]) / size),
                ):
                    fraction = (origin[axis] + index * size - start[axis]) / delta
                    cuts.add(offset + fraction * length)
            offset += length
        return sorted(cuts)
