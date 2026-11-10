"""Un controlador central, tres vehículos y una transición conjunta cada 300 s."""

from collections.abc import Callable, Iterable
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lima_pm25.domain import SpatialDomain
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
    SequentialMapUpdater,
    SequentialUpdateConfig,
)
from lima_pm25.simulation import MeasurementBatch, MeasurementRecord, SensorType

from .episodes import Episode
from .routing import RoadRouter, Route
from .simulation import STEP_SECONDS, VEHICLE_COUNT, Simulation

Observation = dict[str, np.ndarray]


class LimaPM25Env(gym.Env[Observation, np.ndarray]):
    """Gymnasium monoagente con tres asignaciones vehículo–dirección ordenadas.

    Las elecciones se coordinan fuera de step mediante action_masks y reservas.
    Una dirección inviable permanece en el sitio y recibe penalización; nunca
    se ejecuta un movimiento ilegal. Si no hay opciones para un vehículo, se
    mantiene automáticamente sin penalización (no es una novena acción).
    """

    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012 -- contrato de Gymnasium

    def __init__(
        self,
        *,
        domain: SpatialDomain,
        episode_factory: Callable[[int], Episode],
        estimator_config: SequentialUpdateConfig,
        initial_nodes: tuple[object, object, object] | None = None,
        mobile_noise_stddev_ug_m3: float = 0.0,
        invalid_action_penalty: float = 1.0,
        concentration_priority: float = 0.0,
        distance_cost_per_km: float = 0.0,
        arrival_strategy: str = "shortest",
        render_mode: str | None = None,
    ) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Solo se admite render_mode=None o rgb_array")
        if arrival_strategy not in {"shortest", "continuing"}:
            raise ValueError("arrival_strategy debe ser shortest o continuing")
        if not np.isfinite(mobile_noise_stddev_ug_m3) or mobile_noise_stddev_ug_m3 < 0:
            raise ValueError("El ruido móvil debe ser finito y no negativo")
        if not np.isfinite(invalid_action_penalty) or invalid_action_penalty < 0:
            raise ValueError("La penalización debe ser finita y no negativa")
        if any(
            not np.isfinite(value) or value < 0
            for value in (concentration_priority, distance_cost_per_km)
        ):
            raise ValueError(
                "La prioridad y el coste vial deben ser finitos y no negativos"
            )
        self.domain = domain
        self.router = RoadRouter(domain)
        self._episode_factory = episode_factory
        self.estimator_config = estimator_config
        self.initial_nodes = initial_nodes
        self.mobile_noise_stddev_ug_m3 = mobile_noise_stddev_ug_m3
        self.invalid_action_penalty = invalid_action_penalty
        self.concentration_priority = concentration_priority
        self.distance_cost_per_km = distance_cost_per_km
        self.arrival_strategy = arrival_strategy
        self.render_mode = render_mode
        # Cada entrada asigna (vehículo, dirección): 8 * vehículo + dirección.
        # El orden de las tres entradas es el orden de reservas del ciclo.
        self.action_space = spaces.MultiDiscrete([8 * VEHICLE_COUNT] * VEHICLE_COUNT)
        # Gymnasium anota Dict con sus espacios, no con los arrays que devuelve.
        self.observation_space = cast(
            spaces.Space[Observation],
            spaces.Dict(
                {
                    "maps": spaces.Box(
                        -1.0, 1.0, shape=(7, *domain.grid.shape), dtype=np.float32
                    ),
                    "vehicles": spaces.Box(
                        0.0, 1.0, shape=(VEHICLE_COUNT, 2), dtype=np.float32
                    ),
                    "remaining": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                    "action_mask": spaces.MultiBinary((VEHICLE_COUNT, 8)),
                }
            ),
        )
        self._simulation: Simulation | None = None
        self._finished = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        if options and set(options) != {"scenario_seed"}:
            raise ValueError("La única opción admitida es scenario_seed")
        scenario_seed = (options or {}).get("scenario_seed")
        if scenario_seed is None:
            scenario_seed = int(self.np_random.integers(2**32))
        if (
            not isinstance(scenario_seed, int)
            or isinstance(scenario_seed, bool)
            or scenario_seed < 0
        ):
            raise ValueError("scenario_seed debe ser un entero no negativo")
        episode = self._episode_factory(scenario_seed)
        if episode.request.model.grid is not self.domain.grid:
            raise ValueError("El episodio debe usar la misma grilla que el dominio")
        self._nodes = list(self.initial_nodes or self._random_nodes(scenario_seed))
        if (
            len(self._nodes) != VEHICLE_COUNT
            or len({self.router.cell(n) for n in self._nodes}) != VEHICLE_COUNT
        ):
            raise ValueError(
                "Se requieren tres posiciones iniciales en celdas distintas"
            )
        if any(self.router.cell(n) is None for n in self._nodes):
            raise ValueError("Las posiciones iniciales deben estar en celdas activas")
        self._simulation = Simulation(
            episode,
            seed=episode.request.config.seed,
            positions=self.positions,
            noise_stddev_ug_m3=self.mobile_noise_stddev_ug_m3,
        )
        self._updater = SequentialMapUpdater(self.domain.grid, self.estimator_config)
        self._reconstructor = GaussianProcessFieldReconstructor(
            self.estimator_config.reconstruction_config
        )
        self._records: list[MeasurementRecord] = []
        self._routes: list[list[tuple[float, float]]] = [
            [point] for point in self.positions
        ]
        self._distance = np.zeros(VEHICLE_COUNT)
        self._steps = 0
        self._finished = False
        self._assimilate(self._simulation.initial_batch())
        self._posterior = self._predict()
        return self._observation(), self._info()

    @property
    def positions(self) -> tuple[tuple[float, float], ...]:
        return tuple(self.router.position(node) for node in self._nodes)

    def action_masks(
        self, reserved_cells: Iterable[tuple[int, int]] = ()
    ) -> np.ndarray:
        if self._simulation is None:
            raise gym.error.ResetNeeded(
                "Ejecuta reset() antes de consultar movimientos"
            )
        reserved = set(reserved_cells)
        mask = np.zeros((VEHICLE_COUNT, 8), dtype=np.int8)
        occupied = [self.router.cell(node) for node in self._nodes]
        for vehicle, node in enumerate(self._nodes):
            blocked = reserved | {
                cell for i, cell in enumerate(occupied) if i != vehicle
            }
            for action in range(8):
                if self.router.target(node, action) not in blocked:
                    # La variante conserva la llegada más corta como fallback:
                    # no altera la validez de una intención desde esta posición.
                    mask[vehicle, action] = self.router.plan(node, action) is not None
        return mask

    def route(self, vehicle: int, action: int) -> Route | None:
        """Recorrido conocido por el controlador, sin consultar el campo oculto."""
        continuing = (
            self.arrival_strategy == "continuing"
            and self._simulation is not None
            and self._simulation.state.simulation_time_s + STEP_SECONDS
            < self._simulation.duration_s
        )
        return self.router.plan(
            self._nodes[vehicle], action, prefer_continuation=continuing
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._simulation is None or self._finished:
            raise gym.error.ResetNeeded("Ejecuta reset() para iniciar un episodio")
        action = np.asarray(action)
        if (
            action.shape != (VEHICLE_COUNT,)
            or action.dtype.kind not in "iu"
            or not self.action_space.contains(action)
        ):
            raise ValueError("La acción debe contener tres asignaciones entre 0 y 23")
        reserved: set[tuple[int, int]] = set()
        # Ponderación congelada ANTES de mover o recibir nuevas mediciones.
        # Conserva peso >= 1 en todo el dominio y está acotada por 1 + prioridad.
        reference_concentration = max(
            self.estimator_config.reconstruction_config.prior_mean_ug_m3,
            self.estimator_config.reconstruction_config.prior_stddev_ug_m3,
        )
        mean = np.maximum(self._posterior.mean_ug_m3, 0.0)
        reward_weights = (
            1 + self.concentration_priority * mean / (reference_concentration + mean)
            if self.concentration_priority
            else None
        )
        selected: list[Route | None] = [None] * VEHICLE_COUNT
        invalid = np.zeros(VEHICLE_COUNT, dtype=np.int8)
        held = np.zeros(VEHICLE_COUNT, dtype=np.int8)
        assigned = set()
        for assignment in action:
            vehicle, direction = divmod(int(assignment), 8)
            if vehicle in assigned:
                invalid[vehicle] = 1
                continue
            assigned.add(vehicle)
            available = self.action_masks(reserved)[vehicle]
            route = None
            if not available.any():
                held[vehicle] = 1
            elif not available[direction]:
                invalid[vehicle] = 1
            else:
                route = self.route(vehicle, int(direction))
                assert route is not None
                reserved.add(route.cell)
            selected[vehicle] = route
        for vehicle in set(range(VEHICLE_COUNT)) - assigned:
            held[vehicle] = 1

        for vehicle, route in enumerate(selected):
            if route is not None:
                self._nodes[vehicle] = route.destination
                self._distance[vehicle] += route.distance_m
                self._routes[vehicle].extend(route.coordinates[1:])
        gain = 0.0
        for batch in self._simulation.advance(self.positions):
            gain += self._assimilate(batch, reward_weights=reward_weights)
        self._steps += 1
        self._posterior = self._predict()
        terminated = not bool(self.action_masks().any())
        truncated = (
            self._simulation.state.simulation_time_s >= self._simulation.duration_s
        )
        self._finished = terminated or truncated
        distance_cost = (
            self.distance_cost_per_km
            * sum(route.distance_m if route else 0.0 for route in selected)
            / 1000
        )
        reward = gain - distance_cost - self.invalid_action_penalty * int(invalid.sum())
        info = self._info()
        info.update(
            uncertainty_gain=gain,
            distance_cost=distance_cost,
            invalid_actions=invalid,
            held_vehicles=held,
            route_distance_m=np.array(
                [route.distance_m if route else 0.0 for route in selected]
            ),
        )
        return self._observation(), float(reward), terminated, truncated, info

    def _assimilate(
        self, batch: MeasurementBatch | None, *, reward_weights=None
    ) -> float:
        if batch is None:
            return 0.0
        config = self.estimator_config.reconstruction_config
        mobile = [
            r
            for r in batch.records
            if r.sensor_type is SensorType.MOBILE
            and (r.quality_status == "accepted" or config.include_clipped_observations)
        ]
        fixed = tuple(r for r in batch.records if r.sensor_type is SensorType.FIXED)
        reference = None
        if mobile and (fixed or reward_weights is not None):
            reference = self._predict(batch.timestamp_utc, (*self._records, *fixed))
        result = self._updater.update(batch)
        self._records.extend(batch.records)
        if not mobile or not result.applied:
            return 0.0
        assert result.snapshot is not None
        diagnostic = result.snapshot.uncertainty_reduction
        before = (
            self._integrated_variance(reference)
            if reference is not None
            else diagnostic.integrated_variance_before_ug2_m4
        )
        after = diagnostic.integrated_variance_after_ug2_m4
        if reward_weights is not None:
            assert reference is not None
            mask = self.domain.grid.active_mask
            reduction = (
                reference.standard_deviation_ug_m3[mask] ** 2
                - result.snapshot.standard_deviation_ug_m3[mask] ** 2
            )
            return float(
                np.sum(reward_weights[mask] * reduction)
                / (mask.sum() * config.prior_stddev_ug_m3**2)
            )
        scale = (
            self.domain.grid.active_mask.sum()
            * self.domain.grid.cell_area_m2
            * config.prior_stddev_ug_m3**2
        )
        return float((before - after) / scale)

    def _predict(self, timestamp=None, records=None):
        assert self._simulation is not None
        timestamp = timestamp or self._simulation.clock.utc_at(
            self._simulation.state.simulation_time_s
        )
        snapshot = self._updater.current_snapshot
        if (
            records is None
            and snapshot is not None
            and snapshot.reconstruction_timestamp_utc == timestamp
        ):
            return snapshot
        start = timestamp - self.estimator_config.history_duration
        records = self._records if records is None else records
        return self._reconstructor.reconstruct(
            ReconstructionRequest(
                grid=self.domain.grid,
                reconstruction_timestamp_utc=timestamp,
                measurements=tuple(
                    r for r in records if start <= r.timestamp_utc <= timestamp
                ),
            )
        )

    def _integrated_variance(self, posterior) -> float:
        mask = self.domain.grid.active_mask
        return float(
            np.sum(posterior.standard_deviation_ug_m3[mask] ** 2)
            * self.domain.grid.cell_area_m2
        )

    def _observation(self) -> Observation:
        assert self._simulation is not None
        grid = self.domain.grid
        mask = grid.active_mask
        config = self.estimator_config.reconstruction_config
        maps = np.zeros((7, *grid.shape), dtype=np.float32)
        maps[0] = mask
        maps[1] = self.router.navigable_mask
        maps[2, mask] = np.tanh(
            (self._posterior.mean_ug_m3[mask] - config.prior_mean_ug_m3)
            / config.prior_stddev_ug_m3
        )
        maps[3, mask] = np.clip(
            self._posterior.standard_deviation_ug_m3[mask] / config.prior_stddev_ug_m3,
            0,
            1,
        )
        for vehicle, node in enumerate(self._nodes):
            cell = self.router.cell(node)
            assert cell is not None
            maps[4 + vehicle, *cell] = 1
        x0, y0, x1, y1 = grid.grid_bounds_m
        vehicles = (np.asarray(self.positions) - [x0, y0]) / [x1 - x0, y1 - y0]
        return {
            "maps": maps,
            "vehicles": vehicles.astype(np.float32),
            "remaining": np.array(
                [
                    max(
                        0.0,
                        1 - self._steps * STEP_SECONDS / self._simulation.duration_s,
                    )
                ],
                dtype=np.float32,
            ),
            "action_mask": self.action_masks(),
        }

    def _info(self) -> dict[str, Any]:
        assert self._simulation is not None
        return {
            "elapsed_s": self._simulation.state.simulation_time_s,
            "positions_m": np.asarray(self.positions).copy(),
            "distance_m": self._distance.copy(),
            "integrated_variance": self._integrated_variance(self._posterior),
            "measurement_count": len(self._records),
            "step_count": self._steps,
        }

    def _random_nodes(self, seed: int) -> tuple[object, ...]:
        rng = np.random.default_rng([seed, 2])
        candidates = [
            node
            for node in self.router.original_nodes
            if self.router.cell(node) is not None
        ]
        chosen = []
        cells = set()
        for index in rng.permutation(len(candidates)):
            node = candidates[index]
            cell = self.router.cell(node)
            if cell not in cells and any(
                self.router.plan(node, action) is not None for action in range(8)
            ):
                chosen.append(node)
                cells.add(cell)
                if len(chosen) == VEHICLE_COUNT:
                    return tuple(chosen)
        raise ValueError(
            "El dominio necesita tres celdas viales con movimientos posibles"
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._simulation is None:
            raise gym.error.ResetNeeded("Ejecuta reset() antes de visualizar")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10, 4), dpi=100)
        canvas = FigureCanvasAgg(figure)
        grid = self.domain.grid
        extent = grid.grid_bounds_m[[0, 2, 1, 3]] / 1000
        fields = (self._posterior.mean_ug_m3, self._posterior.standard_deviation_ug_m3)
        for axis, field, title in zip(
            figure.subplots(1, 2),
            fields,
            ("Concentración estimada (µg/m³)", "Incertidumbre (µg/m³)"),
            strict=True,
        ):
            image = axis.imshow(field, origin="lower", extent=extent, cmap="viridis")
            for vehicle, points in enumerate(self._routes):
                xy = np.asarray(points) / 1000
                axis.plot(
                    xy[:, 0], xy[:, 1], linewidth=1, label=f"Vehículo {vehicle + 1}"
                )
                axis.scatter(*xy[-1], s=20)
            samples = self._simulation.mobile.history.accepted_records
            if samples:
                axis.scatter(
                    [r.x_m / 1000 for r in samples],
                    [r.y_m / 1000 for r in samples],
                    s=5,
                    c="white",
                    marker=".",
                )
            catalog = self._simulation.request.station_catalog
            if catalog is not None:
                measured = {
                    r.sensor_id
                    for r in self._records
                    if r.sensor_type is SensorType.FIXED
                }
                for site in catalog.sites:
                    axis.scatter(
                        site.x_m / 1000,
                        site.y_m / 1000,
                        marker="^",
                        s=22,
                        edgecolors="red",
                        facecolors="red" if site.site_id in measured else "none",
                    )
            axis.set_title(title)
            axis.set_xlabel("Este UTM (km)")
            axis.set_ylabel("Norte UTM (km)")
            axis.locator_params(axis="both", nbins=5)
            axis.legend(fontsize=6)
            figure.colorbar(image, ax=axis, shrink=0.7)
        figure.suptitle(
            f"Tiempo: {self._steps * 5} min | Distancia: {self._distance.sum():.0f} m"
        )
        figure.tight_layout()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
