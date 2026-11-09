"""Avance interactivo por eventos usando exclusivamente los componentes de core."""

import math
from collections import defaultdict

import numpy as np
from lima_pm25.simulation import (
    BoundaryMode,
    EpisodeClock,
    MeasurementBatch,
    MeasurementRecord,
    PM25Transport,
    SensingService,
    SensorConfig,
    SensorType,
)

from .episodes import Episode

STEP_SECONDS = 300.0
VEHICLE_COUNT = 3
MOBILE_IDS = tuple(f"MOBILE_{index}" for index in range(VEHICLE_COUNT))


class _Positions:
    provider_id = "rl-endpoints"

    def __init__(self, crs: str, positions: tuple[tuple[float, float], ...]) -> None:
        self.crs = crs
        self.positions = dict(zip(MOBILE_IDS, positions, strict=True))

    def position_at(
        self, sensor_id: str, simulation_time_s: float
    ) -> tuple[float, float]:
        return self.positions[sensor_id]


class Simulation:
    """Conserva campo y sensores entre ticks; no selecciona acciones ni estima mapas.

    Los móviles solo se consultan en los extremos de los ticks. Las fronteras
    de integración dependen de calendarios, no de las decisiones ni del ruido
    móvil. El reloj de la física puede incluir eventos interiores a los 300 s.
    """

    def __init__(
        self,
        episode: Episode,
        *,
        seed: int,
        positions: tuple[tuple[float, float], ...],
        noise_stddev_ug_m3: float,
    ) -> None:
        self.request = request = episode.request
        self.is_historical_replay = episode.historical_fixed is not None
        self.clock = EpisodeClock(request.config.start_utc)
        self.duration_s = request.config.duration_s
        if not math.isclose(self.duration_s % STEP_SECONDS, 0, abs_tol=1e-8):
            raise ValueError(
                "La duración del episodio debe contener ticks completos de 300 s"
            )
        if request.initial_state.simulation_time_s != 0:
            raise ValueError("El campo inicial debe corresponder a t=0")
        if request.initial_state.grid is not request.model.grid:
            raise ValueError("El campo y el transporte deben compartir la grilla")
        if (
            request.weather.start_utc != request.config.start_utc
            or request.weather.duration_s < self.duration_s
        ):
            raise ValueError(
                "La meteorología debe cubrir el inicio y la duración del episodio"
            )
        if (
            request.config.boundary_mode is BoundaryMode.REGIONAL_INFLOW
            and request.regional_boundary is None
        ):
            raise ValueError("Falta la condición de frontera regional")
        if len({source.source_id for source in request.sources}) != len(
            request.sources
        ):
            raise ValueError("Las fuentes deben tener identificadores distintos")
        self.model = PM25Transport(request.model.grid, request.model.physics)
        self.state = self.model.initialize_from_array(
            request.initial_state.concentration_ug_m3,
            ambient_baseline=request.initial_state.ambient_baseline,
        )
        self.positions = _Positions(self.model.grid.crs, positions)
        fixed_configs = tuple(
            c for c in request.sensor_configs if c.sensor_type is SensorType.FIXED
        )
        if any(c.sensor_id in MOBILE_IDS for c in fixed_configs):
            raise ValueError("Los identificadores fijos y móviles deben ser distintos")
        streams = np.random.SeedSequence([seed, 1]).spawn(2)
        self.fixed = SensingService(
            self.model.grid,
            fixed_configs if episode.historical_fixed is None else (),
            station_catalog=request.station_catalog,
            rng=np.random.default_rng(streams[0]),
        )
        self.mobile = SensingService(
            self.model.grid,
            tuple(
                SensorConfig(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.MOBILE,
                    sampling_interval_s=STEP_SECONDS,
                    first_due_time_s=STEP_SECONDS,
                    mobile_position_provider_id=self.positions.provider_id,
                    provenance="Medición móvil simulada al final del tick de 300 s",
                    noise_stddev_ug_m3=noise_stddev_ug_m3,
                )
                for sensor_id in MOBILE_IDS
            ),
            mobile_position_providers=(self.positions,),
            rng=np.random.default_rng(streams[1]),
        )
        self._historical: dict[float, list[MeasurementRecord]] = defaultdict(list)
        seen = set()
        for record in episode.historical_fixed or ():
            key = record.sensor_id, record.timestamp_utc
            if key in seen or record.sensor_id in MOBILE_IDS:
                raise ValueError(
                    "El replay contiene identificadores o mediciones duplicados"
                )
            seen.add(key)
            if (
                record.sensor_type is not SensorType.FIXED
                or not record.is_accepted
                or record.timestamp_utc != self.clock.utc_at(record.simulation_time_s)
                or not 0 <= record.simulation_time_s <= self.duration_s
            ):
                raise ValueError(
                    "Las observaciones de replay deben ser fijas y pertenecer a la ventana"
                )
            self._historical[record.simulation_time_s].append(record)

        events = {0.0, self.duration_s, *request.config.output_times_s}
        events.update(
            index * STEP_SECONDS
            for index in range(1, int(self.duration_s / STEP_SECONDS) + 1)
        )
        events.update(request.weather.transition_times_s)
        for source in request.sources:
            events.update(source.transition_times_s)
        if request.regional_boundary is not None:
            events.update(request.regional_boundary.transition_times_s)
        if episode.historical_fixed is None:
            for config in fixed_configs:
                count = math.floor(
                    (self.duration_s - config.first_due_time_s)
                    / config.sampling_interval_s
                )
                events.update(
                    config.first_due_time_s + i * config.sampling_interval_s
                    for i in range(count + 1)
                )
        events.update(self._historical)
        if any(not math.isfinite(t) or t < 0 for t in events):
            raise ValueError(
                "Los calendarios deben contener tiempos finitos no negativos"
            )
        self.events = tuple(sorted(t for t in events if t <= self.duration_s))
        self.attempts: list[MeasurementRecord] = []

    def initial_batch(self) -> MeasurementBatch | None:
        return self._sample(0.0)

    def advance(
        self, positions: tuple[tuple[float, float], ...]
    ) -> list[MeasurementBatch]:
        self.positions.positions = dict(zip(MOBILE_IDS, positions, strict=True))
        start = self.state.simulation_time_s
        end = start + STEP_SECONDS
        if end > self.duration_s:
            raise RuntimeError("El episodio ya terminó")
        batches = []
        for time in self.events:
            if not start < time <= end:
                continue
            current = self.state.simulation_time_s
            self.state, _budget = self.model.step(
                self.state,
                duration_s=time - current,
                forcing=self.request.weather.forcing_at_elapsed_s(current),
                sources=self.request.sources,
                boundary_mode=self.request.config.boundary_mode,
                regional_boundary=(
                    self.request.regional_boundary.boundary_at_elapsed_s(current)
                    if self.request.regional_boundary is not None
                    else None
                ),
            )
            if batch := self._sample(time):
                batches.append(batch)
        return batches

    def _sample(self, time: float) -> MeasurementBatch | None:
        arguments = {
            "timestamp_at": self.clock.utc_at,
            "concentration_at": lambda _time: self.state.concentration_ug_m3,
        }
        records = (
            *self.fixed.sample_due(time, **arguments),
            *self._historical.get(time, ()),
            *self.mobile.sample_due(time, **arguments),
        )
        self.attempts.extend(records)
        accepted = tuple(
            sorted((r for r in records if r.is_accepted), key=lambda r: r.sensor_id)
        )
        return (
            MeasurementBatch(time, self.clock.utc_at(time), accepted)
            if accepted
            else None
        )
