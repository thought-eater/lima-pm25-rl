"""Episodios de core y replay con observaciones históricas sin reinterpretarlas."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from lima_pm25.domain import ScenarioFamilyPurpose, ScenarioRunIdentity
from lima_pm25.estimation import HistoricalPM25Dataset
from lima_pm25.simulation import (
    EpisodeGenerationConfig,
    EpisodeWeatherMode,
    HistoricalWeatherDataset,
    MeasurementRecord,
    SimulationRunRequest,
    compose_episode,
)


@dataclass(frozen=True)
class Episode:
    request: SimulationRunRequest
    # None simula los fijos; una tupla (incluso vacía) usa solo observaciones reales.
    historical_fixed: tuple[MeasurementRecord, ...] | None = None


@dataclass(frozen=True)
class GeneratedEpisodes:
    """Delega meteorología, fuentes, estrés y particiones al generador de core.

    gp_fit_end_utc es el extremo exclusivo de los datos usados para ajustar el
    GP/normalizaciones. Evaluación no puede comenzar antes de ese extremo.
    La selección de rangos nominales y del currículo corresponde al experimento.
    """

    template: SimulationRunRequest
    weather: HistoricalWeatherDataset
    config: EpisodeGenerationConfig
    gp_fit_end_utc: datetime
    partition: str = "training"
    weather_mode: EpisodeWeatherMode | str | None = None

    def __post_init__(self) -> None:
        if self.config.evaluation_start_utc < self.gp_fit_end_utc:
            raise ValueError("La evaluación se solapa con el ajuste del GP")
        if self.partition not in {"training", "evaluation"}:
            raise ValueError("partition debe ser training o evaluation")

    def __call__(self, seed: int) -> Episode:
        return Episode(
            compose_episode(
                self.template,
                self.weather,
                self.config,
                seed=seed,
                partition=self.partition,
                weather_mode=self.weather_mode,
            )
        )


@dataclass(frozen=True)
class HistoricalReplay:
    """Ventana histórica final: todos los fijos observados del catálogo y móviles simulados."""

    template: SimulationRunRequest
    weather: HistoricalWeatherDataset
    pm25: HistoricalPM25Dataset
    start_utc: datetime
    gp_fit_end_utc: datetime
    policy_fit_end_utc: datetime
    historical_noise_stddev_ug_m3: float = 0.0

    def __post_init__(self) -> None:
        if self.start_utc < max(self.gp_fit_end_utc, self.policy_fit_end_utc):
            raise ValueError(
                "El replay debe estar reservado después del ajuste del GP y la política"
            )
        if self.template.station_catalog is None:
            raise ValueError("El replay necesita el catálogo de estaciones del dominio")

    def __call__(self, seed: int) -> Episode:
        duration = timedelta(seconds=self.template.config.duration_s)
        assert self.template.station_catalog is not None
        weather = self.weather.select_window(self.start_utc, duration)
        request = replace(
            self.template,
            weather=weather,
            config=replace(self.template.config, start_utc=self.start_utc, seed=seed),
            scenario_identity=ScenarioRunIdentity(
                "rl-replay",
                ScenarioFamilyPurpose.HISTORICAL_REPLAY,
                self.start_utc.isoformat(),
                True,
            ),
        )
        records = self.pm25.observed_measurements(
            grid=request.model.grid,
            replay_start_utc=self.start_utc,
            end_utc=self.start_utc + duration,
            noise_stddev_ug_m3=self.historical_noise_stddev_ug_m3,
            station_ids=tuple(
                site.site_id for site in self.template.station_catalog.sites
            ),
        )
        return Episode(request, records)
