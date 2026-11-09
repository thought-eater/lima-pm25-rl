from dataclasses import replace
from datetime import UTC, datetime, timedelta

import networkx as nx
import pytest
from lima_pm25.domain import (
    AmbientBaseline,
    AtmosphericForcing,
    DomainConfig,
    FixedStationCatalog,
    FixedStationSite,
    SpatialDomain,
    StationProvenance,
    build_grid,
)
from lima_pm25.estimation import ReconstructionConfig, SequentialUpdateConfig
from lima_pm25.simulation import (
    EpisodeGenerationConfig,
    HistoricalWeatherDataset,
    PM25PhysicsConfig,
    PM25Transport,
    ScheduledWeatherValue,
    SensorConfig,
    SensorType,
    SimulationRunConfig,
    SimulationRunRequest,
    SyntheticWeatherWindow,
    WeatherRecord,
    WeatherValueProvenance,
)
from shapely.geometry import LineString, box

from lima_pm25_rl import Episode, LimaPM25Env


@pytest.fixture
def domain():
    boundary = box(0, 0, 800, 800)
    grid = build_grid(boundary, "EPSG:32718", 200)
    graph = nx.MultiDiGraph(crs=grid.crs)
    for row in range(4):
        for column in range(4):
            graph.add_node((row, column), x=100 + 200 * column, y=100 + 200 * row)
    for row, column in tuple(graph):
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            other = row + dr, column + dc
            if other in graph:
                first, last = graph.nodes[row, column], graph.nodes[other]
                line = LineString(((first["x"], first["y"]), (last["x"], last["y"])))
                graph.add_edge((row, column), other, length=line.length, geometry=line)
    return SpatialDomain(
        DomainConfig(cell_size_m=200), graph, boundary, grid, "test", "test"
    )


@pytest.fixture
def template(domain):
    model = PM25Transport(
        domain.grid,
        PM25PhysicsConfig(
            mixing_height_m=200,
            diffusivity_m2_s=5,
            dry_deposition_velocity_m_s=0,
            wet_scavenging_per_mm=0,
        ),
    )
    initial = model.initialize_uniform(
        AmbientBaseline("test", 8, "Campo sintético de prueba")
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    weather = SyntheticWeatherWindow(
        start,
        900,
        (
            ScheduledWeatherValue(
                0,
                AtmosphericForcing("test", "Meteorología sintética de prueba", 0, 0, 0),
            ),
        ),
    )
    station = FixedStationSite(
        site_id="FIXED",
        longitude_deg=0,
        latitude_deg=0,
        altitude_m=None,
        x_m=700,
        y_m=700,
        crs=domain.grid.crs,
        provenance=StationProvenance("test", "1", "fixture"),
    )
    return SimulationRunRequest(
        model=model,
        initial_state=initial,
        weather=weather,
        config=SimulationRunConfig(start, 900, 300, seed=0),
        sensor_configs=(
            SensorConfig(
                sensor_id="FIXED",
                sensor_type=SensorType.FIXED,
                sampling_interval_s=600,
                first_due_time_s=0,
                provenance="Sensor fijo sintético de prueba",
                station_site_id="FIXED",
            ),
        ),
        station_catalog=FixedStationCatalog((station,)),
    )


@pytest.fixture
def estimator_config():
    return SequentialUpdateConfig(
        reconstruction_config=ReconstructionConfig(
            prior_mean_ug_m3=10,
            prior_stddev_ug_m3=5,
            x_length_scale_m=300,
            y_length_scale_m=300,
            temporal_length_scale_s=3600,
        ),
        history_duration=timedelta(hours=6),
        variance_numerical_tolerance_ug_m3_squared=1e-9,
        profile_sha256="0" * 64,
        profile_status="test",
        profile_source="fixture",
        calibration_artifact_sha256="0" * 64,
        calibration_artifact_source="fixture",
    )


@pytest.fixture
def make_env(domain, template, estimator_config):
    def build(**kwargs):
        return LimaPM25Env(
            domain=domain,
            episode_factory=kwargs.pop(
                "episode_factory",
                lambda seed: Episode(
                    replace(template, config=replace(template.config, seed=seed))
                ),
            ),
            estimator_config=estimator_config,
            initial_nodes=((0, 0), (0, 3), (3, 0)),
            **kwargs,
        )

    return build


@pytest.fixture
def env(make_env):
    return make_env()


@pytest.fixture
def historical_weather():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    provenance = WeatherValueProvenance("synthetic_test")
    records = tuple(
        WeatherRecord(
            station_id="TEST",
            timestamp_utc=start + timedelta(hours=hour),
            temperature_c=20,
            relative_humidity_pct=70,
            eastward_wind_m_s=1 + (hour % 24) / 24 + hour // 24 * 0.05,
            northward_wind_m_s=0.1 + hour % 3 / 10,
            precipitation_mm_h=0,
            temperature_provenance=provenance,
            relative_humidity_provenance=provenance,
            precipitation_provenance=provenance,
            wind_direction_provenance=provenance,
            wind_speed_provenance=provenance,
        )
        for hour in range(97)
    )
    return HistoricalWeatherDataset(
        "TEST", "unit-weather", "Serie artificial para pruebas de particiones", records
    )


@pytest.fixture
def generation_config():
    return EpisodeGenerationConfig(
        training_start_utc=datetime(2025, 1, 1, tzinfo=UTC),
        training_end_utc=datetime(2025, 1, 3, tzinfo=UTC),
        evaluation_start_utc=datetime(2025, 1, 3, tzinfo=UTC),
        evaluation_end_utc=datetime(2025, 1, 5, tzinfo=UTC),
        weather_weights=(1, 1, 1),
        source_variations=(),
        provenance="Configuración sintética de pruebas, no currículo seleccionado",
        stress=True,
    )
