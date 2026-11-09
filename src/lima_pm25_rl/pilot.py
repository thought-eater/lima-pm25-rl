"""Caso artificial pequeño para comprobar aprendizaje, no para evaluar la tesis."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import networkx as nx
import numpy as np
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
    PM25PhysicsConfig,
    PM25Transport,
    ScheduledWeatherValue,
    SensorConfig,
    SensorType,
    SimulationRunConfig,
    SimulationRunRequest,
    SyntheticWeatherWindow,
)
from shapely.geometry import LineString, box

from .env import LimaPM25Env
from .episodes import Episode
from .network import BiLGD3QN, NeuralController


def controlled_environment() -> LimaPM25Env:
    """Dos destinos para el vehículo 0; los otros dos permanecen sin opciones.

    Una estación fija ya informa el este. Medir al oeste debe aportar más
    información. El campo y la configuración son exclusivamente de software.
    """
    boundary = box(0, 0, 1000, 1000)
    grid = build_grid(boundary, "EPSG:32718", 200)
    graph = nx.MultiDiGraph(crs=grid.crs)
    positions = {
        "start": (500, 500),
        "east": (700, 500),
        "west": (300, 500),
        "held-a": (100, 100),
        "held-b": (900, 900),
    }
    for name, (x, y) in positions.items():
        graph.add_node(name, x=x, y=y)
    for name in ("east", "west"):
        graph.add_edge(
            "start",
            name,
            length=200,
            geometry=LineString((positions["start"], positions[name])),
        )
    domain = SpatialDomain(
        DomainConfig(cell_size_m=200),
        graph,
        boundary,
        grid,
        "software-pilot",
        "software-pilot",
    )
    model = PM25Transport(grid, PM25PhysicsConfig(200, 0, 0, 0))
    start = datetime(2025, 1, 1, tzinfo=UTC)
    station = FixedStationSite(
        "PILOT_FIXED",
        0,
        0,
        None,
        700,
        500,
        grid.crs,
        StationProvenance("software-pilot", "1", "artificial"),
    )
    template = SimulationRunRequest(
        model=model,
        initial_state=model.initialize_uniform(
            AmbientBaseline("software-pilot", 10, "Campo uniforme artificial")
        ),
        weather=SyntheticWeatherWindow(
            start,
            300,
            (
                ScheduledWeatherValue(
                    0,
                    AtmosphericForcing(
                        "software-pilot", "Meteorología artificial", 0, 0, 0
                    ),
                ),
            ),
        ),
        config=SimulationRunConfig(start, 300, 300, 0),
        station_catalog=FixedStationCatalog((station,)),
        sensor_configs=(
            SensorConfig(
                sensor_id=station.site_id,
                sensor_type=SensorType.FIXED,
                sampling_interval_s=3600,
                first_due_time_s=0,
                provenance="Sensor artificial de prueba",
                station_site_id=station.site_id,
            ),
        ),
    )
    gp = SequentialUpdateConfig(
        reconstruction_config=ReconstructionConfig(
            prior_mean_ug_m3=10,
            prior_stddev_ug_m3=5,
            x_length_scale_m=180,
            y_length_scale_m=180,
            temporal_length_scale_s=3600,
        ),
        history_duration=timedelta(hours=6),
        variance_numerical_tolerance_ug_m3_squared=1e-9,
        profile_sha256="0" * 64,
        profile_status="software-pilot",
        profile_source="artificial",
        calibration_artifact_sha256="0" * 64,
        calibration_artifact_source="artificial",
    )
    return LimaPM25Env(
        domain=domain,
        estimator_config=gp,
        episode_factory=lambda seed: Episode(
            replace(template, config=replace(template.config, seed=seed))
        ),
        initial_nodes=("start", "held-a", "held-b"),
    )


def controlled_score(network: BiLGD3QN, history_length: int) -> dict[str, float | int]:
    env = controlled_environment()
    observation, _ = env.reset(seed=0)
    policy = NeuralController(network, history_length=history_length)
    action = policy.act(env, observation)
    _, reward, _, _, info = env.step(action)
    if info["invalid_actions"].any():
        raise RuntimeError("El piloto produjo un movimiento inválido")
    return {"action": int(action[0]), "reward": float(reward)}


def controlled_oracle() -> dict[int, float]:
    env = controlled_environment()
    rewards = {}
    for direction in (2, 6):
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(np.array([direction, 8, 16]))
        rewards[direction] = reward
    return rewards
