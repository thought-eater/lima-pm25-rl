"""Carga explícita y de solo lectura de los recursos conservados de core."""

from dataclasses import replace
from pathlib import Path

import osmnx as ox
from lima_pm25.domain import (
    SpatialDomain,
    create_lima_pm25_station_catalog,
    load_grid,
    load_study_environment_config,
)
from lima_pm25.estimation import (
    SequentialUpdateConfig,
    load_change10_evaluation_profile,
)
from lima_pm25.simulation import (
    SensorType,
    SimulationRunRequest,
    create_synthetic_study_family_requests,
)
from shapely.geometry import box


def load_study_resources(
    root: Path | str,
    *,
    member: str = "event",
) -> tuple[SpatialDomain, SimulationRunRequest, SequentialUpdateConfig]:
    """Falla si faltan artefactos; no llama al constructor que descarga mapas.

    La solicitud devuelta es la demostración sintética con evento de core, no
    un currículo nominal calibrado ni un replay histórico.
    """
    root = Path(root).resolve()
    study = load_study_environment_config(root / "config/study_environment.json")
    directory = root / study.domain.cache_directory
    graph = ox.io.load_graphml(directory / "road_graph.graphml")
    grid = load_grid(
        directory / "grid.npz", study.domain.fingerprint, graph.graph["crs"]
    )
    boundary, _ = ox.projection.project_geometry(
        box(*study.domain.bbox), crs="EPSG:4326", to_crs=grid.crs
    )
    domain = SpatialDomain(study.domain, graph, boundary, grid, "retained", "retained")
    stations = create_lima_pm25_station_catalog(domain)
    if member not in {"control", "event"}:
        raise ValueError("member debe ser control o event")
    template = create_synthetic_study_family_requests(
        domain, station_catalog=stations, study_config=study
    )[member]
    template = replace(
        template,
        sensor_configs=tuple(
            c for c in template.sensor_configs if c.sensor_type is SensorType.FIXED
        ),
        mobile_position_providers=(),
    )
    profile = load_change10_evaluation_profile(
        root / "config/estimation_evaluation.json"
    )
    profile = replace(
        profile, calibration_artifact_path=root / profile.calibration_artifact_path
    )
    return domain, template, SequentialUpdateConfig.from_change10_profile(profile)
