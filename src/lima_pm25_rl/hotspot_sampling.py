"""Control de foco persistente/pulsante y geometría de muestreo; sin aprendizaje."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch
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
from lima_pm25.estimation import (
    SequentialUpdateConfig,
    load_change10_evaluation_profile,
)
from lima_pm25.simulation import (
    AreaSource,
    EmissionProfile,
    GaussianKernelSource,
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

from .arrival_experiment import _write_json
from .comparison import _validate_seeds
from .env import LimaPM25Env
from .episodes import Episode
from .gp_diagnostics import spatial_metrics
from .routing import RoadRouter
from .simulation import STEP_SECONDS
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256
from .value_diagnostics import digest

DEFAULT_CONFIG = PROJECT_ROOT / "config/hotspot-sampling.json"
PLANS = ("center", "center_and_surroundings")
REGIMES = ("persistent", "pulsed")


def validate_settings(spec):
    _validate_seeds(spec["sensor_seeds"])
    for key in (
        "grid_cells",
        "steps",
        "initial_common_ticks",
        "excursion_cells",
        "pulse_on_ticks",
        "pulse_period_ticks",
    ):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f"{key} debe ser entero positivo")
    if (
        spec["cell_size_m"] != 200
        or spec["grid_cells"] % 2 != 1
        or spec["grid_cells"] // 2 <= spec["excursion_cells"]
        or spec["initial_common_ticks"] % 2
        or spec["initial_common_ticks"] >= spec["steps"]
        or spec["excursion_cells"] < 3
        or spec["pulse_on_ticks"] >= spec["pulse_period_ticks"]
    ):
        raise ValueError("Geometría, prefijo o pulso incompatibles con la prueba")
    for key in (
        "focus_sigma_m",
        "peak_excess_ug_m3",
        "relaxation_time_s",
        "core_radius_m",
        "background_from_m",
        "high_excess_threshold_ug_m3",
    ):
        if not np.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"{key} debe ser positivo y finito")
    if (
        not 0
        < spec["core_radius_m"]
        < spec["background_from_m"]
        < spec["grid_cells"] * spec["cell_size_m"] / 2
    ):
        raise ValueError(
            "Las regiones de evaluación deben ser distintas y estar dentro del dominio"
        )
    if (
        not np.isfinite(spec["mobile_noise_stddev_ug_m3"])
        or spec["mobile_noise_stddev_ug_m3"] < 0
    ):
        raise ValueError("Ruido móvil inválido")
    for key in ("background_ug_m3", "concentration_priority", "distance_cost_per_km"):
        if not np.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError(f"{key} debe ser finito y no negativo")
    if len(set(spec["map_ticks"])) != len(spec["map_ticks"]) or any(
        type(t) is not int or not 0 <= t <= spec["steps"] for t in spec["map_ticks"]
    ):
        raise ValueError("Ticks de mapas inválidos")


def build_domain(spec):
    count, size = spec["grid_cells"], spec["cell_size_m"]
    boundary = box(0, 0, count * size, count * size)
    grid = build_grid(boundary, "EPSG:32718", size)
    graph = nx.MultiDiGraph(crs=grid.crs)
    for row in range(count):
        for col in range(count):
            x, y = grid.cell_center(row, col)
            graph.add_node((row, col), x=x, y=y)
    for row, col in tuple(graph):
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            other = row + dr, col + dc
            if other in graph:
                points = (grid.cell_center(row, col), grid.cell_center(*other))
                graph.add_edge(
                    (row, col), other, length=size, geometry=LineString(points)
                )
    return SpatialDomain(
        DomainConfig(cell_size_m=size),
        graph,
        boundary,
        grid,
        "controlled-hotspot",
        "controlled-hotspot",
    )


def initial_nodes(spec):
    mid = spec["grid_cells"] // 2
    return ((mid, mid), (mid, mid - 1), (mid, mid + 1))


def prescribed_actions(spec, plan):
    """Plan geométrico previo: no consulta concentración, incertidumbre o verdad."""
    if plan not in PLANS:
        raise ValueError("Plan desconocido")
    steps, prefix, extent = (
        spec["steps"],
        spec["initial_common_ticks"],
        spec["excursion_cells"],
    )
    north = ([0, 4] * steps)[:steps]
    west = ([6, 2] * steps)[:steps]
    if plan == "center_and_surroundings":
        cycle = (
            [2] * (extent - 1)
            + [4] * (2 * extent)
            + [6] * (extent - 1)
            + [0] * (2 * extent)
        )
        west = (west[:prefix] + [6] * (extent - 1) + [0] * extent + cycle * steps)[
            :steps
        ]
    east = [(action + 4) % 8 for action in west]
    return np.asarray(
        [[a, 8 + b, 16 + c] for a, b, c in zip(north, west, east, strict=True)],
        dtype=np.int64,
    )


def preview_plan(router, starts, actions):
    """Comprobar reservas y distancias en el decodificador sin ejecutar la física."""
    nodes = list(starts)
    rows = []
    for joint in actions:
        occupied = [router.cell(node) for node in nodes]
        reserved, routes = set(), []
        for assignment in joint:
            vehicle, direction = divmod(int(assignment), 8)
            route = router.plan(nodes[vehicle], direction)
            if (
                route is None
                or route.cell in reserved
                or route.cell in {c for i, c in enumerate(occupied) if i != vehicle}
            ):
                raise ValueError(
                    "El plan prefijado no respeta conectividad, ocupaciones o reservas"
                )
            reserved.add(route.cell)
            routes.append((vehicle, route))
        for vehicle, route in routes:
            nodes[vehicle] = route.destination
        rows.append(
            {
                "positions_m": [router.position(n) for n in nodes],
                "route_distance_m": [r.distance_m for _, r in routes],
            }
        )
    return rows


def build_episode(domain, spec, regime, sensor_seed):
    if regime not in REGIMES:
        raise ValueError("Régimen desconocido")
    grid = domain.grid
    mid = spec["grid_cells"] // 2
    center = grid.cell_center(mid, mid)
    height = 200.0
    # Control numérico: fuentes compensadoras + retirada exponencial en core.
    # Sin viento/difusión, el foco persistente es equilibrio exacto a cada tick.
    physics = PM25PhysicsConfig(height, 0, height / spec["relaxation_time_s"], 0)
    model = PM25Transport(grid, physics)
    duration = spec["steps"] * STEP_SECONDS
    background = spec["background_ug_m3"]
    provenance = "Control sintético idealizado: balance artificial de fuentes y retirada; no parámetros atmosféricos de Lima"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    compensation = np.expm1(STEP_SECONDS / spec["relaxation_time_s"]) / STEP_SECONDS
    profile = EmissionProfile(
        tuple(
            (
                tick * STEP_SECONDS,
                1.0
                if regime == "persistent"
                or tick % spec["pulse_period_ticks"] < spec["pulse_on_ticks"]
                else 0.0,
            )
            for tick in range(spec["steps"] + 1)
        )
    )
    source = GaussianKernelSource(
        "controlled-focus",
        *center,
        1.0,
        spec["focus_sigma_m"],
        ends_at_s=duration,
        profile=profile,
        provenance=provenance,
    )
    weights = source.spatial_weights(grid)
    shape = weights / weights.max()
    source = replace(
        source,
        emission_rate_ug_s=spec["peak_excess_ug_m3"]
        * grid.cell_area_m2
        * height
        * compensation
        / weights.max(),
    )
    compensation_source = AreaSource(
        "controlled-background",
        domain.boundary,
        background * height * compensation,
        provenance,
        0,
        duration,
    )
    field = background + spec["peak_excess_ug_m3"] * shape
    stations = tuple(
        FixedStationSite(
            f"CONTROL_FIXED_{i}",
            0,
            0,
            None,
            *grid.cell_center(*cell),
            grid.crs,
            StationProvenance(
                "controlled-hotspot", "1", "Estación artificial, no SENAMHI"
            ),
        )
        for i, cell in enumerate(
            (
                (0, 0),
                (0, spec["grid_cells"] - 1),
                (spec["grid_cells"] - 1, 0),
                (spec["grid_cells"] - 1, spec["grid_cells"] - 1),
            )
        )
    )
    return Episode(
        SimulationRunRequest(
            model=model,
            initial_state=model.initialize_from_array(
                field,
                ambient_baseline=AmbientBaseline(
                    "controlled-background", background, provenance
                ),
            ),
            weather=SyntheticWeatherWindow(
                start,
                duration,
                (
                    ScheduledWeatherValue(
                        0, AtmosphericForcing("controlled-calm", provenance, 0, 0, 0)
                    ),
                ),
            ),
            config=SimulationRunConfig(start, duration, 300, seed=sensor_seed),
            sources=(compensation_source, source),
            station_catalog=FixedStationCatalog(stations),
            sensor_configs=tuple(
                SensorConfig(
                    sensor_id=s.site_id,
                    sensor_type=SensorType.FIXED,
                    sampling_interval_s=1800,
                    first_due_time_s=0,
                    provenance="Lectura fija sintética ideal",
                    station_site_id=s.site_id,
                )
                for s in stations
            ),
        )
    )


def region_masks(domain, spec):
    mid = spec["grid_cells"] // 2
    center = domain.grid.cell_center(mid, mid)
    xy = np.asarray(
        [
            domain.grid.cell_center(int(r), int(c))
            for r, c in np.argwhere(domain.grid.active_mask)
        ]
    )
    distances = np.linalg.norm(xy - center, axis=1)
    regions = {
        "core": distances <= spec["core_radius_m"],
        "surroundings": (distances > spec["core_radius_m"])
        & (distances < spec["background_from_m"]),
        "background": distances >= spec["background_from_m"],
    }
    if not all(mask.any() for mask in regions.values()):
        raise ValueError("La geometría debe contener las tres regiones")
    return regions


def frame_metrics(truth, mean, std, regions, threshold, cell_area):
    error = mean - truth
    result = {"global_mse": float(np.mean(error**2))}
    for name, mask in regions.items():
        values = spatial_metrics(truth[mask], mean[mask], std[mask])
        result.update(
            {
                f"{name}_{key}": values[key]
                for key in ("mse", "coverage_95", "mean_variance")
            }
        )
    result["background_bias"] = float(error[regions["background"]].mean())
    result["background_overestimation_mse"] = float(
        np.mean(np.maximum(error[regions["background"]], 0) ** 2)
    )
    true_high, predicted_high = truth >= threshold, mean >= threshold
    result["false_high_area_km2"] = float(
        np.sum(predicted_high & ~true_high) * cell_area / 1e6
    )
    result["true_high_area_km2"] = float(true_high.sum() * cell_area / 1e6)
    result["truth_peak"] = float(truth[regions["core"]].max())
    result["predicted_peak"] = float(mean[regions["core"]].max())
    result["peak_abs_error"] = abs(result["predicted_peak"] - result["truth_peak"])
    result["active_focus"] = bool(result["truth_peak"] >= threshold)
    result["detected_focus"] = bool(result["predicted_peak"] >= threshold)
    return result


def plot_frame(env, truth, path, tick, spec, caption):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    mask = env.domain.grid.active_mask
    mean, std = env._posterior.mean_ug_m3, env._posterior.standard_deviation_ug_m3
    figure = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(figure)
    gp = env.estimator_config.reconstruction_config
    maximum = spec["background_ug_m3"] + 1.5 * spec["peak_excess_ug_m3"]
    maxima = (maximum, maximum, gp.prior_stddev_ug_m3, 1.5 * spec["peak_excess_ug_m3"])
    for i, (axis, values, title) in enumerate(
        zip(
            figure.subplots(1, 4),
            (truth, mean, std, np.abs(mean - truth)),
            ("Verdad privada", "Media GP", "Desviación GP", "Error absoluto"),
            strict=True,
        )
    ):
        image = axis.imshow(
            np.ma.array(values, mask=~mask),
            origin="lower",
            extent=env.domain.grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
            vmin=0,
            vmax=maxima[i],
            cmap="viridis",
        )
        for route in env._routes:
            points = np.asarray(route) / 1000
            axis.plot(points[:, 0], points[:, 1], "w-", linewidth=0.5)
        for kind, color, marker in (
            (SensorType.FIXED, "red", "^"),
            (SensorType.MOBILE, "white", "."),
        ):
            samples = [r for r in env._records if r.sensor_type is kind]
            axis.scatter(
                [r.x_m / 1000 for r in samples],
                [r.y_m / 1000 for r in samples],
                c=color,
                marker=marker,
                s=5,
            )
        axis.set(
            title=title, xlabel="Este sintético (km)", ylabel="Norte sintético (km)"
        )
        figure.colorbar(image, ax=axis, shrink=0.7)
    figure.suptitle(f"{caption} | tick {tick}; µg/m³")
    with path.open("xb") as file:
        figure.savefig(file, format="png", dpi=110)


def run_mission(env, spec, plan, sensor_seed, output, *, maps, regime):
    output.mkdir(parents=True, exist_ok=False)
    env.reset(seed=0, options={"scenario_seed": sensor_seed})
    grid = env.domain.grid
    regions = region_masks(env.domain, spec)
    threshold = spec["background_ug_m3"] + spec["high_excess_threshold_ug_m3"]
    field_hash = hashlib.sha256()
    initial = env.positions
    frames, rows, trace = [], [], []

    def record(tick):
        truth = env._simulation.state.concentration_ug_m3
        field_hash.update(truth[grid.active_mask].tobytes())
        values = [
            a[grid.active_mask].copy()
            for a in (
                truth,
                env._posterior.mean_ug_m3,
                env._posterior.standard_deviation_ug_m3,
            )
        ]
        frames.append(values)
        row = {
            "tick": tick,
            "elapsed_s": tick * STEP_SECONDS,
            **frame_metrics(*values, regions, threshold, grid.cell_area_m2),
        }
        change = row["truth_peak"] - rows[-1]["truth_peak"] if rows else 0
        row["phase"] = (
            "initial"
            if tick == 0
            else "rising"
            if change > 1e-8
            else "declining"
            if change < -1e-8
            else "steady"
        )
        rows.append(row)
        if maps and tick in spec["map_ticks"]:
            plot_frame(
                env,
                truth,
                output / f"tick-{tick:02d}.png",
                tick,
                spec,
                f"{regime} | {plan} | ruido {sensor_seed}",
            )

    record(0)
    for tick, actions in enumerate(prescribed_actions(spec, plan), 1):
        rng_before = digest(env._simulation.mobile._rng.bit_generator.state)
        _, reward, terminated, truncated, info = env.step(actions)
        if (
            info["invalid_actions"].any()
            or info["held_vehicles"].any()
            or (terminated and not truncated)
        ):
            raise RuntimeError(
                "El plan geométrico no pudo ejecutar todos sus movimientos"
            )
        trace.append(
            {
                "tick": tick,
                "actions": actions.tolist(),
                "positions_m": env.positions,
                "route_distance_m": info["route_distance_m"].tolist(),
                "reward": reward,
                "sensor_rng_before_sha256": rng_before,
                "sensor_rng_after_sha256": digest(
                    env._simulation.mobile._rng.bit_generator.state
                ),
            }
        )
        record(tick)
    duration = spec["steps"] * STEP_SECONDS
    summary = {
        "distance_m": float(env._distance.sum()),
        "mobile_records": sum(r.sensor_type is SensorType.MOBILE for r in env._records),
    }
    for name in (
        "global_mse",
        "core_mse",
        "surroundings_mse",
        "background_mse",
        "peak_abs_error",
        "false_high_area_km2",
        "background_overestimation_mse",
    ):
        summary[f"temporal_{name}"] = float(
            np.trapezoid([r[name] for r in rows], dx=STEP_SECONDS) / duration
        )
        summary[f"final_{name}"] = rows[-1][name]
    for phase in ("steady", "rising", "declining"):
        subset = [r for r in rows[1:] if r["phase"] == phase]
        summary[f"{phase}_ticks"] = len(subset)
        summary[f"{phase}_core_mse"] = (
            float(np.mean([r["core_mse"] for r in subset])) if subset else None
        )
    active = [r for r in rows[1:] if r["active_focus"]]
    inactive = [r for r in rows[1:] if not r["active_focus"]]
    summary["active_detection_rate"] = (
        float(np.mean([r["detected_focus"] for r in active])) if active else None
    )
    summary["inactive_false_alarm_rate"] = (
        float(np.mean([r["detected_focus"] for r in inactive])) if inactive else None
    )
    report = {
        "regime": regime,
        "plan": plan,
        "sensor_seed": sensor_seed,
        "physics": asdict(env._simulation.model.physics),
        "sources": [
            source.evidence_descriptor() for source in env._simulation.request.sources
        ],
        "summary": summary,
        "frames": rows,
        "ticks": trace,
        "routes_m": env._routes,
        "measurements": [
            {**asdict(r), "timestamp_utc": r.timestamp_utc.isoformat()}
            for r in env._records
        ],
        "signatures": {
            "field_sha256": field_hash.hexdigest(),
            "initial_positions_sha256": digest(initial),
            "fixed_measurements_sha256": digest(
                [asdict(r) for r in env._records if r.sensor_type is SensorType.FIXED]
            ),
            "horizon_s": duration,
        },
    }
    _write_json(output / "report.json", report)
    array = np.asarray(frames)
    with (output / "fields-private.npz").open("xb") as file:
        np.savez_compressed(
            file,
            truth=array[:, 0],
            mean=array[:, 1],
            std=array[:, 2],
            active_cells=np.argwhere(grid.active_mask),
            grid_shape=grid.shape,
            grid_bounds_m=grid.grid_bounds_m,
            **{f"region_{name}": mask for name, mask in regions.items()},
        )
    return report, array[:, 2]


def run_experiment(spec, gp, output, *, maps=True):
    validate_settings(spec)
    domain = build_domain(spec)
    router = RoadRouter(domain)
    previews = {
        p: preview_plan(router, initial_nodes(spec), prescribed_actions(spec, p))
        for p in PLANS
    }
    distances = {
        p: np.asarray([r["route_distance_m"] for r in previews[p]]) for p in PLANS
    }
    if not np.allclose(distances[PLANS[0]], distances[PLANS[1]], rtol=0, atol=1e-6):
        raise ValueError(
            "Los planes no tienen la misma distancia vial por tick y vehículo"
        )
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            "settings": spec,
            "gp": asdict(gp.reconstruction_config),
            "gp_profile_sha256": gp.profile_sha256,
            "gp_calibration_sha256": gp.calibration_artifact_sha256,
            "history_duration_s": gp.history_duration.total_seconds(),
            "training_executed": False,
            "purpose": "controlled_mechanism_not_lima_validation",
            "code_sha256": _sha256(Path(__file__)),
            "region_cells": {
                k: int(v.sum()) for k, v in region_masks(domain, spec).items()
            },
            "distance_per_plan_m": float(distances[PLANS[0]].sum()),
            "artificial_physics": {
                "mixing_height_m": 200,
                "diffusivity_m2_s": 0,
                "wind_speed_m_s": 0,
                "dry_deposition_velocity_m_s": 200 / spec["relaxation_time_s"],
                "purpose": "Retirada y emisiones compensadoras para un equilibrio discreto controlado; no física calibrada para Lima",
            },
            "criterion": "Menor MSE temporal exterior sin aumentar MSE temporal del núcleo ni error absoluto medio del pico; informar consistencia entre ruidos.",
        },
    )
    _write_json(output / "route-preview.json", previews)
    reports, deviations, paired, rows = {}, {}, [], []
    for regime in REGIMES:
        field_reference = None
        env = LimaPM25Env(
            domain=domain,
            estimator_config=gp,
            episode_factory=lambda seed, regime=regime: build_episode(
                domain, spec, regime, seed
            ),
            initial_nodes=initial_nodes(spec),
            mobile_noise_stddev_ug_m3=spec["mobile_noise_stddev_ug_m3"],
            concentration_priority=spec["concentration_priority"],
            distance_cost_per_km=spec["distance_cost_per_km"],
        )
        try:
            for seed_index, seed in enumerate(spec["sensor_seeds"]):
                for plan in PLANS:
                    folder = output / regime / f"noise-{seed}" / plan
                    report, std = run_mission(
                        env,
                        spec,
                        plan,
                        seed,
                        folder,
                        maps=maps and seed_index == 0,
                        regime=regime,
                    )
                    reports[regime, seed, plan] = report
                    if (
                        field_reference is not None
                        and field_reference != report["signatures"]["field_sha256"]
                    ):
                        raise ValueError(
                            "El ruido instrumental cambió el campo sintético"
                        )
                    field_reference = report["signatures"]["field_sha256"]
                    deviations[regime, seed, plan] = std
                    rows.append(
                        {
                            "regime": regime,
                            "sensor_seed": seed,
                            "plan": plan,
                            **report["summary"],
                        }
                    )
                    print(
                        f"{regime} | ruido {seed} | {plan}: RMSE temporal={np.sqrt(report['summary']['temporal_global_mse']):.4f}",
                        flush=True,
                    )
                a, b = (reports[regime, seed, p] for p in PLANS)
                if a["signatures"] != b["signatures"]:
                    raise ValueError(
                        "Los planes recibieron campos, fijas, inicio u horizonte diferentes"
                    )
                for x, y in zip(a["ticks"], b["ticks"], strict=True):
                    if any(
                        x[key] != y[key]
                        for key in (
                            "sensor_rng_before_sha256",
                            "sensor_rng_after_sha256",
                        )
                    ) or not np.allclose(
                        x["route_distance_m"], y["route_distance_m"], rtol=0, atol=1e-6
                    ):
                        raise ValueError("Ruido o distancia no emparejados")
                if (
                    a["summary"]["mobile_records"] != b["summary"]["mobile_records"]
                    or a["summary"]["mobile_records"] != 3 * spec["steps"]
                ):
                    raise ValueError("Presupuesto de lecturas distinto")
                paired.append(
                    {
                        "regime": regime,
                        "sensor_seed": seed,
                        "paired": True,
                        **{
                            f"surroundings_minus_center_{key}": b["summary"][key]
                            - a["summary"][key]
                            for key in (
                                "temporal_core_mse",
                                "temporal_background_mse",
                                "temporal_global_mse",
                                "temporal_peak_abs_error",
                            )
                        },
                    }
                )
        finally:
            env.close()
    _write_json(output / "cases.json", rows)
    summaries, decisions = [], []
    for regime in REGIMES:
        group = {}
        for plan in PLANS:
            selected = [r for r in rows if r["regime"] == regime and r["plan"] == plan]
            summary = {"regime": regime, "plan": plan}
            for key in selected[0]:
                if key in {"regime", "plan", "sensor_seed"}:
                    continue
                values = [r[key] for r in selected if r[key] is not None]
                summary[key] = float(np.mean(values)) if values else None
                if key.endswith("_mse") and summary[key] is not None:
                    summary[key[:-3] + "rmse"] = float(np.sqrt(summary[key]))
            summaries.append(summary)
            group[plan] = summary
        a, b = (group[p] for p in PLANS)
        decisions.append(
            {
                "regime": regime,
                "lower_background_error": b["temporal_background_mse"]
                < a["temporal_background_mse"],
                "core_error_not_higher": b["temporal_core_mse"]
                <= a["temporal_core_mse"],
                "peak_error_not_higher": b["temporal_peak_abs_error"]
                <= a["temporal_peak_abs_error"],
            }
        )
    covariance_check = max(
        float(
            np.max(np.abs(deviations["persistent", s, p] - deviations["pulsed", s, p]))
        )
        for s in spec["sensor_seeds"]
        for p in PLANS
    )
    result = {
        "summaries": summaries,
        "paired": paired,
        "criteria_by_regime": decisions,
        "max_std_difference_between_regimes": covariance_check,
        "interpretation": "Plan geométrico prefijado con foco inicialmente próximo, no búsqueda aprendida. Un resultado negativo no demuestra que ningún muestreo pueda ayudar. No hay persistencia local adaptativa.",
    }
    _write_json(output / "summary.json", result)
    return result


def load_gp(root):
    profile = load_change10_evaluation_profile(
        root / "config/estimation_evaluation.json"
    )
    return SequentialUpdateConfig.from_change10_profile(
        replace(
            profile, calibration_artifact_path=root / profile.calibration_artifact_path
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Control de foco persistente/pulsante y muestreo de alrededores"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    validate_settings(spec)
    root = (args.config.resolve().parent / spec["core_root"]).resolve()
    gp = load_gp(root)
    if (
        gp.reconstruction_config.x_length_scale_m != 2000
        or gp.reconstruction_config.y_length_scale_m != 2000
        or gp.reconstruction_config.temporal_length_scale_s != 3600
        or gp.history_duration.total_seconds() != 21600
    ):
        parser.error(
            "El control requiere el GP original 2000 m/3600 s e historial de 6 h"
        )
    if args.check:
        if args.output:
            parser.error("--check no escribe salidas")
        domain = build_domain(spec)
        router = RoadRouter(domain)
        previews = [
            preview_plan(router, initial_nodes(spec), prescribed_actions(spec, p))
            for p in PLANS
        ]
        distances = [
            np.asarray([r["route_distance_m"] for r in preview]) for preview in previews
        ]
        if not np.allclose(*distances, rtol=0, atol=1e-6):
            raise ValueError("Los planes no tienen igual distancia por tick y vehículo")
        print(
            json.dumps(
                {
                    "simulation_executed": False,
                    "training_executed": False,
                    "missions": len(REGIMES) * len(PLANS) * len(spec["sensor_seeds"]),
                    "mobile_readings_per_mission": 3 * spec["steps"],
                    "road_distance_per_mission_m": float(distances[0].sum()),
                },
                indent=2,
            )
        )
        return
    torch.set_num_threads(1)
    output = args.output or CACHE_DIRECTORY / "hotspot-sampling" / datetime.now(
        UTC
    ).strftime("%Y%m%d-%H%M%S-%f")
    print(f"Salida: {output}", flush=True)
    run_experiment(spec, gp, output)


if __name__ == "__main__":
    main()
