"""Diagnóstico privado de campos de entrenamiento, sin política ni ajuste del GP."""

import csv
import json
from pathlib import Path

import numpy as np
from lima_pm25.simulation import SensorType, run_simulation

from .episodes import Episode
from .simulation import STEP_SECONDS


def field_diagnostics(fields, active_mask, road_mask):
    """Separar contraste espacial por instante de variación temporal por celda."""
    fields = np.asarray(fields, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    roads = np.asarray(road_mask, dtype=bool)
    if (
        fields.ndim != 3
        or len(fields) < 2
        or fields.shape[1:] != active.shape
        or roads.shape != active.shape
        or not active.any()
        or np.any(roads & ~active)
    ):
        raise ValueError(
            "Se requieren campos [tiempo, filas, columnas] y máscaras válidas"
        )
    if not np.isfinite(fields[:, active]).all() or np.any(fields[:, active] < 0):
        raise ValueError("El campo activo debe ser finito y no negativo")

    ticks = []
    for index, field in enumerate(fields):
        row = {"tick": index, "elapsed_s": index * STEP_SECONDS}
        for name, mask in (("active", active), ("road_support", roads)):
            values = field[mask]
            if not values.size:
                row.update(
                    {
                        f"{name}_{key}": None
                        for key in (
                            "mean",
                            "std",
                            "p05",
                            "p50",
                            "p95",
                            "p99",
                            "max",
                            "p99_minus_p50",
                        )
                    }
                )
                continue
            quantiles = np.percentile(values, (5, 50, 95, 99), method="linear")
            row.update(
                {
                    f"{name}_mean": float(values.mean()),
                    f"{name}_std": float(values.std()),
                    **{
                        f"{name}_{key}": float(value)
                        for key, value in zip(
                            ("p05", "p50", "p95", "p99"), quantiles, strict=True
                        )
                    },
                    f"{name}_max": float(values.max()),
                    f"{name}_p99_minus_p50": float(quantiles[3] - quantiles[1]),
                }
            )
        row["change_rmse_ug_m3"] = (
            float(np.sqrt(np.mean((field[active] - fields[index - 1, active]) ** 2)))
            if index
            else None
        )
        ticks.append(row)

    # Son mapas del evaluador: nunca se añaden a observation ni a info.
    temporal_std = np.full(active.shape, np.nan)
    change_rms = np.full(active.shape, np.nan)
    temporal_std[active] = np.std(fields[:, active], axis=0)
    change_rms[active] = np.sqrt(
        np.mean(np.diff(fields[:, active], axis=0) ** 2, axis=0)
    )
    summary = {
        "active_cells": int(active.sum()),
        "road_support_cells": int(roads.sum()),
        "initial_spatial_contrast_ug_m3": ticks[0]["active_p99_minus_p50"],
        "mean_spatial_contrast_ug_m3": float(
            np.mean([row["active_p99_minus_p50"] for row in ticks[1:]])
        ),
        "final_spatial_contrast_ug_m3": ticks[-1]["active_p99_minus_p50"],
        "mean_spatial_std_ug_m3": float(
            np.mean([row["active_std"] for row in ticks[1:]])
        ),
        "mean_temporal_std_ug_m3": float(temporal_std[active].mean()),
        "p95_temporal_std_ug_m3": float(np.percentile(temporal_std[active], 95)),
        "mean_tick_change_rmse_ug_m3": float(
            np.mean([row["change_rmse_ug_m3"] for row in ticks[1:]])
        ),
        "initial_field_persistence_rmse_ug_m3": float(
            np.sqrt(np.mean((fields[1:, active] - fields[0, active]) ** 2))
        ),
    }
    return ticks, summary, temporal_std, change_rms


def simulate_diagnostic(episode: Episode):
    """Mismo calendario de ticks que RL, sin decisiones ni sensores móviles."""
    from dataclasses import replace

    request = episode.request
    metadata = json.loads(request.episode_parameters_json)
    if episode.historical_fixed is not None or metadata.get("partition") != "training":
        raise ValueError(
            "El diagnóstico solo admite episodios generados de entrenamiento"
        )
    count = round(request.config.duration_s / STEP_SECONDS)
    if count < 1 or not np.isclose(count * STEP_SECONDS, request.config.duration_s):
        raise ValueError("El horizonte debe contener ticks completos de 300 s")
    times = tuple(index * STEP_SECONDS for index in range(count + 1))
    # Mantener también salidas anteriores: cambiar fronteras de integración puede
    # cambiar el campo. Los ticks ya pertenecen al calendario del entorno RL.
    config = replace(
        request.config,
        concentration_output_times_s=tuple(
            sorted({*request.config.concentration_output_times_s, *times})
        ),
    )
    result = run_simulation(
        replace(
            request,
            config=config,
            sensor_configs=tuple(
                c for c in request.sensor_configs if c.sensor_type is SensorType.FIXED
            ),
            mobile_position_providers=(),
        )
    )
    frames = {frame.simulation_time_s: frame for frame in result.frames}
    fields = np.stack([frames[time].concentration_ug_m3 for time in times])
    return fields, result.measurement_attempts, result.run_id


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics(
    env, *, seeds: list[int], fit_cases: int, output: Path, metadata: dict
):
    """Auditar casos de entrenamiento; ajustar el control constante solo en un prefijo.

    No llama reset/step del entorno ni inicializa GP, red o controlador. El
    entorno proporciona el mismo generador y soporte vial usados al entrenar.
    """
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or type(fit_cases) is not int
        or not 0 < fit_cases < len(seeds)
    ):
        raise ValueError(
            "Se necesitan semillas únicas y casos distintos de ajuste y diagnóstico"
        )
    episodes = [env._episode_factory(seed) for seed in seeds]
    for episode in episodes:
        parameters = json.loads(episode.request.episode_parameters_json)
        if (
            episode.historical_fixed is not None
            or parameters.get("partition") != "training"
        ):
            raise ValueError(
                "El diagnóstico solo admite episodios generados de entrenamiento"
            )
    output.mkdir(parents=True, exist_ok=False)
    (output / "fields").mkdir()
    configuration = {
        **metadata,
        "purpose": "training_field_diagnostic_not_final_evaluation",
        "training_executed": False,
        "seeds": seeds,
        "constant_fit_cases": fit_cases,
        "constant_fit_rule": "mean_of_accepted_synthetic_fixed_measurements_in_fit_cases_only",
        "road_support_note": "Celdas con soporte vial; no demuestra alcanzabilidad desde una posición inicial.",
        "spatial_summary_rule": "mean_of_per_tick_statistics_excluding_initial_frame",
        "temporal_summary_rule": "per_cell_statistics_including_initial_frame",
        "concentration_unit": "ug/m3",
        "cases": [json.loads(e.request.episode_parameters_json) for e in episodes],
    }
    (output / "config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    grid = env.domain.grid
    summaries, tick_rows, station_rows = [], [], []
    fit_values = []
    constant = None
    for index, (seed, episode) in enumerate(zip(seeds, episodes, strict=True)):
        fields, measurements, run_id = simulate_diagnostic(episode)
        ticks, summary, temporal_std, change_rms = field_diagnostics(
            fields, grid.active_mask, env.router.navigable_mask
        )
        role = "constant_fit" if index < fit_cases else "diagnostic"
        parameters = json.loads(episode.request.episode_parameters_json)
        context = {
            "case": index,
            "seed": seed,
            "role": role,
            "weather_mode": parameters["weather_mode"],
        }
        tick_rows.extend({**context, **row} for row in ticks)
        for record in measurements:
            station_rows.append(
                {
                    **context,
                    "sensor_id": record.sensor_id,
                    "elapsed_s": record.simulation_time_s,
                    "timestamp_utc": record.timestamp_utc.isoformat(),
                    "x_m": record.x_m,
                    "y_m": record.y_m,
                    "value_ug_m3": record.value_ug_m3,
                    "quality_status": record.quality_status,
                    "provenance": record.provenance,
                }
            )
            if index < fit_cases and record.quality_status == "accepted":
                fit_values.append(record.value_ug_m3)
        if index == fit_cases - 1:
            if not fit_values:
                raise ValueError(
                    "No hay mediciones fijas aceptadas para ajustar el predictor constante"
                )
            constant = float(np.mean(fit_values))
        truth = fields[1:, grid.active_mask]
        prior = env.estimator_config.reconstruction_config.prior_mean_ug_m3
        summaries.append(
            {
                **context,
                "start_utc": episode.request.config.start_utc.isoformat(),
                "duration_s": episode.request.config.duration_s,
                "source_ids": ";".join(
                    source.source_id for source in episode.request.sources
                ),
                "field_run_id": run_id,
                **summary,
                "gp_prior_constant_rmse_ug_m3": float(
                    np.sqrt(np.mean((truth - prior) ** 2))
                ),
                "fitted_constant_mse": float(np.mean((truth - constant) ** 2))
                if index >= fit_cases
                else None,
                "fitted_constant_rmse_ug_m3": float(
                    np.sqrt(np.mean((truth - constant) ** 2))
                )
                if index >= fit_cases
                else None,
            }
        )
        np.savez_compressed(
            output / "fields" / f"case-{index:02d}.npz",
            concentration_ug_m3=fields,
            elapsed_s=np.arange(len(fields)) * STEP_SECONDS,
            active_mask=grid.active_mask,
            road_support_mask=env.router.navigable_mask,
            temporal_std_ug_m3=temporal_std,
            tick_change_rms_ug_m3=change_rms,
            bounds_m=grid.grid_bounds_m,
            crs=np.asarray(grid.crs),
        )
        print(
            f"[{index + 1}/{len(seeds)}] {parameters['weather_mode']} semilla={seed}: contraste medio={summary['mean_spatial_contrast_ug_m3']:.4f} µg/m³",
            flush=True,
        )
    _write_csv(output / "cases.csv", summaries)
    _write_csv(output / "ticks.csv", tick_rows)
    _write_csv(output / "stations.csv", station_rows)
    report = {
        "purpose": configuration["purpose"],
        "training_executed": False,
        "cases": len(seeds),
        "constant_fit_cases": fit_cases,
        "constant_value_ug_m3": constant,
        "constant_diagnostic_rmse_ug_m3": float(
            np.sqrt(
                np.mean([row["fitted_constant_mse"] for row in summaries[fit_cases:]])
            )
        ),
        "mean_spatial_contrast_ug_m3": float(
            np.mean([row["mean_spatial_contrast_ug_m3"] for row in summaries])
        ),
        "mean_temporal_std_ug_m3": float(
            np.mean([row["mean_temporal_std_ug_m3"] for row in summaries])
        ),
        "limitations": [
            "Diagnóstico de entrenamiento, no prueba final ni validación atmosférica de Lima.",
            "El predictor constante se ajusta con fijas sintéticas de otros casos; no es el GP de solo fijas.",
            "La variación temporal incluye el transitorio desde el estado inicial.",
            "El soporte vial no equivale a cobertura alcanzable dentro del presupuesto.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return report
