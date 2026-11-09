"""Comparación pareada de muestreo; diagnóstico sintético, no prueba histórica final."""

import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from lima_pm25.simulation import SensorType

from .env import LimaPM25Env
from .evaluation import evaluate_episode, fixed_posterior
from .policies import BaselinePolicy
from .simulation import STEP_SECONDS

STRATEGIES = ("fixed", "random", "greedy", "sweep", "neural")
SIGNATURES = (
    "field_sha256",
    "fixed_measurements_sha256",
    "initial_positions_sha256",
    "start_utc",
    "elapsed_s",
)


def _reduction(value: float, reference: float) -> float | None:
    return 100 * (1 - value / reference) if reference > 0 else None


def _mean(rows, key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def summarize_comparison(rows: list[dict[str, Any]]):
    """Promedia primero réplicas por caso; RMSE = raíz del MSE agrupado.

    La dispersión entre semillas es descriptiva y separada de la variación
    entre escenarios. Las doce realizaciones no son una muestra de toda Lima.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["strategy"], row["case_seed"]].append(row)
    keys = (
        "synthetic_mse",
        "synthetic_mae_ug_m3",
        "final_integrated_variance",
        "mean_integrated_variance",
        "reward",
        "distance_m",
        "sampling_coverage",
    )
    cases = {
        key: {name: _mean(items, name) for name in keys}
        for key, items in groups.items()
    }
    summaries: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        selected = [value for (name, _), value in cases.items() if name == strategy]
        raw = [row for row in rows if row["strategy"] == strategy]
        deviations = [
            float(np.std([item["synthetic_rmse_ug_m3"] for item in items], ddof=1))
            for (name, _), items in groups.items()
            if name == strategy and len(items) > 1
        ]
        summaries.append(
            {
                "strategy": strategy,
                "cases": len(selected),
                "rollouts": len(raw),
                "synthetic_rmse_ug_m3": float(
                    np.sqrt(_mean(selected, "synthetic_mse"))
                ),
                **{key: _mean(selected, key) for key in keys if key != "synthetic_mse"},
                "policy_seed_rmse_sd_mean": float(np.mean(deviations))
                if deviations
                else None,
                "early_terminations": sum(row["early_termination"] for row in raw),
                "invalid_actions": sum(row["invalid_actions"] for row in raw),
            }
        )
    fixed = summaries[0]
    for summary in summaries:
        summary["rmse_reduction_vs_fixed_percent"] = _reduction(
            summary["synthetic_rmse_ug_m3"], fixed["synthetic_rmse_ug_m3"]
        )
        summary["variance_reduction_vs_fixed_percent"] = _reduction(
            summary["final_integrated_variance"], fixed["final_integrated_variance"]
        )

    paired = []
    for (name, seed), neural in cases.items():
        if name != "neural":
            continue
        for strategy in ("fixed", "random", "greedy", "sweep"):
            reference = cases[strategy, seed]
            neural_rmse = float(np.sqrt(neural["synthetic_mse"]))
            reference_rmse = float(np.sqrt(reference["synthetic_mse"]))
            paired.append(
                {
                    "case_seed": seed,
                    "reference": strategy,
                    "neural_rmse_ug_m3": neural_rmse,
                    "reference_rmse_ug_m3": reference_rmse,
                    "rmse_reduction_percent": _reduction(neural_rmse, reference_rmse),
                    "neural_variance": neural["final_integrated_variance"],
                    "reference_variance": reference["final_integrated_variance"],
                    "variance_reduction_percent": _reduction(
                        neural["final_integrated_variance"],
                        reference["final_integrated_variance"],
                    ),
                }
            )
    return summaries, paired


def save_comparison_map(
    env: LimaPM25Env, path: Path, title: str, *, fixed_only: bool = False
) -> None:
    """Mapas humanos con escala común del GP y coordenadas reales de las muestras."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    assert env._simulation is not None
    posterior = fixed_posterior(env) if fixed_only else env._posterior
    grid = env.domain.grid
    config = env.estimator_config.reconstruction_config
    figure = Figure(figsize=(11, 4), layout="constrained")
    FigureCanvasAgg(figure)
    fields = (posterior.mean_ug_m3, posterior.standard_deviation_ug_m3)
    maxima = (
        max(1.0, config.prior_mean_ug_m3 + 3 * config.prior_stddev_ug_m3),
        config.prior_stddev_ug_m3,
    )
    for axis, field, maximum, label in zip(
        figure.subplots(1, 2),
        fields,
        maxima,
        ("Concentración estimada (µg/m³)", "Incertidumbre (µg/m³)"),
        strict=True,
    ):
        image = axis.imshow(
            np.ma.array(field, mask=~grid.active_mask),
            origin="lower",
            extent=grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
            vmin=0,
            vmax=maximum,
            cmap="viridis",
        )
        if not fixed_only:
            for vehicle, points in enumerate(env._routes):
                xy = np.asarray(points) / 1000
                axis.plot(
                    xy[:, 0], xy[:, 1], linewidth=1, label=f"Vehículo {vehicle + 1}"
                )
            samples = [r for r in env._records if r.sensor_type is SensorType.MOBILE]
            if samples:
                axis.scatter(
                    [r.x_m / 1000 for r in samples],
                    [r.y_m / 1000 for r in samples],
                    s=5,
                    c="white",
                )
            axis.legend(fontsize=6)
        catalog = env._simulation.request.station_catalog
        if catalog is not None:
            axis.scatter(
                [site.x_m / 1000 for site in catalog.sites],
                [site.y_m / 1000 for site in catalog.sites],
                marker="^",
                c="red",
                s=22,
            )
        axis.set(title=label, xlabel="Este UTM (km)", ylabel="Norte UTM (km)")
        axis.locator_params(nbins=5)
        figure.colorbar(image, ax=axis, shrink=0.7)
    figure.suptitle(f"{title} | t={env._simulation.state.simulation_time_s / 60:g} min")
    with path.open("xb") as file:
        figure.savefig(file, format="png", dpi=110)


def _validate_seeds(seeds: list[int]) -> None:
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "Las semillas deben ser enteros no negativos, distintos y explícitos"
        )


def run_comparison(
    env: LimaPM25Env,
    neural_policy,
    *,
    case_seeds: list[int],
    policy_seeds: list[int],
    output: Path,
    metadata: dict[str, Any],
    maps: bool = True,
) -> dict[str, Any]:
    _validate_seeds(case_seeds)
    _validate_seeds(policy_seeds)
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        **metadata,
        "purpose": "diagnostic_validation_not_final",
        "training_executed": False,
        "case_seeds": case_seeds,
        "policy_seeds": policy_seeds,
        "representative_case_seed": case_seeds[0],
        "representative_policy_seed": policy_seeds[0],
        "early_termination_rule": "no_more_mobile_observations; evaluator_advances_field_and_fixed_to_common_horizon",
        "aggregation": "equal_case_weight; average_replica_MSE_before_pooled_RMSE; policy_seed_sample_SD_is_descriptive",
    }
    (output / "config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if maps:
        (output / "maps").mkdir()
    rows = []
    references = {}
    total = len(case_seeds) * (2 + 2 * len(policy_seeds))
    completed = 0
    with ExitStack() as stack:
        cases_file = stack.enter_context((output / "cases.csv").open("x", newline=""))
        samples_file = stack.enter_context(
            (output / "measurements.csv").open("x", newline="")
        )
        routes_file = stack.enter_context(
            (output / "trajectories.csv").open("x", newline="")
        )
        cases_writer = None
        samples_writer = csv.DictWriter(
            samples_file,
            fieldnames=[
                "case_seed",
                "strategy",
                "policy_seed",
                *(
                    "sensor_id",
                    "sensor_type",
                    "timestamp_utc",
                    "simulation_time_s",
                    "x_m",
                    "y_m",
                    "crs",
                    "value_ug_m3",
                    "quality_status",
                    "noise_stddev_ug_m3",
                    "position_source_id",
                    "provenance",
                ),
            ],
        )
        samples_writer.writeheader()
        routes_writer = csv.writer(routes_file)
        routes_writer.writerow(
            ["case_seed", "strategy", "policy_seed", "vehicle", "vertex", "x_m", "y_m"]
        )

        def write_row(row):
            nonlocal cases_writer
            if cases_writer is None:
                cases_writer = csv.DictWriter(cases_file, fieldnames=list(row))
                cases_writer.writeheader()
            cases_writer.writerow(row)
            cases_file.flush()
            rows.append(row)

        for case_seed in case_seeds:
            for strategy in ("neural", "random", "greedy", "sweep"):
                for policy_seed in (
                    policy_seeds if strategy in {"random", "greedy"} else [None]
                ):
                    policy = (
                        neural_policy
                        if strategy == "neural"
                        else BaselinePolicy(strategy, seed=policy_seed or 0)
                    )
                    metrics = evaluate_episode(
                        env, policy, seed=case_seed, comparison=True
                    )
                    if metrics["invalid_actions"]:
                        raise RuntimeError(
                            "Una referencia produjo acciones inválidas; no se publica una comparación válida"
                        )
                    if case_seed in references:
                        reference = references[case_seed]
                        if any(metrics[key] != reference[key] for key in SIGNATURES):
                            raise RuntimeError(
                                "Las políticas no recibieron el mismo campo, lecturas fijas, inicio y horizonte"
                            )
                    else:
                        reference = metrics.copy()
                        references[case_seed] = reference
                        fixed_row: dict[str, Any] = {
                            "case_seed": case_seed,
                            "strategy": "fixed",
                            "policy_seed": None,
                            **metrics,
                        }
                        for key in (
                            "synthetic_mse",
                            "synthetic_rmse_ug_m3",
                            "synthetic_mae_ug_m3",
                        ):
                            fixed_row[key] = metrics[f"fixed_{key}"]
                        fixed_row.update(
                            final_integrated_variance=metrics[
                                "fixed_integrated_variance"
                            ],
                            mean_integrated_variance=metrics[
                                "fixed_mean_integrated_variance"
                            ],
                            steps=round(metrics["elapsed_s"] / STEP_SECONDS),
                            reward=0.0,
                            distance_m=0.0,
                            sampling_coverage=0.0,
                            mobile_measurements=0,
                            early_termination=False,
                            decision_end_elapsed_s=metrics["elapsed_s"],
                        )
                        fixed_row["net_variance_reduction_percent"] = _reduction(
                            fixed_row["final_integrated_variance"],
                            fixed_row["initial_integrated_variance"],
                        )
                        write_row(fixed_row)
                        if maps and case_seed == case_seeds[0]:
                            save_comparison_map(
                                env,
                                output / "maps/fixed.png",
                                f"Solo fijas | caso {case_seed}",
                                fixed_only=True,
                            )
                    row: dict[str, Any] = {
                        "case_seed": case_seed,
                        "strategy": strategy,
                        "policy_seed": policy_seed,
                        **metrics,
                    }
                    # CSV acepta vacíos, pero nunca se ocultan NaN/inf en métricas sintéticas.
                    if any(
                        not np.isfinite(row[key])
                        for key in (
                            "synthetic_mse",
                            "synthetic_mae_ug_m3",
                            "final_integrated_variance",
                        )
                    ):
                        raise FloatingPointError(
                            "La comparación contiene métricas no finitas"
                        )
                    write_row(row)
                    for record in env._records:
                        samples_writer.writerow(
                            {
                                "case_seed": case_seed,
                                "strategy": strategy,
                                "policy_seed": policy_seed,
                                **asdict(record),
                                "timestamp_utc": record.timestamp_utc.isoformat(),
                            }
                        )
                    for vehicle, route in enumerate(env._routes):
                        routes_writer.writerows(
                            [case_seed, strategy, policy_seed, vehicle, index, x, y]
                            for index, (x, y) in enumerate(route)
                        )
                    if (
                        maps
                        and case_seed == case_seeds[0]
                        and (policy_seed is None or policy_seed == policy_seeds[0])
                    ):
                        save_comparison_map(
                            env,
                            output / f"maps/{strategy}.png",
                            f"{strategy} | caso {case_seed} | semilla {policy_seed}",
                        )
                    completed += 1
                    print(
                        f"[{completed}/{total}] caso={case_seed} política={strategy} semilla={policy_seed}: RMSE={row['synthetic_rmse_ug_m3']:.4f}",
                        flush=True,
                    )

    summaries, paired = summarize_comparison(rows)
    for name, items in (("summary.csv", summaries), ("paired.csv", paired)):
        with (output / name).open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    neural = next(row for row in summaries if row["strategy"] == "neural")
    comparisons = []
    for reference in summaries:
        if reference["strategy"] == "neural":
            continue
        pairs = [row for row in paired if row["reference"] == reference["strategy"]]
        comparisons.append(
            {
                "reference": reference["strategy"],
                "rmse_reduction_percent": _reduction(
                    neural["synthetic_rmse_ug_m3"], reference["synthetic_rmse_ug_m3"]
                ),
                "variance_reduction_percent": _reduction(
                    neural["final_integrated_variance"],
                    reference["final_integrated_variance"],
                ),
                "cases_with_lower_rmse": sum(
                    row["neural_rmse_ug_m3"] < row["reference_rmse_ug_m3"]
                    for row in pairs
                ),
                "cases": len(pairs),
            }
        )
    summary = {
        "purpose": configuration["purpose"],
        "training_executed": False,
        "strategies": summaries,
        "neural_vs_references": comparisons,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return summary
