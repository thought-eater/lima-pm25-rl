"""Incertidumbre, error y aporte móvil: información privada del evaluador."""

import argparse
import hashlib
import io
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from lima_pm25.simulation import MeasurementBatch, SensorType

from .architecture_experiment import DEFAULT_CONFIG
from .arrival_experiment import _write_json
from .comparison import _validate_seeds
from .evaluate_cli import _compatible
from .mobility_diagnostics import compare_fixed_measurements
from .train_cli import CACHE_DIRECTORY, prepare_experiment
from .training import TrainingConfig, load_policy
from .value_diagnostics import digest, discounted_returns


def correlation(x, y, *, ranks=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if ranks:

        def rank(values):
            _, inverse, counts = np.unique(
                values, return_inverse=True, return_counts=True
            )
            return (np.cumsum(counts) - (counts - 1) / 2)[inverse]

        x, y = rank(x), rank(y)
    return float(np.corrcoef(x, y)[0, 1])


def spatial_metrics(truth, mean, std):
    """Cobertura marginal del campo latente, sin añadir ruido de lectura."""
    truth, mean, std = (np.asarray(a, dtype=float) for a in (truth, mean, std))
    if (
        truth.ndim != 1
        or not len(truth)
        or mean.shape != truth.shape
        or std.shape != truth.shape
    ):
        raise ValueError("Se requieren vectores del mismo dominio activo")
    if not all(np.isfinite(a).all() for a in (truth, mean, std)) or (std < 0).any():
        raise ValueError("Campos finitos y desviaciones no negativas requeridos")
    error = mean - truth
    squared, variance = error**2, std**2
    count = max(1, int(np.ceil(0.2 * len(truth))))
    threshold = np.sort(variance)[-count]
    # Repartir el peso del corte entre empates, sin favorecer índices espaciales.
    weights = (variance > threshold).astype(float)
    tied = variance == threshold
    weights[tied] = (count - weights.sum()) / tied.sum()
    share = float(np.sum(weights * squared) / squared.sum()) if squared.sum() else None
    return {
        "mse": float(squared.mean()),
        "mean_variance": float(variance.mean()),
        "coverage_68": float(np.mean(np.abs(error) <= std)),
        "coverage_95": float(np.mean(np.abs(error) <= 1.959963984540054 * std)),
        "pearson_variance_squared_error": correlation(variance, squared),
        "spearman_variance_squared_error": correlation(variance, squared, ranks=True),
        "top_uncertainty_fraction": count / len(truth),
        "top_uncertainty_error_share": share,
        "top_uncertainty_error_enrichment": share / (count / len(truth))
        if share is not None
        else None,
        "zero_std_cells": int(np.sum(std == 0)),
    }


def mobile_comparison(env, previous_records, weights):
    """Predecir en t con fijas nuevas y móviles antiguos, sin actualizar el GP."""
    simulation = env._simulation
    new = env._records[len(previous_records) :]
    mobile = [r for r in new if r.sensor_type is SensorType.MOBILE]
    if any(r.simulation_time_s != simulation.state.simulation_time_s for r in mobile):
        raise ValueError("El diagnóstico requiere móviles solo al final del tick")
    fixed = tuple(r for r in new if r.sensor_type is SensorType.FIXED)
    before = env._predict(records=(*previous_records, *fixed))
    after = env._posterior
    mask = env.domain.grid.active_mask
    truth = simulation.state.concentration_ug_m3[mask]
    reduction = (
        before.standard_deviation_ug_m3[mask] ** 2
        - after.standard_deviation_ug_m3[mask] ** 2
    )
    scale = env.estimator_config.reconstruction_config.prior_stddev_ug_m3**2
    before_mse = float(np.mean((before.mean_ug_m3[mask] - truth) ** 2))
    after_mse = float(np.mean((after.mean_ug_m3[mask] - truth) ** 2))
    return before, {
        "new_fixed_records": len(fixed),
        "new_mobile_records": len(mobile),
        "unweighted_gain": float(reduction.mean() / scale),
        "weighted_gain": float(np.mean(weights * reduction) / scale),
        "mse_before_mobile": before_mse,
        "mse_after_mobile": after_mse,
        "mse_reduction": before_mse - after_mse,
    }


def save_map(env, truth, before, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    after = env._posterior
    mask = env.domain.grid.active_mask
    fields = (
        truth,
        after.mean_ug_m3,
        after.standard_deviation_ug_m3,
        np.abs(before.mean_ug_m3 - truth),
        np.abs(after.mean_ug_m3 - truth),
        before.standard_deviation_ug_m3**2 - after.standard_deviation_ug_m3**2,
    )
    titles = (
        "Campo sintético privado",
        "Media GP",
        "Desviación GP",
        "Error absoluto antes de móviles",
        "Error absoluto después de móviles",
        "Reducción de varianza móvil",
    )
    concentration_max = max(float(np.max(fields[i][mask])) for i in (0, 1))
    error_max = max(float(np.max(fields[i][mask])) for i in (3, 4))
    maxima = (
        concentration_max,
        concentration_max,
        float(np.max(fields[2][mask])),
        error_max,
        error_max,
        float(np.max(np.abs(fields[5][mask]))),
    )
    figure = Figure(figsize=(13, 7), layout="constrained")
    FigureCanvasAgg(figure)
    for index, (axis, field, title, maximum) in enumerate(
        zip(figure.subplots(2, 3).ravel(), fields, titles, maxima, strict=True)
    ):
        maximum = max(maximum, 1e-12)
        image = axis.imshow(
            np.ma.array(field, mask=~mask),
            origin="lower",
            extent=env.domain.grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
            vmin=-maximum if index == 5 else 0,
            vmax=maximum,
            cmap="coolwarm" if index == 5 else "viridis",
        )
        for points in env._routes:
            xy = np.asarray(points) / 1000
            axis.plot(xy[:, 0], xy[:, 1], color="white", linewidth=0.5)
        fixed = [r for r in env._records if r.sensor_type is SensorType.FIXED]
        axis.scatter(
            [r.x_m / 1000 for r in fixed],
            [r.y_m / 1000 for r in fixed],
            c="red",
            marker="^",
            s=10,
        )
        axis.set(title=title, xlabel="Este UTM (km)", ylabel="Norte UTM (km)")
        figure.colorbar(image, ax=axis, shrink=0.7)
    figure.suptitle(
        f"Evaluador | t={env._simulation.state.simulation_time_s / 60:g} min; µg/m³, varianza (µg/m³)²"
    )
    with path.open("xb") as file:
        figure.savefig(file, format="png", dpi=110)


def diagnose_case(env, policy, *, seed, gamma, output, map_ticks=()):
    if env.arrival_strategy != "shortest":
        raise ValueError("El control usa shortest")
    output.mkdir(parents=True, exist_ok=False)
    observation, _ = env.reset(seed=seed)
    if env._simulation.is_historical_replay:
        raise ValueError("Se requiere campo sintético; no replay histórico")
    policy.reset()
    initial = np.asarray(env.positions)
    mask = env.domain.grid.active_mask
    field_hash = hashlib.sha256()
    spatial, ticks, frames = [], [], []

    def record(before, weights, tick, decision):
        truth = env._simulation.state.concentration_ug_m3
        field_hash.update(truth[mask].tobytes())
        after = env._posterior
        frame = [
            a[mask].copy()
            for a in (
                truth,
                before.mean_ug_m3,
                before.standard_deviation_ug_m3,
                after.mean_ug_m3,
                after.standard_deviation_ug_m3,
            )
        ]
        frames.append((*frame, weights.copy()))
        spatial.append(
            {
                "tick": tick,
                "decision_executed": decision,
                "elapsed_s": float(env._simulation.state.simulation_time_s),
                "before_mobile": spatial_metrics(frame[0], frame[1], frame[2]),
                "after_mobile": spatial_metrics(frame[0], frame[3], frame[4]),
            }
        )
        if tick in map_ticks:
            save_map(env, truth, before, output / f"tick-{tick:02d}.png")

    record(env._posterior, np.full(mask.sum(), np.nan), 0, False)
    while True:
        previous = tuple(env._records)
        config = env.estimator_config.reconstruction_config
        mean = np.maximum(env._posterior.mean_ug_m3[mask], 0)
        weights = 1 + env.concentration_priority * mean / (
            max(config.prior_mean_ug_m3, config.prior_stddev_ug_m3) + mean
        )
        action = policy.act(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if info["invalid_actions"].any():
            raise RuntimeError("La política produjo acciones inválidas")
        before, contribution = mobile_comparison(env, previous, weights)
        if not np.isclose(
            contribution["weighted_gain"],
            info["uncertainty_gain"],
            rtol=1e-7,
            atol=1e-9,
        ):
            raise RuntimeError(
                "La reconstrucción contrafactual no reproduce la ganancia del entorno"
            )
        if not np.isclose(
            reward,
            contribution["weighted_gain"] - info["distance_cost"],
            rtol=1e-7,
            atol=1e-9,
        ):
            raise RuntimeError("No coincide la descomposición de recompensa")
        ticks.append(
            {
                "tick": env._steps,
                "actions": action.tolist(),
                "reward": reward,
                "distance_cost": info["distance_cost"],
                "positions_m": env.positions,
                **contribution,
            }
        )
        record(before, weights, env._steps, True)
        if terminated or truncated:
            break
    decision_end = float(info["elapsed_s"])
    while env._simulation.state.simulation_time_s < env._simulation.duration_s:
        for batch in env._simulation.advance(env.positions):
            fixed = tuple(r for r in batch.records if r.sensor_type is SensorType.FIXED)
            if fixed:
                env._assimilate(
                    MeasurementBatch(
                        batch.simulation_time_s, batch.timestamp_utc, fixed
                    )
                )
        env._posterior = env._predict()
        record(env._posterior, np.full(mask.sum(), np.nan), len(spatial), False)
    mses = [r["after_mobile"]["mse"] for r in spatial]
    variances = [
        r["after_mobile"]["mean_variance"] * mask.sum() * env.domain.grid.cell_area_m2
        for r in spatial
    ]
    report = {
        "case_seed": seed,
        "ticks": ticks,
        "spatial": spatial,
        "routes_m": env._routes,
        "measurements": [
            {**asdict(r), "timestamp_utc": r.timestamp_utc.isoformat()}
            for r in env._records
        ],
        "signatures": {
            "field_sha256": field_hash.hexdigest(),
            "fixed_measurements_sha256": digest(
                [asdict(r) for r in env._records if r.sensor_type is SensorType.FIXED]
            ),
            "initial_positions_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "horizon_s": float(env._simulation.duration_s),
        },
        "metrics": {
            "reward": sum(r["reward"] for r in ticks),
            "discounted_return": discounted_returns(
                [r["reward"] for r in ticks], gamma
            )[0],
            "synthetic_mse": mses[-1],
            "synthetic_rmse_ug_m3": float(np.sqrt(mses[-1])),
            "temporal_mse": float(np.trapezoid(mses) / (len(mses) - 1)),
            "final_integrated_variance": variances[-1],
            "mean_integrated_variance": float(
                np.trapezoid(variances) / (len(variances) - 1)
            ),
            "distance_m": float(env._distance.sum()),
            "decision_end_elapsed_s": decision_end,
        },
    }
    _write_json(output / "report.json", report)
    arrays = np.asarray(frames)
    with (output / "fields-private.npz").open("xb") as file:
        np.savez_compressed(
            file,
            active_cells=np.argwhere(mask),
            grid_shape=mask.shape,
            grid_bounds_m=env.domain.grid.grid_bounds_m,
            elapsed_s=[r["elapsed_s"] for r in spatial],
            **{
                name: arrays[:, index]
                for index, name in enumerate(
                    (
                        "truth",
                        "mean_before",
                        "std_before",
                        "mean_after",
                        "std_after",
                        "weights_before_tick",
                    )
                )
            },
        )
    return report


def verify_reproduction(report, reference):
    for key in ("field_sha256", "initial_positions_sha256", "horizon_s"):
        if report["signatures"][key] != reference["signatures"][key]:
            raise ValueError(f"La reproducción difiere en {key}")
    for key, value in report["metrics"].items():
        if not np.isclose(value, reference["metrics"][key], rtol=1e-7, atol=1e-8):
            raise ValueError(f"La reproducción difiere en {key}")
    if len(report["ticks"]) != len(reference["ticks"]):
        raise ValueError("La reproducción cambió la duración de las decisiones")
    for actual, expected in zip(report["ticks"], reference["ticks"], strict=True):
        if actual["actions"] != expected["actions"] or not np.allclose(
            actual["positions_m"], expected["positions_m"], rtol=0, atol=1e-6
        ):
            raise ValueError("La reproducción cambió las decisiones o posiciones")
    fixed = compare_fixed_measurements(
        [r for r in report["measurements"] if r["sensor_type"] == SensorType.FIXED],
        [r for r in reference["measurements"] if r["sensor_type"] == SensorType.FIXED],
    )
    if not fixed["match"]:
        raise ValueError("La reproducción cambió las lecturas fijas")
    return {"match": True, "fixed_measurements": fixed}


def summarize_case(report):
    ticks = report["ticks"]
    result = {
        "case_seed": report["case_seed"],
        "decision_ticks": len(ticks),
        "ticks_with_worse_mse": sum(r["mse_reduction"] < -1e-9 for r in ticks),
        "weighted_gain_positive_mse_worse": sum(
            r["weighted_gain"] > 1e-9 and r["mse_reduction"] < -1e-9 for r in ticks
        ),
    }
    for key in (
        "unweighted_gain",
        "weighted_gain",
        "distance_cost",
        "reward",
        "mse_reduction",
    ):
        values = [r[key] for r in ticks]
        result[f"mean_{key}"] = float(np.mean(values))
        for ranks in (False, True):
            result[f"{'spearman' if ranks else 'pearson'}_{key}_mse_reduction"] = (
                correlation(values, [r["mse_reduction"] for r in ticks], ranks=ranks)
            )
    for phase in ("before_mobile", "after_mobile"):
        rows = [r[phase] for r in report["spatial"] if r["decision_executed"]]
        for key in rows[0]:
            values = [r[key] for r in rows if r[key] is not None]
            result[f"{phase}_{key}"] = float(np.mean(values)) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Doce misiones de diagnóstico GP, sin entrenamiento"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe resultados")
    spec = json.loads(args.config.read_text())
    config = TrainingConfig(**spec["training"])
    _validate_seeds(spec["validation_seeds"])
    if (
        spec["kind"] != "generated"
        or config.architecture != "bilg"
        or config.seed != 42
        or spec["episodes"] != 200
        or spec.get("arrival_strategy", "shortest") != "shortest"
    ):
        parser.error(
            "El control prefijado es BiLG, semilla 42, episodio 200, escenarios generados y shortest"
        )
    candidates = (
        [args.checkpoint]
        if args.checkpoint
        else sorted(
            CACHE_DIRECTORY.glob("architecture-experiments/*/bilg-42/policy-last.pt"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
    )
    train, env, protocol = prepare_experiment(
        spec, args.config.resolve().parent, check=args.check
    )
    train.close()
    assert env is not None
    try:
        chosen = None
        for path in candidates:
            payload = path.read_bytes()
            state = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=True
            )
            if (
                _compatible(state, config, protocol)
                and state["completed_episodes"] == 200
            ):
                chosen = path
                break
        if chosen is None:
            raise ValueError(
                "No hay policy-last de BiLG-42 episodio 200 compatible; indicar --checkpoint"
            )
        reference_dir = chosen.parent / "evaluation-last"
        reference_metadata = json.loads((reference_dir / "policy.json").read_text())
        if reference_metadata["policy_sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("Las trazas de referencia no corresponden a estos pesos")
        for seed in spec["validation_seeds"]:
            if not (reference_dir / f"case-{seed}.json").is_file():
                raise ValueError(
                    f"Falta la traza de referencia del caso {seed} en {reference_dir}"
                )
        if args.check:
            print(
                json.dumps(
                    {
                        "training_executed": False,
                        "simulation_executed": False,
                        "checkpoint": str(chosen),
                        "missions": len(spec["validation_seeds"]),
                    },
                    indent=2,
                )
            )
            return
        torch.set_num_threads(config.cpu_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        policy, _ = load_policy(io.BytesIO(payload), device=args.device)
        output = args.output or CACHE_DIRECTORY / "gp-diagnostics" / datetime.now(
            UTC
        ).strftime("%Y%m%d-%H%M%S-%f")
        output.mkdir(parents=True, exist_ok=False)
        _write_json(
            output / "config.json",
            {
                "training_executed": False,
                "policy_sha256": hashlib.sha256(payload).hexdigest(),
                "checkpoint": str(chosen),
                "training": asdict(config),
                "protocol": protocol,
                "case_seeds": spec["validation_seeds"],
                "map_case": spec["validation_seeds"][0],
                "map_ticks": [1, 12, 24],
                "torch_version": str(torch.__version__),
                "device": str(next(policy.network.parameters()).device),
                "diagnostic_code_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
        )
        print(f"Salida: {output}", flush=True)
        summaries = []
        for index, seed in enumerate(spec["validation_seeds"]):
            report = diagnose_case(
                env,
                policy,
                seed=seed,
                gamma=config.gamma,
                output=output / f"case-{seed}",
                map_ticks=(1, 12, 24) if index == 0 else (),
            )
            reference = json.loads((reference_dir / f"case-{seed}.json").read_text())
            checks = verify_reproduction(report, reference)
            _write_json(output / f"case-{seed}" / "reproduction.json", checks)
            summaries.append(summarize_case(report))
            print(
                f"[{index + 1}/{len(spec['validation_seeds'])}] caso={seed}: reproducción correcta",
                flush=True,
            )
        averages = {}
        for key in summaries[0]:
            if key == "case_seed":
                continue
            values = [r[key] for r in summaries if r[key] is not None]
            averages[key] = {
                "mean": float(np.mean(values)) if values else None,
                "defined_cases": len(values),
            }
        _write_json(
            output / "summary.json",
            {
                "by_case": summaries,
                "equal_case_means": averages,
                "scope": "Asociaciones descriptivas en una política fija; no identifica el orden de acciones alternativas ni prueba causalidad de una fórmula de recompensa.",
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
