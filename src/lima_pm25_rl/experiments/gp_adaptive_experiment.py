"""Comprobación acotada: lecturas de entrenamiento y cuatro episodios nuevos."""

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from time import perf_counter

import numpy as np
import torch
from lima_pm25.estimation import (
    AdaptiveMapUpdater,
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import MeasurementBatch, SensorType
from threadpoolctl import threadpool_limits

from ..adaptive_env import model_change_terms
from ..files import _write_json, product
from ..hotspot_evaluation import field_protocol
from ..study import prepare_experiment
from ..training import Trainer, TrainingConfig, resolve_device
from .gp_diagnostics import spatial_metrics
from .gp_spatial_experiment import load_frozen_case, partition_bounds
from .gp_temporal_experiment import check_control


def score_case(env, frozen, threshold):
    """La verdad se consulta después de cada posterior, solo para puntuar."""
    report, fields, records, origin = frozen
    grid, settings = env.domain.grid, env.estimator_config
    base = settings.reconstruction_config
    adaptive = AdaptiveMapUpdater(grid, base, settings.history_duration, env.gp_fit)
    fixed = GaussianProcessFieldReconstructor(base)
    groups = defaultdict(list)
    for record in records:
        groups[record.timestamp_utc].append(record)
    batches = [
        MeasurementBatch(group[0].simulation_time_s, timestamp, tuple(group))
        for timestamp, group in sorted(groups.items())
    ]
    cursor, rows, traces = 0, [], []
    for index, time in enumerate(fields["elapsed_s"]):
        now = origin + timedelta(seconds=float(time))
        mean_change = std_change = signed_gain = 0.0
        while cursor < len(batches) and batches[cursor].timestamp_utc <= now:
            batch = batches[cursor]
            before, after = adaptive.update(batch)
            for branch, result in (("counterfactual", before), ("factual", after)):
                if result is not None:
                    traces.append(
                        {
                            "elapsed_s": batch.simulation_time_s,
                            "branch": branch,
                            **asdict(result.parameters),
                            **asdict(result.fit),
                        }
                    )
            if before is not None and after is not None:
                eligible = set(after.posterior.observations.used_measurement_keys)
                positions = [
                    (r.x_m, r.y_m)
                    for r in batch.records
                    if r.sensor_type is SensorType.MOBILE
                    and (r.sensor_id, r.timestamp_utc) in eligible
                ]
                dm, ds = model_change_terms(
                    before.posterior,
                    after.posterior,
                    grid,
                    positions,
                    env.model_change,
                    base.prior_stddev_ug_m3,
                )
                mean_change += dm
                std_change += ds
                signed_gain += float(
                    np.mean(
                        before.posterior.standard_deviation_ug_m3[grid.active_mask] ** 2
                        - after.posterior.standard_deviation_ug_m3[grid.active_mask]
                        ** 2
                    )
                    / base.prior_stddev_ug_m3**2
                )
            cursor += 1
        eligible = tuple(
            r
            for r in records
            if now - settings.history_duration <= r.timestamp_utc <= now
        )
        control = fixed.reconstruct(
            ReconstructionRequest(
                grid=grid,
                reconstruction_timestamp_utc=now,
                measurements=eligible,
            )
        )
        for name, posterior in (
            ("fixed_gp", control),
            ("adaptive_gp", adaptive.predict(now)),
        ):
            mean = posterior.mean_ug_m3[grid.active_mask]
            std = posterior.standard_deviation_ug_m3[grid.active_mask]
            if name == "fixed_gp" and (
                not np.allclose(mean, fields["mean_after"][index], rtol=1e-7, atol=1e-8)
                or not np.allclose(
                    std, fields["std_after"][index], rtol=1e-7, atol=1e-8
                )
            ):
                raise ValueError("El control no reproduce los posteriores conservados")
            truth = fields["truth"][index]
            hot = truth >= threshold
            rows.append(
                {
                    "estimator": name,
                    "elapsed_s": float(time),
                    **spatial_metrics(truth, mean, std),
                    "hot_squared_error_sum": float(
                        np.sum((truth[hot] - mean[hot]) ** 2)
                    ),
                    "hot_cells": int(hot.sum()),
                    "model_mean_change": mean_change if name == "adaptive_gp" else None,
                    "model_uncertainty_change": std_change
                    if name == "adaptive_gp"
                    else None,
                    "signed_variance_gain": signed_gain
                    if name == "adaptive_gp"
                    else None,
                }
            )
    summary = []
    for name in ("fixed_gp", "adaptive_gp"):
        selected = [r for r in rows if r["estimator"] == name]
        times = fields["elapsed_s"]
        hot_count = float(np.trapezoid([r["hot_cells"] for r in selected], times))
        summary.append(
            {
                "case_seed": report["case_seed"],
                "estimator": name,
                "temporal_mse": float(
                    np.trapezoid([r["mse"] for r in selected], times) / times[-1]
                ),
                "final_mse": selected[-1]["mse"],
                "hot_temporal_mse": float(
                    np.trapezoid([r["hot_squared_error_sum"] for r in selected], times)
                    / hot_count
                )
                if hot_count
                else None,
                "coverage_95": float(np.mean([r["coverage_95"] for r in selected[1:]])),
            }
        )
    return {
        "summary": summary,
        "ticks": rows,
        "fits": traces,
        "input_signatures": report["signatures"],
    }


def run(args):
    spec = json.loads(args.config.read_text())
    config = TrainingConfig(**spec["training"])
    device = resolve_device(config.device)
    print(f"Dispositivo configurado: {device}", flush=True)
    train, validation, protocol = prepare_experiment(spec, args.config.resolve().parent)
    try:
        source = product("gp-espacial")
        metadata = json.loads((source / "config.json").read_text())
        if field_protocol(metadata["protocol"]) != field_protocol(
            json.loads(json.dumps(protocol))
        ):
            raise ValueError("Los casos congelados no comparten el protocolo físico")
        check_control(train.domain.grid, train.estimator_config)
        seeds = metadata["calibration_seeds"]
        if not seeds or set(seeds) & set(spec["validation_seeds"]):
            raise ValueError(
                "El diagnóstico requiere casos exclusivos de entrenamiento"
            )
        bounds = partition_bounds(protocol, "calibration")
        frozen = [
            load_frozen_case(
                source / "calibration" / f"case-{seed}", train.domain.grid, **bounds
            )
            for seed in seeds
        ]
        if [case[0]["case_seed"] for case in frozen] != seeds:
            raise ValueError("Las semillas congeladas no coinciden con el manifiesto")
        threshold = json.loads((product("duracion") / "protocol.json").read_text())[
            "threshold_ug_m3"
        ]
        if args.check:
            print(
                f"Preparación válida: {len(seeds)} casos de entrenamiento; cuatro episodios integrados al ejecutar."
            )
            return
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
        _write_json(
            output / "config.json",
            {
                "settings": spec,
                "protocol": protocol,
                "source": str(source),
                "case_seeds": seeds,
                "partition": "training",
                "threshold_ug_m3": threshold,
                "integrated_training_episodes": 4,
            },
        )
        torch.set_num_threads(config.cpu_threads)
        summaries, fits, ticks = [], [], []
        with threadpool_limits(limits=config.cpu_threads):
            started = perf_counter()
            for case in frozen:
                report = score_case(train, case, threshold)
                seed = case[0]["case_seed"]
                _write_json(output / f"case-{seed}.json", report)
                summaries.extend(report["summary"])
                fits.extend(report["fits"])
                ticks.extend(
                    r
                    for r in report["ticks"]
                    if r["estimator"] == "adaptive_gp" and r["elapsed_s"] > 0
                )
                print(f"Reconstrucción de entrenamiento: {seed}", flush=True)
            frozen_seconds = perf_counter() - started
            trainer = Trainer(train, config, protocol=protocol)
            for index in range(4):
                started = perf_counter()
                previous_updates = trainer.updates
                row = trainer.train_episode()
                episode_seconds = perf_counter() - started
                _write_json(
                    output / f"integrated-episode-{index + 1}.json",
                    {
                        "metrics": row,
                        "wall_seconds": episode_seconds,
                        "device": str(trainer.device),
                        "fits": train.gp_trace,
                    },
                )
            if trainer.updates == previous_updates:
                raise ValueError(
                    "La medición no alcanzó actualizaciones Adam; no extrapolar coste de entrenamiento"
                )
        attempted = [r for r in fits if r["attempted"]]
        aggregate = []
        for name in ("fixed_gp", "adaptive_gp"):
            selected = [r for r in summaries if r["estimator"] == name]
            aggregate.append(
                {
                    "estimator": name,
                    **{
                        key.replace("mse", "rmse"): float(
                            np.sqrt(
                                np.mean(
                                    [r[key] for r in selected if r[key] is not None]
                                )
                            )
                        )
                        for key in ("temporal_mse", "final_mse", "hot_temporal_mse")
                    },
                    "coverage_95": float(np.mean([r["coverage_95"] for r in selected])),
                }
            )
        summary = {
            "cases": summaries,
            "aggregate": aggregate,
            "frozen_seconds": frozen_seconds,
            "fit_attempts": len(attempted),
            "nonconverged": sum(not r["converged"] for r in attempted),
            "at_bounds_amplitude_space_time": np.sum(
                [r["at_bounds"] for r in attempted], axis=0
            ).tolist(),
            "negative_signed_gain_ticks": sum(
                r["signed_variance_gain"] < 0 for r in ticks
            ),
            "mean_change_sum": sum(r["model_mean_change"] for r in ticks),
            "uncertainty_change_sum": sum(r["model_uncertainty_change"] for r in ticks),
            "integrated_episode_seconds": episode_seconds,
            "training_only_20000_hours_linear_estimate": episode_seconds * 20000 / 3600,
            "integrated_optimizer_updates": trainer.updates,
            "scope": "Doce casos congelados de entrenamiento y cuatro episodios nuevos. Estimación local del coste del cuarto episodio, con Adam, sin validación, checkpoints ni transferencia al hardware del servidor. No evalúa una política adaptativa de 20 000 episodios ni demuestra la meta del 20 %.",
        }
        _write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Salida: {output}")
    finally:
        train.close()
        if validation is not None:
            validation.close()
