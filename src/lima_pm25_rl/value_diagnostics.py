"""Auditoría de Q y alternativas con continuación congelada, fuera del agente."""

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

from .arrival_experiment import _write_json
from .evaluate_cli import _compatible
from .evaluation import _spatial_errors
from .mobility_diagnostics import archive_files
from .train_cli import CACHE_DIRECTORY, DEFAULT_CONFIG, PROJECT_ROOT, prepare_experiment
from .training import TrainingConfig, load_policy


def digest(value):
    return hashlib.sha256(
        json.dumps(value, default=str, sort_keys=True).encode()
    ).hexdigest()


def discounted_returns(rewards, gamma):
    result = [0.0] * len(rewards)
    total = 0.0
    for index in reversed(range(len(rewards))):
        total = float(rewards[index]) + gamma * total
        result[index] = total
    return result


def observed_plan(env, policy, observation, forced_assignment=None):
    """Intervenir una elección mediante hook; usar el planificador real después.

    Los Q guardados son las salidas originales. El tensor devuelto al selector
    solo cambia en el primer forward; no cambia pesos, entradas ni máscaras.
    """
    if forced_assignment is not None and (
        not 0 <= forced_assignment < 24
        or not observation["action_mask"].ravel()[forced_assignment]
    ):
        raise ValueError("La alternativa debe ser un par legal del primer contexto")
    captured = []

    def hook(module, arguments, output):
        fingerprint = hashlib.sha256()
        for argument in arguments:
            fingerprint.update(argument.detach().cpu().numpy().tobytes())
        captured.append((output.detach().cpu().numpy().copy(), fingerprint.hexdigest()))
        if len(captured) == 1 and forced_assignment is not None:
            replacement = output.clone()
            replacement.reshape(-1)[forced_assignment] = output.max() + 1
            return replacement
        return None

    handle = policy.network.register_forward_hook(hook)
    try:
        actions, decisions = policy.plan(env, observation)
    finally:
        handle.remove()
    if len(captured) != len(decisions):
        raise RuntimeError("Se requiere una inferencia determinista por subdecisión")
    stages = []
    for (q, fingerprint), decision in zip(captured, decisions, strict=True):
        assignment = 8 * decision.vehicle + decision.action
        stages.append(
            {
                "assignment": assignment,
                "q": float(q.ravel()[assignment]),
                "legal_assignments": np.flatnonzero(decision.mask.ravel()).tolist(),
                "q_values": q.ravel().tolist(),
                "input_sha256": fingerprint,
                "pending": decision.pending,
                "reserved_cells": decision.reserved_cells,
            }
        )
    if forced_assignment is not None and stages[0]["assignment"] != forced_assignment:
        raise RuntimeError("La intervención no fue ejecutada")
    return actions, stages


def rollout(
    env, policy, *, case_seed, gamma, intervention=None, future_noise_seed=None
):
    """Reproducir desde reset: preserva historia, GP, escenarios y estado sensorial.

    intervention=(tick, assignment), con tick basado en 1. Una semilla futura
    sustituye únicamente el RNG móvil justo antes de esa decisión; el pasado
    observado y las lecturas fijas permanecen idénticos.
    """
    if env.arrival_strategy != "shortest":
        raise ValueError("Este diagnóstico usa el control shortest")
    observation, info = env.reset(seed=case_seed)
    if env._simulation.is_historical_replay:
        raise ValueError("El diagnóstico requiere campos sintéticos de validación")
    policy.reset()
    initial_positions = np.asarray(env.positions).copy()
    field = hashlib.sha256()
    errors, variances, ticks, times = [], [], [], []

    def record():
        field.update(
            env._simulation.state.concentration_ug_m3[
                env.domain.grid.active_mask
            ].tobytes()
        )
        errors.append(_spatial_errors(env, env._posterior))
        variances.append(env._integrated_variance(env._posterior))
        times.append(float(env._simulation.state.simulation_time_s))

    record()
    while True:
        tick = len(ticks) + 1
        force = intervention[1] if intervention and tick == intervention[0] else None
        if force is not None and future_noise_seed is not None:
            env._simulation.mobile._rng = np.random.default_rng(
                [case_seed, tick, 741, future_noise_seed]
            )
        sensor_before = digest(env._simulation.mobile._rng.bit_generator.state)
        actions, stages = observed_plan(env, policy, observation, force)
        observation, reward, terminated, truncated, info = env.step(actions)
        if info["invalid_actions"].any():
            raise RuntimeError("El diagnóstico ejecutó una acción inválida")
        record()
        ticks.append(
            {
                "tick": tick,
                "actions": actions.tolist(),
                "stages": stages,
                "reward": reward,
                "information_gain": info["uncertainty_gain"],
                "distance_cost": info["distance_cost"],
                "distance_m": float(info["route_distance_m"].sum()),
                "positions_m": env.positions,
                "held_vehicles": info["held_vehicles"].tolist(),
                "sensor_rng_before_sha256": sensor_before,
                "sensor_rng_after_sha256": digest(
                    env._simulation.mobile._rng.bit_generator.state
                ),
                "integrated_variance": variances[-1],
                **errors[-1],
            }
        )
        if terminated or truncated:
            break
    if intervention and intervention[0] > len(ticks):
        raise ValueError("El episodio terminó antes del contexto solicitado")
    decision_end = float(info["elapsed_s"])
    # Igual horizonte para calidad, sin prolongar el retorno de un MDP terminado.
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
        record()
    returns = discounted_returns([t["reward"] for t in ticks], gamma)
    for index, tick in enumerate(ticks):
        for stage_index, stage in enumerate(tick["stages"]):
            stage["return"] = returns[index]
            stage["q_minus_return"] = stage["q"] - returns[index]
            if stage_index + 1 < len(tick["stages"]):
                target = tick["stages"][stage_index + 1]["q"]
            else:
                following = ticks[index + 1]["stages"] if index + 1 < len(ticks) else []
                target = tick["reward"] + gamma * (
                    following[0]["q"] if following else 0
                )
            stage["online_bellman_residual"] = stage["q"] - target
    fixed = [asdict(r) for r in env._records if r.sensor_type is SensorType.FIXED]
    return {
        "case_seed": case_seed,
        "intervention": intervention,
        "future_noise_seed": future_noise_seed,
        "ticks": ticks,
        "evaluation_series": [
            {"elapsed_s": time, "integrated_variance": variance, **error}
            for time, variance, error in zip(times, variances, errors, strict=True)
        ],
        "measurements": [
            {**asdict(r), "timestamp_utc": r.timestamp_utc.isoformat()}
            for r in env._records
        ],
        "signatures": {
            "field_sha256": field.hexdigest(),
            "fixed_measurements_sha256": digest(fixed),
            "initial_positions_sha256": hashlib.sha256(
                initial_positions.tobytes()
            ).hexdigest(),
            "horizon_s": float(env._simulation.duration_s),
        },
        "metrics": {
            "reward": sum(t["reward"] for t in ticks),
            "discounted_return": returns[0],
            "distance_m": float(env._distance.sum()),
            "decision_end_elapsed_s": decision_end,
            "final_integrated_variance": variances[-1],
            "mean_integrated_variance": float(
                np.trapezoid(variances) / (len(variances) - 1)
            ),
            "temporal_mse": float(
                np.trapezoid([e["synthetic_mse"] for e in errors]) / (len(errors) - 1)
            ),
            **errors[-1],
        },
    }


def check_pair(factual, branch, tick, *, same_noise):
    if factual["signatures"] != branch["signatures"]:
        raise RuntimeError("No coinciden campo, fijas, inicio u horizonte")
    if factual["ticks"][: tick - 1] != branch["ticks"][: tick - 1]:
        # Retornos y residuos contienen futuro; el prefijo físico debe coincidir.
        for before, after in zip(
            factual["ticks"][: tick - 1], branch["ticks"][: tick - 1], strict=True
        ):
            for key in ("actions", "positions_m", "reward", "sensor_rng_after_sha256"):
                if before[key] != after[key]:
                    raise RuntimeError("La rama no conserva el pasado factual")
    original = factual["ticks"][tick - 1]["stages"][0]
    alternative = branch["ticks"][tick - 1]["stages"][0]
    if (
        original["input_sha256"] != alternative["input_sha256"]
        or original["q_values"] != alternative["q_values"]
    ):
        raise RuntimeError("Las alternativas no comparten contexto y Q originales")
    if same_noise:
        for before, after in zip(factual["ticks"], branch["ticks"]):
            if before["sensor_rng_after_sha256"] != after["sensor_rng_after_sha256"]:
                raise RuntimeError("Las ramas no conservan el flujo de ruido móvil")


def order_agreement(q, returns):
    """Fracción de pares no empatados cuyo orden coincide, dentro de un contexto."""
    q, returns = np.asarray(q), np.asarray(returns)
    i, j = np.triu_indices(len(q), 1)
    dq, dr = q[i] - q[j], returns[i] - returns[j]
    eligible = (np.abs(dq) > 1e-7) & (np.abs(dr) > 1e-7)
    return float(np.mean(dq[eligible] * dr[eligible] > 0)) if eligible.any() else None


def context_summary(rows):
    assignments = sorted({r["assignment"] for r in rows})
    averaged = []
    for assignment in assignments:
        replicas = [r for r in rows if r["assignment"] == assignment]
        averaged.append(
            {
                "assignment": assignment,
                "q": replicas[0]["q"],
                "selected": replicas[0]["selected"],
                **{
                    name: float(np.mean([r[name] for r in replicas]))
                    for name in (
                        "return",
                        "synthetic_mse",
                        "temporal_mse",
                        "final_integrated_variance",
                    )
                },
            }
        )
    selected = next(r for r in averaged if r["selected"])
    best = max(averaged, key=lambda r: r["return"])
    deltas = []
    for replica in sorted({r["replica"] for r in rows}):
        pair = {r["assignment"]: r for r in rows if r["replica"] == replica}
        deltas.append(
            pair[best["assignment"]]["return"] - pair[selected["assignment"]]["return"]
        )
    return {
        "case_seed": rows[0]["case_seed"],
        "tick": rows[0]["tick"],
        "alternatives": len(averaged),
        "selected_assignment": selected["assignment"],
        "best_sampled_return_assignment": best["assignment"],
        "selected_q": selected["q"],
        "selected_return": selected["return"],
        "return_regret": best["return"] - selected["return"],
        "paired_return_deltas": deltas,
        "better_return_alternatives": sum(
            r["return"] > selected["return"] + 1e-7 for r in averaged
        ),
        "order_agreement": order_agreement(
            [r["q"] for r in averaged], [r["return"] for r in averaged]
        ),
        "return_vs_negative_temporal_mse_order_agreement": order_agreement(
            [r["return"] for r in averaged], [-r["temporal_mse"] for r in averaged]
        ),
        "best_return_final_rmse_change": float(
            np.sqrt(best["synthetic_mse"]) - np.sqrt(selected["synthetic_mse"])
        ),
        "best_return_temporal_mse_change": best["temporal_mse"]
        - selected["temporal_mse"],
        "averaged_alternatives": averaged,
    }


def calibration(report):
    stages = [s for t in report["ticks"] for s in t["stages"]]
    difference = np.asarray([s["q_minus_return"] for s in stages])
    return {
        "case_seed": report["case_seed"],
        "decisions": len(stages),
        "mean_q": float(np.mean([s["q"] for s in stages])),
        "mean_return": float(np.mean([s["return"] for s in stages])),
        "bias": float(difference.mean()),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "last_tick_q": [s["q"] for s in report["ticks"][-1]["stages"]],
        "last_tick_return": report["ticks"][-1]["reward"],
    }


def run_diagnostic(
    env, policy, *, case_seeds, ticks, noise_replicas, gamma, output, metadata
):
    if (
        noise_replicas < 1
        or not ticks
        or min(ticks) < 1
        or len(set(ticks)) != len(ticks)
    ):
        raise ValueError("Se requieren ticks distintos positivos y réplicas positivas")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            **metadata,
            "training_executed": False,
            "arrival_strategy": "shortest",
            "case_seeds": case_seeds,
            "ticks": ticks,
            "noise_replicas": noise_replicas,
            "gamma": gamma,
            "intervention_stage": 0,
            "noise_protocol": "replica 0: original; replicas >=1: futuro móvil reseed [case,tick,741,replica]",
            "scope": "Continuación de política congelada; no estima Q óptimo ni integra incertidumbre del campo oculto",
        },
    )
    calibrations, contexts, skipped = [], [], []
    with (output / "alternatives.jsonl").open("x") as file:
        for seed in case_seeds:
            factual = rollout(env, policy, case_seed=seed, gamma=gamma)
            _write_json(output / f"factual-{seed}.json", factual)
            calibrations.append(calibration(factual))
            for tick in ticks:
                if (
                    tick > len(factual["ticks"])
                    or not factual["ticks"][tick - 1]["stages"]
                ):
                    skipped.append({"case_seed": seed, "tick": tick})
                    continue
                stage = factual["ticks"][tick - 1]["stages"][0]
                rows = []
                for replica in range(noise_replicas):
                    paired_reference = None
                    # Ejecutar primero la elección factual como control de cada ruido futuro.
                    alternatives = [
                        stage["assignment"],
                        *[
                            a
                            for a in stage["legal_assignments"]
                            if a != stage["assignment"]
                        ],
                    ]
                    for assignment in alternatives:
                        branch = rollout(
                            env,
                            policy,
                            case_seed=seed,
                            gamma=gamma,
                            intervention=(tick, assignment),
                            future_noise_seed=replica if replica else None,
                        )
                        check_pair(factual, branch, tick, same_noise=replica == 0)
                        if paired_reference is None:
                            paired_reference = branch
                            if replica == 0 and branch["ticks"] != factual["ticks"]:
                                raise RuntimeError(
                                    "Forzar la elección factual debe reproducirla exactamente"
                                )
                        check_pair(paired_reference, branch, tick, same_noise=True)
                        row = {
                            "case_seed": seed,
                            "tick": tick,
                            "replica": replica,
                            "assignment": assignment,
                            "selected": assignment == stage["assignment"],
                            "q": stage["q_values"][assignment],
                            "return": branch["ticks"][tick - 1]["stages"][0]["return"],
                            **branch["metrics"],
                        }
                        rows.append(row)
                        # Guardar retornos y trayectorias completas sin duplicar todos los Q.
                        file.write(
                            json.dumps(
                                {
                                    **row,
                                    "trajectory": [
                                        {k: v for k, v in t.items() if k != "stages"}
                                        for t in branch["ticks"]
                                    ],
                                    "measurements": branch["measurements"],
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    file.flush()
                summary = context_summary(rows)
                contexts.append(summary)
                print(
                    f"caso={seed} tick={tick} alternativas={len(alternatives)} réplicas={noise_replicas} regret={summary['return_regret']:.6f}",
                    flush=True,
                )
    result = {
        "training_executed": False,
        "calibration_by_case": calibrations,
        "contexts": contexts,
        "skipped_contexts": skipped,
        "contexts_with_better_sampled_return": sum(
            c["return_regret"] > 1e-7 for c in contexts
        ),
        "warning": "Máximos retrospectivos sobre pocas réplicas: no garantizan mejora esperada ni constituyen otra política evaluada.",
    }
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico de Q y retornos, sin entrenamiento"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Política exportada específica")
    source.add_argument(
        "--training-archive", type=Path, default=PROJECT_ROOT / "training.tar.zst"
    )
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--ticks", type=int, nargs="+", default=[1, 12, 24])
    parser.add_argument("--noise-replicas", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if spec["kind"] != "generated":
        parser.error("Se requieren escenarios generados de validación")
    payload = (
        args.checkpoint.read_bytes()
        if args.checkpoint
        else archive_files(args.training_archive, {"policy-best.pt"})["policy-best.pt"]
    )
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = TrainingConfig(**spec.get("training", {}))
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, env, protocol = prepare_experiment(spec, args.config.resolve().parent)
    train.close()
    assert env is not None
    try:
        if not _compatible(state, config, protocol):
            raise ValueError("La política no coincide con el protocolo configurado")
        policy, _ = load_policy(io.BytesIO(payload), device=args.device)
        output = args.output or CACHE_DIRECTORY / "value-diagnostics" / datetime.now(
            UTC
        ).strftime("%Y%m%d-%H%M%S-%f")
        print(f"Salida: {output}", flush=True)
        result = run_diagnostic(
            env,
            policy,
            case_seeds=spec["validation_seeds"]
            if args.all_cases
            else spec["validation_seeds"][:1],
            ticks=args.ticks,
            noise_replicas=args.noise_replicas,
            gamma=config.gamma,
            output=output,
            metadata={
                "policy_sha256": hashlib.sha256(payload).hexdigest(),
                "protocol": protocol,
                "torch_version": str(torch.__version__),
                "device": str(next(policy.network.parameters()).device),
                "code_sha256": {
                    name: hashlib.sha256(
                        (Path(__file__).parent / name).read_bytes()
                    ).hexdigest()
                    for name in (
                        "value_diagnostics.py",
                        "network.py",
                        "env.py",
                        "simulation.py",
                        "policies.py",
                        "routing.py",
                    )
                },
            },
        )
        print(
            json.dumps(
                {
                    "cases": len(result["calibration_by_case"]),
                    "contexts": len(result["contexts"]),
                    "contexts_with_better_sampled_return": result[
                        "contexts_with_better_sampled_return"
                    ],
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
