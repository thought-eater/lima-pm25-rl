"""Métricas del evaluador; el campo completo nunca se entrega a la política."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np
from lima_pm25.simulation import MeasurementBatch, SensorType

from .env import LimaPM25Env


def fixed_posterior(env: LimaPM25Env):
    """Reconstruye con TODO el historial fijo elegible, sin ningún móvil previo."""
    return env._predict(
        records=tuple(r for r in env._records if r.sensor_type is SensorType.FIXED)
    )


def _spatial_errors(env: LimaPM25Env, posterior) -> dict[str, float | None]:
    assert env._simulation is not None
    if env._simulation.is_historical_replay:
        return {
            "synthetic_mse": None,
            "synthetic_rmse_ug_m3": None,
            "synthetic_mae_ug_m3": None,
        }
    mask = env.domain.grid.active_mask
    error = posterior.mean_ug_m3[mask] - env._simulation.state.concentration_ug_m3[mask]
    mse = float(np.mean(error**2))
    return {
        "synthetic_mse": mse,
        "synthetic_rmse_ug_m3": float(np.sqrt(mse)),
        "synthetic_mae_ug_m3": float(np.mean(np.abs(error))),
    }


def evaluate_episode(
    env: LimaPM25Env, policy, *, seed: int, comparison: bool = False
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    assert env._simulation is not None
    simulation = env._simulation
    if comparison and env._simulation.is_historical_replay:
        raise ValueError("La comparación diagnóstica no abre el replay histórico final")
    if hasattr(policy, "reset"):
        policy.reset()
    initial = info["integrated_variance"]
    variances = [initial]
    fixed_variances = [initial]
    reward_sum = 0.0
    invalid_count = 0
    # Huella privada del evaluador: no se añade a observaciones ni a info.
    field_hash = hashlib.sha256()

    def record_field() -> None:
        if comparison:
            field_hash.update(
                simulation.state.concentration_ug_m3[
                    env.domain.grid.active_mask
                ].tobytes()
            )

    record_field()
    initial_positions = env.positions
    while True:
        observation, reward, terminated, truncated, info = env.step(
            policy.act(env, observation)
        )
        reward_sum += reward
        invalid_count += int(info["invalid_actions"].sum())
        variances.append(info["integrated_variance"])
        if comparison:
            fixed_variances.append(env._integrated_variance(fixed_posterior(env)))
        record_field()
        if terminated or truncated:
            break
    decision_end = float(info["elapsed_s"])
    early = bool(terminated and not truncated)
    if comparison:
        # La misión terminó: no inventar más decisiones ni nuevas lecturas móviles.
        # Solo el evaluador avanza hasta el horizonte común y conserva las fijas.
        while env._simulation.state.simulation_time_s < env._simulation.duration_s:
            for batch in env._simulation.advance(env.positions):
                fixed = tuple(
                    r for r in batch.records if r.sensor_type is SensorType.FIXED
                )
                if fixed:
                    env._assimilate(
                        MeasurementBatch(
                            batch.simulation_time_s, batch.timestamp_utc, fixed
                        )
                    )
            record_field()
            variances.append(env._integrated_variance(env._predict()))
            fixed_variances.append(env._integrated_variance(fixed_posterior(env)))
        env._posterior = env._predict()
        info = env._info()
    sampled_cells = {
        env.domain.grid.xy_to_cell(record.x_m, record.y_m)
        for record in env._records
        if record.sensor_type is SensorType.MOBILE
    }
    final = info["integrated_variance"]
    metrics = {
        "steps": info["step_count"],
        "reward": reward_sum,
        "invalid_actions": invalid_count,
        "distance_m": float(info["distance_m"].sum()),
        "initial_integrated_variance": initial,
        "final_integrated_variance": final,
        "mean_integrated_variance": float(
            np.trapezoid(variances) / (len(variances) - 1)
        ),
        "net_variance_reduction_percent": 100 * (initial - final) / initial
        if initial > 0
        else None,
        "sampling_coverage": len(sampled_cells)
        / int(env.domain.grid.active_mask.sum()),
        **_spatial_errors(env, env._posterior),
    }
    if comparison:
        fixed = tuple(r for r in env._records if r.sensor_type is SensorType.FIXED)
        reference = fixed_posterior(env)
        metrics.update(
            elapsed_s=float(info["elapsed_s"]),
            decision_end_elapsed_s=decision_end,
            early_termination=early,
            fixed_measurements=len(fixed),
            mobile_measurements=sum(
                r.sensor_type is SensorType.MOBILE for r in env._records
            ),
            start_utc=env._simulation.request.config.start_utc.isoformat(),
            field_sha256=field_hash.hexdigest(),
            fixed_measurements_sha256=hashlib.sha256(
                json.dumps(
                    [asdict(r) for r in fixed], default=str, sort_keys=True
                ).encode()
            ).hexdigest(),
            initial_positions_sha256=hashlib.sha256(
                np.asarray(initial_positions).tobytes()
            ).hexdigest(),
            fixed_integrated_variance=env._integrated_variance(reference),
            fixed_mean_integrated_variance=float(
                np.trapezoid(fixed_variances) / (len(fixed_variances) - 1)
            ),
            **{
                f"fixed_{key}": value
                for key, value in _spatial_errors(env, reference).items()
            },
        )
    return metrics
