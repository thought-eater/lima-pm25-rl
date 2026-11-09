"""Métricas del evaluador; el campo completo nunca se entrega a la política."""

import numpy as np

from .env import LimaPM25Env


def evaluate_episode(
    env: LimaPM25Env, policy, *, seed: int
) -> dict[str, float | int | None]:
    observation, info = env.reset(seed=seed)
    assert env._simulation is not None
    if hasattr(policy, "reset"):
        policy.reset()
    initial = info["integrated_variance"]
    reward_sum = 0.0
    invalid_count = 0
    while True:
        observation, reward, terminated, truncated, info = env.step(
            policy.act(env, observation)
        )
        reward_sum += reward
        invalid_count += int(info["invalid_actions"].sum())
        if terminated or truncated:
            break
    # Acceso privilegiado exclusivamente aquí, fuera del contrato de observación.
    rmse = None
    if not env._simulation.is_historical_replay:
        mask = env.domain.grid.active_mask
        error = (
            env._posterior.mean_ug_m3[mask]
            - env._simulation.state.concentration_ug_m3[mask]
        )
        rmse = float(np.sqrt(np.mean(error**2)))
    sampled_cells = {
        env.domain.grid.xy_to_cell(record.x_m, record.y_m)
        for record in env._simulation.mobile.history.accepted_records
    }
    final = info["integrated_variance"]
    return {
        "steps": info["step_count"],
        "reward": reward_sum,
        "invalid_actions": invalid_count,
        "distance_m": float(info["distance_m"].sum()),
        "initial_integrated_variance": initial,
        "final_integrated_variance": final,
        "net_variance_reduction_percent": 100 * (initial - final) / initial
        if initial > 0
        else None,
        "sampling_coverage": len(sampled_cells)
        / int(env.domain.grid.active_mask.sum()),
        "synthetic_rmse_ug_m3": rmse,
    }
