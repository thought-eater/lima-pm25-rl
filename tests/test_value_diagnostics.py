from copy import deepcopy

import numpy as np
import pytest
import torch

from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.network import BiLGD3QN, NeuralController
from lima_pm25_rl.value_diagnostics import (
    check_pair,
    context_summary,
    discounted_returns,
    observed_plan,
    rollout,
)


def policy():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    return NeuralController(BiLGD3QN(hidden_size=8).eval())


def test_returns_are_fleet_returns_once_per_tick():
    assert discounted_returns([1, 2, 3], 0.5) == [2.75, 3.5, 3]


def test_factual_matches_evaluator_and_terminal_q_has_no_bootstrap(env):
    controller = policy()
    report = rollout(env, controller, case_seed=42, gamma=0.9)
    reference = evaluate_episode(env, controller, seed=42, comparison=True)
    for key in ("reward", "distance_m", "synthetic_mse", "mean_integrated_variance"):
        assert report["metrics"][key] == pytest.approx(reference[key])
    for key in (
        "field_sha256",
        "initial_positions_sha256",
        "fixed_measurements_sha256",
    ):
        assert report["signatures"][key] == reference[key]
    terminal = report["ticks"][-1]
    assert terminal["stages"][-1]["online_bellman_residual"] == pytest.approx(
        terminal["stages"][-1]["q"] - terminal["reward"]
    )
    for tick in report["ticks"]:
        assert len({s["return"] for s in tick["stages"]}) == 1


def test_intervention_keeps_weights_past_and_original_q_and_cleans_hook(make_env):
    env = make_env(mobile_noise_stddev_ug_m3=1)
    controller = policy()
    weights = deepcopy(controller.network.state_dict())
    factual = rollout(env, controller, case_seed=42, gamma=0.9)
    stage = factual["ticks"][1]["stages"][0]
    same = rollout(
        env, controller, case_seed=42, gamma=0.9, intervention=(2, stage["assignment"])
    )
    assert same["ticks"] == factual["ticks"]
    alternative = next(
        a for a in stage["legal_assignments"] if a != stage["assignment"]
    )
    branch = rollout(
        env, controller, case_seed=42, gamma=0.9, intervention=(2, alternative)
    )
    check_pair(factual, branch, 2, same_noise=True)
    assert branch["ticks"][1]["actions"][0] == alternative
    assert branch["ticks"][1]["stages"][0]["q"] == stage["q_values"][alternative]
    assert all(
        torch.equal(value, controller.network.state_dict()[name])
        for name, value in weights.items()
    )
    assert not controller.network._forward_hooks
    # Nueva realización futura: mismo pasado y contexto, solo cambia ruido móvil.
    resampled = rollout(
        env,
        controller,
        case_seed=42,
        gamma=0.9,
        intervention=(2, alternative),
        future_noise_seed=1,
    )
    check_pair(factual, resampled, 2, same_noise=False)
    assert (
        resampled["ticks"][1]["sensor_rng_after_sha256"]
        != branch["ticks"][1]["sensor_rng_after_sha256"]
    )
    with pytest.raises(RuntimeError, match="ruido"):
        check_pair(factual, resampled, 2, same_noise=True)
    changed = deepcopy(branch)
    changed["signatures"]["field_sha256"] = "incorrect"
    with pytest.raises(RuntimeError, match="campo"):
        check_pair(factual, changed, 2, same_noise=True)


def test_forced_illegal_assignment_is_rejected_without_leaking_hooks(env):
    controller = policy()
    observation, _ = env.reset(seed=42)
    illegal = int(np.flatnonzero(~observation["action_mask"].astype(bool).ravel())[0])
    with pytest.raises(ValueError, match="legal"):
        observed_plan(env, controller, observation, illegal)
    assert not controller.network._forward_hooks


def test_early_terminal_quality_advances_without_fabricating_return(env, monkeypatch):
    original = env.action_masks
    monkeypatch.setattr(
        env,
        "action_masks",
        lambda reserved=(): (
            np.zeros((3, 8), dtype=np.int8) if env._steps >= 1 else original(reserved)
        ),
    )
    report = rollout(env, policy(), case_seed=42, gamma=0.9)
    assert report["metrics"]["decision_end_elapsed_s"] == 300
    assert report["signatures"]["horizon_s"] == 900
    assert len(report["ticks"]) == 1
    assert report["metrics"]["discounted_return"] == report["ticks"][0]["reward"]
    assert sum(r["sensor_type"] == "mobile" for r in report["measurements"]) == 3


def test_context_summary_averages_replicas_before_selecting_return_maximum():
    rows = []
    for replica, returns in enumerate(((1, 4), (3, 2))):
        for assignment, total in enumerate(returns):
            rows.append(
                {
                    "case_seed": 42,
                    "tick": 1,
                    "replica": replica,
                    "assignment": assignment,
                    "selected": assignment == 0,
                    "q": 2 - assignment,
                    "return": total,
                    "synthetic_mse": 4 + assignment,
                    "temporal_mse": 4 + assignment,
                    "final_integrated_variance": 10,
                }
            )
    result = context_summary(rows)
    assert result["return_regret"] == 1
    assert result["paired_return_deltas"] == [3, -1]
    assert result["order_agreement"] == 0
    assert result["best_return_final_rmse_change"] > 0
