import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import SensorType

from lima_pm25_rl import Episode
from lima_pm25_rl.comparison import STRATEGIES, run_comparison, summarize_comparison
from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.network import BiLGD3QN, NeuralController
from lima_pm25_rl.policies import BaselinePolicy


def controller():
    torch.manual_seed(42)
    network = BiLGD3QN(hidden_size=8).eval()
    return NeuralController(network, history_length=2)


def test_fixed_reference_has_no_mobile_history_and_uses_the_same_final_time(env):
    row = evaluate_episode(
        env, BaselinePolicy("random", seed=3), seed=42, comparison=True
    )
    fixed = tuple(r for r in env._records if r.sensor_type is SensorType.FIXED)
    assert len(fixed) == row["fixed_measurements"] == 2
    assert row["mobile_measurements"] == 9
    assert row["elapsed_s"] == row["decision_end_elapsed_s"] == 900
    assert not row["early_termination"]
    reference = GaussianProcessFieldReconstructor(
        env.estimator_config.reconstruction_config
    ).reconstruct(
        ReconstructionRequest(
            grid=env.domain.grid,
            reconstruction_timestamp_utc=env._simulation.clock.utc_at(900),
            measurements=fixed,
        )
    )
    assert row["fixed_integrated_variance"] == pytest.approx(
        env._integrated_variance(reference)
    )
    assert row["fixed_integrated_variance"] > row["final_integrated_variance"]
    mask = env.domain.grid.active_mask
    expected = (
        reference.mean_ug_m3[mask] - env._simulation.state.concentration_ug_m3[mask]
    )
    assert row["fixed_synthetic_mse"] == pytest.approx(np.mean(expected**2))


def test_field_fixed_and_initial_position_signatures_do_not_depend_on_policy(make_env):
    a = evaluate_episode(
        make_env(), BaselinePolicy("random", seed=1), seed=42, comparison=True
    )
    b = evaluate_episode(
        make_env(), BaselinePolicy("greedy", seed=2), seed=42, comparison=True
    )
    for key in (
        "field_sha256",
        "fixed_measurements_sha256",
        "initial_positions_sha256",
    ):
        assert a[key] == b[key]
    assert a["fixed_synthetic_mse"] == b["fixed_synthetic_mse"]


def test_early_termination_ages_old_mobile_data_but_adds_only_fixed_until_horizon(
    make_env, monkeypatch
):
    env = make_env()
    original_masks = env.action_masks

    def masks(reserved=()):
        return (
            np.zeros((3, 8), dtype=np.int8)
            if env._steps >= 1
            else original_masks(reserved)
        )

    monkeypatch.setattr(env, "action_masks", masks)
    row = evaluate_episode(env, BaselinePolicy("random"), seed=42, comparison=True)
    assert row["early_termination"] and row["steps"] == 1
    assert row["decision_end_elapsed_s"] == 300 and row["elapsed_s"] == 900
    assert row["mobile_measurements"] == 3
    assert {
        r.simulation_time_s for r in env._records if r.sensor_type is SensorType.MOBILE
    } == {300}
    assert {
        r.simulation_time_s for r in env._records if r.sensor_type is SensorType.FIXED
    } == {0, 600}
    assert env._posterior.reconstruction_timestamp_utc == env._simulation.clock.utc_at(
        900
    )
    full = evaluate_episode(
        make_env(), BaselinePolicy("random"), seed=42, comparison=True
    )
    assert row["field_sha256"] == full["field_sha256"]
    assert row["fixed_measurements_sha256"] == full["fixed_measurements_sha256"]
    assert row["fixed_integrated_variance"] == full["fixed_integrated_variance"]


def test_comparison_refuses_historical_replay(make_env, template):
    env = make_env(episode_factory=lambda seed: Episode(template, ()))
    with pytest.raises(ValueError, match="replay histórico final"):
        evaluate_episode(env, BaselinePolicy("random"), seed=42, comparison=True)


def summary_row(strategy, seed, mse):
    return {
        "strategy": strategy,
        "case_seed": seed,
        "synthetic_mse": mse,
        "synthetic_rmse_ug_m3": float(np.sqrt(mse)),
        "synthetic_mae_ug_m3": mse / 2,
        "final_integrated_variance": mse * 100,
        "mean_integrated_variance": mse * 100,
        "reward": 0.1,
        "distance_m": 20,
        "sampling_coverage": 0.1,
        "early_termination": False,
        "invalid_actions": 0,
    }


def test_summary_uses_equal_case_weights_and_pooled_mse_not_mean_rmse():
    rows = [summary_row(name, seed, 4 * seed) for name in STRATEGIES for seed in (1, 4)]
    rows = [row for row in rows if row["strategy"] != "random"]
    rows.extend(
        summary_row("random", seed, mse)
        for seed, mse in ((1, 1), (1, 9), (4, 4), (4, 16))
    )
    summaries, paired = summarize_comparison(rows)
    random = next(row for row in summaries if row["strategy"] == "random")
    assert random["cases"] == 2 and random["rollouts"] == 4
    assert random["synthetic_rmse_ug_m3"] == pytest.approx(np.sqrt(7.5))
    assert random["policy_seed_rmse_sd_mean"] == pytest.approx(np.sqrt(2))
    first = next(
        row for row in paired if row["case_seed"] == 1 and row["reference"] == "random"
    )
    assert first["reference_rmse_ug_m3"] == pytest.approx(np.sqrt(5))


def test_zero_reference_errors_produce_no_infinite_percentages():
    rows = [summary_row(name, 1, 0) for name in STRATEGIES]
    summaries, paired = summarize_comparison(rows)
    assert all(row["rmse_reduction_vs_fixed_percent"] is None for row in summaries)
    assert all(row["variance_reduction_percent"] is None for row in paired)


def test_runner_writes_paired_metrics_samples_routes_and_five_preselected_maps(
    env, tmp_path
):
    policy = controller()
    weights = {key: value.clone() for key, value in policy.network.state_dict().items()}
    output = tmp_path / "comparison"
    summary = run_comparison(
        env, policy, case_seeds=[42], policy_seeds=[0, 1], output=output, metadata={}
    )
    assert len(summary["strategies"]) == 5
    with (output / "cases.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 7  # Seis rollouts y una referencia de solo fijas.
    assert len({row["field_sha256"] for row in rows}) == 1
    assert {row["strategy"] for row in rows} == set(STRATEGIES)
    assert all(row["elapsed_s"] == "900.0" for row in rows)
    with (output / "measurements.csv").open() as file:
        records = list(csv.DictReader(file))
    assert {row["sensor_type"] for row in records} == {"fixed", "mobile"}
    assert all(row["timestamp_utc"].endswith("+00:00") for row in records)
    assert (output / "trajectories.csv").stat().st_size > 100
    assert len(list((output / "maps").glob("*.png"))) == 5
    assert (output / "paired.csv").is_file() and (output / "summary.json").is_file()
    assert (
        json.loads((output / "config.json").read_text())["representative_case_seed"]
        == 42
    )
    assert all(
        torch.equal(value, policy.network.state_dict()[key])
        for key, value in weights.items()
    )
    assert all(parameter.grad is None for parameter in policy.network.parameters())
    saved = (output / "cases.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_comparison(
            env, policy, case_seeds=[42], policy_seeds=[0], output=output, metadata={}
        )
    assert (output / "cases.csv").read_bytes() == saved


def test_repetitions_create_fresh_policy_rngs_and_preserve_case_measurements(
    env, tmp_path
):
    policy = controller()
    rows = []
    for name in ("a", "b"):
        output = tmp_path / name
        run_comparison(
            env,
            policy,
            case_seeds=[42, 43],
            policy_seeds=[0, 1],
            output=output,
            metadata={},
            maps=False,
        )
        rows.append((output / "cases.csv").read_bytes())
    assert rows[0] == rows[1]


def test_unpaired_fields_are_rejected_instead_of_silently_averaged(
    env, tmp_path, monkeypatch
):
    from lima_pm25_rl import comparison

    original = comparison.evaluate_episode
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            result["field_sha256"] = "different-field"
        return result

    monkeypatch.setattr(comparison, "evaluate_episode", changed)
    output = tmp_path / "unpaired"
    with pytest.raises(RuntimeError, match="mismo campo"):
        run_comparison(
            env,
            controller(),
            case_seeds=[42],
            policy_seeds=[0],
            output=output,
            metadata={},
            maps=False,
        )
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [True]])
def test_bad_policy_seeds_are_rejected_before_creating_output(env, tmp_path, seeds):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="semillas"):
        run_comparison(
            env,
            controller(),
            case_seeds=[42],
            policy_seeds=seeds,
            output=output,
            metadata={},
        )
    assert not output.exists()


def test_greedy_uses_the_largest_current_legal_cell_uncertainty(env):
    observation, _ = env.reset(seed=42)
    legal = np.flatnonzero(env.action_masks()[0])
    expected = int(legal[-1])
    observation["maps"][3] = 0
    observation["maps"][3, *env.route(0, expected).cell] = 1
    action = BaselinePolicy("greedy", seed=0).act(env, observation)
    assert action[0] == expected


def test_sweep_alternates_rows_and_reset_restores_its_directions(
    domain, template, estimator_config, monkeypatch
):
    from lima_pm25_rl import LimaPM25Env

    template = replace(
        template,
        config=replace(template.config, duration_s=2400),
        weather=replace(template.weather, duration_s=2400),
    )
    env = LimaPM25Env(
        domain=domain,
        estimator_config=estimator_config,
        episode_factory=lambda seed: Episode(template),
        initial_nodes=((0, 0), (3, 2), (3, 3)),
    )
    original_masks = env.action_masks

    def masks(reserved=()):
        result = original_masks(reserved)
        result[1:] = 0
        return result

    monkeypatch.setattr(env, "action_masks", masks)
    observation, _ = env.reset(seed=42)
    policy = BaselinePolicy("sweep")
    cells = []
    for _ in range(8):
        observation, _, _, _, info = env.step(policy.act(env, observation))
        assert info["invalid_actions"].sum() == 0
        cells.append(env.router.cell(env._nodes[0]))
    assert cells == [(0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0)]
    policy.reset()
    assert policy.horizontal == [2, 6, 2] and policy.vertical == [0, 0, 0]
