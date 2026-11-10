import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lima_pm25.simulation import SensorType

from lima_pm25_rl import gp_diagnostics as diagnostic
from lima_pm25_rl.network import BiLGD3QN, NeuralController
from lima_pm25_rl.training import TrainingConfig
from lima_pm25_rl.value_diagnostics import rollout


def controller():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return NeuralController(BiLGD3QN(hidden_size=8).eval())


def test_spatial_coverage_and_uniform_uncertainty_do_not_break_ties_by_position():
    result = diagnostic.spatial_metrics([0, 2, 4], [0, 0, 0], [1, 1, 1])
    assert result["coverage_68"] == result["coverage_95"] == pytest.approx(1 / 3)
    assert result["spearman_variance_squared_error"] is None
    assert result["top_uncertainty_error_share"] == pytest.approx(1 / 3)
    assert result["top_uncertainty_error_enrichment"] == pytest.approx(1)
    exact = diagnostic.spatial_metrics([0, 0], [0, 0], [0, 0])
    assert exact["top_uncertainty_error_share"] is None
    assert exact["coverage_95"] == 1
    assert diagnostic.correlation([1, 1, 3], [2, 2, 9], ranks=True) == pytest.approx(1)


def test_lower_variance_may_coexist_with_worse_error():
    before = SimpleNamespace(
        mean_ug_m3=np.array([0.0]), standard_deviation_ug_m3=np.array([2.0])
    )
    after = SimpleNamespace(
        mean_ug_m3=np.array([3.0]), standard_deviation_ug_m3=np.array([1.0])
    )
    mobile = SimpleNamespace(sensor_type=SensorType.MOBILE, simulation_time_s=300)
    env = SimpleNamespace(
        _records=[mobile],
        _predict=lambda **kwargs: before,
        _posterior=after,
        _simulation=SimpleNamespace(
            state=SimpleNamespace(
                simulation_time_s=300, concentration_ug_m3=np.array([0.0])
            )
        ),
        domain=SimpleNamespace(grid=SimpleNamespace(active_mask=np.array([True]))),
        estimator_config=SimpleNamespace(
            reconstruction_config=SimpleNamespace(prior_stddev_ug_m3=2)
        ),
    )
    _, result = diagnostic.mobile_comparison(env, (), np.array([1.5]))
    assert result["unweighted_gain"] == 0.75
    assert result["weighted_gain"] == 1.125
    assert result["mse_reduction"] == -9


def test_counterfactual_keeps_old_mobiles_and_new_fixed_without_mutating_gp(
    make_env, monkeypatch
):
    env = make_env(concentration_priority=1, mobile_noise_stddev_ug_m3=1)
    policy = controller()
    observation, _ = env.reset(seed=42)
    observation, *_ = env.step(policy.act(env, observation))
    previous = tuple(env._records)
    action = policy.act(env, observation)
    mean = np.maximum(env._posterior.mean_ug_m3[env.domain.grid.active_mask], 0)
    config = env.estimator_config.reconstruction_config
    weights = 1 + mean / (
        max(config.prior_mean_ug_m3, config.prior_stddev_ug_m3) + mean
    )
    _, _, _, _, info = env.step(action)
    snapshot = env._updater.current_snapshot
    records = tuple(env._records)
    predict = env._predict
    captured = []

    def spy(**kwargs):
        captured.extend(kwargs["records"])
        return predict(**kwargs)

    monkeypatch.setattr(env, "_predict", spy)
    before, result = diagnostic.mobile_comparison(env, previous, weights)
    assert result["new_fixed_records"] == 1
    assert result["new_mobile_records"] == 3
    assert sum(r.sensor_type is SensorType.MOBILE for r in captured) == 3
    assert all(
        r.simulation_time_s == 300
        for r in captured
        if r.sensor_type is SensorType.MOBILE
    )
    assert any(
        r.simulation_time_s == 600
        for r in captured
        if r.sensor_type is SensorType.FIXED
    )
    assert (
        before.reconstruction_timestamp_utc
        == env._posterior.reconstruction_timestamp_utc
    )
    assert result["weighted_gain"] == pytest.approx(info["uncertainty_gain"])
    assert env._updater.current_snapshot is snapshot
    assert tuple(env._records) == records


def test_diagnostic_reproduces_original_rollout_and_keeps_fields_private(
    make_env, tmp_path
):
    env = make_env(
        concentration_priority=1,
        distance_cost_per_km=0.001,
        mobile_noise_stddev_ug_m3=1,
    )
    policy = controller()
    reference = rollout(env, policy, case_seed=42, gamma=0.99)
    report = diagnostic.diagnose_case(
        env, policy, seed=42, gamma=0.99, output=tmp_path / "case", map_ticks=(1,)
    )
    assert diagnostic.verify_reproduction(report, reference)["match"]
    assert len(report["spatial"]) == 4
    assert diagnostic.summarize_case(report)["decision_ticks"] == 3
    with np.load(tmp_path / "case/fields-private.npz") as fields:
        assert fields["truth"].shape == (4, 16)
        assert np.isnan(fields["weights_before_tick"][0]).all()
        assert np.isfinite(fields["weights_before_tick"][1:]).all()
    assert (tmp_path / "case/tick-01.png").is_file()
    assert set(env._observation()) == {"maps", "vehicles", "remaining", "action_mask"}
    changed = deepcopy(reference)
    changed["ticks"][0]["actions"][0] = -1
    with pytest.raises(ValueError, match="decisiones"):
        diagnostic.verify_reproduction(report, changed)


def test_early_termination_adds_spatial_assessments_but_no_mobile_contributions(
    env, tmp_path, monkeypatch
):
    original = env.action_masks
    monkeypatch.setattr(
        env,
        "action_masks",
        lambda reserved=(): (
            np.zeros((3, 8), dtype=np.int8) if env._steps >= 1 else original(reserved)
        ),
    )
    report = diagnostic.diagnose_case(
        env, controller(), seed=42, gamma=0.99, output=tmp_path / "early"
    )
    assert len(report["ticks"]) == 1
    assert len(report["spatial"]) == 4
    assert [r["decision_executed"] for r in report["spatial"]] == [
        False,
        True,
        False,
        False,
    ]
    assert (
        sum(r["sensor_type"] is SensorType.MOBILE for r in report["measurements"]) == 3
    )


def test_check_does_not_load_policy_or_simulate(tmp_path, monkeypatch, capsys):
    spec = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    path = tmp_path / "policy-last.pt"
    torch.save(
        {
            "version": 2,
            "kind": "policy",
            "training_config": asdict(TrainingConfig(**spec["training"])),
            "protocol": {},
            "completed_episodes": 200,
        },
        path,
    )
    references = tmp_path / "evaluation-last"
    references.mkdir()
    (references / "policy.json").write_text(
        json.dumps({"policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    for seed in spec["validation_seeds"]:
        (references / f"case-{seed}.json").write_text("{}")
    env = SimpleNamespace(close=lambda: None)

    def prepare(*args, check):
        assert check
        return env, env, {}

    monkeypatch.setattr(diagnostic, "prepare_experiment", prepare)
    monkeypatch.setattr(
        diagnostic, "load_policy", lambda *a, **k: pytest.fail("No cargar red")
    )
    monkeypatch.setattr(
        diagnostic, "diagnose_case", lambda *a, **k: pytest.fail("No simular")
    )
    monkeypatch.setattr(sys, "argv", ["audit-gp", "--checkpoint", str(path), "--check"])
    diagnostic.main()
    result = json.loads(capsys.readouterr().out)
    assert result["simulation_executed"] is False
    assert result["missions"] == 12
