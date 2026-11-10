import hashlib
import json
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lima_pm25_rl import gp_feedback_experiment as experiment
from lima_pm25_rl.network import BiLGD3QN, NeuralController
from lima_pm25_rl.training import TrainingConfig
from lima_pm25_rl.value_diagnostics import rollout


def controller():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return NeuralController(BiLGD3QN(hidden_size=8).eval())


def test_gp_feedback_reaches_both_estimators_and_policy_with_frozen_weights(
    make_env, tmp_path, monkeypatch
):
    env = make_env(concentration_priority=1, mobile_noise_stddev_ug_m3=1)
    policy = controller()
    reference = rollout(env, policy, case_seed=42, gamma=0.99)
    original = env.estimator_config
    weights = experiment.weights_digest(policy.network)
    seen = []
    plan = policy.plan

    def spy(environment, observation, **kwargs):
        gp = environment.estimator_config.reconstruction_config
        assert environment._updater.config.reconstruction_config == gp
        assert environment._reconstructor.config == gp
        expected = asdict(original.reconstruction_config)
        expected["temporal_length_scale_s"] = gp.temporal_length_scale_s
        assert asdict(gp) == expected
        assert set(observation) == {"maps", "vehicles", "remaining", "action_mask"}
        seen.append(gp.temporal_length_scale_s)
        return plan(environment, observation, **kwargs)

    monkeypatch.setattr(policy, "plan", spy)
    output = tmp_path / "paired"
    series = reference["evaluation_series"]
    metrics = {
        **reference["metrics"],
        **experiment.temporal_thirds(
            [r["synthetic_mse"] for r in series],
            np.asarray([r["elapsed_s"] for r in series]),
        ),
    }
    # Referencia numérica artificial para comprobar la descomposición, no GP ajustado.
    offline = {
        42: {
            "signatures": reference["signatures"],
            3600.0: {k: metrics[k] for k in experiment.ERROR_METRICS},
            7200.0: {k: metrics[k] + 1 for k in experiment.ERROR_METRICS},
        }
    }
    result = experiment.run_feedback(
        env,
        policy,
        seeds=[42],
        gamma=0.99,
        references={42: reference},
        output=output,
        metadata={},
        maps=True,
        offline=offline,
    )
    assert seen == [3600] * 3 + [7200] * 3
    assert env.estimator_config is original
    assert experiment.weights_digest(policy.network) == weights
    pair = result["paired"][0]
    assert pair["signatures_match"]
    assert pair["noise_verified_common_decision_ticks"] == 3
    assert pair["first_input_difference_tick"] == 2
    assert result["feedback_vs_frozen_candidate"]["synthetic_mse"][
        "change"
    ] == pytest.approx(
        result["variants"][1]["synthetic_mse"] - metrics["synthetic_mse"] - 1
    )
    assert (output / "case-42/seconds-7200/map.png").is_file()
    before = json.loads((output / "case-42/seconds-3600/report.json").read_text())
    after = json.loads((output / "case-42/seconds-7200/report.json").read_text())
    after["ticks"][0]["sensor_rng_after_sha256"] = "different"
    with pytest.raises(ValueError, match="ruido"):
        experiment.compare_pair(before, after)
    after["signatures"]["field_sha256"] = "different"
    with pytest.raises(ValueError, match="campo"):
        experiment.compare_pair(before, after)
    original_rollout = experiment.rollout

    def failing(environment, *args, **kwargs):
        if (
            environment.estimator_config.reconstruction_config.temporal_length_scale_s
            == 7200
        ):
            raise RuntimeError("fixture failure")
        return original_rollout(environment, *args, **kwargs)

    monkeypatch.setattr(experiment, "rollout", failing)
    with pytest.raises(RuntimeError, match="fixture failure"):
        experiment.run_feedback(
            env,
            policy,
            seeds=[42],
            gamma=0.99,
            references={42: reference},
            output=tmp_path / "failed",
            metadata={},
            maps=False,
        )
    assert env.estimator_config is original
    assert not (tmp_path / "failed/summary.json").exists()


def test_early_stop_preserves_evaluation_horizon_without_extra_rewards(
    env, tmp_path, monkeypatch
):
    masks = env.action_masks
    monkeypatch.setattr(
        env,
        "action_masks",
        lambda reserved=(): (
            np.zeros((3, 8), dtype=np.int8) if env._steps >= 1 else masks(reserved)
        ),
    )
    policy = controller()
    reference = rollout(env, policy, case_seed=42, gamma=0.99)
    experiment.run_feedback(
        env,
        policy,
        seeds=[42],
        gamma=0.99,
        references={42: reference},
        output=tmp_path / "early",
        metadata={},
        maps=False,
    )
    report = json.loads(
        (tmp_path / "early/case-42/seconds-7200/report.json").read_text()
    )
    assert len(report["ticks"]) == 1 and len(report["evaluation_series"]) == 4
    assert report["metrics"]["discounted_return"] == report["ticks"][0]["reward"]
    assert report["evaluation_series"][-1]["elapsed_s"] == 900
    assert sum(r["sensor_type"] == "mobile" for r in report["measurements"]) == 3
    thirds = [
        report["metrics"][f"{name}_third_mse"] for name in ("first", "middle", "last")
    ]
    assert np.mean(thirds) == pytest.approx(report["metrics"]["temporal_mse"])


def test_summary_pools_mse_and_rejects_incomplete_pairs():
    rows = []
    for scale, values in ((3600, (1, 9)), (7200, (4, 16))):
        for seed, mse in zip((42, 43), values, strict=True):
            rows.append(
                {
                    **dict.fromkeys(experiment.METRICS, 0),
                    "case_seed": seed,
                    "temporal_scale_s": scale,
                    "synthetic_mse": mse,
                }
            )
    result = experiment.summarize(rows, {42: {}, 43: {}})
    assert result["variants"][0]["synthetic_rmse_ug_m3"] == pytest.approx(np.sqrt(5))
    assert result["variants"][1]["synthetic_rmse_ug_m3"] == pytest.approx(np.sqrt(10))
    with pytest.raises(ValueError, match="pares completos"):
        experiment.summarize(rows[:-1], {42: {}, 43: {}})


def test_frozen_reference_loader_checks_consistency(tmp_path):
    rows = []
    for scale in experiment.VARIANTS:
        row = {k: 1.0 for k in experiment.ERROR_METRICS if k != "synthetic_mse"}
        row.update(case_seed=42, temporal_scale_s=scale, final_mse=2.0)
        rows.append(row)
        folder = tmp_path / "validation/case-42" / f"seconds-{scale:g}"
        folder.mkdir(parents=True)
        (folder / "report.json").write_text(
            json.dumps({"summary": row, "input_signatures": {"field_sha256": "same"}})
        )
    (tmp_path / "validation.json").write_text(json.dumps(rows))
    assert experiment.load_offline(tmp_path, [42])[42][7200]["synthetic_mse"] == 2
    rows[1]["final_mse"] = 3
    (tmp_path / "validation.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="reportes"):
        experiment.load_offline(tmp_path, [42])


def test_check_does_not_simulate_or_load_controller(
    domain, estimator_config, tmp_path, monkeypatch, capsys
):
    spec = json.loads(experiment.DEFAULT_CONFIG.read_text())
    gp = replace(
        estimator_config.reconstruction_config,
        x_length_scale_m=2000,
        y_length_scale_m=2000,
    )
    env = SimpleNamespace(
        domain=domain,
        estimator_config=replace(estimator_config, reconstruction_config=gp),
        arrival_strategy="shortest",
        close=lambda: None,
    )
    source, diagnostic = tmp_path / "temporal", tmp_path / "diagnostic"
    source.mkdir()
    diagnostic.mkdir()
    checkpoint = tmp_path / "policy-last.pt"
    torch.save(
        {
            "version": 2,
            "kind": "policy",
            "training_config": asdict(TrainingConfig(**spec["training"])),
            "protocol": {},
            "completed_episodes": 200,
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (source / "summary.json").write_text("{}")
    (source / "config.json").write_text(
        json.dumps(
            {
                "protocol": {},
                "original_gp": asdict(gp),
                "history_duration_s": 21600,
                "selection_partition": "training",
                "validation_seeds": spec["validation_seeds"],
                "diagnostic_source": str(diagnostic),
                "policy_sha256": digest,
            }
        )
    )
    (source / "selection.json").write_text(
        json.dumps(
            {
                "selection_partition": "training",
                "original_scale_s": 3600,
                "selected_scale_s": 7200,
                "selected_gp": asdict(replace(gp, temporal_length_scale_s=7200)),
            }
        )
    )
    (diagnostic / "config.json").write_text(
        json.dumps({"policy_sha256": digest, "checkpoint": str(checkpoint)})
    )
    reference = tmp_path / "evaluation-last"
    reference.mkdir()
    (reference / "policy.json").write_text(json.dumps({"policy_sha256": digest}))
    for seed in spec["validation_seeds"]:
        (reference / f"case-{seed}.json").write_text("{}")

    def prepare(*args, check):
        assert check
        return env, env, {}

    monkeypatch.setattr(experiment, "prepare_experiment", prepare)
    monkeypatch.setattr(
        experiment, "load_policy", lambda *a, **k: pytest.fail("No cargar red")
    )
    monkeypatch.setattr(
        experiment, "run_feedback", lambda *a, **k: pytest.fail("No simular")
    )
    monkeypatch.setattr(experiment, "load_offline", lambda *a: {})
    monkeypatch.setattr(
        sys, "argv", ["feedback", "--temporal-experiment", str(source), "--check"]
    )
    experiment.main()
    report = json.loads(capsys.readouterr().out)
    assert report["missions"] == 24 and report["simulation_executed"] is False
