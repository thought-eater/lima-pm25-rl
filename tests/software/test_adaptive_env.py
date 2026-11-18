from dataclasses import asdict, replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from lima_pm25.estimation import GPFitConfig
from lima_pm25.simulation import SensorType

from lima_pm25_rl.adaptive_env import (
    AdaptiveGPEnv,
    ModelChangeReward,
    model_change_terms,
)
from lima_pm25_rl.episodes import Episode
from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.policies import BaselinePolicy
from lima_pm25_rl.training import Trainer, TrainingConfig


def build(domain, template, estimator_config, **kwargs):
    return AdaptiveGPEnv(
        domain=domain,
        estimator_config=estimator_config,
        episode_factory=lambda seed: Episode(
            replace(template, config=replace(template.config, seed=seed))
        ),
        gp_fit=GPFitConfig(max_iterations=3, min_measurements=4, min_time_span_s=300),
        model_change=ModelChangeReward(),
        mobile_noise_stddev_ug_m3=1.0,
        **kwargs,
    )


def test_absolute_changes_are_per_cell_and_overlap_is_not_counted_twice(domain):
    before = SimpleNamespace(
        mean_ug_m3=np.zeros(domain.grid.shape),
        standard_deviation_ug_m3=np.ones(domain.grid.shape) * 3,
    )
    after = SimpleNamespace(
        mean_ug_m3=np.ones(domain.grid.shape),
        standard_deviation_ug_m3=np.ones(domain.grid.shape) * 3,
    )
    after.mean_ug_m3[::2] = -1
    after.standard_deviation_ug_m3[::2] += 1
    after.standard_deviation_ug_m3[1::2] -= 1
    settings = ModelChangeReward()
    one = model_change_terms(before, after, domain.grid, [(100, 100)], settings, 5)
    three = model_change_terms(
        before, after, domain.grid, [(100, 100)] * 3, settings, 5
    )
    assert one == three == pytest.approx((0.2, 0.2))


def test_adaptive_reward_keeps_signed_gain_and_fixed_reference_independent(
    domain, template, estimator_config
):
    env = build(domain, template, estimator_config, distance_cost_per_km=0.0005)
    observation, info = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    policy = BaselinePolicy("random", seed=1)
    for _ in range(3):
        observation, reward, _, _, info = env.step(policy.act(env, observation))
        assert env.observation_space.contains(observation)
        assert reward == pytest.approx(
            info["model_mean_change"]
            + 0.5 * info["model_uncertainty_change"]
            - info["distance_cost"]
        )
        assert "truth" not in info and "synthetic_mse" not in info
        assert all(
            r.sensor_type is SensorType.FIXED for r in env._fixed_adaptive.records
        )
    params = env._adaptive.parameters
    env.predict_fixed()
    assert env._adaptive.parameters == params
    env.reset(seed=42)
    assert (
        env._adaptive.parameters.spatial_scale_m
        == estimator_config.reconstruction_config.x_length_scale_m
    )
    assert len(env._adaptive.records) == 1


def test_fixed_only_is_identical_under_different_mobile_routes(
    domain, template, estimator_config
):
    environments = [build(domain, template, estimator_config) for _ in range(2)]
    finals = []
    for seed, env in enumerate(environments):
        evaluate_episode(
            env, BaselinePolicy("random", seed=seed), seed=42, comparison=True
        )
        finals.append(env.predict_fixed())
    mask = domain.grid.active_mask
    np.testing.assert_array_equal(
        finals[0].mean_ug_m3[mask], finals[1].mean_ug_m3[mask]
    )
    np.testing.assert_array_equal(
        finals[0].standard_deviation_ug_m3[mask],
        finals[1].standard_deviation_ug_m3[mask],
    )


def test_empty_mobile_batch_does_not_reward_aging(domain, template, estimator_config):
    env = build(domain, template, estimator_config)
    env.reset(seed=42)
    now = template.config.start_utc + timedelta(seconds=300)
    env._adaptive.predict(now)
    assert env._assimilate(None) == 0
    assert env._mean_change == env._std_change == 0


def test_trainer_records_model_change_components_and_refuses_fixed_protocol(
    domain, template, estimator_config
):
    env = build(domain, template, estimator_config)
    config = TrainingConfig(
        device="cpu",
        hidden_size=8,
        history_length=2,
        capacity_steps=8,
        batch_size=2,
        learning_starts=1,
    )
    with pytest.raises(ValueError, match="recompensa"):
        Trainer(env, config)
    trainer = Trainer(
        env,
        config,
        protocol={"adaptive_gp": asdict(env.gp_fit), "reward": env.reward_protocol},
    )
    row = trainer.train_episode()
    assert row["optimizer_updates"] == 3 and row["gp_fit_attempts"] > 0
    assert row["reward"] == pytest.approx(
        row["model_mean_change"] + 0.5 * row["model_uncertainty_change"]
    )


def test_saved_adaptive_missions_reproduce_fixed_and_mobile_posteriors(
    domain, template, estimator_config, make_env, tmp_path, monkeypatch
):
    import csv
    import json

    import lima_pm25_rl.training_report as reporting
    from lima_pm25_rl.comparison import run_comparison
    from lima_pm25_rl.experiments.gp_diagnostics import diagnose_case
    from lima_pm25_rl.experiments.gp_spatial_experiment import load_frozen_case
    from lima_pm25_rl.hotspot_evaluation import evaluate_saved

    fixed_env = make_env()
    diagnose_case(
        fixed_env,
        BaselinePolicy("random", seed=7),
        seed=42,
        gamma=0.99,
        output=tmp_path / "frozen",
    )
    origin = template.config.start_utc
    frozen = load_frozen_case(
        tmp_path / "frozen",
        domain.grid,
        start_utc=origin,
        end_utc=origin + timedelta(days=1),
    )
    env = build(
        domain, template, estimator_config, initial_nodes=((0, 0), (0, 3), (3, 0))
    )
    protocol = {"adaptive_gp": asdict(env.gp_fit), "reward": env.reward_protocol}
    run_comparison(
        env,
        BaselinePolicy("random", seed=7),
        case_seeds=[42],
        policy_seeds=[0],
        output=tmp_path / "missions",
        metadata={"protocol": protocol},
        maps=False,
    )
    trace = [
        json.loads(row)
        for row in (tmp_path / "missions/gp-fits.jsonl").read_text().splitlines()
    ]
    assert {row["branch"] for row in trace} == {
        "factual",
        "counterfactual",
        "fixed_only",
    }
    rows = evaluate_saved(
        tmp_path / "missions",
        {42: frozen},
        domain.grid,
        estimator_config,
        {"protocol": {}, "threshold_ug_m3": 20.0},
        tmp_path / "focos",
        map_ticks=[],
    )
    assert len(rows) == 5 and all(row["reproduction_passed"] for row in rows)
    run_comparison(
        fixed_env,
        BaselinePolicy("random", seed=7),
        case_seeds=[42],
        policy_seeds=[0],
        output=tmp_path / "fixed-missions",
        metadata={"protocol": {}},
        maps=False,
    )
    evaluate_saved(
        tmp_path / "fixed-missions",
        {42: frozen},
        domain.grid,
        estimator_config,
        {"protocol": {}, "threshold_ug_m3": 20.0},
        tmp_path / "fixed-focos",
        map_ticks=[],
    )
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            {
                "version": 2,
                "protocol": {},
                "threshold_ug_m3": 20.0,
            }
        )
    )
    products = {
        "protocol": tmp_path,
        "fixed": tmp_path / "fixed-focos",
        "adaptive": tmp_path / "focos",
    }
    monkeypatch.setattr(reporting, "product", products.__getitem__)
    output = tmp_path / "report"
    output.mkdir()
    html, _ = reporting.comparison_section(
        {
            "title": "Comparación de métodos",
            "protocol_product": "protocol",
            "products": {"fixed": "GP fijo", "adaptive": "GP adaptativo"},
        },
        output,
        1,
    )
    with (output / "comparison-1/summary.csv").open() as file:
        compared = list(csv.DictReader(file))
    assert len(compared) == 10  # Dos políticas y cuatro referencias propias por método.
    assert "Cada método conserva sus propias referencias" in html


def test_adaptive_training_exports_final_without_variance_selected_best(
    domain, template, estimator_config, tmp_path
):
    from lima_pm25_rl.study import run_training

    env = build(domain, template, estimator_config)
    config = TrainingConfig(
        device="cpu",
        hidden_size=8,
        history_length=2,
        capacity_steps=8,
        batch_size=2,
        learning_starts=1,
    )
    protocol = {"adaptive_gp": asdict(env.gp_fit), "reward": env.reward_protocol}
    trainer = Trainer(env, config, protocol=protocol)
    output = tmp_path / "training"
    row = run_training(
        trainer,
        episodes=1,
        output=output,
        validation_env=build(domain, template, estimator_config),
        validation_seeds=[42],
        validation_every=1,
        policy_snapshot_episodes=[1],
    )
    assert row["validation_mse"] >= 0
    assert (output / "policy-last.pt").exists()
    assert (output / "policy-episode-0001.pt").exists()
    assert not (output / "policy-best.pt").exists()
    other = Trainer(
        build(domain, template, estimator_config), config, protocol=protocol
    )
    other.load_checkpoint(output / "checkpoint-last.pt")
    assert other.completed_episodes == 1
    changed = {**protocol, "adaptive_gp": {**asdict(env.gp_fit), "max_iterations": 4}}
    with pytest.raises(ValueError, match="ajuste del GP"):
        Trainer(env, config, protocol=changed)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_interrupted_adaptive_training_resumes_best_rmse_and_generates_report(
    domain, template, estimator_config, tmp_path, monkeypatch, capsys, device
):
    import csv
    import json

    import torch

    import lima_pm25_rl.training_report as reporting
    from lima_pm25_rl.study import run_training
    from lima_pm25_rl.training import load_policy

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA no disponible")
    config = TrainingConfig(
        device=device,
        hidden_size=8,
        history_length=2,
        capacity_steps=12,
        batch_size=2,
        learning_starts=1,
    )
    env = build(domain, template, estimator_config)
    protocol = {"adaptive_gp": asdict(env.gp_fit), "reward": env.reward_protocol}
    trainer = Trainer(env, config, protocol=protocol)
    first, second = tmp_path / "interrupted", tmp_path / "resumed"
    original_episode = trainer.train_episode

    def interrupt():
        if trainer.completed_episodes == 3:
            raise KeyboardInterrupt
        return original_episode()

    monkeypatch.setattr(trainer, "train_episode", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_training(
            trainer,
            episodes=4,
            output=first,
            checkpoint_every=2,
            validation_env=build(domain, template, estimator_config),
            validation_seeds=[42],
            validation_every=2,
        )
    original_csv = (first / "metrics.csv").read_bytes()
    state = torch.load(first / "checkpoint-last.pt", weights_only=True)
    assert state["completed_episodes"] == 2
    assert state["best_validation_rmse"] is not None
    assert state["best_validation_episode"] <= 2
    resumed = Trainer(
        build(domain, template, estimator_config), config, protocol=protocol
    )
    resumed.load_checkpoint(first / "checkpoint-last.pt")
    assert resumed.best_validation_rmse == state["best_validation_rmse"]
    assert resumed.best_validation_episode == state["best_validation_episode"]
    run_training(
        resumed,
        episodes=4,
        output=second,
        checkpoint_every=2,
        validation_env=build(domain, template, estimator_config),
        validation_seeds=[42],
        validation_every=2,
        resumed_from=str(first / "checkpoint-last.pt"),
        policy_snapshot_episodes=[4],
    )
    # El episodio repetido reproduce recompensa y aprendizaje desde el checkpoint.
    with (first / "metrics.csv").open() as file:
        old = list(csv.DictReader(file))[-1]
    with (second / "metrics.csv").open() as file:
        repeated = next(csv.DictReader(file))
    for key in (
        "reward",
        "loss",
        "physical_steps",
        "optimizer_updates",
        "validation_mse",
    ):
        if old[key]:
            assert float(repeated[key]) == pytest.approx(float(old[key]), rel=1e-6)
        else:
            assert repeated[key] == ""
    monkeypatch.setattr(reporting, "product", lambda name: tmp_path / name)
    page = reporting.generate_report(
        {
            "training_runs": [
                {
                    "label": "Adaptativo",
                    "sources": [first.name, second.name],
                    "end_episodes": [2, None],
                }
            ],
            "smoothing_episodes": 2,
            "block_episodes": 2,
            "comparisons": [],
            "pending_note": "Prueba de software.",
        },
        tmp_path / "report",
    )
    assert page.is_file()
    assert (first / "metrics.csv").read_bytes() == original_csv
    with (tmp_path / "report/validation-1.csv").open() as file:
        validations = list(csv.DictReader(file))
    best = min(validations, key=lambda row: float(row["synthetic_mse"]))
    assert resumed.best_validation_rmse == pytest.approx(
        np.sqrt(float(best["synthetic_mse"]))
    )
    assert resumed.best_validation_episode == int(best["episode"])
    assert not (second / "policy-best.pt").exists()
    assert (second / "policy-episode-0004.pt").is_file()
    best_record = json.loads((second / "best-rmse.json").read_text())
    policy, _ = load_policy(second / best_record["policy_file"], device=device)
    reproduced = evaluate_episode(
        build(domain, template, estimator_config),
        policy,
        seed=42,
        comparison=True,
    )
    assert np.sqrt(reproduced["synthetic_mse"]) == pytest.approx(
        best_record["rmse"], rel=1e-7
    )
    checkpoint = torch.load(second / best_record["checkpoint_file"], weights_only=True)
    assert checkpoint["completed_episodes"] == best_record["episode"]
    assert all(
        torch.equal(v.cpu(), checkpoint["online"][key].cpu())
        for key, v in policy.network.state_dict().items()
    )
    text = capsys.readouterr().out
    assert "mejor recuperable=" in text and "en episodio" in text
