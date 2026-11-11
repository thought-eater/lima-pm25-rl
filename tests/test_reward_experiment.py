import json
import sys
import tarfile
from dataclasses import asdict

import numpy as np
import pytest

from lima_pm25_rl import reward_experiment as experiment
from lima_pm25_rl.reward_env import SyntheticErrorRewardEnv
from lima_pm25_rl.train_cli import run_training
from lima_pm25_rl.training import Trainer, TrainingConfig, load_policy
from lima_pm25_rl.value_diagnostics import rollout


@pytest.fixture
def controls(make_env, tmp_path):
    cfg = TrainingConfig(
        device="cpu", hidden_size=8, capacity_steps=12, batch_size=2, learning_starts=1
    )
    spec = {
        "kind": "generated",
        "training": asdict(cfg),
        "episodes": 2,
        "paired_seeds": [42],
        "validation_seeds": [100, 101],
        "steps_per_episode": 3,
        "checkpoint_every": 1,
        "validation_every": 1,
        "core_root": ".",
        "episode_configs": [],
    }
    source = tmp_path / "controls"
    source.mkdir()
    code = {"fixture": "audited"}
    (source / "experiment.json").write_text(json.dumps({"code_sha256": code}))
    trainer = Trainer(make_env(), cfg)
    folder = source / "bilg-42"
    run_training(
        trainer,
        episodes=2,
        output=folder,
        validation_env=make_env(),
        validation_seeds=[100, 101],
        checkpoint_every=1,
        validation_every=1,
    )
    for phase in ("initial", "last"):
        path = folder / f"policy-{phase}.pt"
        policy, _ = load_policy(path, device="cpu")
        target = folder / f"evaluation-{phase}"
        target.mkdir()
        (target / "policy.json").write_text(
            json.dumps({"policy_sha256": experiment._sha256(path)})
        )
        for case in spec["validation_seeds"]:
            report = rollout(make_env(), policy, case_seed=case, gamma=cfg.gamma)
            (target / f"case-{case}.json").write_text(json.dumps(report))
    return source, spec, {"control_code_sha256": code, "reference_policy_seeds": [7, 8]}


def test_small_reward_experiment_reuses_controls_and_trains_only_new_arm(
    controls, make_env, tmp_path, monkeypatch
):
    source, spec, settings = controls
    before = experiment.read_controls(source, spec)
    calls = []
    train_episode = Trainer.train_episode

    def spy(trainer):
        assert isinstance(trainer.env, SyntheticErrorRewardEnv)
        calls.append(trainer.config.seed)
        return train_episode(trainer)

    monkeypatch.setattr(Trainer, "train_episode", spy)
    output = tmp_path / "reward"
    result = experiment.run_experiment(
        spec,
        directory=tmp_path,
        output=output,
        source=source,
        settings=settings,
        train=make_env(),
        validation=make_env(),
        protocol={},
        maps=True,
    )
    assert calls == [42, 42]
    assert result["primary_snapshot"] == "last"
    assert len(result["summaries"]) == 4 and len(result["paired_last"]) == 2
    assert result["training_runs"][0]["episodes"] == 2
    assert result["training_runs"][0]["initial_weights_match"]
    assert sum(r["rollouts"] for r in result["references"]) == 10
    assert all(
        "native_reward" not in r["synthetic_error_minus_uncertainty"]
        for r in result["paired_last"]
    )
    assert (output / "references/greedy/policy-7/map.png").is_file()
    assert (output / "evaluations/synthetic-error-42/last/map.png").is_file()
    config = json.loads((output / "configs/synthetic-error-42.json").read_text())
    assert config["reward_mode"] == "synthetic_error"
    policy, protocol = load_policy(
        output / "synthetic-error-42/policy-last.pt", device="cpu"
    )
    assert (
        protocol["reward"]["mode"] == "synthetic_error"
        and policy.history_mode == "history"
    )
    rows = json.loads((output / "cases.json").read_text())
    initial = [r for r in rows if r["snapshot"] == "initial" and r["case_seed"] == 100]
    assert initial[0]["reference_reward"] == pytest.approx(
        initial[1]["reference_reward"]
    )
    assert initial[0]["temporal_mse"] == pytest.approx(initial[1]["temporal_mse"])
    assert initial[0]["native_reward"] != pytest.approx(initial[1]["native_reward"])
    for name, data in before.items():
        assert (source / name).read_bytes() == data
        assert (output / "controls" / name).read_bytes() == data
    references = json.loads((output / "references.json").read_text())
    with pytest.raises(ValueError, match="pares"):
        experiment.summarize(rows[:-1], [42], [100, 101], references)
    with pytest.raises(ValueError, match="referencias"):
        experiment.summarize(rows, [42], [100, 101], references[:-1])
    with pytest.raises(FileExistsError):
        experiment.run_experiment(
            spec,
            directory=tmp_path,
            output=output,
            source=source,
            settings=settings,
            train=make_env(),
            validation=make_env(),
            protocol={},
            maps=False,
        )


def test_tar_controls_match_directory_without_extracting(controls, tmp_path):
    source, spec, settings = controls
    path = tmp_path / "eval3.tar.zst"
    with tarfile.open(path, "w:zst") as archive:
        archive.add(source, arcname="cache/architecture-experiments/example")
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    raw = experiment.read_controls(path, spec)
    assert raw == experiment.read_controls(source, spec)
    records, initials, hashes = experiment.validate_control_files(
        raw.__getitem__, spec, {}, settings["control_code_sha256"]
    )
    assert len(records) == 4 and set(initials) == {42}
    assert set(hashes) == set(raw)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_initial_reproduction_failure_stops_before_any_new_training(
    controls, make_env, tmp_path, monkeypatch
):
    source, spec, settings = controls
    path = source / "bilg-42/evaluation-initial/case-100.json"
    report = json.loads(path.read_text())
    report["metrics"]["synthetic_mse"] += 1
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        Trainer,
        "train_episode",
        lambda self: pytest.fail("No entrenar con control incompatible"),
    )
    with pytest.raises(ValueError, match="métricas iniciales"):
        experiment.run_experiment(
            spec,
            directory=tmp_path,
            output=tmp_path / "bad",
            source=source,
            settings=settings,
            train=make_env(),
            validation=make_env(),
            protocol={},
            maps=False,
        )
    assert not list((tmp_path / "bad").rglob("checkpoint-last.pt"))


def test_check_does_not_construct_trainer_or_simulate(
    make_env, tmp_path, monkeypatch, capsys
):
    env, validation = make_env(), make_env()

    def prepare(*args, check):
        assert check
        return env, validation, {}

    monkeypatch.setattr(experiment, "prepare_experiment", prepare)
    monkeypatch.setattr(experiment, "check_control", lambda *a: None)
    monkeypatch.setattr(experiment, "read_controls", lambda *a: {})
    monkeypatch.setattr(experiment, "validate_control_files", lambda *a: ({}, {}, {}))
    monkeypatch.setattr(
        experiment, "Trainer", lambda *a, **k: pytest.fail("No crear entrenador")
    )
    monkeypatch.setattr(
        experiment, "run_experiment", lambda *a, **k: pytest.fail("No simular")
    )
    monkeypatch.setattr(
        sys, "argv", ["compare-rewards", "--controls", str(tmp_path), "--check"]
    )
    experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert result["new_training_episodes"] == 600
    assert (
        result["candidate_evaluation_missions"] == 72
        and result["reference_missions"] == 84
    )
    assert env._simulation is None and validation._simulation is None


def test_common_returns_are_comparable_but_native_returns_are_not_aggregated_across_modes():
    rows, references = [], []
    for case, mse in ((100, 1.0), (101, 9.0)):
        values = {key: mse for key in experiment.METRICS}
        for mode in experiment.MODES:
            for phase in ("initial", "last"):
                rows.append(
                    {
                        "reward_mode": mode,
                        "seed": 42,
                        "snapshot": phase,
                        "case_seed": case,
                        "native_reward": 100 if mode == "uncertainty" else -50,
                        "native_return": 80,
                        **values,
                    }
                )
        for strategy in ("random", "sweep", "greedy"):
            for seed in (7, 8):
                references.append(
                    {
                        "strategy": strategy,
                        "policy_seed": seed,
                        "case_seed": case,
                        **values,
                    }
                )
    result = experiment.summarize(rows, [42], [100, 101], references)
    assert result["summaries"][0]["temporal_rmse"] == pytest.approx(np.sqrt(5))
    assert result["references"][0]["temporal_rmse"] == pytest.approx(np.sqrt(5))
    assert (
        result["paired_last"][0]["synthetic_error_minus_uncertainty"][
            "reference_return"
        ]
        == 0
    )
    assert (
        "native_reward"
        not in result["paired_last"][0]["synthetic_error_minus_uncertainty"]
    )
