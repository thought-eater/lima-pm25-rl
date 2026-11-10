import json
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lima_pm25_rl import history_experiment as experiment
from lima_pm25_rl.policies import sequence_input
from lima_pm25_rl.replay import SequenceReplay
from lima_pm25_rl.train_cli import run_training
from lima_pm25_rl.training import Trainer, TrainingConfig, load_policy
from lima_pm25_rl.value_diagnostics import rollout


def test_repeat_current_removes_past_but_keeps_last_planning_context(env):
    observation, _ = env.reset(seed=42)
    history = []
    for tick in range(3):
        item = {k: v.copy() for k, v in observation.items()}
        item["maps"][2] = tick
        item["vehicles"][:] = tick / 10
        item["remaining"][:] = 1 - tick / 5
        history.append(item)
    kwargs = {
        "vehicle": 1,
        "reserved_cells": ((0, 0),),
        "history_length": 4,
        "pending": (1, 2),
        "assigned_positions": ((0.2, 0.3), (0, 0), (0, 0)),
        "action_mask": np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    }
    control = sequence_input(history, **kwargs)
    repeated = sequence_input(history, **kwargs, history_mode="repeat_current")
    np.testing.assert_array_equal(control[0][:, 2, 0, 0], [0, 0, 1, 2])
    assert np.all(repeated[0][:, 2] == 2)
    assert np.all(repeated[1][:, 1:7] == history[-1]["vehicles"].ravel())
    assert np.all(repeated[0][:-1, 6] == 0) and repeated[0][-1, 6, 0, 0] == 1
    assert np.all(repeated[1][:-1, 10:13] == 1)
    np.testing.assert_array_equal(repeated[1][-1, 10:13], [0, 1, 1])
    np.testing.assert_array_equal(control[0][-1], repeated[0][-1])
    np.testing.assert_array_equal(control[1][-1], repeated[1][-1])
    history[0]["maps"][:] = 99
    after = sequence_input(history, **kwargs, history_mode="repeat_current")
    for a, b in zip(repeated, after, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(sequence_input(history[-1:], **kwargs), repeated, strict=True):
        np.testing.assert_array_equal(a, b)


def test_inference_replay_and_double_targets_use_identical_current_windows(
    make_env, tmp_path
):
    config = TrainingConfig(
        device="cpu",
        hidden_size=8,
        history_length=4,
        history_mode="repeat_current",
        capacity_steps=12,
        batch_size=2,
        learning_starts=100,
        epsilon_start=0,
        epsilon_end=0,
    )
    trainer = Trainer(make_env(), config)
    captured = []
    hook = trainer.online.register_forward_pre_hook(
        lambda module, args: captured.append(
            tuple(t.detach().cpu().numpy().copy() for t in args)
        )
    )
    trainer.train_episode()
    hook.remove()
    episode = trainer.replay.episodes[0]
    decisions = [d for tick in episode.decisions for d in tick]
    assert len(captured) == len(decisions) == len(trainer.replay)
    for index, decision in enumerate(decisions):
        transition = trainer.replay.transition(index)
        for part in (0, 1):
            np.testing.assert_array_equal(
                transition["observations"][part],
                captured[index][part][decision.vehicle],
            )
            if index + 1 < len(decisions):
                np.testing.assert_array_equal(
                    transition["next_observations"][part], captured[index + 1][part]
                )
    # El objetivo del último par del tick 0 repite la observación del tick 1, no la terminal.
    transition = trainer.replay.transition(len(episode.decisions[0]) - 1)
    expected = episode.observations[1]["maps"][2]
    np.testing.assert_array_equal(
        transition["next_observations"][0][0, :, 2], np.stack([expected] * 4)
    )
    trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    with pytest.raises(ValueError, match="hiperparámetros"):
        Trainer(make_env(), replace(config, history_mode="history")).load_checkpoint(
            tmp_path / "checkpoint.pt"
        )
    with pytest.raises(ValueError, match="buffer"):
        SequenceReplay(12, 4, config.gamma, seed=1).load_state_dict(
            trainer.replay.state_dict()
        )
    trainer.export_policy(tmp_path / "policy.pt")
    restored, _ = load_policy(tmp_path / "policy.pt", device="cpu")
    assert restored.history_mode == "repeat_current"


def test_validate_propagates_mode_and_does_not_touch_gp_or_buffer(
    make_env, monkeypatch
):
    from lima_pm25_rl import training

    config = TrainingConfig(device="cpu", hidden_size=8, history_mode="repeat_current")
    trainer = Trainer(make_env(), config)
    validation = make_env()
    original_gp = validation.estimator_config
    evaluate = training.evaluate_episode
    seen = []

    def spy(env, policy, **kwargs):
        seen.append(policy.history_mode)
        return evaluate(env, policy, **kwargs)

    monkeypatch.setattr(training, "evaluate_episode", spy)
    trainer.validate(validation, [42])
    assert seen == ["repeat_current"]
    assert validation.estimator_config is original_gp
    assert trainer.updates == 0 and len(trainer.replay) == 0


def test_small_full_experiment_reuses_controls_and_never_warm_starts(
    make_env, tmp_path, monkeypatch
):
    spec = {
        "training": asdict(
            TrainingConfig(
                device="cpu",
                hidden_size=8,
                capacity_steps=12,
                batch_size=2,
                learning_starts=1,
            )
        ),
        "episodes": 2,
        "steps_per_episode": 3,
        "paired_seeds": [42],
        "validation_seeds": [100],
        "validation_every": 1,
        "checkpoint_every": 1,
        "core_root": ".",
        "episode_configs": [],
    }
    source = tmp_path / "controls"
    source.mkdir()
    code = {"fixture": "audited"}
    (source / "experiment.json").write_text(json.dumps({"code_sha256": code}))
    (source / "summary.json").write_text("{}")
    config = TrainingConfig(**spec["training"])
    trainer = Trainer(make_env(), config)
    run = source / "bilg-42"
    run_training(
        trainer,
        episodes=2,
        output=run,
        validation_env=make_env(),
        validation_seeds=[100],
        checkpoint_every=1,
        validation_every=1,
    )
    saved = json.loads((run / "config.json").read_text())
    saved["training"].pop("history_mode")
    (run / "config.json").write_text(json.dumps(saved))
    for phase in ("initial", "last"):
        path = run / f"policy-{phase}.pt"
        state = torch.load(path, weights_only=True)
        state["training_config"].pop("history_mode")
        torch.save(state, path)
        policy, _ = load_policy(run / f"policy-{phase}.pt", device="cpu")
        report = rollout(make_env(), policy, case_seed=100, gamma=config.gamma)
        # Formato de las trazas archivadas de eval3, sin serie inicial explícita.
        expected = experiment.case_row(
            report, seed=42, snapshot=phase, history_mode="history"
        )
        report.pop("evaluation_series")
        actual = experiment.case_row(
            report, seed=42, snapshot=phase, history_mode="history"
        )
        assert actual["last_third_mse"] == pytest.approx(expected["last_third_mse"])
        folder = run / f"evaluation-{phase}"
        folder.mkdir()
        (folder / "case-100.json").write_text(json.dumps(report))
        (folder / "policy.json").write_text(
            json.dumps(
                {"policy_sha256": experiment._sha256(run / f"policy-{phase}.pt")}
            )
        )
    _, _, hashes = experiment.validate_controls(source, spec, {}, code)
    constructions = []

    def build(env, cfg, **kwargs):
        constructions.append(cfg.history_mode)
        return Trainer(env, cfg, **kwargs)

    monkeypatch.setattr(experiment, "Trainer", build)
    output = tmp_path / "result"
    result = experiment.run_experiment(
        spec,
        directory=tmp_path,
        output=output,
        source=source,
        settings={"control_code_sha256": code},
        train=make_env(),
        validation=make_env(),
        protocol={},
        maps=False,
    )
    assert constructions == ["repeat_current"]
    assert result["training_runs"][0]["initial_weights_match"]
    assert result["training_runs"][0]["episodes"] == 2
    assert len(result["summaries"]) == 4
    assert len(result["paired_last"]) == 1
    assert (
        json.loads((output / "repeat-current-42/config.json").read_text())["training"][
            "history_mode"
        ]
        == "repeat_current"
    )
    assert all(
        experiment._sha256(source / name) == value for name, value in hashes.items()
    )
    with pytest.raises(ValueError, match="presupuesto"):
        experiment.validate_controls(source, {**spec, "episodes": 3}, {}, code)
    rows = json.loads((output / "cases.json").read_text())
    with pytest.raises(ValueError, match="pares completos"):
        experiment.summarize(rows[:-1], [42], [100])
    expanded = []
    for row in rows:
        expanded.extend(
            [{**row, "synthetic_mse": 1}, {**row, "case_seed": 101, "synthetic_mse": 9}]
        )
    assert experiment.summarize(expanded, [42], [100, 101])["summaries"][0][
        "final_rmse"
    ] == pytest.approx(np.sqrt(5))


def test_check_never_constructs_trainer_or_runs_experiment(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "controls"
    source.mkdir()
    (source / "summary.json").write_text("{}")
    env = SimpleNamespace(
        close=lambda: None, domain=SimpleNamespace(grid=None), estimator_config=None
    )

    def prepare(*args, check):
        assert check
        return env, env, {}

    monkeypatch.setattr(experiment, "prepare_experiment", prepare)
    monkeypatch.setattr(experiment, "check_control", lambda *a: None)
    monkeypatch.setattr(experiment, "validate_controls", lambda *a: ({}, {}, {}))
    monkeypatch.setattr(
        experiment, "Trainer", lambda *a, **k: pytest.fail("No crear entrenador")
    )
    monkeypatch.setattr(
        experiment, "run_experiment", lambda *a, **k: pytest.fail("No entrenar")
    )
    monkeypatch.setattr(
        sys, "argv", ["compare-history", "--controls", str(source), "--check"]
    )
    experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert result["new_training_episodes"] == 600 and result["reused_controls"] == 3
    assert not result["training_executed"]
