import json
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from lima_pm25_rl import architecture_experiment as experiment
from lima_pm25_rl.evaluate_cli import _compatible
from lima_pm25_rl.network import build_network
from lima_pm25_rl.training import Trainer, TrainingConfig


def test_window_model_uses_all_frames_and_auxiliaries_without_recurrence():
    torch.manual_seed(42)
    model = build_network("window", hidden_size=8, history_length=4)
    assert not any(isinstance(m, torch.nn.RNNBase) for m in model.modules())
    maps = torch.randn(1, 4, 7, 8, 8, requires_grad=True)
    aux = torch.randn(1, 4, 27, requires_grad=True)
    model(maps, aux).sum().backward()
    assert (maps.grad.abs().sum(dim=(2, 3, 4)) > 0).all()
    assert (aux.grad.abs().sum(dim=2) > 0).all()
    with pytest.raises(ValueError, match="ventana"):
        model(maps[:, -1:], aux[:, -1:])


def test_paired_configuration_changes_only_architecture_and_preserves_environment():
    spec = json.loads(experiment.DEFAULT_CONFIG.read_text())
    configs = list(experiment.pilot_configs(spec, experiment.DEFAULT_CONFIG.parent))
    assert len(configs) == 6
    assert sum(c["episodes"] for c in configs) == 1200
    for first, second in zip(configs[::2], configs[1::2], strict=True):
        assert first["training"].pop("architecture") == "bilg"
        assert second["training"].pop("architecture") == "window"
        assert first == second
    original = json.loads(
        (experiment.DEFAULT_CONFIG.parent / "training-server.json").read_text()
    )
    for key in (
        "field_variation",
        "noise_stddev_ug_m3",
        "concentration_priority",
        "distance_cost_per_km",
        "episode_configs",
        "validation_seeds",
    ):
        assert spec[key] == original[key]


def test_checkpoint_architecture_is_checked_before_loading_weights(make_env, tmp_path):
    config = TrainingConfig(device="cpu", hidden_size=8, history_length=2)
    trainer = Trainer(make_env(), config)
    path = tmp_path / "state.pt"
    trainer.save_checkpoint(path)
    other = Trainer(make_env(), replace(config, architecture="window"))
    with pytest.raises(ValueError, match="hiperparámetros"):
        other.load_checkpoint(path)
    trainer.export_policy(tmp_path / "policy.pt")
    state = torch.load(tmp_path / "policy.pt", weights_only=True)
    assert not _compatible(state, other.config, {})
    assert _compatible(state, config, {})


def test_tiny_full_pilot_pairs_seeds_saves_both_snapshots_and_pools_mse(
    make_env, monkeypatch, tmp_path
):
    spec = json.loads(experiment.DEFAULT_CONFIG.read_text())
    spec.update(
        episodes=1, steps_per_episode=3, paired_seeds=[42], validation_seeds=[100]
    )
    spec["training"].update(
        device="cpu",
        hidden_size=8,
        history_length=2,
        capacity_steps=6,
        batch_size=2,
        learning_starts=1,
    )
    monkeypatch.setattr(
        experiment, "prepare_experiment", lambda *a, **k: (make_env(), make_env(), {})
    )
    output = tmp_path / "pilot"
    summary = experiment.run_pilot(spec, directory=tmp_path, output=output, maps=False)
    assert len(summary["summaries"]) == 4
    assert len(summary["training_runs"]) == 2
    assert [r["episodes"] for r in summary["training_runs"]] == [1, 1]
    assert len(summary["paired_last"]) == 1
    for name in experiment.ARCHITECTURES:
        assert (output / f"{name}-42/policy-initial.pt").is_file()
        assert (output / f"{name}-42/evaluation-last/policy.json").is_file()
    rows = json.loads((output / "cases.json").read_text())
    changed = deepcopy(rows)
    changed[-1]["signatures"]["field_sha256"] = "different"
    with pytest.raises(ValueError, match="campo"):
        experiment.summarize_results(changed)
    with pytest.raises(ValueError, match="completos"):
        experiment.summarize_results(rows[:-1])
    with pytest.raises(FileExistsError):
        experiment.run_pilot(spec, directory=tmp_path, output=output, maps=False)
    expanded = []
    for row in rows:
        for seed, mse in ((42, 1), (43, 9)):
            expanded.append({**row, "seed": seed, "synthetic_mse": mse})
    result = experiment.summarize_results(expanded)
    assert result["summaries"][0]["synthetic_rmse_ug_m3"] == pytest.approx(np.sqrt(5))


def test_check_composes_only_without_creating_a_trainer(monkeypatch, capsys):
    class Environment:
        def close(self):
            pass

    def prepare(spec, directory, *, check):
        assert check
        return Environment(), Environment(), {}

    monkeypatch.setattr(experiment, "prepare_experiment", prepare)
    monkeypatch.setattr(
        experiment, "Trainer", lambda *a, **k: pytest.fail("No construir entrenador")
    )
    monkeypatch.setattr(
        experiment, "run_pilot", lambda *a, **k: pytest.fail("No iniciar el piloto")
    )
    monkeypatch.setattr(sys, "argv", ["compare-architectures", "--check"])
    experiment.main()
    result = json.loads(capsys.readouterr().out)
    assert result["training_executed"] is False
    assert result["total_training_episodes"] == 1200
