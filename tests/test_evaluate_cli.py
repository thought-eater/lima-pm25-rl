import json
import os
import sys
from dataclasses import asdict, replace

import pytest
import torch

import lima_pm25_rl.evaluate_cli as entry
from lima_pm25_rl.training import Trainer, TrainingConfig


@pytest.fixture
def project(tmp_path, monkeypatch, make_env, domain):
    config = TrainingConfig(
        device="cpu", hidden_size=8, history_length=2, capacity_steps=8, batch_size=2
    )
    protocol = {"kind": "generated", "grid_shape": list(domain.grid.shape)}
    spec = {
        "kind": "generated",
        "training": asdict(config),
        "validation_seeds": [42],
        "evaluation_policy_seeds": [0],
    }
    path = tmp_path / "training-server.json"
    path.write_text(json.dumps(spec))
    Trainer(make_env(), config, protocol=protocol).export_policy(
        tmp_path / "policy-best.pt"
    )
    monkeypatch.setattr(entry, "DEFAULT_CONFIG", path)
    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(entry, "CACHE_DIRECTORY", tmp_path / "cache")
    monkeypatch.setattr(
        entry, "prepare_experiment", lambda *a, **kw: (make_env(), make_env(), protocol)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["lima-pm25-evaluate"])
    return config, protocol


def test_default_evaluation_needs_no_flags_and_never_constructs_a_trainer(
    project, monkeypatch, capsys
):
    monkeypatch.setattr(
        Trainer,
        "__init__",
        lambda *a, **kw: pytest.fail("No se debe construir un entrenador"),
    )
    entry.main()
    [configuration] = list((entry.CACHE_DIRECTORY / "evaluation").glob("*/config.json"))
    config = json.loads(configuration.read_text())
    assert config["partition"] == "validation"
    assert config["checkpoint_path"] == str(entry.PROJECT_ROOT / "policy-best.pt")
    assert config["case_seeds"] == [42] and config["policy_seeds"] == [0]
    assert len(config["checkpoint_sha256"]) == 64
    assert (configuration.parent / "summary.csv").is_file()
    assert "Salida:" in capsys.readouterr().out


def test_latest_policy_must_match_protocol_and_training_parameters(project):
    config, protocol = project
    correct = entry.PROJECT_ROOT / "policy-best.pt"
    os.utime(correct, (1, 1))
    path = entry.CACHE_DIRECTORY / "other/policy-best.pt"
    path.parent.mkdir(parents=True)
    state = torch.load(correct, weights_only=True)
    state["training_config"] = asdict(replace(config, seed=99))
    torch.save(state, path)
    os.utime(path, (2, 2))
    assert entry.find_policy(config, protocol) == correct
    state["training_config"] = asdict(replace(config, device="cuda", cpu_threads=2))
    torch.save(state, path)
    os.utime(path, (3, 3))
    assert entry.find_policy(config, protocol) == path
    assert not entry._compatible(state, config, {"kind": "other"})


@pytest.mark.parametrize("kind", ["controlled", "demonstration", "historical_replay"])
def test_cli_refuses_non_generated_experiments(project, kind):
    entry.DEFAULT_CONFIG.write_text(json.dumps({"kind": kind}))
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert not entry.CACHE_DIRECTORY.exists()
