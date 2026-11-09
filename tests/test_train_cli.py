import csv
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import lima_pm25_rl.train_cli as entry
from lima_pm25_rl.training import TrainingConfig


@pytest.fixture
def project(tmp_path, monkeypatch):
    config = tmp_path / "config/training-server.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "kind": "controlled",
                "episodes": 1,
                "steps_per_episode": 1,
                "checkpoint_every": 1,
                "training": {
                    "device": "cpu",
                    "hidden_size": 8,
                    "history_length": 2,
                    "capacity_steps": 8,
                    "batch_size": 2,
                    "learning_starts": 1,
                },
            }
        )
    )
    monkeypatch.setattr(entry, "DEFAULT_CONFIG", config)
    monkeypatch.setattr(entry, "CACHE_DIRECTORY", tmp_path / "cache")
    monkeypatch.setattr(sys, "argv", ["lima-pm25-train"])
    return config


def runs() -> list[Path]:
    return sorted((entry.CACHE_DIRECTORY / "training").glob("*/config.json"))


def set_budget(config: Path, episodes: int) -> None:
    spec = json.loads(config.read_text())
    spec["episodes"] = episodes
    config.write_text(json.dumps(spec))


def test_no_flags_uses_default_config_and_prints_automatic_output(
    project, tmp_path, monkeypatch, capsys
):
    # La configuración y la salida pertenecen al proyecto, no al cwd accidental.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    entry.main()
    [metadata_path] = runs()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["target_episodes"] == 1 and metadata["start_episode"] == 0
    assert metadata["resumed_from"] is None
    assert (metadata_path.parent / "checkpoint-last.pt").is_file()
    assert (metadata_path.parent / "policy-last.pt").is_file()
    assert f"Salida: {metadata_path.parent}" in capsys.readouterr().out


def test_starting_again_creates_a_new_run_without_resuming_or_overwriting(project):
    entry.main()
    [first] = runs()
    checkpoint = first.parent / "checkpoint-last.pt"
    saved = checkpoint.read_bytes()
    entry.main()
    assert len(runs()) == 2
    assert checkpoint.read_bytes() == saved
    assert all(json.loads(path.read_text())["start_episode"] == 0 for path in runs())


def test_bare_resume_continues_in_a_new_directory(project, monkeypatch, capsys):
    entry.main()
    [first] = runs()
    checkpoint = first.parent / "checkpoint-last.pt"
    previous_metrics = (first.parent / "metrics.csv").read_bytes()
    set_budget(project, 2)
    monkeypatch.setattr(sys, "argv", ["lima-pm25-train", "--resume"])
    entry.main()
    [continued] = [path for path in runs() if path != first]
    metadata = json.loads(continued.read_text())
    assert metadata["resumed_from"] == str(checkpoint)
    assert metadata["start_episode"] == 1 and metadata["target_episodes"] == 2
    with (continued.parent / "metrics.csv").open() as file:
        assert [row["episode"] for row in csv.DictReader(file)] == ["2"]
    assert (first.parent / "metrics.csv").read_bytes() == previous_metrics
    assert f"Reanudando: {checkpoint}" in capsys.readouterr().out


def test_explicit_resume_and_output_remain_available(project, tmp_path, monkeypatch):
    original = tmp_path / "custom-output"
    monkeypatch.setattr(sys, "argv", ["lima-pm25-train", "--output", str(original)])
    entry.main()
    checkpoint = original / "checkpoint-last.pt"
    set_budget(project, 2)
    destination = tmp_path / "custom-continuation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lima-pm25-train",
            "--resume",
            str(checkpoint),
            "--output",
            str(destination),
        ],
    )
    monkeypatch.setattr(
        entry, "_latest_checkpoint", lambda *args: pytest.fail("La ruta ya está indicada")
    )
    entry.main()
    metadata = json.loads((destination / "config.json").read_text())
    assert metadata["resumed_from"] == str(checkpoint)
    assert metadata["start_episode"] == 1
    assert (original / "metrics.csv").is_file()


def test_missing_automatic_resume_fails_without_starting_a_fresh_run(
    project, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["lima-pm25-train", "--resume"])
    monkeypatch.setattr(
        entry, "Trainer", lambda *a, **kw: pytest.fail("No debe empezar desde cero")
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert "No hay un checkpoint compatible" in capsys.readouterr().err
    assert not entry.CACHE_DIRECTORY.exists()


def test_resume_does_not_increase_the_configured_budget(project, monkeypatch, capsys):
    entry.main()
    [original] = runs()
    checkpoint = (original.parent / "checkpoint-last.pt").read_bytes()
    monkeypatch.setattr(sys, "argv", ["lima-pm25-train", "--resume"])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert "Aumenta 'episodes'" in capsys.readouterr().err
    assert runs() == [original]
    assert (original.parent / "checkpoint-last.pt").read_bytes() == checkpoint


def test_check_also_uses_the_default_config_without_creating_output(
    project, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["lima-pm25-train", "--check"])
    monkeypatch.setattr(
        entry, "Trainer", lambda *a, **kw: pytest.fail("No debe construir el entrenador")
    )
    entry.main()
    result = json.loads(capsys.readouterr().out)
    assert result["training_executed"] is False
    assert not entry.CACHE_DIRECTORY.exists()


def write_candidate(directory, config, protocol, modified):
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint-last.pt"
    checkpoint.write_bytes(b"No debe cargarse el buffer para buscar candidatos")
    (directory / "config.json").write_text(
        json.dumps({"training": asdict(config), "protocol": protocol})
    )
    os.utime(checkpoint, (modified, modified))
    return checkpoint


def test_automatic_resume_selects_the_latest_compatible_checkpoint(project):
    config = TrainingConfig(device="cpu", hidden_size=8)
    protocol = {"experiment": "server", "grid_shape": (2, 3)}
    write_candidate(entry.CACHE_DIRECTORY / "z-older", config, protocol, 1)
    latest = write_candidate(
        entry.CACHE_DIRECTORY / "training/a-newer",
        replace(config, device="cuda", cpu_threads=4),
        protocol,
        2,
    )
    wrong_protocol = write_candidate(
        entry.CACHE_DIRECTORY / "another-experiment",
        config,
        {"experiment": "other"},
        3,
    )
    wrong_config = write_candidate(
        entry.CACHE_DIRECTORY / "another-seed",
        replace(config, seed=99),
        protocol,
        4,
    )
    broken = write_candidate(entry.CACHE_DIRECTORY / "broken", config, protocol, 5)
    broken.with_name("config.json").write_text("{incomplete")
    assert entry._latest_checkpoint(config, protocol) == latest
    assert wrong_protocol.is_file() and wrong_config.is_file() and broken.is_file()


@pytest.mark.parametrize("metadata", [None, [], {"training": []}])
def test_automatic_resume_ignores_missing_or_invalid_metadata(project, metadata):
    config = TrainingConfig(device="cpu")
    path = write_candidate(entry.CACHE_DIRECTORY / "run", config, {}, 1)
    if metadata is None:
        # Un checkpoint suelto sigue siendo utilizable mediante ruta explícita.
        path.with_name("config.json").unlink()
    else:
        path.with_name("config.json").write_text(json.dumps(metadata))
    with pytest.raises(FileNotFoundError, match="compatible"):
        entry._latest_checkpoint(config, {})
