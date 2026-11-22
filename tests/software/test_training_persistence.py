"""Recuperar los pesos y el estado anunciados, no solo el número del récord."""

import json

import numpy as np
import pytest
import torch

from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.files import atomic_output
from lima_pm25_rl.study import run_training
from lima_pm25_rl.training import (
    Trainer,
    TrainingConfig,
    load_policy,
    read_training_checkpoint,
)
from lima_pm25_rl.training_artifacts import latest_checkpoint


def trainer_for(make_env):
    return Trainer(
        make_env(),
        TrainingConfig(
            device="cpu",
            hidden_size=8,
            history_length=2,
            capacity_steps=18,
            batch_size=2,
            learning_starts=1,
            target_update_interval=2,
        ),
    )


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_cadence_change_preserves_scientific_protocol_and_best_selection(
    make_env, tmp_path
):
    trainer = trainer_for(make_env)
    trainer.protocol = {
        "validation_every": 20,
        "validation_seeds": [41, 42],
        "gp_profile": "fixed",
    }
    target = {**trainer.protocol, "validation_every": 200}
    path = tmp_path / "state.pt"
    trainer.save_checkpoint(path)
    read_training_checkpoint(path, trainer.config, target)
    for changed in ({"validation_seeds": [41, 43]}, {"gp_profile": "adaptive"}):
        with pytest.raises(ValueError, match="protocolo"):
            read_training_checkpoint(path, trainer.config, {**target, **changed})
    trainer.best_validation = {"episode": 20, "rmse": 4.9}
    trainer.save_checkpoint(path)
    with pytest.raises(ValueError, match="otra cadencia"):
        read_training_checkpoint(path, trainer.config, target)


def scripted_validation(trainer, monkeypatch, mse_by_episode):
    original = trainer.validate

    def validate(env, seeds):
        result = original(env, seeds)
        # Secuencia artificial para forzar mejoras y deterioros de la selección.
        return {**result, "synthetic_mse": mse_by_episode[trainer.completed_episodes]}

    monkeypatch.setattr(trainer, "validate", validate)


def test_compact_nonfinal_best_survives_worsening_and_resume(
    make_env, tmp_path, monkeypatch
):
    trainer = trainer_for(make_env)
    first = tmp_path / "first"
    scripted_validation(
        trainer, monkeypatch, {0: 9.0, 1: 4.0, 2: 16.0, 3: 1.0, 4: 25.0}
    )
    run_training(
        trainer,
        episodes=4,
        output=first,
        checkpoint_every=2,
        validation_env=make_env(),
        validation_seeds=[42],
        validation_every=1,
    )
    best = json.loads((first / "best-rmse.json").read_text())
    assert best["episode"] == 3 and best["rmse"] == 1
    assert not list(first.glob("policy-episode-*.pt"))
    assert sorted(p.name for p in first.glob("checkpoint-*.pt")) == [
        "checkpoint-best.pt",
        "checkpoint-final.pt",
        "checkpoint-last.pt",
    ]
    best_policy = load(first / best["policy_file"])
    best_checkpoint = load(first / best["checkpoint_file"])
    assert best_checkpoint["best_validation"] == best
    assert (
        best_policy["completed_episodes"] == best_checkpoint["completed_episodes"] == 3
    )
    assert all(
        torch.equal(v, best_checkpoint["online"][k])
        for k, v in best_policy["online"].items()
    )
    assert load(first / "policy-last.pt")["completed_episodes"] == 4

    # Reanudar DESDE EL MEJOR reproduce el siguiente episodio y sus pesos.
    from_best = trainer_for(make_env)
    from_best.load_checkpoint(first / best["checkpoint_file"])
    from_best.train_episode()
    expected = load(first / "checkpoint-final.pt")
    for network in ("online", "target"):
        assert all(
            torch.equal(v, expected[network][k])
            for k, v in getattr(from_best, network).state_dict().items()
        )
    assert from_best.updates == expected["updates"]
    assert from_best.replay.rng.bit_generator.state == expected["replay"]["rng"]

    # Reanudar DESDE EL ÚLTIMO también conserva el mejor anterior, completo.
    resumed = trainer_for(make_env)
    resumed.load_checkpoint(first / "checkpoint-last.pt")
    scripted_validation(resumed, monkeypatch, {5: 9.0, 6: 16.0})
    second = tmp_path / "second"
    run_training(
        resumed,
        episodes=6,
        output=second,
        checkpoint_every=2,
        validation_env=make_env(),
        validation_seeds=[42],
        validation_every=1,
        resumed_from=str(first / "checkpoint-last.pt"),
    )
    assert json.loads((second / "best-rmse.json").read_text()) == best
    for key in ("policy_file", "checkpoint_file"):
        original, copied = load(first / best[key]), load(second / best[key])
        assert copied["completed_episodes"] == original["completed_episodes"]
        assert all(
            torch.equal(v, copied["online"][k]) for k, v in original["online"].items()
        )
    assert load(second / "checkpoint-last.pt")["completed_episodes"] == 6


def test_best_and_final_reproduce_from_their_actual_policy(make_env, tmp_path):
    trainer = trainer_for(make_env)
    run_training(
        trainer,
        episodes=3,
        output=tmp_path / "run",
        checkpoint_every=2,
        validation_env=make_env(),
        validation_seeds=[42, 43],
        validation_every=1,
    )
    folder = tmp_path / "run"
    records = [
        (
            json.loads((folder / "best-rmse.json").read_text()),
            "policy-best.pt",
            "checkpoint-best.pt",
        ),
        (
            json.loads((folder / "validation-episode-0003.json").read_text()),
            "policy-last.pt",
            "checkpoint-final.pt",
        ),
    ]
    for record, policy_file, checkpoint_file in records:
        policy, _ = load_policy(folder / policy_file, device="cpu")
        metrics = [
            evaluate_episode(make_env(), policy, seed=seed, comparison=True)
            for seed in record["case_seeds"]
        ]
        rmse = np.sqrt(np.mean([row["synthetic_mse"] for row in metrics]))
        assert rmse == pytest.approx(record["rmse"], rel=1e-10)
        assert load(folder / policy_file)["completed_episodes"] == record["episode"]
        checkpoint = load(folder / checkpoint_file)
        assert checkpoint["completed_episodes"] == record["episode"]
        assert {
            "optimizer",
            "target",
            "replay",
            "torch_rng",
            "controller_rng",
        } <= checkpoint.keys()


def test_missing_historical_best_does_not_block_saving_recoverable_best(
    make_env, tmp_path, monkeypatch
):
    trainer = trainer_for(make_env)
    trainer.train_episode()
    trainer.save_checkpoint(tmp_path / "old.pt")
    old = load(tmp_path / "old.pt")
    old.pop("best_validation")
    old["best_validation_rmse"], old["best_validation_episode"] = 0.01, 0
    torch.save(old, tmp_path / "old.pt")
    resumed = trainer_for(make_env)
    resumed.load_checkpoint(tmp_path / "old.pt")
    scripted_validation(resumed, monkeypatch, {1: 9.0, 2: 4.0})
    run_training(
        resumed,
        episodes=2,
        output=tmp_path / "new",
        checkpoint_every=2,
        validation_env=make_env(),
        validation_seeds=[42],
        validation_every=1,
        resumed_from=str(tmp_path / "old.pt"),
    )
    best = json.loads((tmp_path / "new/best-rmse.json").read_text())
    assert not hasattr(resumed, "best_validation_rmse")
    assert best["episode"] == 2 and best["rmse"] == 2.0
    assert (tmp_path / "new" / best["policy_file"]).is_file()
    assert (tmp_path / "new" / best["checkpoint_file"]).is_file()


def test_persistence_failure_does_not_announce_or_replace_best(
    make_env, tmp_path, monkeypatch, capsys
):
    trainer = trainer_for(make_env)
    scripted_validation(trainer, monkeypatch, {0: 9.0, 1: 1.0})
    original = trainer.save_checkpoint

    def fail(path, **kwargs):
        if trainer.completed_episodes == 1:
            raise OSError("disco lleno artificial")
        return original(path, **kwargs)

    monkeypatch.setattr(trainer, "save_checkpoint", fail)
    with pytest.raises(OSError, match="disco lleno"):
        run_training(
            trainer,
            episodes=2,
            output=tmp_path / "run",
            checkpoint_every=2,
            validation_env=make_env(),
            validation_seeds=[42],
            validation_every=1,
        )
    folder = tmp_path / "run"
    best = json.loads((folder / "best-rmse.json").read_text())
    assert best["episode"] == 0
    assert load(folder / best["checkpoint_file"])["completed_episodes"] == 0
    assert load(folder / "policy-best.pt")["completed_episodes"] == 0
    assert "en episodio 1;" not in capsys.readouterr().out


def test_partial_torch_save_keeps_previous_file_and_cleans_temporary(
    make_env, tmp_path, monkeypatch
):
    trainer = trainer_for(make_env)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    previous = path.read_bytes()

    def fail(state, file):
        file.write(b"partial")
        raise OSError("fallo de escritura artificial")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="escritura"):
        trainer.save_checkpoint(path)
    assert path.read_bytes() == previous
    assert not path.with_name(path.name + ".tmp").exists()


def test_episode_files_cannot_be_overwritten(tmp_path):
    path = tmp_path / "policy-episode-0001.pt"
    with atomic_output(path) as file:
        file.write(b"first")
    with pytest.raises(FileExistsError), atomic_output(path) as file:
        file.write(b"second")
    assert path.read_bytes() == b"first"
    assert not path.with_name(path.name + ".tmp").exists()


def test_resume_refuses_missing_saved_best_before_training(make_env, tmp_path):
    trainer = trainer_for(make_env)
    source = tmp_path / "first"
    run_training(
        trainer,
        episodes=1,
        output=source,
        validation_env=make_env(),
        validation_seeds=[42],
        validation_every=1,
    )
    resumed = trainer_for(make_env)
    resumed.load_checkpoint(source / "checkpoint-last.pt")
    best = resumed.best_validation
    (source / best["checkpoint_file"]).unlink()
    with pytest.raises(FileNotFoundError):
        run_training(
            resumed,
            episodes=2,
            output=tmp_path / "second",
            validation_env=make_env(),
            validation_seeds=[42],
            validation_every=1,
            resumed_from=str(source / "checkpoint-last.pt"),
        )
    assert resumed.completed_episodes == 1


def test_completed_run_returns_without_constructing_trainer(
    make_env, tmp_path, monkeypatch, capsys
):
    from argparse import Namespace
    from dataclasses import asdict

    from lima_pm25_rl import study

    trainer = trainer_for(make_env)
    trainer.train_episode()
    source = tmp_path / "source"
    trainer.save_checkpoint(source / "checkpoint-last.pt")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "kind": "controlled",
                "episodes": 1,
                "steps_per_episode": 3,
                "training": asdict(trainer.config),
            }
        )
    )
    monkeypatch.setattr(
        study, "prepare_experiment", lambda *a, **kw: (make_env(), None, {})
    )
    monkeypatch.setattr(
        study, "Trainer", lambda *a, **kw: pytest.fail("No construir entrenador")
    )
    before = set(tmp_path.iterdir())
    study.run(
        Namespace(
            config=config,
            output=None,
            resume=source,
        )
    )
    assert "ya alcanzó 1 episodios" in capsys.readouterr().out
    assert set(tmp_path.iterdir()) == before


def test_server_continuation_only_changes_budget_and_resume_settings():
    from lima_pm25_rl.files import PROJECT_ROOT

    first = json.loads((PROJECT_ROOT / "config/adaptativo.json").read_text())
    continuation = json.loads(
        (PROJECT_ROOT / "config/adaptativo-continuacion.json").read_text()
    )
    assert continuation["episodes"] == 30000
    assert continuation["minimum_resume_episode"] == 20000
    allowed = {
        "episodes",
        "minimum_resume_episode",
        "resume_checkpoint",
        "policy_snapshot_episodes",
    }
    assert {k: v for k, v in first.items() if k not in allowed} == {
        k: v for k, v in continuation.items() if k not in allowed
    }


def test_resume_uses_latest_complete_episode_not_modification_time(make_env, tmp_path):
    import os

    trainer = trainer_for(make_env)
    trainer.train_episode()
    trainer.save_checkpoint(tmp_path / "checkpoint-final.pt")
    trainer.train_episode()
    trainer.save_checkpoint(tmp_path / "checkpoint-last.pt")
    trainer.train_episode()
    trainer.save_checkpoint(tmp_path / "checkpoint-best.pt")
    os.utime(tmp_path / "checkpoint-best.pt", (1, 1))
    assert (
        latest_checkpoint(tmp_path, trainer.config, trainer.protocol).name
        == "checkpoint-best.pt"
    )


def test_full_best_recovers_even_if_policy_export_fails(
    make_env, tmp_path, monkeypatch
):
    trainer = trainer_for(make_env)
    scripted_validation(trainer, monkeypatch, {0: 9.0, 1: 1.0})
    original = trainer.export_policy

    def fail(path, **kwargs):
        if trainer.completed_episodes == 1:
            raise OSError("exportación interrumpida")
        return original(path, **kwargs)

    monkeypatch.setattr(trainer, "export_policy", fail)
    first = tmp_path / "first"
    with pytest.raises(OSError, match="interrumpida"):
        run_training(
            trainer,
            episodes=2,
            output=first,
            checkpoint_every=2,
            validation_env=make_env(),
            validation_seeds=[42],
            validation_every=1,
        )
    path = latest_checkpoint(first, trainer.config, trainer.protocol)
    assert path.name == "checkpoint-best.pt"
    restored = trainer_for(make_env)
    restored.load_checkpoint(path)
    assert restored.completed_episodes == 1
    scripted_validation(restored, monkeypatch, {2: 4.0})
    second = tmp_path / "second"
    run_training(
        restored,
        episodes=2,
        output=second,
        resumed_from=str(path),
        validation_env=make_env(),
        validation_seeds=[42],
        validation_every=1,
    )
    best = json.loads((second / "best-rmse.json").read_text())
    assert best["episode"] == 1
    assert load(second / best["policy_file"])["completed_episodes"] == 1
    assert load(second / best["checkpoint_file"])["completed_episodes"] == 1
