"""Mejor y final conservados, más un checkpoint rotativo para continuar."""

from pathlib import Path

import numpy as np
import torch

from .files import write_json_atomic
from .training import read_training_checkpoint


def latest_checkpoint(folder, config, protocol):
    """Elegir por episodio interno entre los tres estados completos de esa corrida."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("--reanudar requiere la carpeta de una corrida")
    candidates = []
    for name in ("checkpoint-last.pt", "checkpoint-best.pt", "checkpoint-final.pt"):
        path = folder / name
        if path.is_file():
            state = read_training_checkpoint(path, config, protocol)
            candidates.append((state["completed_episodes"], path))
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoint completo en {folder}")
    return max(candidates, key=lambda item: item[0])[1]


def validation_record(trainer, result, seeds):
    """El RMSE de validación es el único criterio de selección del mejor."""
    mse = result["synthetic_mse"]
    if not np.isfinite(mse) or mse < 0:
        raise ValueError("La validación debe producir un MSE finito y no negativo")
    rmse = float(np.sqrt(mse))
    episode = trainer.completed_episodes
    record = {
        "episode": episode,
        "rmse": rmse,
        "case_seeds": list(seeds),
        "metrics": result,
    }
    best = trainer.best_validation
    improved = best is None or rmse < best["rmse"]
    if improved:
        trainer.best_validation = {
            **record,
            "policy_file": "policy-best.pt",
            "checkpoint_file": "checkpoint-best.pt",
        }
    return record, improved


def save_checkpoint(trainer, output):
    """Rotar un solo estado de recuperación; no acumular checkpoints periódicos."""
    trainer.save_checkpoint(output / "checkpoint-last.pt")


def save_validation(trainer, output, record, improved):
    """El checkpoint del mejor es completo y atómico, incluso si falla su exportación."""
    if improved:
        trainer.save_checkpoint(output / "checkpoint-best.pt")
        trainer.export_policy(output / "policy-best.pt")
        write_json_atomic(
            output / "best-rmse.json", trainer.best_validation, overwrite=True
        )
    write_json_atomic(
        output / f"validation-episode-{record['episode']:04d}.json", record
    )


def save_final(trainer, output):
    trainer.save_checkpoint(output / "checkpoint-final.pt")
    trainer.export_policy(output / "policy-last.pt")
    save_checkpoint(trainer, output)


def inherit_best(trainer, resumed_from, output, seeds):
    """Exportar el mejor desde su estado completo; la política es derivada de él."""
    best = trainer.best_validation
    if best is None:
        return
    if resumed_from is None:
        raise ValueError("El mejor guardado requiere la carpeta del checkpoint")
    if best["episode"] > trainer.completed_episodes or best["case_seeds"] != list(
        seeds
    ):
        raise ValueError("El mejor guardado no corresponde al checkpoint o validación")
    source = Path(resumed_from).parent
    if Path(best["checkpoint_file"]).name != best["checkpoint_file"]:
        raise ValueError("El checkpoint del mejor debe pertenecer a la corrida")
    checkpoint = read_training_checkpoint(
        source / best["checkpoint_file"], trainer.config, trainer.protocol
    )
    if (
        checkpoint["completed_episodes"] != best["episode"]
        or checkpoint.get("best_validation", checkpoint.get("best_saved_validation"))
        != best
        or not np.isclose(
            best["rmse"], np.sqrt(best["metrics"]["synthetic_mse"]), rtol=0, atol=1e-12
        )
    ):
        raise ValueError("Los archivos del mejor no corresponden al registro guardado")
    if best["episode"] == trainer.completed_episodes and any(
        not torch.equal(value.cpu(), checkpoint["online"][key])
        for key, value in trainer.online.state_dict().items()
    ):
        raise ValueError("El mejor y el checkpoint reanudado tienen pesos distintos")
    best = {k: best[k] for k in ("episode", "rmse", "case_seeds", "metrics")}
    best.update(policy_file="policy-best.pt", checkpoint_file="checkpoint-best.pt")
    trainer.best_validation = best
    for key in (
        "best_saved_validation",
        "best_validation_variance",
        "best_validation_rmse",
        "best_validation_episode",
    ):
        checkpoint.pop(key, None)
    checkpoint["best_validation"] = best
    trainer._save(output / "checkpoint-best.pt", checkpoint, overwrite=False)
    policy = {
        k: checkpoint[k]
        for k in (
            "version",
            "training_config",
            "protocol",
            "online",
            "torch_version",
            "device",
            "completed_episodes",
            "physical_steps",
        )
    }
    trainer._save(
        output / "policy-best.pt", {**policy, "kind": "policy"}, overwrite=False
    )
    write_json_atomic(output / "best-rmse.json", best)


def validation_progress(trainer, mse):
    current = (
        f"RMSE validación={np.sqrt(mse):.6f} µg/m³"
        if mse is not None
        else "sin nueva validación"
    )
    saved = trainer.best_validation
    best = (
        f"mejor RMSE={saved['rmse']:.6f} µg/m³ en episodio {saved['episode']}"
        if saved is not None
        else "sin validación registrada"
    )
    return f"{current}; {best}"
