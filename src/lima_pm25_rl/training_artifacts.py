"""Políticas por episodio, checkpoints retenidos y mejor RMSE recuperable."""

import json
from pathlib import Path

import numpy as np
import torch

from .files import copy_atomic, write_json_atomic
from .training import read_training_checkpoint


def policy_name(episode):
    return f"policy-episode-{episode:04d}.pt"


def checkpoint_name(episode):
    return f"checkpoint-episode-{episode:04d}.pt"


def validation_record(trainer, result, seeds):
    """Separar récord histórico de un mejor que sí tiene pesos y estado completo."""
    mse = result["synthetic_mse"]
    if not np.isfinite(mse) or mse < 0:
        raise ValueError("La validación debe producir un MSE finito y no negativo")
    rmse = float(np.sqrt(mse))
    episode = trainer.completed_episodes
    if trainer.best_validation_rmse is None or rmse < trainer.best_validation_rmse:
        trainer.best_validation_rmse = rmse
        trainer.best_validation_episode = episode
    record = {
        "episode": episode,
        "rmse": rmse,
        "case_seeds": list(seeds),
        "metrics": result,
        "policy_file": policy_name(episode),
        "checkpoint_file": checkpoint_name(episode),
        "validation_file": f"validation-episode-{episode:04d}.json",
    }
    best = trainer.best_saved_validation
    improved = best is None or rmse < best["rmse"]
    if improved:
        trainer.best_saved_validation = record
    return record, improved


def save_checkpoint(trainer, output):
    """El último es una copia de acceso rápido, nunca el único checkpoint."""
    path = output / checkpoint_name(trainer.completed_episodes)
    trainer.save_checkpoint(path, overwrite=False)
    copy_atomic(path, output / "checkpoint-last.pt", overwrite=True)


def save_validation(trainer, output, record, improved):
    """Publicar el mejor solo después de conservar política, checkpoint y métricas."""
    if not (output / record["policy_file"]).is_file():
        raise FileNotFoundError("Faltan los pesos del episodio validado")
    trainer.save_checkpoint(output / record["checkpoint_file"], overwrite=False)
    write_json_atomic(output / record["validation_file"], record)
    if improved:
        # Este único registro atómico apunta al par de archivos inmutables.
        write_json_atomic(output / "best-rmse.json", record, overwrite=True)
    copy_atomic(
        output / record["checkpoint_file"],
        output / "checkpoint-last.pt",
        overwrite=True,
    )


def inherit_best(trainer, resumed_from, output, seeds):
    """Conservar el mejor del checkpoint usado, no el de una ejecución posterior."""
    best = trainer.best_saved_validation
    if best is None:
        return
    if resumed_from is None:
        raise ValueError("El mejor recuperable requiere la carpeta del checkpoint")
    if best["episode"] > trainer.completed_episodes or best["case_seeds"] != list(
        seeds
    ):
        raise ValueError("El mejor guardado no corresponde al checkpoint o validación")
    source = Path(resumed_from).parent
    if (
        best["policy_file"] != policy_name(best["episode"])
        or best["checkpoint_file"] != checkpoint_name(best["episode"])
        or best["validation_file"] != f"validation-episode-{best['episode']:04d}.json"
    ):
        raise ValueError("Las rutas del mejor deben identificar su episodio")
    policy = torch.load(
        source / best["policy_file"], map_location="cpu", weights_only=True
    )
    checkpoint = read_training_checkpoint(
        source / best["checkpoint_file"], trainer.config, trainer.protocol
    )
    if (
        policy.get("kind") != "policy"
        or checkpoint.get("kind") != "training"
        or any(s["completed_episodes"] != best["episode"] for s in (policy, checkpoint))
        or any(s["protocol"] != trainer.protocol for s in (policy, checkpoint))
        or checkpoint.get("best_saved_validation") != best
        or policy["training_config"] != checkpoint["training_config"]
        or not np.isclose(
            best["rmse"], np.sqrt(best["metrics"]["synthetic_mse"]), rtol=0, atol=1e-12
        )
        or policy["online"].keys() != checkpoint["online"].keys()
        or any(
            not torch.equal(value, checkpoint["online"][key])
            for key, value in policy["online"].items()
        )
    ):
        raise ValueError("Los archivos del mejor no corresponden al registro guardado")
    if best["episode"] == trainer.completed_episodes and any(
        not torch.equal(value.cpu(), policy["online"][key])
        for key, value in trainer.online.state_dict().items()
    ):
        raise ValueError("El mejor y el checkpoint reanudado tienen pesos distintos")
    if json.loads((source / best["validation_file"]).read_text()) != best:
        raise ValueError("Las métricas del mejor no coinciden con su checkpoint")
    for key in ("policy_file", "checkpoint_file", "validation_file"):
        copy_atomic(source / best[key], output / best[key])
    write_json_atomic(output / "best-rmse.json", best)


def validation_progress(trainer, mse):
    current = (
        f"RMSE validación={np.sqrt(mse):.6f} µg/m³"
        if mse is not None
        else "sin nueva validación"
    )
    saved = trainer.best_saved_validation
    best = (
        f"mejor recuperable={saved['rmse']:.6f} µg/m³ en episodio {saved['episode']}; "
        f"pesos={saved['policy_file']}; checkpoint={saved['checkpoint_file']}"
        if saved is not None
        else "mejor recuperable todavía no guardado"
    )
    historical = ""
    if trainer.best_validation_rmse is not None and (
        saved is None or trainer.best_validation_rmse < saved["rmse"]
    ):
        historical = (
            f"; récord histórico={trainer.best_validation_rmse:.6f} "
            f"en episodio {trainer.best_validation_episode} "
            "(sin pesos recuperables registrados)"
        )
    return f"{current}; {best}{historical}"
