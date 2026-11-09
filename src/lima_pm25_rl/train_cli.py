"""Entrenamiento explícito con configuración JSON, CSV y checkpoints entre episodios."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lima_pm25.domain import PM25SourceRole

from .env import LimaPM25Env
from .episodes import Episode, GeneratedEpisodes
from .pilot import controlled_environment, controlled_oracle, controlled_score
from .resources import load_study_resources
from .training import Trainer, TrainingConfig


def _utc(value: str) -> datetime:
    date = datetime.fromisoformat(value)
    if date.utcoffset() != timedelta(0):
        raise ValueError("Las fechas del experimento deben incluir UTC (+00:00)")
    return date


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare_experiment(spec: dict[str, Any], directory: Path):
    """Los paths son relativos al JSON. No toca particiones de replay final."""
    kind = spec["kind"]
    if kind not in {"controlled", "demonstration", "generated"}:
        raise ValueError("kind debe ser controlled, demonstration o generated")
    steps = spec["steps_per_episode"]
    if type(steps) is not int or steps < 1:
        raise ValueError("steps_per_episode debe ser un entero positivo")
    if kind == "controlled":
        if steps != 1:
            raise ValueError("El piloto controlado tiene exactamente un tick")
        env = controlled_environment()
        return (
            env,
            None,
            {
                "kind": kind,
                "grid_shape": list(env.domain.grid.shape),
                "gp_profile_sha256": env.estimator_config.profile_sha256,
                "initialization": "zero_advantage_for_controlled_software_check",
            },
        )

    root = (directory / spec["core_root"]).resolve()
    domain, event_template, gp = load_study_resources(root, member="event")

    def selected_template(member):
        if member == "event":
            return event_template
        if member != "control":
            raise ValueError("Cada member debe ser control o event")
        identity = event_template.scenario_identity
        return replace(
            event_template,
            sources=tuple(
                source
                for source in event_template.sources
                if source.role is not PM25SourceRole.COUNTERFACTUAL_EVENT
            ),
            scenario_identity=replace(
                identity, member_id="control", is_control=True, changed_factors=()
            )
            if identity is not None
            else None,
        )

    template = selected_template(spec.get("member", "event"))
    if steps * 300 > template.weather.duration_s:
        raise ValueError("La plantilla conservada cubre como máximo 36 ticks")
    duration = steps * 300
    protocol: dict[str, Any] = {
        "kind": kind,
        "steps_per_episode": steps,
        "noise_stddev_ug_m3": spec.get("noise_stddev_ug_m3", 0.0),
        "member": spec.get("member", "event"),
        "grid_shape": list(domain.grid.shape),
        "gp_profile_sha256": gp.profile_sha256,
        "gp_calibration_sha256": gp.calibration_artifact_sha256,
        "graph_sha256": _sha256(
            root / domain.config.cache_directory / "road_graph.graphml"
        ),
        "grid_sha256": _sha256(root / domain.config.cache_directory / "grid.npz"),
    }

    def demonstration(seed):
        return Episode(replace(template, config=replace(template.config, seed=seed)))

    training_factory = demonstration
    validation_factory = None
    if kind == "generated":
        from lima_pm25.simulation import (
            load_episode_generation_config,
            load_historical_weather_csv,
        )

        fit_end = _utc(spec["gp_fit_end_utc"])
        final_start = _utc(spec["reserved_final_start_utc"])
        candidates = spec["episode_configs"]
        if not candidates:
            raise ValueError(
                "El experimento necesita configuraciones explícitas de episodios"
            )
        weights = np.asarray([item["weight"] for item in candidates], dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(
                "Los pesos de escenarios deben ser finitos, no negativos y sumar más de cero"
            )
        weather = load_historical_weather_csv(
            root / "data/var_meteorologicas_lima_2025_2026_imputed.csv",
            station_id="CAMPO_DE_MARTE",
        )
        configs = [
            load_episode_generation_config(directory / item["path"])
            for item in candidates
        ]
        periods = {
            (
                item.training_start_utc,
                item.training_end_utc,
                item.evaluation_start_utc,
                item.evaluation_end_utc,
            )
            for item in configs
        }
        if len(periods) != 1:
            raise ValueError(
                "Los escenarios de la mezcla deben compartir las particiones temporales"
            )
        if any(
            final_start < max(config.evaluation_end_utc, fit_end) for config in configs
        ):
            raise ValueError(
                "El replay final reservado se solapa con calibración o validación"
            )
        training = [
            GeneratedEpisodes(
                selected_template(item.get("member", spec.get("member", "event"))),
                weather,
                config,
                fit_end,
                partition="training",
            )
            for config, item in zip(configs, candidates, strict=True)
        ]
        validation = [replace(factory, partition="evaluation") for factory in training]
        protocol.update(
            gp_fit_end_utc=fit_end.isoformat(),
            policy_fit_end_utc=max(
                config.evaluation_end_utc for config in configs
            ).isoformat(),
            reserved_final_start_utc=final_start.isoformat(),
            weather_sha256=weather.resource_sha256,
            scenarios=[
                {
                    "weight": float(weight),
                    "member": item.get("member", spec.get("member", "event")),
                    "configuration": json.loads((directory / item["path"]).read_text()),
                }
                for weight, item in zip(weights, candidates, strict=True)
            ],
            validation_seeds=spec["validation_seeds"],
            validation_every=spec.get("validation_every", 10),
        )

        def choose(seed, factories):
            rng = np.random.default_rng([seed, 7])
            return factories[
                int(rng.choice(len(factories), p=weights / weights.sum()))
            ](seed)

        training_factory = lambda seed: choose(seed, training)
        validation_factory = lambda seed: choose(seed, validation)

    def environment(factory):
        def shortened(seed):
            episode = factory(seed)
            config = replace(
                episode.request.config,
                duration_s=duration,
                concentration_output_times_s=tuple(
                    t
                    for t in episode.request.config.concentration_output_times_s
                    if t <= duration
                ),
            )
            return Episode(replace(episode.request, config=config))

        return LimaPM25Env(
            domain=domain,
            estimator_config=gp,
            episode_factory=shortened,
            mobile_noise_stddev_ug_m3=protocol["noise_stddev_ug_m3"],
        )

    return (
        environment(training_factory),
        environment(validation_factory) if validation_factory else None,
        protocol,
    )


def run_training(
    trainer: Trainer,
    *,
    episodes: int,
    output: Path,
    checkpoint_every: int = 10,
    validation_env: LimaPM25Env | None = None,
    validation_seeds: list[int] | None = None,
    validation_every: int = 10,
    resumed_from: str | None = None,
) -> dict[str, Any]:
    if type(episodes) is not int or episodes <= trainer.completed_episodes:
        raise ValueError(
            "episodes es el total deseado y debe superar los episodios ya completados"
        )
    if any(
        type(value) is not int or value < 1
        for value in (checkpoint_every, validation_every)
    ):
        raise ValueError(
            "Las frecuencias de checkpoint y validación deben ser positivas"
        )
    seeds = validation_seeds or []
    if validation_env is not None and (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "La validación requiere semillas distintas, explícitas y no negativas"
        )
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "training": asdict(trainer.config),
        "protocol": trainer.protocol,
        "target_episodes": episodes,
        "start_episode": trainer.completed_episodes,
        "checkpoint_every": checkpoint_every,
        "validation_every": validation_every,
        "validation_seeds": seeds,
        "resumed_from": resumed_from,
    }
    (output / "config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n"
    )
    initial_score = None
    if trainer.protocol.get("kind") == "controlled":
        initial_score = controlled_score(trainer.online, trainer.config.history_length)
    with (output / "metrics.csv").open("x", newline="") as file:
        writer = None
        while trainer.completed_episodes < episodes:
            row = trainer.train_episode()
            row.update(validation_reward=None, validation_variance=None)
            if validation_env is not None and (
                trainer.completed_episodes % validation_every == 0
                or trainer.completed_episodes == episodes
            ):
                result = trainer.validate(validation_env, seeds)
                row.update(
                    validation_reward=result["reward"],
                    validation_variance=result["final_integrated_variance"],
                )
                if (
                    result["final_integrated_variance"]
                    < trainer.best_validation_variance
                ):
                    trainer.best_validation_variance = result[
                        "final_integrated_variance"
                    ]
                    trainer.export_policy(output / "policy-best.pt")
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            if (
                trainer.completed_episodes % checkpoint_every == 0
                or trainer.completed_episodes == episodes
            ):
                trainer.save_checkpoint(output / "checkpoint-last.pt")
                print(
                    f"Episodio {row['episode']}/{episodes}: recompensa={row['reward']:.5f}, pérdida={row['loss']}, actualizaciones={row['optimizer_updates']}",
                    flush=True,
                )
    trainer.export_policy(output / "policy-last.pt")
    if initial_score is not None:
        final_score = controlled_score(trainer.online, trainer.config.history_length)
        oracle = controlled_oracle()
        best_action = max(oracle, key=oracle.__getitem__)
        report = {
            "purpose": "Comprobación artificial de software; no es un resultado de la tesis",
            "start_episode": configuration["start_episode"],
            "completed_episodes": trainer.completed_episodes,
            "before": initial_score,
            "after": final_score,
            "oracle_rewards": oracle,
            "random_expected_reward": float(np.mean(list(oracle.values()))),
            "learned_optimal_action": final_score["action"] == best_action,
            "optimizer_updates": trainer.updates,
            "device": str(trainer.device),
        }
        (output / "pilot.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenador mínimo BiLG-D3QN. No inicia replay histórico final."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Configuración JSON explícita"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directorio nuevo, también al reanudar",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint completo para continuar al siguiente episodio",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Total de episodios deseado, no episodios adicionales",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    parameters = dict(spec.get("training", {}))
    if args.device:
        parameters["device"] = args.device
    config = TrainingConfig(**parameters)
    env, validation_env, protocol = prepare_experiment(
        spec, args.config.resolve().parent
    )
    trainer = Trainer(env, config, protocol=protocol)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif spec["kind"] == "controlled":
        # Empate inicial conocido: el argmax elige este. Solo para este piloto.
        with torch.no_grad():
            for parameter in trainer.online.advantage[-1].parameters():
                parameter.zero_()
        trainer.target.load_state_dict(trainer.online.state_dict())
    run_training(
        trainer,
        episodes=args.episodes if args.episodes is not None else spec["episodes"],
        output=args.output,
        checkpoint_every=spec.get("checkpoint_every", 10),
        validation_env=validation_env,
        validation_seeds=spec.get("validation_seeds"),
        validation_every=spec.get("validation_every", 10),
        resumed_from=str(args.resume.resolve()) if args.resume else None,
    )
    env.close()
    if validation_env is not None:
        validation_env.close()
