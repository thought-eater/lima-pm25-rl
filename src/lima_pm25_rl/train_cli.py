"""Entrenamiento explícito con configuración JSON, CSV y checkpoints entre episodios."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lima_pm25.domain import PM25SourceRole

from .env import LimaPM25Env
from .episodes import Episode, GeneratedEpisodes
from .pilot import controlled_environment, controlled_oracle, controlled_score
from .resources import load_study_resources
from .scenario_fields import FieldVariation
from .training import Trainer, TrainingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/training-server.json"
CACHE_DIRECTORY = PROJECT_ROOT / "cache"


def _utc(value: str) -> datetime:
    date = datetime.fromisoformat(value)
    if date.utcoffset() != timedelta(0):
        raise ValueError("Las fechas del experimento deben incluir UTC (+00:00)")
    return date


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare_experiment(spec: dict[str, Any], directory: Path, *, check: bool = False):
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
    field_variation = (
        FieldVariation(**spec["field_variation"]) if "field_variation" in spec else None
    )
    if field_variation is not None and kind != "generated":
        raise ValueError(
            "La variación del campo se configura solo para episodios generados"
        )
    protocol: dict[str, Any] = {
        "kind": kind,
        "steps_per_episode": steps,
        "noise_stddev_ug_m3": spec.get("noise_stddev_ug_m3", 0.0),
        "concentration_priority": spec.get("concentration_priority", 0.0),
        "distance_cost_per_km": spec.get("distance_cost_per_km", 0.0),
        "planning_schema": "global_vehicle_direction_v2",
        "member": spec.get("member", "event"),
        "grid_shape": list(domain.grid.shape),
        "gp_profile_sha256": gp.profile_sha256,
        "gp_calibration_sha256": gp.calibration_artifact_sha256,
        "graph_sha256": _sha256(
            root / domain.config.cache_directory / "road_graph.graphml"
        ),
        "grid_sha256": _sha256(root / domain.config.cache_directory / "grid.npz"),
    }
    if field_variation is not None:
        protocol["field_variation"] = asdict(field_variation)

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
        final_end = (
            _utc(spec["reserved_final_end_utc"])
            if "reserved_final_end_utc" in spec
            else None
        )
        if final_end is not None and final_end <= final_start:
            raise ValueError("El periodo final reservado debe tener duración positiva")
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
        if "gp_partition_manifest" in spec:
            manifest_path = root / spec["gp_partition_manifest"]
            manifest = json.loads(manifest_path.read_text())
            if _utc(manifest["calibration_end_utc"]) != fit_end:
                raise ValueError("gp_fit_end_utc no coincide con el manifiesto del GP")
            if manifest["weather_resource_sha256"] != weather.resource_sha256:
                raise ValueError("La meteorología no coincide con el manifiesto del GP")
            if final_start < _utc(manifest["replay_start_utc"]) or (
                final_end is not None and final_end > _utc(manifest["replay_end_utc"])
            ):
                raise ValueError("El periodo final excede la reserva histórica del GP")
            protocol["gp_partition_sha256"] = _sha256(manifest_path)
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
        if check:
            from lima_pm25.simulation import EpisodeWeatherMode

            # Componer no avanza el campo ni inicializa el GP o la red. Probar
            # cada combinación, no solo la que salga en un sorteo de la mezcla.
            for factory in (*training, *validation):
                for mode, weight in zip(
                    EpisodeWeatherMode, factory.config.weather_weights, strict=True
                ):
                    if weight > 0:
                        episode = replace(factory, weather_mode=mode)(0)
                        if field_variation is not None:
                            field_variation.apply(episode.request, 0)
        if final_end is not None:
            protocol["reserved_final_end_utc"] = final_end.isoformat()
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
            if field_variation is not None:
                episode = replace(
                    episode, request=field_variation.apply(episode.request, seed)
                )
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
            concentration_priority=protocol["concentration_priority"],
            distance_cost_per_km=protocol["distance_cost_per_km"],
        )

    return (
        environment(training_factory),
        environment(validation_factory) if validation_factory else None,
        protocol,
    )


def _validate_run_settings(
    *,
    episodes: int,
    completed_episodes: int,
    checkpoint_every: int,
    validation_every: int,
    validation: bool,
    validation_seeds: list[int] | None,
) -> list[int]:
    if type(episodes) is not int or episodes <= completed_episodes:
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
    if validation and (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "La validación requiere semillas distintas, explícitas y no negativas"
        )
    return seeds


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
    seeds = _validate_run_settings(
        episodes=episodes,
        completed_episodes=trainer.completed_episodes,
        checkpoint_every=checkpoint_every,
        validation_every=validation_every,
        validation=validation_env is not None,
        validation_seeds=validation_seeds,
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
    if validation_env is not None and trainer.completed_episodes == 0:
        initial_validation = trainer.validate(validation_env, seeds)
        (output / "validation-initial.json").write_text(
            json.dumps(initial_validation, ensure_ascii=False, indent=2) + "\n"
        )
        trainer.export_policy(output / "policy-initial.pt")
    if trainer.protocol.get("kind") == "controlled":
        initial_score = controlled_score(trainer.online, trainer.config.history_length)
    with (output / "metrics.csv").open("x", newline="") as file:
        writer = None
        while trainer.completed_episodes < episodes:
            row = trainer.train_episode()
            row.update(
                validation_reward=None,
                validation_variance=None,
                validation_mean_variance=None,
            )
            if validation_env is not None and (
                trainer.completed_episodes % validation_every == 0
                or trainer.completed_episodes == episodes
            ):
                result = trainer.validate(validation_env, seeds)
                row.update(
                    validation_reward=result["reward"],
                    validation_variance=result["final_integrated_variance"],
                    validation_mean_variance=result["mean_integrated_variance"],
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


def _latest_checkpoint(config: TrainingConfig, protocol: dict[str, Any]) -> Path:
    """Busca en las ejecuciones locales sin cargar todos sus buffers en memoria."""
    hardware = {"device", "cpu_threads"}
    expected = {
        key: value for key, value in asdict(config).items() if key not in hardware
    }
    protocol = json.loads(json.dumps(protocol))
    candidates = sorted(
        (
            path
            for path in CACHE_DIRECTORY.rglob("checkpoint-last.pt")
            if path.is_file()
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for path in candidates:
        try:
            metadata = json.loads(path.with_name("config.json").read_text())
        except OSError, json.JSONDecodeError:
            continue
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("training"), dict
        ):
            continue
        actual = {
            key: value
            for key, value in metadata["training"].items()
            if key not in hardware
        }
        if actual == expected and metadata.get("protocol") == protocol:
            return path
    raise FileNotFoundError(
        f"No hay un checkpoint compatible en {CACHE_DIRECTORY}. "
        "Ejecuta sin --resume para iniciar un entrenamiento."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenador mínimo BiLG-D3QN. No inicia replay histórico final."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Configuración JSON (por defecto: config/training-server.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directorio nuevo (por defecto: cache/training/<fecha UTC>)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validar recursos y componer escenarios sin simular ni entrenar",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        nargs="?",
        const=True,
        metavar="CHECKPOINT",
        help="Continuar desde el último checkpoint compatible, o desde la ruta indicada",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Total de episodios deseado, no episodios adicionales",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--seed",
        type=int,
        help="Semilla de una ejecución independiente; no cambia los casos de validación",
    )
    args = parser.parse_args()
    if args.check and (
        args.output is not None or args.resume is not None or args.episodes is not None
    ):
        parser.error("--check no admite --output, --resume ni --episodes")
    spec = json.loads(args.config.read_text())
    parameters = dict(spec.get("training", {}))
    if args.device:
        parameters["device"] = args.device
    if args.seed is not None:
        parameters["seed"] = args.seed
    config = TrainingConfig(**parameters)
    if config.capacity_steps < spec["steps_per_episode"]:
        raise ValueError("Un episodio completo debe caber en el replay buffer")
    env, validation_env, protocol = prepare_experiment(
        spec, args.config.resolve().parent, check=args.check
    )
    if args.check:
        _validate_run_settings(
            episodes=spec["episodes"],
            completed_episodes=0,
            checkpoint_every=spec.get("checkpoint_every", 10),
            validation_every=spec.get("validation_every", 10),
            validation=validation_env is not None,
            validation_seeds=spec.get("validation_seeds"),
        )
        print(
            json.dumps(
                {
                    "status": "configuration_checked",
                    "training_executed": False,
                    "cuda_available": torch.cuda.is_available(),
                    "requested_device": config.device,
                    "protocol": protocol,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        env.close()
        if validation_env is not None:
            validation_env.close()
        return
    resume = None
    if args.resume is True:
        try:
            resume = _latest_checkpoint(config, protocol)
        except FileNotFoundError as error:
            parser.error(str(error))
    elif args.resume is not None:
        resume = args.resume.resolve()
    if resume is not None:
        print(f"Reanudando: {resume}", flush=True)
    trainer = Trainer(env, config, protocol=protocol)
    if resume is not None:
        trainer.load_checkpoint(resume)
    elif spec["kind"] == "controlled":
        # Empate inicial conocido: el argmax elige este. Solo para este piloto.
        with torch.no_grad():
            for parameter in trainer.online.advantage[-1].parameters():
                parameter.zero_()
        trainer.target.load_state_dict(trainer.online.state_dict())
    episodes = args.episodes if args.episodes is not None else spec["episodes"]
    if (
        resume is not None
        and type(episodes) is int
        and episodes <= trainer.completed_episodes
    ):
        parser.error(
            f"El checkpoint ya alcanzó {trainer.completed_episodes} episodios. "
            f"Aumenta 'episodes' en {args.config} para continuar."
        )
    output = args.output or (
        CACHE_DIRECTORY / "training" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    )
    print(f"Salida: {output}", flush=True)
    run_training(
        trainer,
        episodes=episodes,
        output=output,
        checkpoint_every=spec.get("checkpoint_every", 10),
        validation_env=validation_env,
        validation_seeds=spec.get("validation_seeds"),
        validation_every=spec.get("validation_every", 10),
        resumed_from=str(resume) if resume is not None else None,
    )
    env.close()
    if validation_env is not None:
        validation_env.close()
