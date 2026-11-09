"""Demostración corta del entorno; no inicia entrenamiento."""

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from lima_pm25.estimation import load_historical_pm25_csv
from lima_pm25.simulation import (
    load_episode_generation_config,
    load_historical_weather_csv,
)

from .env import LimaPM25Env
from .episodes import Episode, GeneratedEpisodes, HistoricalReplay
from .evaluation import evaluate_episode
from .policies import BaselinePolicy
from .resources import load_study_resources
from .simulation import STEP_SECONDS


def _utc(value: str) -> datetime:
    date = datetime.fromisoformat(value)
    if date.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError("La fecha debe incluir zona UTC (+00:00)")
    return date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demostración del controlador central de tres sensores, sin entrenamiento"
    )
    parser.add_argument(
        "--core-root",
        type=Path,
        default=Path("../lima-pm25-core"),
        help="Raíz de recursos conservados, de solo lectura",
    )
    parser.add_argument(
        "--steps", type=int, default=3, help="Cantidad de ticks de cinco minutos"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Política entrenada; requiere --policy neural",
    )
    parser.add_argument(
        "--scenario",
        choices=("control", "event"),
        default="event",
        help="Miembro de la demostración sintética de core",
    )
    parser.add_argument(
        "--policy", choices=("random", "greedy", "sweep", "neural"), default="greedy"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Ruido móvil común, en µg/m³; usa el de la política cargada o cero en demostración",
    )
    parser.add_argument(
        "--image", type=Path, help="Guardar una imagen final; no sobrescribe archivos"
    )
    parser.add_argument(
        "--episode-config",
        type=Path,
        help="Configuración del generador de episodios de core",
    )
    parser.add_argument(
        "--partition", choices=("training", "evaluation"), default="training"
    )
    parser.add_argument(
        "--weather", choices=("synthetic", "historical", "historical_perturbed")
    )
    parser.add_argument(
        "--gp-fit-end",
        type=_utc,
        help="Extremo exclusivo real del ajuste del GP y sus normalizaciones",
    )
    parser.add_argument(
        "--replay-start", type=_utc, help="Inicio de una ventana histórica reservada"
    )
    parser.add_argument(
        "--policy-fit-end",
        type=_utc,
        help="Extremo exclusivo real del entrenamiento de la política",
    )
    args = parser.parse_args()
    loaded_policy = None
    policy_protocol = {}
    if args.checkpoint:
        if args.policy != "neural":
            parser.error("--checkpoint requiere --policy neural")
        from .training import load_policy

        loaded_policy, policy_protocol = load_policy(args.checkpoint, seed=args.seed)
        if args.replay_start:
            if "policy_fit_end_utc" not in policy_protocol:
                parser.error(
                    "El checkpoint no declara un protocolo experimental válido para replay final"
                )
            fitted = _utc(policy_protocol["policy_fit_end_utc"])
            if args.policy_fit_end is not None and args.policy_fit_end < fitted:
                parser.error(
                    "--policy-fit-end no puede omitir el periodo de validación del modelo"
                )
            args.policy_fit_end = args.policy_fit_end or fitted
            gp_end = _utc(policy_protocol["gp_fit_end_utc"])
            if args.gp_fit_end is not None and args.gp_fit_end < gp_end:
                parser.error(
                    "--gp-fit-end es anterior al ajuste declarado por el checkpoint"
                )
            args.gp_fit_end = args.gp_fit_end or gp_end
            if args.replay_start < _utc(policy_protocol["reserved_final_start_utc"]):
                parser.error(
                    "La fecha solicitada no pertenece al replay final reservado"
                )
            if "reserved_final_end_utc" in policy_protocol and (
                args.replay_start + timedelta(seconds=args.steps * STEP_SECONDS)
                > _utc(policy_protocol["reserved_final_end_utc"])
            ):
                parser.error("La ventana solicitada excede el extremo final reservado")
    if args.steps < 1:
        parser.error("--steps debe ser positivo")
    if args.episode_config and args.replay_start:
        parser.error("Elige episodios generados o replay, no ambos")
    if args.image and args.image.exists():
        parser.error("La imagen de destino ya existe")
    domain, template, gp_config = load_study_resources(
        args.core_root, member=args.scenario
    )
    if loaded_policy is not None and (
        policy_protocol.get("grid_shape") != list(domain.grid.shape)
        or policy_protocol.get("gp_profile_sha256") != gp_config.profile_sha256
    ):
        parser.error(
            "El checkpoint no corresponde a esta grilla y normalización del GP"
        )
    duration = args.steps * STEP_SECONDS
    if duration > template.weather.duration_s:
        parser.error("La demostración base cubre como máximo tres horas")

    def demonstration(seed: int) -> Episode:
        return Episode(replace(template, config=replace(template.config, seed=seed)))

    source_factory: Callable[[int], Episode] = demonstration

    if args.episode_config or args.replay_start:
        if args.gp_fit_end is None:
            parser.error(
                "Es obligatorio declarar --gp-fit-end para impedir fugas temporales"
            )
        weather = load_historical_weather_csv(
            args.core_root / "data/var_meteorologicas_lima_2025_2026_imputed.csv",
            station_id="CAMPO_DE_MARTE",
        )
        if args.episode_config:
            source_factory = GeneratedEpisodes(
                template,
                weather,
                load_episode_generation_config(args.episode_config),
                args.gp_fit_end,
                args.partition,
                args.weather,
            )
        else:
            if args.policy_fit_end is None:
                parser.error("El replay requiere --policy-fit-end")
            pm25 = load_historical_pm25_csv(
                args.core_root / "data/pm25_lima_2025_2026_imputed.csv"
            )
            source_factory = HistoricalReplay(
                template,
                weather,
                pm25,
                args.replay_start,
                args.gp_fit_end,
                args.policy_fit_end,
            )

    def factory(seed):
        # Generar primero la ventana completa: un ensayo corto no altera el
        # calendario original de fuentes ni invalida eventos posteriores.
        episode = source_factory(seed)
        config = replace(
            episode.request.config,
            duration_s=duration,
            concentration_output_times_s=tuple(
                t
                for t in episode.request.config.concentration_output_times_s
                if t <= duration
            ),
        )
        fixed = episode.historical_fixed
        if fixed is not None:
            fixed = tuple(r for r in fixed if r.simulation_time_s <= duration)
        return Episode(replace(episode.request, config=config), fixed)

    env = LimaPM25Env(
        domain=domain,
        episode_factory=factory,
        estimator_config=gp_config,
        mobile_noise_stddev_ug_m3=(
            args.noise
            if args.noise is not None
            else policy_protocol.get("noise_stddev_ug_m3", 0.0)
        ),
        concentration_priority=policy_protocol.get("concentration_priority", 0.0),
        distance_cost_per_km=policy_protocol.get("distance_cost_per_km", 0.0),
        render_mode="rgb_array" if args.image else None,
    )
    if loaded_policy is not None:
        policy = loaded_policy
    elif args.policy == "neural":
        import torch

        from .network import BiLGD3QN, NeuralController

        torch.manual_seed(args.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = NeuralController(BiLGD3QN().to(device), seed=args.seed)
        print("Demostración con red sin entrenar; no constituye una política evaluada.")
    else:
        policy = BaselinePolicy(args.policy, seed=args.seed)
    result = evaluate_episode(env, policy, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.image:
        from PIL import Image

        args.image.parent.mkdir(parents=True, exist_ok=True)
        image = env.render()
        assert image is not None
        with args.image.open("xb") as output:
            Image.fromarray(image).save(output, format="PNG")
    env.close()
