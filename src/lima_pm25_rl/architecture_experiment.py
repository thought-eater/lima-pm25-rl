"""Piloto pareado de arquitectura para el servidor; sin gestores de ejecuciones."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .arrival_experiment import _write_json
from .comparison import _validate_seeds, save_comparison_map
from .train_cli import (
    CACHE_DIRECTORY,
    PROJECT_ROOT,
    _sha256,
    _validate_run_settings,
    prepare_experiment,
    run_training,
)
from .training import Trainer, TrainingConfig, load_policy
from .value_diagnostics import calibration, rollout

DEFAULT_CONFIG = PROJECT_ROOT / "config/architecture-pilot.json"
ARCHITECTURES = ("bilg", "window")
SNAPSHOTS = ("initial", "last")
METRICS = (
    "discounted_return",
    "reward",
    "synthetic_mse",
    "temporal_mse",
    "final_integrated_variance",
    "mean_integrated_variance",
    "distance_m",
    "q_return_mae",
    "q_return_bias",
)


def pilot_configs(spec, directory):
    """JSON autocontenidos: solo architecture cambia dentro de cada par."""
    _validate_seeds(spec["paired_seeds"])
    _validate_seeds(spec["validation_seeds"])
    if (
        spec["kind"] != "generated"
        or spec.get("arrival_strategy", "shortest") != "shortest"
    ):
        raise ValueError("La ablación requiere escenarios generados y shortest")
    _validate_run_settings(
        episodes=spec["episodes"],
        completed_episodes=0,
        checkpoint_every=spec["checkpoint_every"],
        validation_every=spec["validation_every"],
        validation=True,
        validation_seeds=spec["validation_seeds"],
    )
    for seed in spec["paired_seeds"]:
        for architecture in ARCHITECTURES:
            parameters = replace(
                TrainingConfig(**spec["training"]), seed=seed, architecture=architecture
            )
            if parameters.capacity_steps < spec["steps_per_episode"]:
                raise ValueError("Un episodio completo debe caber en el replay buffer")
            item = json.loads(json.dumps(spec))
            item.pop("paired_seeds")
            item["training"] = asdict(parameters)
            item["core_root"] = str((directory / item["core_root"]).resolve())
            for episode in item["episode_configs"]:
                episode["path"] = str((directory / episode["path"]).resolve())
            yield item


def evaluate_snapshot(env, policy, *, config, case_seeds, snapshot, output, maps=True):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case_seed in case_seeds:
        report = rollout(env, policy, case_seed=case_seed, gamma=config.gamma)
        report["routes_m"] = env._routes
        _write_json(output / f"case-{case_seed}.json", report)
        diagnostic = calibration(report)
        rows.append(
            {
                "architecture": config.architecture,
                "seed": config.seed,
                "snapshot": snapshot,
                "case_seed": case_seed,
                "signatures": report["signatures"],
                **report["metrics"],
                "q_return_mae": diagnostic["mae"],
                "q_return_bias": diagnostic["bias"],
            }
        )
        if maps and case_seed == case_seeds[0]:
            save_comparison_map(
                env,
                output / "map.png",
                f"{config.architecture} | semilla {config.seed} | {snapshot}",
            )
    _write_json(output / "cases.json", rows)
    return rows


def summarize_results(rows):
    """Igual peso por caso y semilla; conservar pares y raíces de MSE agrupado."""
    indexed = {
        (r["architecture"], r["seed"], r["snapshot"], r["case_seed"]): r for r in rows
    }
    seeds = sorted({r["seed"] for r in rows})
    cases = sorted({r["case_seed"] for r in rows})
    expected = {
        (a, s, p, c)
        for a in ARCHITECTURES
        for s in seeds
        for p in SNAPSHOTS
        for c in cases
    }
    if not rows or len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("La comparación necesita pares completos, sin duplicados")
    for case in cases:
        signatures = [r["signatures"] for r in rows if r["case_seed"] == case]
        if any(item != signatures[0] for item in signatures):
            raise ValueError("Los pares no comparten campo, fijas, inicio y horizonte")

    def means(selected):
        result = {key: float(np.mean([r[key] for r in selected])) for key in METRICS}
        result["synthetic_rmse_ug_m3"] = float(np.sqrt(result["synthetic_mse"]))
        result["temporal_rmse_ug_m3"] = float(np.sqrt(result["temporal_mse"]))
        return result

    summaries, per_seed, paired = [], [], []
    for architecture in ARCHITECTURES:
        for snapshot in SNAPSHOTS:
            selected = [
                r
                for r in rows
                if r["architecture"] == architecture and r["snapshot"] == snapshot
            ]
            summaries.append(
                {"architecture": architecture, "snapshot": snapshot, **means(selected)}
            )
            for seed in seeds:
                per_seed.append(
                    {
                        "architecture": architecture,
                        "snapshot": snapshot,
                        "seed": seed,
                        **means([r for r in selected if r["seed"] == seed]),
                    }
                )
    for seed in seeds:
        for case in cases:
            control, variant = (indexed[a, seed, "last", case] for a in ARCHITECTURES)
            paired.append(
                {
                    "seed": seed,
                    "case_seed": case,
                    **{
                        f"window_minus_bilg_{key}": variant[key] - control[key]
                        for key in METRICS
                    },
                }
            )
    return {
        "purpose": "architecture_pilot_validation_not_final",
        "primary_snapshot": "last",
        "seed_count": len(seeds),
        "case_count": len(cases),
        "summaries": summaries,
        "per_seed": per_seed,
        "paired_last": paired,
        "interpretation": "Comparación de arquitecturas completas, no igualada por parámetros o tiempo. Q–retorno mide calibración, no orden contrafactual de acciones.",
    }


def run_pilot(spec, *, directory, output, maps=True):
    configurations = list(pilot_configs(spec, directory))
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    _write_json(
        output / "experiment.json",
        {
            **spec,
            "torch_version": str(torch.__version__),
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "architecture_experiment.py",
                    "network.py",
                    "training.py",
                    "replay.py",
                    "value_diagnostics.py",
                    "env.py",
                    "policies.py",
                    "routing.py",
                )
            },
        },
    )
    for item in configurations:
        training = item["training"]
        _write_json(
            output / "configs" / f"{training['architecture']}-{training['seed']}.json",
            item,
        )
    rows, runs, scenarios = [], [], {}
    for item in configurations:
        config = TrainingConfig(**item["training"])
        label = f"{config.architecture}-{config.seed}"
        env, validation, protocol = prepare_experiment(item, directory)
        assert validation is not None
        try:
            trainer = Trainer(env, config, protocol=protocol)
            run = output / label
            print(
                f"Entrenamiento {label}: {item['episodes']} episodios | {run}",
                flush=True,
            )
            start = perf_counter()
            run_training(
                trainer,
                episodes=item["episodes"],
                output=run,
                checkpoint_every=item["checkpoint_every"],
                validation_env=validation,
                validation_seeds=item["validation_seeds"],
                validation_every=item["validation_every"],
            )
            seconds = perf_counter() - start
            with (run / "metrics.csv").open() as file:
                episode_seeds = [
                    int(row["scenario_seed"]) for row in csv.DictReader(file)
                ]
            if config.seed in scenarios and scenarios[config.seed] != episode_seeds:
                raise RuntimeError(
                    "Las arquitecturas no compartieron la secuencia de escenarios"
                )
            scenarios[config.seed] = episode_seeds
            runs.append(
                {
                    "architecture": config.architecture,
                    "seed": config.seed,
                    "parameters": sum(p.numel() for p in trainer.online.parameters()),
                    "episodes": trainer.completed_episodes,
                    "physical_steps": trainer.physical_steps,
                    "optimizer_updates": trainer.updates,
                    "training_and_validation_seconds": seconds,
                }
            )
            evaluation_start = perf_counter()
            for snapshot in SNAPSHOTS:
                print(f"Evaluación {label}: {snapshot}", flush=True)
                path = run / f"policy-{snapshot}.pt"
                policy, _ = load_policy(path, device=config.device)
                phase_rows = evaluate_snapshot(
                    validation,
                    policy,
                    config=config,
                    case_seeds=item["validation_seeds"],
                    snapshot=snapshot,
                    output=run / f"evaluation-{snapshot}",
                    maps=maps,
                )
                for row in phase_rows:
                    row["policy_sha256"] = _sha256(path)
                _write_json(
                    run / f"evaluation-{snapshot}" / "policy.json",
                    {
                        "policy_sha256": _sha256(path),
                        "training": asdict(config),
                        "snapshot": snapshot,
                        "protocol": protocol,
                    },
                )
                rows.extend(phase_rows)
            runs[-1]["evaluation_seconds"] = perf_counter() - evaluation_start
            _write_json(run / "run-summary.json", runs[-1])
        finally:
            env.close()
            validation.close()
    result = {**summarize_results(rows), "training_runs": runs}
    _write_json(output / "cases.json", rows)
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Piloto pareado BiLG frente a D3QN sin recurrencia"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Componer escenarios sin simular ni entrenar",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if args.check:
        if args.output:
            parser.error("--check no escribe resultados")
        configurations = list(pilot_configs(spec, args.config.resolve().parent))
        # Las seis configuraciones comparten entorno; componerlo una sola vez.
        env, validation, _ = prepare_experiment(
            configurations[0], args.config.resolve().parent, check=True
        )
        env.close()
        if validation is not None:
            validation.close()
        print(
            json.dumps(
                {
                    "training_executed": False,
                    "runs": len(configurations),
                    "episodes_per_run": spec["episodes"],
                    "total_training_episodes": len(configurations) * spec["episodes"],
                    "architectures": ARCHITECTURES,
                    "paired_seeds": spec["paired_seeds"],
                },
                indent=2,
            )
        )
        return
    output = args.output or CACHE_DIRECTORY / "architecture-experiments" / datetime.now(
        UTC
    ).strftime("%Y%m%d-%H%M%S-%f")
    print(f"Salida: {output}", flush=True)
    result = run_pilot(spec, directory=args.config.resolve().parent, output=output)
    for row in result["summaries"]:
        if row["snapshot"] == "last":
            print(
                f"{row['architecture']}: RMSE={row['synthetic_rmse_ug_m3']:.4f}, "
                f"RMSE temporal={row['temporal_rmse_ug_m3']:.4f}, "
                f"retorno={row['discounted_return']:.5f}"
            )


if __name__ == "__main__":
    main()
