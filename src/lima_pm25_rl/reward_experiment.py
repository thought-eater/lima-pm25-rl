"""Comparación de recompensa con el GP y la arquitectura BiLG originales."""

import argparse
import csv
import io
import json
import tarfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .arrival_experiment import _write_json
from .comparison import _validate_seeds, save_comparison_map
from .gp_temporal_experiment import check_control
from .history_experiment import case_row as history_case_row
from .history_experiment import pair_case, validate_control_files
from .policies import BaselinePolicy
from .reward_env import error_environment, error_reward_protocol
from .train_cli import (
    CACHE_DIRECTORY,
    PROJECT_ROOT,
    _sha256,
    prepare_experiment,
    run_training,
)
from .training import Trainer, TrainingConfig, load_policy
from .value_diagnostics import discounted_returns, rollout

DEFAULT_CONFIG = PROJECT_ROOT / "config/reward-pilot.json"
MODES = ("uncertainty", "synthetic_error")
METRICS = (
    "synthetic_mse",
    "temporal_mse",
    "last_third_mse",
    "synthetic_mae_ug_m3",
    "distance_m",
    "final_integrated_variance",
    "mean_integrated_variance",
    "held_vehicle_ticks",
    "early_termination",
    "reference_reward",
    "reference_return",
)


def read_controls(source, spec):
    """Cargar solo controles BiLG auditables; nunca checkpoints con buffers ni Window."""
    names = {"experiment.json"}
    for seed in spec["paired_seeds"]:
        prefix = f"bilg-{seed}"
        names.update((f"{prefix}/config.json", f"{prefix}/metrics.csv"))
        for phase in ("initial", "last"):
            names.update(
                (
                    f"{prefix}/policy-{phase}.pt",
                    f"{prefix}/evaluation-{phase}/policy.json",
                )
            )
            names.update(
                f"{prefix}/evaluation-{phase}/case-{case}.json"
                for case in spec["validation_seeds"]
            )
    if source.is_dir():
        return {name: (source / name).read_bytes() for name in sorted(names)}
    raw, roots = {}, set()
    with tarfile.open(source, "r|zst") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            matches = [
                name
                for name in names
                if str(path) == name or str(path).endswith("/" + name)
            ]
            if not matches:
                continue
            name = matches[0]
            if path.is_absolute() or ".." in path.parts or name in raw:
                raise ValueError("Control duplicado o ruta inválida en el archivo")
            roots.add(path.parents[len(PurePosixPath(name).parts) - 1])
            raw[name] = archive.extractfile(member).read()
    if len(roots) != 1 or set(raw) != names:
        raise ValueError("Se requiere una sola ejecución con todos los controles BiLG")
    return raw


def case_row(report, *, seed, snapshot, mode, gamma):
    row = history_case_row(report, seed=seed, snapshot=snapshot, history_mode="history")
    row.pop("history_mode")
    row["reward_mode"] = mode
    row["native_reward"] = row.pop("reward")
    row["native_return"] = row.pop("discounted_return")
    reference_rewards = [
        t["information_gain"] - t["distance_cost"] for t in report["ticks"]
    ]
    row["reference_reward"] = float(sum(reference_rewards))
    row["reference_return"] = discounted_returns(reference_rewards, gamma)[0]
    return row


def _means(rows):
    values = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    for name, key in (
        ("final_rmse", "synthetic_mse"),
        ("temporal_rmse", "temporal_mse"),
        ("last_third_rmse", "last_third_mse"),
    ):
        values[name] = float(np.sqrt(values[key]))
    return values


def summarize(rows, seeds, cases, references):
    indexed = {
        (r["reward_mode"], r["seed"], r["snapshot"], r["case_seed"]): r for r in rows
    }
    expected = {
        (mode, seed, phase, case)
        for mode in MODES
        for seed in seeds
        for phase in ("initial", "last")
        for case in cases
    }
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("La comparación de recompensa requiere todos los pares")
    summaries, per_seed, paired = [], [], []
    for mode in MODES:
        for phase in ("initial", "last"):
            group = [
                indexed[mode, seed, phase, case] for seed in seeds for case in cases
            ]
            summaries.append(
                {
                    "reward_mode": mode,
                    "snapshot": phase,
                    **_means(group),
                    "native_reward": float(
                        np.mean([r["native_reward"] for r in group])
                    ),
                    "native_return": float(
                        np.mean([r["native_return"] for r in group])
                    ),
                }
            )
            per_seed.extend(
                {
                    "reward_mode": mode,
                    "snapshot": phase,
                    "seed": seed,
                    **_means([indexed[mode, seed, phase, case] for case in cases]),
                }
                for seed in seeds
            )
    for seed in seeds:
        for case in cases:
            a, b = (indexed[mode, seed, "last", case] for mode in MODES)
            paired.append(
                {
                    "seed": seed,
                    "case_seed": case,
                    "synthetic_error_minus_uncertainty": {
                        k: b[k] - a[k] for k in METRICS
                    },
                }
            )
    ref_groups, ref_pairs = [], []
    for strategy in ("random", "sweep", "greedy"):
        expected_seeds = {
            r["policy_seed"] for r in references if r["strategy"] == strategy
        }
        grouped = []
        for case in cases:
            selected = [
                r
                for r in references
                if r["strategy"] == strategy and r["case_seed"] == case
            ]
            if (
                not selected
                or {r["policy_seed"] for r in selected} != expected_seeds
                or len(expected_seeds) != len(selected)
            ):
                raise ValueError("Faltan referencias o hay réplicas duplicadas")
            average = _means(selected)
            grouped.append(average)
            for mode in MODES:
                neural = _means([indexed[mode, seed, "last", case] for seed in seeds])
                ref_pairs.append(
                    {
                        "case_seed": case,
                        "reward_mode": mode,
                        "reference": strategy,
                        "neural_minus_reference": {
                            k: neural[k] - average[k] for k in METRICS
                        },
                    }
                )
        ref_groups.append(
            {
                "strategy": strategy,
                "rollouts": sum(r["strategy"] == strategy for r in references),
                **_means(grouped),
            }
        )
    control = next(
        r
        for r in summaries
        if r["reward_mode"] == "uncertainty" and r["snapshot"] == "last"
    )
    candidate = next(
        r
        for r in summaries
        if r["reward_mode"] == "synthetic_error" and r["snapshot"] == "last"
    )
    greedy = next(r for r in ref_groups if r["strategy"] == "greedy")
    return {
        "primary_snapshot": "last",
        "summaries": summaries,
        "per_seed": per_seed,
        "paired_last": paired,
        "references": ref_groups,
        "paired_references": ref_pairs,
        "criteria": {
            "lower_temporal_error_vs_control": candidate["temporal_mse"]
            < control["temporal_mse"],
            "final_error_not_higher_vs_control": candidate["synthetic_mse"]
            <= control["synthetic_mse"],
            "lower_temporal_error_vs_greedy": candidate["temporal_mse"]
            < greedy["temporal_mse"],
            "final_error_not_higher_vs_greedy": candidate["synthetic_mse"]
            <= greedy["synthetic_mse"],
            "distance_not_higher_vs_control": candidate["distance_m"]
            <= control["distance_m"],
        },
        "interpretation": "Retornos nativos usan recompensas diferentes y no se comparan entre brazos. La decisión usa errores sintéticos comunes, distancia y consistencia entre casos/semillas.",
    }


def check_initial_equivalence(reference, current):
    if len(reference["ticks"]) != len(current["ticks"]):
        raise ValueError("La política inicial ya no reproduce el horizonte del control")
    for a, b in zip(reference["ticks"], current["ticks"], strict=True):
        if a["actions"] != b["actions"] or [s["input_sha256"] for s in a["stages"]] != [
            s["input_sha256"] for s in b["stages"]
        ]:
            raise ValueError(
                "La política inicial no reproduce acciones/observaciones de eval3"
            )
        if not np.isclose(
            a["information_gain"], b["information_gain"], rtol=1e-7, atol=1e-9
        ):
            raise ValueError("La ganancia GP inicial difiere del control")
    for key in (
        "synthetic_mse",
        "temporal_mse",
        "distance_m",
        "final_integrated_variance",
    ):
        if not np.isclose(
            reference["metrics"][key], current["metrics"][key], rtol=1e-7, atol=1e-8
        ):
            raise ValueError("Las métricas iniciales no reproducen eval3")


def check_initial_weights(trainer, expected):
    actual = trainer.online.state_dict()
    if actual.keys() != expected.keys() or any(
        not torch.equal(value.detach().cpu(), expected[key])
        for key, value in actual.items()
    ):
        raise ValueError("La inicialización nueva no coincide con el control auditado")


def evaluate_candidate(env, policy, *, spec, seed, phase, controls, output, maps):
    output.mkdir(parents=True, exist_ok=False)
    gamma = TrainingConfig(**spec["training"]).gamma
    rows = []
    for index, case in enumerate(spec["validation_seeds"]):
        report = rollout(env, policy, case_seed=case, gamma=gamma)
        report["pairing"] = pair_case(controls[seed, phase, case], report)
        if phase == "initial":
            check_initial_equivalence(controls[seed, phase, case], report)
        report["routes_m"] = env._routes
        report["training_reward_mode"] = "synthetic_error"
        report["evaluation_reward_mode"] = "synthetic_error"
        _write_json(output / f"case-{case}.json", report)
        rows.append(
            case_row(
                report, seed=seed, snapshot=phase, mode="synthetic_error", gamma=gamma
            )
        )
        if maps and index == 0:
            save_comparison_map(
                env,
                output / "map.png",
                f"Recompensa de error | semilla {seed} | {phase}",
            )
    return rows


def run_experiment(
    spec, *, directory, output, source, settings, train, validation, protocol, maps=True
):
    raw = read_controls(source, spec)
    controls, initials, hashes = validate_control_files(
        raw.__getitem__, spec, protocol, settings["control_code_sha256"]
    )
    candidate_train, candidate_validation = (
        error_environment(train),
        error_environment(validation),
    )
    candidate_protocol = error_reward_protocol(protocol)
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    _write_json(
        output / "config.json",
        {
            "spec": spec,
            "settings": settings,
            "control_protocol": protocol,
            "candidate_protocol": candidate_protocol,
            "control_source": str(source),
            "new_training_episodes": len(spec["paired_seeds"]) * spec["episodes"],
            "primary_snapshot": "last",
            "best_selection": "final_integrated_variance_unchanged",
            "torch_version": str(torch.__version__),
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "reward_experiment.py",
                    "reward_env.py",
                    "env.py",
                    "routing.py",
                    "simulation.py",
                    "scenario_fields.py",
                    "training.py",
                    "network.py",
                    "replay.py",
                    "policies.py",
                    "train_cli.py",
                    "history_experiment.py",
                    "value_diagnostics.py",
                )
            },
        },
    )
    for name, data in raw.items():
        path = output / "controls" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(data)
    _write_json(output / "control-files.json", hashes)
    config = TrainingConfig(**spec["training"])
    rows = [
        case_row(r, seed=s, snapshot=p, mode="uncertainty", gamma=config.gamma)
        for (s, p, _), r in controls.items()
    ]
    references, runs = [], []
    try:
        # Todos los controles iniciales se comprueban antes del primer entrenamiento.
        for seed in spec["paired_seeds"]:
            probe = Trainer(
                candidate_train, replace(config, seed=seed), protocol=candidate_protocol
            )
            check_initial_weights(probe, initials[seed])
            probe.online.eval()
            folder = output / "evaluations" / f"synthetic-error-{seed}" / "initial"
            rows.extend(
                evaluate_candidate(
                    candidate_validation,
                    probe.controller,
                    spec=spec,
                    seed=seed,
                    phase="initial",
                    controls=controls,
                    output=folder,
                    maps=maps,
                )
            )
            assert probe.completed_episodes == 0 and probe.updates == 0
            del probe
        for strategy in ("random", "sweep", "greedy"):
            policy_seeds = (
                settings["reference_policy_seeds"][:1]
                if strategy == "sweep"
                else settings["reference_policy_seeds"]
            )
            for policy_seed in policy_seeds:
                for index, case in enumerate(spec["validation_seeds"]):
                    report = rollout(
                        validation,
                        BaselinePolicy(strategy, seed=policy_seed),
                        case_seed=case,
                        gamma=config.gamma,
                    )
                    report["pairing"] = pair_case(
                        controls[spec["paired_seeds"][0], "last", case], report
                    )
                    report["routes_m"] = validation._routes
                    folder = output / "references" / strategy / f"policy-{policy_seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    _write_json(folder / f"case-{case}.json", report)
                    references.append(
                        {
                            **case_row(
                                report,
                                seed=policy_seed,
                                snapshot="reference",
                                mode="uncertainty",
                                gamma=config.gamma,
                            ),
                            "strategy": strategy,
                            "policy_seed": policy_seed,
                        }
                    )
                    if (
                        maps
                        and index == 0
                        and policy_seed == settings["reference_policy_seeds"][0]
                    ):
                        save_comparison_map(
                            validation,
                            folder / "map.png",
                            f"{strategy} | semilla {policy_seed}",
                        )
        _write_json(output / "references.json", references)
        for seed in spec["paired_seeds"]:
            cfg = replace(config, seed=seed)
            trainer = Trainer(candidate_train, cfg, protocol=candidate_protocol)
            check_initial_weights(trainer, initials[seed])
            saved_spec = json.loads(json.dumps(spec))
            saved_spec["reward_mode"] = "synthetic_error"
            saved_spec["training"] = asdict(cfg)
            saved_spec["core_root"] = str((directory / spec["core_root"]).resolve())
            for item in saved_spec["episode_configs"]:
                item["path"] = str((directory / item["path"]).resolve())
            _write_json(output / "configs" / f"synthetic-error-{seed}.json", saved_spec)
            folder = output / f"synthetic-error-{seed}"
            print(
                f"Entrenamiento con error sintético | semilla {seed}: {spec['episodes']} episodios",
                flush=True,
            )
            run_training(
                trainer,
                episodes=spec["episodes"],
                output=folder,
                validation_env=candidate_validation,
                validation_seeds=spec["validation_seeds"],
                checkpoint_every=spec["checkpoint_every"],
                validation_every=spec["validation_every"],
            )
            with (folder / "metrics.csv").open() as file:
                trained = list(csv.DictReader(file))
            original = list(
                csv.DictReader(io.StringIO(raw[f"bilg-{seed}/metrics.csv"].decode()))
            )
            if [r["scenario_seed"] for r in trained] != [
                r["scenario_seed"] for r in original
            ]:
                raise ValueError(
                    "Las secuencias de escenarios no coinciden con el control"
                )
            run = {
                "seed": seed,
                "episodes": trainer.completed_episodes,
                "physical_steps": trainer.physical_steps,
                "optimizer_updates": trainer.updates,
                "parameters": sum(p.numel() for p in trainer.online.parameters()),
                "initial_weights_match": True,
                "control_episodes": len(original),
                "control_physical_steps": int(original[-1]["physical_steps"]),
                "control_optimizer_updates": int(original[-1]["optimizer_updates"]),
            }
            runs.append(run)
            _write_json(folder / "run-summary.json", run)
            policy, restored_protocol = load_policy(
                folder / "policy-last.pt", device=cfg.device
            )
            if restored_protocol != json.loads(json.dumps(candidate_protocol)):
                raise ValueError(
                    "La política exportada no conserva el protocolo de recompensa"
                )
            target = output / "evaluations" / f"synthetic-error-{seed}" / "last"
            rows.extend(
                evaluate_candidate(
                    candidate_validation,
                    policy,
                    spec=spec,
                    seed=seed,
                    phase="last",
                    controls=controls,
                    output=target,
                    maps=maps,
                )
            )
            for phase in ("initial", "last"):
                _write_json(
                    output
                    / "evaluations"
                    / f"synthetic-error-{seed}"
                    / phase
                    / "policy.json",
                    {
                        "policy_sha256": _sha256(folder / f"policy-{phase}.pt"),
                        "protocol": candidate_protocol,
                        "training": asdict(cfg),
                    },
                )
        _write_json(output / "cases.json", rows)
        result = {
            **summarize(
                rows, spec["paired_seeds"], spec["validation_seeds"], references
            ),
            "training_runs": runs,
            "scope": "Comparación de recompensa en validación sintética ya utilizada. No prueba histórica final ni despliegue físico.",
        }
        _write_json(output / "summary.json", result)
        return result
    finally:
        candidate_train.close()
        candidate_validation.close()


def main():
    parser = argparse.ArgumentParser(
        description="Comparar recompensa de incertidumbre y error sintético; reutilizar controles BiLG"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe resultados")
    settings = json.loads(args.config.read_text())
    spec_path = (args.config.resolve().parent / settings["experiment_config"]).resolve()
    spec = json.loads(spec_path.read_text())
    cfg = TrainingConfig(**spec["training"])
    if (
        settings["candidate_reward_mode"] != "synthetic_error"
        or spec["kind"] != "generated"
        or spec["episodes"] != 200
        or spec["paired_seeds"] != [42, 43, 44]
        or cfg.architecture != "bilg"
        or cfg.history_mode != "history"
        or cfg.history_length != 4
        or spec.get("reward_mode", "uncertainty") != "uncertainty"
        or spec.get("arrival_strategy", "shortest") != "shortest"
    ):
        parser.error(
            "Se requiere el piloto BiLG original: 200 episodios, semillas 42/43/44, historia completa y shortest"
        )
    _validate_seeds(settings["reference_policy_seeds"])
    for name in ("env.py", "routing.py"):
        if (
            _sha256(Path(__file__).parent / name)
            != settings["control_code_sha256"][name]
        ):
            raise ValueError(
                "El entorno o decodificador original difiere del control auditado"
            )
    train, validation, protocol = prepare_experiment(
        spec, spec_path.parent, check=args.check
    )
    assert validation is not None
    try:
        check_control(train.domain.grid, train.estimator_config)
        source = args.controls or PROJECT_ROOT / "eval3.tar.zst"
        if args.controls is None and not source.is_file():
            candidates = sorted(
                CACHE_DIRECTORY.glob("architecture-experiments/*/experiment.json")
            )
            if not candidates:
                parser.error(
                    "Falta eval3.tar.zst o una ejecución de arquitectura; no se entrenarán controles sustitutos"
                )
            source = candidates[-1].parent
        raw = read_controls(source, spec)
        validate_control_files(
            raw.__getitem__, spec, protocol, settings["control_code_sha256"]
        )
        if args.check:
            print(
                json.dumps(
                    {
                        "training_executed": False,
                        "simulation_executed": False,
                        "controls": str(source),
                        "reused_controls": 3,
                        "new_training_episodes": 600,
                        "candidate_evaluation_missions": 2
                        * len(spec["paired_seeds"])
                        * len(spec["validation_seeds"]),
                        "reference_missions": (
                            2 * len(settings["reference_policy_seeds"]) + 1
                        )
                        * len(spec["validation_seeds"]),
                        "candidate_reward": error_reward_protocol({})["reward"],
                    },
                    indent=2,
                )
            )
            return
        output = args.output or CACHE_DIRECTORY / "reward-experiments" / datetime.now(
            UTC
        ).strftime("%Y%m%d-%H%M%S-%f")
        print(f"Salida: {output}", flush=True)
        run_experiment(
            spec,
            directory=spec_path.parent,
            output=output,
            source=source,
            settings=settings,
            train=train,
            validation=validation,
            protocol=protocol,
        )
    finally:
        train.close()
        validation.close()


if __name__ == "__main__":
    main()
