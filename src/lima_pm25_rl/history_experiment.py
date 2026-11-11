"""Historia explícita frente a presente repetido, reutilizando controles auditados."""

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .arrival_experiment import _write_json
from .comparison import save_comparison_map
from .evaluate_cli import _compatible
from .gp_temporal_experiment import check_control
from .mobility_diagnostics import compare_fixed_measurements
from .policies import HISTORY_MODES
from .train_cli import (
    CACHE_DIRECTORY,
    PROJECT_ROOT,
    _sha256,
    prepare_experiment,
    run_training,
)
from .training import Trainer, TrainingConfig, load_policy
from .value_diagnostics import rollout

DEFAULT_CONFIG = PROJECT_ROOT / "config/history-pilot.json"
METRICS = (
    "synthetic_mse",
    "temporal_mse",
    "last_third_mse",
    "discounted_return",
    "reward",
    "distance_m",
    "final_integrated_variance",
    "mean_integrated_variance",
    "held_vehicle_ticks",
    "early_termination",
)


def case_row(report, *, seed, snapshot, history_mode):
    """El último tercio se obtiene de las muestras, sin inventar el t=0 ausente."""
    series = report.get("evaluation_series") or [
        {"elapsed_s": t["tick"] * 300, "synthetic_mse": t["synthetic_mse"]}
        for t in report["ticks"]
    ]
    times = np.asarray([r["elapsed_s"] for r in series], dtype=float)
    errors = np.asarray([r["synthetic_mse"] for r in series], dtype=float)
    end = report["signatures"]["horizon_s"]
    start = 2 * end / 3
    if (
        len(times) < 2
        or times[0] > start
        or times[-1] != end
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("La traza no permite reconstruir el último tercio completo")
    x = np.concatenate(([start], times[(times > start) & (times < end)], [end]))
    return {
        "seed": seed,
        "snapshot": snapshot,
        "history_mode": history_mode,
        "case_seed": report["case_seed"],
        "signatures": report["signatures"],
        **report["metrics"],
        "last_third_mse": float(
            np.trapezoid(np.interp(x, times, errors), x) / (end - start)
        ),
        "held_vehicle_ticks": sum(sum(t["held_vehicles"]) for t in report["ticks"]),
        "early_termination": int(report["metrics"]["decision_end_elapsed_s"] < end),
    }


def validate_controls(source, spec, protocol, expected_code):
    return validate_control_files(
        lambda name: (source / name).read_bytes(), spec, protocol, expected_code
    )


def validate_control_files(read_file, spec, protocol, expected_code):
    """Auditar los mismos controles desde carpeta o miembros de un tar en memoria."""
    records, files, states = {}, {}, {}

    def read(name):
        data = read_file(name)
        files[name] = hashlib.sha256(data).hexdigest()
        return data

    experiment = json.loads(read("experiment.json"))
    if any(
        experiment["code_sha256"].get(name) != value
        for name, value in expected_code.items()
    ):
        raise ValueError("El código de los controles no coincide con eval3 auditado")
    for seed in spec["paired_seeds"]:
        folder = f"bilg-{seed}"
        config = replace(TrainingConfig(**spec["training"]), seed=seed)
        metadata = json.loads(read(f"{folder}/config.json"))
        actual = TrainingConfig(**metadata["training"])
        if (
            replace(actual, device=config.device, cpu_threads=config.cpu_threads)
            != config
            or metadata["protocol"] != json.loads(json.dumps(protocol))
            or metadata["target_episodes"] != spec["episodes"]
            or metadata["start_episode"] != 0
            or metadata["resumed_from"] is not None
            or metadata["validation_every"] != spec["validation_every"]
            or metadata["checkpoint_every"] != spec["checkpoint_every"]
            or metadata["validation_seeds"] != spec["validation_seeds"]
        ):
            raise ValueError("El protocolo o presupuesto de los controles no coincide")
        metrics = list(
            csv.DictReader(io.StringIO(read(f"{folder}/metrics.csv").decode()))
        )
        scenarios = [
            int(np.random.SeedSequence([seed, 0, i]).generate_state(1)[0])
            for i in range(spec["episodes"])
        ]
        if (
            [int(r["episode"]) for r in metrics] != list(range(1, spec["episodes"] + 1))
            or [int(r["scenario_seed"]) for r in metrics] != scenarios
            or any(int(r["invalid_actions"]) for r in metrics)
        ):
            raise ValueError(
                "El historial de entrenamiento del control está incompleto o no emparejado"
            )
        for phase, episode in (("initial", 0), ("last", spec["episodes"])):
            policy_path = f"{folder}/policy-{phase}.pt"
            state = torch.load(
                io.BytesIO(read(policy_path)), map_location="cpu", weights_only=True
            )
            meta_path = f"{folder}/evaluation-{phase}/policy.json"
            evaluation = json.loads(read(meta_path))
            if (
                not _compatible(state, config, protocol)
                or state["completed_episodes"] != episode
                or evaluation["policy_sha256"] != files[policy_path]
                or state["physical_steps"]
                != (0 if phase == "initial" else int(metrics[-1]["physical_steps"]))
            ):
                raise ValueError(
                    "Los pesos del control no corresponden al snapshot y protocolo requeridos"
                )
            if phase == "initial":
                states[seed] = state["online"]
            for case in spec["validation_seeds"]:
                path = f"{folder}/evaluation-{phase}/case-{case}.json"
                report = json.loads(read(path))
                if (
                    report["case_seed"] != case
                    or report["signatures"]["horizon_s"]
                    != spec["steps_per_episode"] * 300
                ):
                    raise ValueError(
                        "La evaluación del control usa otros casos u horizonte"
                    )
                case_row(report, seed=seed, snapshot=phase, history_mode="history")
                records[seed, phase, case] = report
    return records, states, files


def pair_case(reference, report):
    for key in ("field_sha256", "initial_positions_sha256", "horizon_s"):
        if reference["signatures"][key] != report["signatures"][key]:
            raise ValueError("Los brazos no comparten campo, inicio u horizonte")
    fixed = compare_fixed_measurements(
        [r for r in report["measurements"] if r["sensor_type"] == "fixed"],
        [r for r in reference["measurements"] if r["sensor_type"] == "fixed"],
    )
    if not fixed["match"]:
        raise ValueError("Las lecturas fijas no coinciden")
    for before, after in zip(reference["ticks"], report["ticks"]):
        if before["sensor_rng_after_sha256"] != after["sensor_rng_after_sha256"]:
            raise ValueError("El ruido móvil no está emparejado")
    return {
        "match": True,
        "fixed_measurements": fixed,
        "common_decision_ticks": min(len(reference["ticks"]), len(report["ticks"])),
    }


def summarize(rows, seeds, cases):
    phases = ("initial", "last")
    index = {
        (r["history_mode"], r["seed"], r["snapshot"], r["case_seed"]): r for r in rows
    }
    expected = {
        (mode, seed, phase, case)
        for mode in HISTORY_MODES
        for seed in seeds
        for phase in phases
        for case in cases
    }
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError("La comparación necesita pares completos sin duplicados")

    def means(group):
        values = {k: float(np.mean([r[k] for r in group])) for k in METRICS}
        values["final_rmse"] = float(np.sqrt(values["synthetic_mse"]))
        values["temporal_rmse"] = float(np.sqrt(values["temporal_mse"]))
        values["last_third_rmse"] = float(np.sqrt(values["last_third_mse"]))
        return values

    aggregates, per_seed, paired = [], [], []
    for mode in HISTORY_MODES:
        for phase in phases:
            group = [index[mode, s, phase, c] for s in seeds for c in cases]
            aggregates.append({"history_mode": mode, "snapshot": phase, **means(group)})
            for seed in seeds:
                per_seed.append(
                    {
                        "history_mode": mode,
                        "snapshot": phase,
                        "seed": seed,
                        **means([index[mode, seed, phase, c] for c in cases]),
                    }
                )
    for seed in seeds:
        for case in cases:
            a, b = (index[mode, seed, "last", case] for mode in HISTORY_MODES)
            paired.append(
                {
                    "seed": seed,
                    "case_seed": case,
                    **{
                        f"repeat_current_minus_history_{k}": b[k] - a[k]
                        for k in METRICS
                    },
                }
            )
    return {
        "primary_snapshot": "last",
        "summaries": aggregates,
        "per_seed": per_seed,
        "paired_last": paired,
        "scope": "Ablación de historia explícita, no del historial GP. Validación, no prueba histórica final.",
    }


def run_experiment(
    spec, *, directory, output, source, settings, train, validation, protocol, maps=True
):
    controls, initials, hashes = validate_controls(
        source, spec, protocol, settings["control_code_sha256"]
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    _write_json(
        output / "config.json",
        {
            "spec": spec,
            "settings": settings,
            "protocol": protocol,
            "control_source": str(source),
            "new_training_episodes": len(spec["paired_seeds"]) * spec["episodes"],
            "torch_version": str(torch.__version__),
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "history_experiment.py",
                    "policies.py",
                    "network.py",
                    "replay.py",
                    "training.py",
                    "env.py",
                    "routing.py",
                )
            },
        },
    )
    _write_json(output / "control-files.json", hashes)
    rows = [
        case_row(r, seed=s, snapshot=p, history_mode="history")
        for (s, p, _), r in controls.items()
    ]
    runs = []
    for seed in spec["paired_seeds"]:
        config = replace(
            TrainingConfig(**spec["training"]), seed=seed, history_mode="repeat_current"
        )
        saved_spec = json.loads(json.dumps(spec))
        saved_spec["training"] = asdict(config)
        saved_spec["core_root"] = str((directory / spec["core_root"]).resolve())
        for item in saved_spec["episode_configs"]:
            item["path"] = str((directory / item["path"]).resolve())
        _write_json(output / "configs" / f"repeat-current-{seed}.json", saved_spec)
        trainer = Trainer(train, config, protocol=protocol)
        initial = trainer.online.state_dict()
        if initial.keys() != initials[seed].keys() or any(
            not torch.equal(v.detach().cpu(), initials[seed][k])
            for k, v in initial.items()
        ):
            raise ValueError(
                "La inicialización de la red no coincide exactamente con el control reutilizado"
            )
        folder = output / f"repeat-current-{seed}"
        print(
            f"Entrenamiento repeat_current semilla {seed}: {spec['episodes']} episodios",
            flush=True,
        )
        run_training(
            trainer,
            episodes=spec["episodes"],
            output=folder,
            validation_env=validation,
            validation_seeds=spec["validation_seeds"],
            checkpoint_every=spec["checkpoint_every"],
            validation_every=spec["validation_every"],
        )
        with (folder / "metrics.csv").open() as file:
            trained = list(csv.DictReader(file))
        with (source / f"bilg-{seed}/metrics.csv").open() as file:
            original = list(csv.DictReader(file))
        if [r["scenario_seed"] for r in trained] != [
            r["scenario_seed"] for r in original
        ]:
            raise ValueError("Las secuencias de escenarios no coinciden")
        runs.append(
            {
                "seed": seed,
                "history_mode": config.history_mode,
                "episodes": trainer.completed_episodes,
                "physical_steps": trainer.physical_steps,
                "updates": trainer.updates,
                "parameters": sum(p.numel() for p in trainer.online.parameters()),
                "initial_weights_match": True,
                "control_episodes": len(original),
                "control_physical_steps": int(original[-1]["physical_steps"]),
                "control_optimizer_updates": int(original[-1]["optimizer_updates"]),
            }
        )
        for phase in ("initial", "last"):
            path = folder / f"policy-{phase}.pt"
            policy, _ = load_policy(path, device=config.device)
            target = folder / f"evaluation-{phase}"
            target.mkdir()
            _write_json(
                target / "policy.json",
                {
                    "policy_sha256": _sha256(path),
                    "training": asdict(config),
                    "protocol": protocol,
                },
            )
            for case_index, case in enumerate(spec["validation_seeds"]):
                report = rollout(validation, policy, case_seed=case, gamma=config.gamma)
                pairing = pair_case(controls[seed, phase, case], report)
                report["pairing"] = pairing
                report["routes_m"] = validation._routes
                _write_json(target / f"case-{case}.json", report)
                rows.append(
                    case_row(
                        report,
                        seed=seed,
                        snapshot=phase,
                        history_mode=config.history_mode,
                    )
                )
                if maps and case_index == 0:
                    save_comparison_map(
                        validation,
                        target / "map.png",
                        f"Presente repetido | semilla {seed} | {phase}",
                    )
            print(
                f"Evaluación {phase}, semilla {seed}: {len(spec['validation_seeds'])} casos",
                flush=True,
            )
        _write_json(folder / "run-summary.json", runs[-1])
    _write_json(output / "cases.json", rows)
    result = {
        **summarize(rows, spec["paired_seeds"], spec["validation_seeds"]),
        "training_runs": runs,
    }
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="BiLG con historia frente a presente repetido; reutiliza controles"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe archivos")
    settings = json.loads(args.config.read_text())
    spec_path = (args.config.resolve().parent / settings["experiment_config"]).resolve()
    spec = json.loads(spec_path.read_text())
    config = TrainingConfig(**spec["training"])
    if (
        settings["candidate_history_mode"] != "repeat_current"
        or config.history_mode != "history"
        or config.architecture != "bilg"
        or config.history_length != 4
        or spec["kind"] != "generated"
        or spec["episodes"] != 200
        or spec["paired_seeds"] != [42, 43, 44]
        or spec.get("arrival_strategy", "shortest") != "shortest"
    ):
        parser.error(
            "Se requiere el piloto BiLG de 200 episodios, cuatro observaciones y semillas 42/43/44"
        )
    for name in ("env.py", "routing.py"):
        if (
            _sha256(Path(__file__).parent / name)
            != settings["control_code_sha256"][name]
        ):
            raise ValueError("El entorno o decodificador difiere del control auditado")
    train, validation, protocol = prepare_experiment(
        spec, spec_path.parent, check=args.check
    )
    assert validation is not None
    try:
        check_control(train.domain.grid, train.estimator_config)
        candidates = (
            [args.controls]
            if args.controls
            else sorted(
                CACHE_DIRECTORY.glob("architecture-experiments/*"),
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )
        )
        source = None
        for folder in candidates:
            if not (folder / "summary.json").is_file():
                continue
            try:
                validate_controls(
                    folder, spec, protocol, settings["control_code_sha256"]
                )
            except ValueError, KeyError, OSError:
                if args.controls:
                    raise
                continue
            source = folder
            break
        if source is None:
            raise ValueError(
                "No hay controles compatibles; indicar --controls. No se iniciarán entrenamientos sustitutos"
            )
        if args.check:
            print(
                json.dumps(
                    {
                        "training_executed": False,
                        "simulation_executed": False,
                        "reused_controls": 3,
                        "new_training_episodes": 600,
                        "controls": str(source),
                    },
                    indent=2,
                )
            )
            return
        output = args.output or CACHE_DIRECTORY / "history-experiments" / datetime.now(
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
