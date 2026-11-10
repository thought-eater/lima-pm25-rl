"""Comparación aislada de llegadas viales con pesos congelados."""

import argparse
import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .comparison import SIGNATURES, run_comparison, save_comparison_map
from .evaluate_cli import _compatible
from .mobility_diagnostics import archive_files, trace_episode
from .train_cli import CACHE_DIRECTORY, DEFAULT_CONFIG, PROJECT_ROOT, prepare_experiment
from .training import TrainingConfig, load_policy

VARIANTS = ("shortest", "continuing")
METRICS = (
    "synthetic_rmse_ug_m3",
    "final_integrated_variance",
    "mean_integrated_variance",
    "distance_m",
    "reward",
)


def _write_json(path, value):
    with path.open("x") as file:
        file.write(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )


def metric_changes(before, after):
    return {
        name: {
            "shortest": float(before[name]),
            "continuing": float(after[name]),
            "change": float(after[name]) - float(before[name]),
            "reduction_percent": 100 * (1 - float(after[name]) / float(before[name]))
            if float(before[name]) > 0
            else None,
        }
        for name in METRICS
    }


def run_case_trial(env, policy, *, case_seed, output, metadata, maps=True):
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            **metadata,
            "stage": "single_case",
            "case_seed": case_seed,
            "training_executed": False,
            "intervention": "arrival_selection_only; existing_policy_weights_frozen",
        },
    )
    reports = {}
    previous = env.arrival_strategy
    try:
        for variant in VARIANTS:
            env.arrival_strategy = variant
            report = trace_episode(
                env, policy, case_seed=case_seed, vehicle=0, complete_horizon=True
            )
            reports[variant] = report
            _write_json(output / f"{variant}.json", report)
            if maps:
                save_comparison_map(
                    env, output / f"{variant}.png", f"{variant} | caso {case_seed}"
                )
    finally:
        env.arrival_strategy = previous
    before, after = (reports[name] for name in VARIANTS)
    if before["signatures"] != after["signatures"]:
        raise RuntimeError(
            "Los campos, posiciones iniciales o fijas difieren entre variantes"
        )
    if before["metrics"]["elapsed_s"] != after["metrics"]["elapsed_s"]:
        raise RuntimeError("Los horizontes de comparación deben coincidir")
    result = {
        "training_executed": False,
        "case_seed": case_seed,
        "signatures_match": True,
        "metric_changes": metric_changes(before["metrics"], after["metrics"]),
        "held_ticks_by_vehicle": {
            name: np.sum(
                [tick["held_vehicles"] for tick in report["ticks"]], axis=0
            ).tolist()
            for name, report in reports.items()
        },
        "first_arrival": {
            name: {
                "assignment": report["first_arrival_audit"]["executed_arrival"],
                "structural_directions_after_tick": report["ticks"][0][
                    "audited_vehicle_structural_directions"
                ],
            }
            if report["first_arrival_audit"]
            else None
            for name, report in reports.items()
        },
        "interpretation": "Prueba con pesos existentes, no rendimiento de una política reentrenada ni evaluación final.",
    }
    _write_json(output / "summary.json", result)
    return result


def compare_outputs(output):
    """Emparejar réplicas por caso y resumir con igual peso por caso."""
    cases, summaries = {}, {}
    for variant in VARIANTS:
        with (output / variant / "cases.csv").open() as file:
            rows = list(csv.DictReader(file))
        indexed = {
            (row["case_seed"], row["strategy"], row["policy_seed"]): row for row in rows
        }
        if len(indexed) != len(rows):
            raise ValueError("Hay réplicas duplicadas en la comparación")
        cases[variant] = indexed
        summaries[variant] = {
            row["strategy"]: row
            for row in json.loads((output / variant / "summary.json").read_text())[
                "strategies"
            ]
        }
    before, after = (cases[name] for name in VARIANTS)
    if before.keys() != after.keys():
        raise ValueError("Las variantes deben contener los mismos casos y semillas")
    if any(
        before[key][name] != after[key][name] for key in before for name in SIGNATURES
    ):
        raise ValueError(
            "La intervención cambió el campo, las fijas, el inicio o el horizonte"
        )
    grouped = {}
    for variant, rows in cases.items():
        grouped[variant] = {}
        for seed, strategy, _ in rows:
            key = seed, strategy
            if key not in grouped[variant]:
                replicas = [r for k, r in rows.items() if k[:2] == key]
                grouped[variant][key] = {
                    metric: float(np.mean([float(r[metric]) for r in replicas]))
                    for metric in (
                        "synthetic_mse",
                        "held_vehicle_ticks",
                        "structural_no_exit_vehicle_ticks",
                        "final_integrated_variance",
                    )
                }
    result = {
        "purpose": "paired_arrival_intervention_on_validation",
        "training_executed": False,
        "signatures_match": True,
        "strategies": {},
    }
    for strategy in summaries["shortest"]:
        keys = [key for key in grouped["shortest"] if key[1] == strategy]
        result["strategies"][strategy] = {
            "metric_changes": metric_changes(
                summaries["shortest"][strategy], summaries["continuing"][strategy]
            ),
            "cases": len(keys),
            "cases_with_lower_rmse": sum(
                grouped["continuing"][key]["synthetic_mse"]
                < grouped["shortest"][key]["synthetic_mse"]
                for key in keys
            ),
            "cases_with_lower_final_variance": sum(
                grouped["continuing"][key]["final_integrated_variance"]
                < grouped["shortest"][key]["final_integrated_variance"]
                for key in keys
            ),
            "mobility": {
                variant: {
                    metric: float(
                        np.mean([grouped[variant][key][metric] for key in keys])
                    )
                    for metric in (
                        "held_vehicle_ticks",
                        "structural_no_exit_vehicle_ticks",
                    )
                }
                for variant in VARIANTS
            },
        }
    return result


def run_full_trial(
    env, policy, *, case_seeds, policy_seeds, output, metadata, maps=True
):
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            **metadata,
            "stage": "full_validation",
            "training_executed": False,
            "case_seeds": case_seeds,
            "policy_seeds": policy_seeds,
            "intervention": "arrival_selection_only; existing_policy_weights_frozen",
        },
    )
    previous = env.arrival_strategy
    try:
        for variant in VARIANTS:
            env.arrival_strategy = variant
            run_comparison(
                env,
                policy,
                case_seeds=case_seeds,
                policy_seeds=policy_seeds,
                output=output / variant,
                metadata={**metadata, "effective_arrival_strategy": variant},
                maps=maps,
            )
    finally:
        env.arrival_strategy = previous
    result = compare_outputs(output)
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comparar dos criterios de llegada sin reentrenar"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--training-archive", type=Path, default=PROJECT_ROOT / "training.tar.zst"
    )
    parser.add_argument("--case", type=int, default=100002)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if spec["kind"] != "generated" or args.case not in spec["validation_seeds"]:
        parser.error("Se requiere un experimento generado y un caso de validación")
    output = args.output or (
        CACHE_DIRECTORY
        / "arrival-experiments"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    )
    if output.exists():
        parser.error("La salida debe ser nueva")
    payload = archive_files(args.training_archive, {"policy-best.pt"})["policy-best.pt"]
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = TrainingConfig(**spec.get("training", {}))
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, env, protocol = prepare_experiment(spec, args.config.resolve().parent)
    train.close()
    assert env is not None
    try:
        if not _compatible(state, config, protocol):
            raise ValueError("La política no coincide con la configuración base")
        if env.arrival_strategy != "shortest":
            raise ValueError("El control de este experimento debe usar shortest")
        policy, _ = load_policy(io.BytesIO(payload), device=args.device)
        metadata = {
            "policy_sha256": hashlib.sha256(payload).hexdigest(),
            "policy_training_protocol": protocol,
            "policy_training_arrival_strategy": "shortest",
            "routing_code_sha256": hashlib.sha256(
                (Path(__file__).parent / "routing.py").read_bytes()
            ).hexdigest(),
            "torch_version": str(torch.__version__),
            "device": str(next(policy.network.parameters()).device),
        }
        print(f"Salida: {output}", flush=True)
        if args.all_cases:
            result = run_full_trial(
                env,
                policy,
                case_seeds=spec["validation_seeds"],
                policy_seeds=spec.get("evaluation_policy_seeds", [0, 1, 2, 3, 4]),
                output=output,
                metadata=metadata,
            )
        else:
            result = run_case_trial(
                env, policy, case_seed=args.case, output=output, metadata=metadata
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
