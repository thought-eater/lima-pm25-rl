"""Intervención GP temporal en inferencia con pesos congelados y rutas libres."""

import argparse
import hashlib
import io
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .architecture_experiment import DEFAULT_CONFIG
from .arrival_experiment import _write_json
from .comparison import _validate_seeds, save_comparison_map
from .evaluate_cli import _compatible
from .gp_diagnostics import verify_reproduction
from .gp_spatial_experiment import gp_variant, temporal_thirds
from .gp_temporal_experiment import check_control
from .train_cli import CACHE_DIRECTORY, _sha256, prepare_experiment
from .training import TrainingConfig, load_policy
from .value_diagnostics import rollout

VARIANTS = (3600.0, 7200.0)
METRICS = (
    "synthetic_mse",
    "temporal_mse",
    "first_third_mse",
    "middle_third_mse",
    "last_third_mse",
    "synthetic_mae_ug_m3",
    "reward",
    "discounted_return",
    "distance_m",
    "final_integrated_variance",
    "mean_integrated_variance",
    "decision_ticks",
    "held_vehicle_ticks",
    "early_termination",
)
ERROR_METRICS = (
    "synthetic_mse",
    "temporal_mse",
    "first_third_mse",
    "middle_third_mse",
    "last_third_mse",
)


def load_offline(source, seeds):
    rows = json.loads((source / "validation.json").read_text())
    indexed = {(r["case_seed"], r["temporal_scale_s"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {
        (s, t) for s in seeds for t in VARIANTS
    }:
        raise ValueError("Falta la comparación congelada completa de gp4")
    result = {}
    for seed in seeds:
        result[seed] = {}
        signatures = None
        for scale in VARIANTS:
            row = indexed[seed, scale]
            report = json.loads(
                (
                    source
                    / "validation"
                    / f"case-{seed}"
                    / f"seconds-{scale:g}"
                    / "report.json"
                ).read_text()
            )
            if report["summary"] != row:
                raise ValueError(
                    "Las métricas congeladas no coinciden con sus reportes por caso"
                )
            if signatures is not None and report["input_signatures"] != signatures:
                raise ValueError("gp4 no conserva las mismas entradas entre variantes")
            signatures = report["input_signatures"]
            result[seed][scale] = {
                key: row["final_mse" if key == "synthetic_mse" else key]
                for key in ERROR_METRICS
            }
        result[seed]["signatures"] = signatures
    return result


def weights_digest(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare_pair(control, candidate):
    if control["signatures"] != candidate["signatures"]:
        raise ValueError("Las variantes no comparten campo, fijas, inicio u horizonte")
    common = min(len(control["ticks"]), len(candidate["ticks"]))
    input_difference = assignment_difference = position_difference = None
    for before, after in zip(control["ticks"], candidate["ticks"]):
        for key in ("sensor_rng_before_sha256", "sensor_rng_after_sha256"):
            if before[key] != after[key]:
                raise ValueError("Las variantes no conservan el flujo de ruido móvil")
        if (
            input_difference is None
            and before["stages"]
            and after["stages"]
            and before["stages"][0]["input_sha256"]
            != after["stages"][0]["input_sha256"]
        ):
            input_difference = before["tick"]
        if assignment_difference is None and before["actions"] != after["actions"]:
            assignment_difference = before["tick"]
        if position_difference is None and not np.allclose(
            before["positions_m"], after["positions_m"], rtol=0, atol=1e-6
        ):
            position_difference = before["tick"]
    return {
        "signatures_match": True,
        "noise_verified_common_decision_ticks": common,
        "first_input_difference_tick": input_difference,
        "first_assignment_difference_tick": assignment_difference,
        "first_position_difference_tick": position_difference,
        "decision_tick_counts": [len(control["ticks"]), len(candidate["ticks"])],
    }


def summarize(rows, pairs, offline=None):
    seeds = sorted({r["case_seed"] for r in rows})
    indexed = {(r["temporal_scale_s"], r["case_seed"]): r for r in rows}
    if (
        not seeds
        or len(indexed) != len(rows)
        or set(indexed) != {(scale, seed) for scale in VARIANTS for seed in seeds}
    ):
        raise ValueError("Se requieren pares completos sin duplicados")
    summaries = []
    for scale in VARIANTS:
        records = [indexed[scale, seed] for seed in seeds]
        summary = {"temporal_scale_s": scale}
        summary.update(
            {key: float(np.mean([r[key] for r in records])) for key in METRICS}
        )
        summary["synthetic_rmse_ug_m3"] = float(np.sqrt(summary["synthetic_mse"]))
        for name in ("temporal", "first_third", "middle_third", "last_third"):
            summary[f"{name}_rmse"] = float(np.sqrt(summary[f"{name}_mse"]))
        summaries.append(summary)
    before, after = summaries
    keys = [key for key in before if key != "temporal_scale_s"]
    paired = []
    for seed in seeds:
        control, candidate = (indexed[scale, seed] for scale in VARIANTS)
        paired.append(
            {
                "case_seed": seed,
                **pairs[seed],
                **{
                    f"candidate_minus_control_{key}": candidate[key] - control[key]
                    for key in METRICS
                },
            }
        )
    result = {
        "training_executed": False,
        "scope": "Intervención de inferencia; no política reentrenada ni evaluación histórica final.",
        "variants": summaries,
        "changes": {
            key: {
                "change": after[key] - before[key],
                "percent_change": 100 * (after[key] / before[key] - 1)
                if before[key] > 0
                else None,
            }
            for key in keys
        },
        "cases_with_lower_temporal_mse": sum(
            r["candidate_minus_control_temporal_mse"] < 0 for r in paired
        ),
        "cases_with_lower_final_mse": sum(
            r["candidate_minus_control_synthetic_mse"] < 0 for r in paired
        ),
        "paired": paired,
    }
    if offline is not None:
        frozen = {
            key: float(np.mean([offline[s][7200.0][key] for s in seeds]))
            for key in ERROR_METRICS
        }
        frozen["synthetic_rmse_ug_m3"] = float(np.sqrt(frozen["synthetic_mse"]))
        for name in ("temporal", "first_third", "middle_third", "last_third"):
            frozen[f"{name}_rmse"] = float(np.sqrt(frozen[f"{name}_mse"]))
        result["frozen_candidate_reference"] = frozen
        result["feedback_vs_frozen_candidate"] = {
            key: {
                "change": after[key] - value,
                "percent_change": 100 * (after[key] / value - 1) if value > 0 else None,
            }
            for key, value in frozen.items()
        }
        for pair in paired:
            seed = pair["case_seed"]
            pair.update(
                {
                    f"feedback_minus_frozen_candidate_{key}": indexed[7200.0, seed][key]
                    - offline[seed][7200.0][key]
                    for key in ERROR_METRICS
                }
            )
    return result


def run_feedback(
    env, policy, *, seeds, gamma, references, output, metadata, maps=True, offline=None
):
    _validate_seeds(seeds)
    original = env.estimator_config
    base = original.reconstruction_config
    if (
        base.temporal_length_scale_s != VARIANTS[0]
        or env.arrival_strategy != "shortest"
    ):
        raise ValueError("El control debe usar GP de 3600 s y shortest")
    output.mkdir(parents=True, exist_ok=False)
    weight_hash = weights_digest(policy.network)
    _write_json(
        output / "config.json",
        {
            **metadata,
            "training_executed": False,
            "case_seeds": seeds,
            "temporal_scales_s": VARIANTS,
            "weights_sha256": weight_hash,
            "original_gp": asdict(base),
            "history_duration_s": original.history_duration.total_seconds(),
            "source_gp_profile_sha256": original.profile_sha256,
            "effective_gp_configs": [
                asdict(gp_variant(base, s, "temporal_scale_s")) for s in VARIANTS
            ],
            "intervention": "GP temporal en inferencia: mapas, decisiones y mediciones posteriores pueden cambiar",
            "reward_interpretation": "Misma fórmula, evaluada con el GP de cada brazo; no es una métrica común de error real.",
        },
    )
    rows, pairs = [], {}
    try:
        for index, seed in enumerate(seeds):
            reports = {}
            for scale in VARIANTS:
                # reset() crea tanto el updater como el reconstructor con este GP.
                # Los identificadores del perfil conservan la procedencia original;
                # effective_gp_configs declara la intervención, no un perfil validado.
                env.estimator_config = replace(
                    original,
                    reconstruction_config=gp_variant(base, scale, "temporal_scale_s"),
                )
                report = rollout(env, policy, case_seed=seed, gamma=gamma)
                folder = output / f"case-{seed}" / f"seconds-{scale:g}"
                folder.mkdir(parents=True, exist_ok=False)
                if scale == VARIANTS[0]:
                    _write_json(
                        folder / "reproduction.json",
                        verify_reproduction(report, references[seed]),
                    )
                series = report["evaluation_series"]
                report["metrics"].update(
                    temporal_thirds(
                        [r["synthetic_mse"] for r in series],
                        np.asarray([r["elapsed_s"] for r in series]),
                    )
                )
                if offline is not None and scale == VARIANTS[0]:
                    for key in (
                        "field_sha256",
                        "initial_positions_sha256",
                        "horizon_s",
                    ):
                        if (
                            report["signatures"][key]
                            != offline[seed]["signatures"][key]
                        ):
                            raise ValueError(
                                "El control difiere de las entradas congeladas de gp4"
                            )
                    if any(
                        not np.isclose(
                            report["metrics"][key], value, rtol=1e-7, atol=1e-8
                        )
                        for key, value in offline[seed][scale].items()
                    ):
                        raise ValueError(
                            "El control difiere de las métricas congeladas de gp4"
                        )
                report["metrics"].update(
                    decision_ticks=len(report["ticks"]),
                    held_vehicle_ticks=sum(
                        sum(r["held_vehicles"]) for r in report["ticks"]
                    ),
                    early_termination=int(
                        report["metrics"]["decision_end_elapsed_s"]
                        < report["signatures"]["horizon_s"]
                    ),
                )
                report["routes_m"] = env._routes
                report["effective_gp"] = asdict(
                    env.estimator_config.reconstruction_config
                )
                _write_json(folder / "report.json", report)
                if maps and index == 0:
                    save_comparison_map(
                        env,
                        folder / "map.png",
                        f"GP {scale:g} s | caso {seed} | pesos congelados",
                    )
                reports[scale] = report
                rows.append(
                    {"case_seed": seed, "temporal_scale_s": scale, **report["metrics"]}
                )
                if weights_digest(policy.network) != weight_hash:
                    raise RuntimeError("Los pesos cambiaron durante la comparación")
                print(
                    f"Caso {seed}, GP {scale:g} s: RMSE={report['metrics']['synthetic_rmse_ug_m3']:.4f}",
                    flush=True,
                )
            pairs[seed] = compare_pair(reports[VARIANTS[0]], reports[VARIANTS[1]])
            _write_json(output / f"case-{seed}" / "pairing.json", pairs[seed])
    finally:
        env.estimator_config = original
    _write_json(output / "cases.json", rows)
    result = summarize(rows, pairs, offline)
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comparar GP temporal con decisiones libres y pesos congelados"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--temporal-experiment", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe resultados")
    spec = json.loads(args.config.read_text())
    config = TrainingConfig(**spec["training"])
    _validate_seeds(spec["validation_seeds"])
    if (
        spec["kind"] != "generated"
        or config.architecture != "bilg"
        or config.seed != 42
        or spec["episodes"] != 200
    ):
        parser.error("Se requiere BiLG-42 del episodio 200 en escenarios generados")
    train, env, protocol = prepare_experiment(
        spec, args.config.resolve().parent, check=args.check
    )
    train.close()
    assert env is not None
    try:
        check_control(env.domain.grid, env.estimator_config)
        if env.arrival_strategy != "shortest":
            raise ValueError("Se requiere el decodificador shortest")
        candidates = (
            [args.temporal_experiment]
            if args.temporal_experiment
            else sorted(
                CACHE_DIRECTORY.glob("gp-temporal-experiments/*"),
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )
        )
        source = None
        for path in candidates:
            if not (path / "summary.json").is_file():
                continue
            saved = json.loads((path / "config.json").read_text())
            if (
                saved["protocol"] == json.loads(json.dumps(protocol))
                and saved["original_gp"]
                == asdict(env.estimator_config.reconstruction_config)
                and saved["validation_seeds"] == spec["validation_seeds"]
                and saved["history_duration_s"]
                == env.estimator_config.history_duration.total_seconds()
                and saved["selection_partition"] == "training"
            ):
                source = path
                break
        if source is None:
            raise ValueError(
                "Falta el experimento temporal compatible; indicar --temporal-experiment"
            )
        selection = json.loads((source / "selection.json").read_text())
        expected_gp = asdict(
            gp_variant(
                env.estimator_config.reconstruction_config, 7200, "temporal_scale_s"
            )
        )
        if (
            selection["selection_partition"] != "training"
            or selection["original_scale_s"] != 3600
            or selection["selected_scale_s"] != 7200
            or selection["selected_gp"] != expected_gp
        ):
            raise ValueError(
                "La intervención acordada requiere 7200 s seleccionados solo en entrenamiento"
            )
        diagnostic = args.diagnostic or Path(saved["diagnostic_source"])
        diagnostic_metadata = json.loads((diagnostic / "config.json").read_text())
        if diagnostic_metadata["policy_sha256"] != saved["policy_sha256"]:
            raise ValueError("La política de origen no coincide")
        checkpoint = args.checkpoint or Path(diagnostic_metadata["checkpoint"])
        payload = checkpoint.read_bytes()
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if (
            not _compatible(state, config, protocol)
            or state["completed_episodes"] != 200
            or hashlib.sha256(payload).hexdigest() != saved["policy_sha256"]
        ):
            raise ValueError("Los pesos no coinciden con el control original")
        reference_dir = checkpoint.parent / "evaluation-last"
        if (
            json.loads((reference_dir / "policy.json").read_text())["policy_sha256"]
            != saved["policy_sha256"]
        ):
            raise ValueError("Las trazas de control no corresponden a estos pesos")
        paths = {
            seed: reference_dir / f"case-{seed}.json"
            for seed in spec["validation_seeds"]
        }
        if not all(p.is_file() for p in paths.values()):
            raise ValueError("Faltan trazas originales de control en evaluation-last")
        offline = load_offline(source, spec["validation_seeds"])
        if args.check:
            print(
                json.dumps(
                    {
                        "simulation_executed": False,
                        "training_executed": False,
                        "missions": 2 * len(paths),
                        "temporal_scales_s": VARIANTS,
                        "checkpoint": str(checkpoint),
                    },
                    indent=2,
                )
            )
            return
        torch.set_num_threads(config.cpu_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        policy, _ = load_policy(io.BytesIO(payload), device=args.device)
        output = (
            args.output
            or CACHE_DIRECTORY
            / "gp-feedback-experiments"
            / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        )
        print(f"Salida: {output}", flush=True)
        run_feedback(
            env,
            policy,
            seeds=spec["validation_seeds"],
            gamma=config.gamma,
            references={seed: json.loads(p.read_text()) for seed, p in paths.items()},
            offline=offline,
            output=output,
            metadata={
                "policy_sha256": saved["policy_sha256"],
                "training": asdict(config),
                "policy_training_protocol": protocol,
                "temporal_experiment": str(source),
                "selection_sha256": _sha256(source / "selection.json"),
                "offline_validation_sha256": _sha256(source / "validation.json"),
                "reference_hashes": {
                    str(seed): _sha256(p) for seed, p in paths.items()
                },
                "torch_version": str(torch.__version__),
                "device": str(next(policy.network.parameters()).device),
                "code_sha256": {
                    name: _sha256(Path(__file__).parent / name)
                    for name in (
                        "gp_feedback_experiment.py",
                        "value_diagnostics.py",
                        "env.py",
                        "network.py",
                        "routing.py",
                    )
                },
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
