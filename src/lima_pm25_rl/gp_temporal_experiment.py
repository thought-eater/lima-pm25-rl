"""Ablación temporal del GP sobre datos guardados, sin simular ni cargar políticas."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .arrival_experiment import _write_json
from .gp_spatial_experiment import (
    gp_variant,
    load_frozen_case,
    partition_bounds,
    plot_pair,
    score_frozen_case,
    select_scale,
    summarize_validation,
)
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256, prepare_experiment
from .training import TrainingConfig

DEFAULT_CONFIG = PROJECT_ROOT / "config/gp-temporal-pilot.json"
SCALE_KEY = "temporal_scale_s"


def frozen_files(folder):
    names = ("report.json", "fields-private.npz")
    if not all((folder / name).is_file() for name in names):
        raise ValueError(f"Faltan datos congelados en {folder}")
    return {name: _sha256(folder / name) for name in names}


def check_control(grid, updater):
    gp = updater.reconstruction_config
    if (
        grid.cell_size_m != 200
        or gp.x_length_scale_m != 2000
        or gp.y_length_scale_m != 2000
        or gp.temporal_length_scale_s != 3600
        or updater.history_duration.total_seconds() != 21600
    ):
        raise ValueError(
            "El control requiere grilla 200 m, espacio 2000 m, tiempo 3600 s e historial 6 h"
        )


def run_comparison(grid, updater, *, settings, source, diagnostic, metadata, output):
    """Seleccionar en calibración antes de leer posteriors/verdad de validación."""
    check_control(grid, updater)
    original = updater.reconstruction_config
    baseline = original.temporal_length_scale_s
    scales = settings["temporal_scales_s"]
    calibration_seeds = metadata["calibration_seeds"]
    validation_seeds = metadata["validation_seeds"]
    protocol = metadata["protocol"]
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            "settings": settings,
            "protocol": protocol,
            "original_gp": asdict(original),
            "history_duration_s": updater.history_duration.total_seconds(),
            "spatial_source": str(source),
            "diagnostic_source": str(diagnostic),
            "calibration_seeds": calibration_seeds,
            "validation_seeds": validation_seeds,
            "policy_sha256": metadata["policy_sha256"],
            "simulation_executed": False,
            "neural_training_executed": False,
            "selection_partition": "training",
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "gp_temporal_experiment.py",
                    "gp_spatial_experiment.py",
                    "gp_diagnostics.py",
                )
            },
            "source_config_sha256": _sha256(source / "config.json"),
            "diagnostic_config_sha256": _sha256(diagnostic / "config.json"),
        },
    )
    calibration_rows, manifest = [], []
    bounds = partition_bounds(protocol, "calibration")
    for index, seed in enumerate(calibration_seeds):
        folder = source / "calibration" / f"case-{seed}"
        frozen = load_frozen_case(folder, grid, **bounds)
        if frozen[0]["case_seed"] != seed:
            raise ValueError(
                "La semilla del archivo no coincide con el caso solicitado"
            )
        manifest.append(
            {"partition": "calibration", "case_seed": seed, **frozen_files(folder)}
        )
        for scale in scales:
            calibration_rows.append(
                score_frozen_case(
                    grid,
                    updater,
                    frozen,
                    scale=scale,
                    scale_key=SCALE_KEY,
                    control=scale == baseline,
                    output=output
                    / "calibration"
                    / f"case-{seed}"
                    / f"seconds-{scale:g}",
                )
            )
        print(f"Calibración congelada {index + 1}/{len(calibration_seeds)}", flush=True)
    selection = select_scale(
        calibration_rows, scales, calibration_seeds, baseline, scale_key=SCALE_KEY
    )
    selected = selection["selected_scale_s"]
    _write_json(output / "calibration.json", calibration_rows)
    _write_json(
        output / "selection.json",
        {
            **selection,
            "selected_gp": asdict(gp_variant(original, selected, SCALE_KEY)),
            "selection_partition": "training",
        },
    )
    # El manifiesto enlaza exactamente la validación usada en el experimento espacial.
    expected = json.loads((source / "source-files.json").read_text())
    hashes = {row["case_seed"]: row for row in expected}
    if len(hashes) != len(expected) or set(hashes) != set(validation_seeds):
        raise ValueError("El manifiesto de validación no contiene los casos prefijados")
    validation_rows = []
    bounds = partition_bounds(protocol, "validation")
    for index, seed in enumerate(validation_seeds):
        folder = diagnostic / f"case-{seed}"
        actual = frozen_files(folder)
        if any(value != hashes[seed][name] for name, value in actual.items()):
            raise ValueError(
                "Las lecturas o campos de validación difieren del experimento espacial"
            )
        manifest.append({"partition": "validation", "case_seed": seed, **actual})
        frozen = load_frozen_case(folder, grid, **bounds)
        if frozen[0]["case_seed"] != seed:
            raise ValueError(
                "La semilla del archivo no coincide con el caso solicitado"
            )
        target = output / "validation" / f"case-{seed}"
        for scale in dict.fromkeys((baseline, selected)):
            validation_rows.append(
                score_frozen_case(
                    grid,
                    updater,
                    frozen,
                    scale=scale,
                    scale_key=SCALE_KEY,
                    control=scale == baseline,
                    output=target / f"seconds-{scale:g}",
                )
            )
        if index == 0:
            plot_pair(
                grid,
                frozen,
                target / f"seconds-{baseline:g}/posterior-private.npz",
                target / f"seconds-{selected:g}/posterior-private.npz",
                ticks=settings["map_ticks"],
                output=target,
            )
        print(f"Validación congelada {index + 1}/{len(validation_seeds)}", flush=True)
    _write_json(output / "validation.json", validation_rows)
    _write_json(output / "source-files.json", manifest)
    result = {
        "selection": selection,
        **summarize_validation(
            validation_rows, baseline, selected, validation_seeds, scale_key=SCALE_KEY
        ),
        "scope": "Cambio solo de correlación temporal; mismos datos e historial. No evalúa una política realimentada con el GP candidato.",
    }
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comparar escalas temporales GP sin generar misiones"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spatial-experiment", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe archivos")
    settings = json.loads(args.config.read_text())
    spec_path = (args.config.resolve().parent / settings["experiment_config"]).resolve()
    spec = json.loads(spec_path.read_text())
    config = TrainingConfig(**spec["training"])
    scales = settings["temporal_scales_s"]
    if (
        not scales
        or 3600 not in scales
        or len(set(scales)) != len(scales)
        or any(
            type(s) not in (int, float) or not np.isfinite(s) or s <= 0 for s in scales
        )
        or type(settings["calibration_cases"]) is not int
        or settings["calibration_cases"] < 1
        or settings["selection_metric"] != "mean_case_temporal_mse"
    ):
        parser.error(
            "Configurar escalas positivas distintas con control 3600 s, casos positivos y criterio MSE temporal"
        )
    if (
        spec["kind"] != "generated"
        or config.architecture != "bilg"
        or config.seed != 42
        or spec["episodes"] != 200
        or spec.get("arrival_strategy", "shortest") != "shortest"
    ):
        parser.error(
            "Se requieren los datos del protocolo BiLG-42 episodio 200 con shortest"
        )
    if len(set(settings["map_ticks"])) != len(settings["map_ticks"]) or any(
        type(t) is not int or not 1 <= t <= spec["steps_per_episode"]
        for t in settings["map_ticks"]
    ):
        parser.error("Ticks de mapas inválidos")
    # Solo recursos y contratos: no reset, composición de episodios ni step.
    train, validation, protocol = prepare_experiment(spec, spec_path.parent)
    assert validation is not None
    try:
        check_control(train.domain.grid, train.estimator_config)
        seeds = [
            int(np.random.SeedSequence([42, 0, i]).generate_state(1)[0])
            for i in range(settings["calibration_cases"])
        ]
        candidates = (
            [args.spatial_experiment]
            if args.spatial_experiment
            else sorted(
                CACHE_DIRECTORY.glob("gp-spatial-experiments/*"),
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )
        )
        source = None
        for path in candidates:
            if not (path / "summary.json").is_file():
                continue
            metadata = json.loads((path / "config.json").read_text())
            if (
                metadata["protocol"] == json.loads(json.dumps(protocol))
                and metadata["training"] == asdict(config)
                and metadata["original_gp"]
                == asdict(train.estimator_config.reconstruction_config)
                and metadata["calibration_seeds"] == seeds
                and metadata["validation_seeds"] == spec["validation_seeds"]
                and metadata["selection_partition"] == "training"
            ):
                source = path
                break
        if source is None:
            raise ValueError(
                "Falta el experimento espacial compatible; indicar --spatial-experiment"
            )
        diagnostic = args.diagnostic or Path(metadata["diagnostic_source"])
        info = json.loads((diagnostic / "config.json").read_text())
        if (
            info["protocol"] != metadata["protocol"]
            or info["policy_sha256"] != metadata["policy_sha256"]
            or info["case_seeds"] != metadata["validation_seeds"]
        ):
            raise ValueError("El diagnóstico no corresponde a los datos congelados")
        if set(seeds) & set(spec["validation_seeds"]):
            raise ValueError("Las semillas de calibración y validación deben diferir")
        partition_bounds(protocol, "calibration")
        partition_bounds(protocol, "validation")
        for root, cases in (
            (source / "calibration", seeds),
            (diagnostic, spec["validation_seeds"]),
        ):
            for seed in cases:
                if not all(
                    (root / f"case-{seed}" / name).is_file()
                    for name in ("report.json", "fields-private.npz")
                ):
                    raise ValueError(f"Faltan datos congelados del caso {seed}")
        if not (source / "source-files.json").is_file():
            raise ValueError("Falta el manifiesto de la validación espacial")
        if args.check:
            print(
                json.dumps(
                    {
                        "simulation_executed": False,
                        "neural_training_executed": False,
                        "new_missions": 0,
                        "calibration_cases": len(seeds),
                        "validation_cases": len(spec["validation_seeds"]),
                        "temporal_scales_s": scales,
                        "spatial_source": str(source),
                    },
                    indent=2,
                )
            )
            return
        torch.set_num_threads(config.cpu_threads)
        output = (
            args.output
            or CACHE_DIRECTORY
            / "gp-temporal-experiments"
            / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        )
        print(f"Salida: {output}", flush=True)
        result = run_comparison(
            train.domain.grid,
            train.estimator_config,
            settings=settings,
            source=source,
            diagnostic=diagnostic,
            metadata=metadata,
            output=output,
        )
        print(
            f"Escala temporal seleccionada en entrenamiento: {result['selection']['selected_scale_s']:g} s"
        )
    finally:
        train.close()
        validation.close()


if __name__ == "__main__":
    main()
