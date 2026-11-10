"""Seleccionar un alcance espacial del GP con entradas y trayectorias congeladas."""

import argparse
import hashlib
import io
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import MeasurementRecord

from .arrival_experiment import _write_json
from .evaluate_cli import _compatible
from .gp_diagnostics import diagnose_case, spatial_metrics
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256, prepare_experiment
from .training import TrainingConfig, load_policy

DEFAULT_CONFIG = PROJECT_ROOT / "config/gp-spatial-pilot.json"


def gp_variant(base, scale, scale_key):
    """Cambiar exactamente el factor declarado, sin modificar el historial."""
    if scale_key == "spatial_scale_m":
        if base.x_length_scale_m != base.y_length_scale_m:
            raise ValueError("Esta prueba requiere el GP espacial isotrópico original")
        return replace(base, x_length_scale_m=scale, y_length_scale_m=scale)
    if scale_key == "temporal_scale_s":
        return replace(base, temporal_length_scale_s=scale)
    raise ValueError("Factor GP desconocido")


def scale_unit(scale_key):
    if scale_key not in {"spatial_scale_m", "temporal_scale_s"}:
        raise ValueError("Factor GP desconocido")
    return "m" if scale_key == "spatial_scale_m" else "s"


def temporal_thirds(mse, times):
    """Integrar tres tramos iguales; incluir los extremos con interpolación lineal."""
    times = np.asarray(times, dtype=float)
    boundaries = np.linspace(times[0], times[-1], 4)
    result = {}
    for index, name in enumerate(("first", "middle", "last")):
        low, high = boundaries[index : index + 2]
        interior = np.asarray(times)[(times > low) & (times < high)]
        x = np.concatenate(([low], interior, [high]))
        result[f"{name}_third_mse"] = float(
            np.trapezoid(np.interp(x, times, mse), x) / (high - low)
        )
    return result


def load_frozen_case(path, grid, *, start_utc, end_utc):
    """Validar reloj, grilla, verdad y partición antes de evaluar un GP."""
    report = json.loads((path / "report.json").read_text())
    with np.load(path / "fields-private.npz", allow_pickle=False) as file:
        fields = {key: file[key] for key in file.files}
    if (
        not np.array_equal(fields["active_cells"], np.argwhere(grid.active_mask))
        or not np.array_equal(fields["grid_shape"], grid.shape)
        or not np.allclose(
            fields["grid_bounds_m"], grid.grid_bounds_m, rtol=0, atol=1e-6
        )
    ):
        raise ValueError("La grilla de las mediciones congeladas no coincide")
    records = tuple(
        MeasurementRecord(
            **{**row, "timestamp_utc": datetime.fromisoformat(row["timestamp_utc"])}
        )
        for row in report["measurements"]
    )
    if not records or len({(r.sensor_id, r.timestamp_utc) for r in records}) != len(
        records
    ):
        raise ValueError(
            "Se requieren mediciones sin duplicados para reconstruir el reloj"
        )
    origins = {
        r.timestamp_utc - timedelta(seconds=r.simulation_time_s) for r in records
    }
    if len(origins) != 1:
        raise ValueError("Los tiempos de las mediciones no comparten reloj")
    origin = origins.pop()
    times = fields["elapsed_s"]
    if len(times) < 2 or times[0] != 0 or not np.all(np.diff(times) == 300):
        raise ValueError("Se requiere el horizonte completo en ticks de 300 s")
    if times[-1] != report["signatures"]["horizon_s"]:
        raise ValueError("El horizonte no coincide con el informe")
    if (
        not start_utc <= origin
        or origin + timedelta(seconds=float(times[-1])) > end_utc
    ):
        raise ValueError("El caso no pertenece a la partición temporal solicitada")
    if any(r.simulation_time_s > times[-1] for r in records):
        raise ValueError("Hay mediciones posteriores al horizonte guardado")
    shape = (len(times), int(grid.active_mask.sum()))
    for key in ("truth", "mean_after", "std_after"):
        if fields[key].shape != shape or not np.isfinite(fields[key]).all():
            raise ValueError("Campos congelados incompletos o no finitos")
    if (
        hashlib.sha256(fields["truth"].tobytes()).hexdigest()
        != report["signatures"]["field_sha256"]
    ):
        raise ValueError("El campo guardado no coincide con su huella")
    return report, fields, records, origin


def score_frozen_case(
    grid, updater_config, frozen, *, scale, scale_key, output, control=False
):
    """Reconstruir solo desde lecturas disponibles; la verdad solo puntúa después."""
    report, fields, records, origin = frozen
    config = gp_variant(updater_config.reconstruction_config, scale, scale_key)
    model = GaussianProcessFieldReconstructor(config)
    predictions, deviations, rows = [], [], []
    for index, time in enumerate(fields["elapsed_s"]):
        now = origin + timedelta(seconds=float(time))
        lower = now - updater_config.history_duration
        eligible = tuple(r for r in records if lower <= r.timestamp_utc <= now)
        posterior = model.reconstruct(
            ReconstructionRequest(
                grid=grid,
                reconstruction_timestamp_utc=now,
                measurements=eligible,
            )
        )
        mean = posterior.mean_ug_m3[grid.active_mask]
        std = posterior.standard_deviation_ug_m3[grid.active_mask]
        if control and (
            not np.allclose(mean, fields["mean_after"][index], rtol=1e-7, atol=1e-8)
            or not np.allclose(std, fields["std_after"][index], rtol=1e-7, atol=1e-8)
        ):
            raise ValueError("El GP original no reproduce los posteriores congelados")
        predictions.append(mean)
        deviations.append(std)
        rows.append(
            {
                "elapsed_s": float(time),
                "eligible_records": len(eligible),
                **spatial_metrics(fields["truth"][index], mean, std),
            }
        )
    duration = float(fields["elapsed_s"][-1])
    mse = [r["mse"] for r in rows]
    summary = {
        "case_seed": report["case_seed"],
        scale_key: scale,
        "temporal_mse": float(np.trapezoid(mse, fields["elapsed_s"]) / duration),
        "final_mse": mse[-1],
        "control_reproduced": control,
        **temporal_thirds(mse, fields["elapsed_s"]),
    }
    for key in (
        "coverage_68",
        "coverage_95",
        "mean_variance",
        "spearman_variance_squared_error",
        "top_uncertainty_error_enrichment",
    ):
        values = [r[key] for r in rows[1:] if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "report.json",
        {
            "reconstruction_config": asdict(config),
            "summary": summary,
            "ticks": rows,
            "input_signatures": report["signatures"],
        },
    )
    with (output / "posterior-private.npz").open("xb") as file:
        np.savez_compressed(
            file,
            mean=predictions,
            std=deviations,
            elapsed_s=fields["elapsed_s"],
            active_cells=fields["active_cells"],
        )
    return summary


def select_scale(rows, scales, seeds, baseline, *, scale_key="spatial_scale_m"):
    """Esta función recibe solo resultados de calibración, nunca validación."""
    unit = scale_unit(scale_key)
    indexed = {(r[scale_key], r["case_seed"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {
        (s, c) for s in scales for c in seeds
    }:
        raise ValueError(
            "La selección requiere todos los casos y candidatos sin duplicados"
        )
    scores = {
        s: float(np.mean([indexed[s, c]["temporal_mse"] for c in seeds]))
        for s in scales
    }
    if not all(np.isfinite(v) and v >= 0 for v in scores.values()):
        raise ValueError(
            "Las puntuaciones de calibración deben ser finitas y no negativas"
        )
    minimum = min(scores.values())
    tied = [s for s in scales if np.isclose(scores[s], minimum, rtol=1e-10, atol=1e-12)]
    selected = min(tied, key=lambda s: (abs(np.log(s / baseline)), s))
    return {
        f"selected_scale_{unit}": selected,
        f"original_scale_{unit}": baseline,
        "calibration_scores": [
            {scale_key: s, "mean_case_temporal_mse": scores[s]} for s in scales
        ],
        "criterion": "lowest mean case temporal MSE; numerical ties prefer original or nearest log-scale",
    }


def summarize_validation(
    rows, original, selected, seeds, *, scale_key="spatial_scale_m"
):
    scale_unit(scale_key)
    variants = list(dict.fromkeys((original, selected)))
    index = {(r[scale_key], r["case_seed"]): r for r in rows}
    if len(index) != len(rows) or set(index) != {
        (s, c) for s in variants for c in seeds
    }:
        raise ValueError("La evaluación requiere pares completos")
    summaries = []
    for scale in variants:
        records = [index[scale, seed] for seed in seeds]
        summary = {scale_key: scale}
        for key in (
            "temporal_mse",
            "final_mse",
            "first_third_mse",
            "middle_third_mse",
            "last_third_mse",
            "coverage_68",
            "coverage_95",
            "mean_variance",
            "spearman_variance_squared_error",
            "top_uncertainty_error_enrichment",
        ):
            values = [r[key] for r in records if r[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
        summary["temporal_rmse"] = float(np.sqrt(summary["temporal_mse"]))
        summary["final_rmse"] = float(np.sqrt(summary["final_mse"]))
        for name in ("first", "middle", "last"):
            summary[f"{name}_third_rmse"] = float(np.sqrt(summary[f"{name}_third_mse"]))
        summaries.append(summary)
    return {
        "variants": summaries,
        "paired": [
            {
                "case_seed": seed,
                **{
                    f"selected_minus_original_{key}": index[selected, seed][key]
                    - index[original, seed][key]
                    for key in (
                        "temporal_mse",
                        "final_mse",
                        "first_third_mse",
                        "middle_third_mse",
                        "last_third_mse",
                        "coverage_68",
                        "coverage_95",
                        "mean_variance",
                    )
                },
            }
            for seed in seeds
        ],
    }


def plot_pair(grid, frozen, original_file, selected_file, *, ticks, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report, fields, _, _ = frozen
    with np.load(original_file, allow_pickle=False) as f:
        original = {k: f[k] for k in ("mean", "std")}
    with np.load(selected_file, allow_pickle=False) as f:
        selected = {k: f[k] for k in ("mean", "std")}
    for tick in ticks:
        truth = fields["truth"][tick]
        errors = [np.abs(r["mean"][tick] - truth) for r in (original, selected)]
        values = (
            truth,
            original["mean"][tick],
            selected["mean"][tick],
            errors[0] - errors[1],
            original["std"][tick],
            selected["std"][tick],
            *errors,
        )
        labels = (
            "Verdad sintética privada",
            "Media original",
            "Media seleccionada",
            "Reducción del error absoluto",
            "Desviación original",
            "Desviación seleccionada",
            "Error original",
            "Error seleccionado",
        )
        maximum_mean = max(float(v.max()) for v in values[:3])
        maximum_std = max(float(v.max()) for v in values[4:6])
        maximum_error = max(float(v.max()) for v in errors)
        maxima = (
            maximum_mean,
            maximum_mean,
            maximum_mean,
            float(np.abs(values[3]).max()),
            maximum_std,
            maximum_std,
            maximum_error,
            maximum_error,
        )
        figure = Figure(figsize=(15, 7), layout="constrained")
        FigureCanvasAgg(figure)
        for i, (axis, value, label, maximum) in enumerate(
            zip(figure.subplots(2, 4).ravel(), values, labels, maxima, strict=True)
        ):
            raster = np.full(grid.shape, np.nan)
            raster[grid.active_mask] = value
            maximum = max(maximum, 1e-12)
            image = axis.imshow(
                raster,
                origin="lower",
                extent=grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
                vmin=-maximum if i == 3 else 0,
                vmax=maximum,
                cmap="coolwarm" if i == 3 else "viridis",
            )
            samples = [
                r
                for r in report["measurements"]
                if r["simulation_time_s"] <= fields["elapsed_s"][tick]
            ]
            axis.scatter(
                [r["x_m"] / 1000 for r in samples],
                [r["y_m"] / 1000 for r in samples],
                s=2,
                c="white",
            )
            axis.set(title=label, xlabel="Este UTM (km)", ylabel="Norte UTM (km)")
            figure.colorbar(image, ax=axis, shrink=0.6)
        figure.suptitle(
            f"Caso {report['case_seed']} | tick {tick} | mismas lecturas; µg/m³"
        )
        with (output / f"tick-{tick:02d}.png").open("xb") as file:
            figure.savefig(file, format="png", dpi=110)


def partition_bounds(protocol, partition):
    if partition not in {"calibration", "validation"}:
        raise ValueError("Partición desconocida")
    prefix = "training" if partition == "calibration" else "evaluation"
    periods = {
        (
            item["configuration"][f"{prefix}_start_utc"],
            item["configuration"][f"{prefix}_end_utc"],
        )
        for item in protocol["scenarios"]
    }
    if len(periods) != 1:
        raise ValueError("Los escenarios deben compartir la partición temporal")
    low, high = (datetime.fromisoformat(t) for t in periods.pop())
    limit = datetime.fromisoformat(
        protocol["gp_fit_end_utc"]
        if partition == "calibration"
        else protocol["reserved_final_start_utc"]
    )
    if high > limit:
        raise ValueError("La partición invade datos reservados")
    return {"start_utc": low, "end_utc": high}


def main():
    parser = argparse.ArgumentParser(
        description="Comparar alcances GP en trayectorias congeladas, sin entrenar al agente"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--diagnostic", type=Path, help="Carpeta del diagnóstico GP de validación"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe archivos")
    settings = json.loads(args.config.read_text())
    spec_path = (args.config.resolve().parent / settings["experiment_config"]).resolve()
    spec = json.loads(spec_path.read_text())
    training = TrainingConfig(**spec["training"])
    if (
        spec["kind"] != "generated"
        or training.architecture != "bilg"
        or training.seed != 42
        or spec["episodes"] != 200
        or spec.get("arrival_strategy", "shortest") != "shortest"
    ):
        parser.error(
            "Se requiere el protocolo de BiLG-42, episodio 200, generado y shortest"
        )
    scales = settings["spatial_scales_m"]
    if (
        not scales
        or len(scales) != len(set(scales))
        or any(not np.isfinite(s) or s <= 0 for s in scales)
    ):
        parser.error("Los alcances deben ser positivos, finitos y distintos")
    if (
        type(settings["calibration_cases"]) is not int
        or settings["calibration_cases"] < 1
        or settings["selection_metric"] != "mean_case_temporal_mse"
    ):
        parser.error("Configurar casos positivos y el criterio temporal MSE prefijado")
    if len(settings["map_ticks"]) != len(set(settings["map_ticks"])) or any(
        type(t) is not int or not 1 <= t <= spec["steps_per_episode"]
        for t in settings["map_ticks"]
    ):
        parser.error("Los ticks de mapas deben pertenecer a la misión")
    train, validation, protocol = prepare_experiment(
        spec, spec_path.parent, check=args.check
    )
    assert validation is not None
    try:
        original = train.estimator_config.reconstruction_config
        if original.x_length_scale_m not in scales:
            raise ValueError("La selección debe incluir el GP original")
        for scale in scales:
            gp_variant(original, scale, "spatial_scale_m")
        train_bounds = partition_bounds(protocol, "calibration")
        validation_bounds = partition_bounds(protocol, "validation")
        candidates = (
            [args.diagnostic]
            if args.diagnostic
            else sorted(
                CACHE_DIRECTORY.glob("gp-diagnostics/*"),
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )
        )
        source = None
        for candidate in candidates:
            if not (candidate / "summary.json").is_file():
                continue
            metadata = json.loads((candidate / "config.json").read_text())
            if metadata["protocol"] == json.loads(json.dumps(protocol)) and asdict(
                TrainingConfig(**metadata["training"])
            ) == asdict(training):
                source = candidate
                break
        if source is None:
            raise ValueError("No hay diagnóstico GP compatible; indicar --diagnostic")
        checkpoint = args.checkpoint or Path(metadata["checkpoint"])
        payload = checkpoint.read_bytes()
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if (
            not _compatible(state, training, protocol)
            or state["completed_episodes"] != 200
            or hashlib.sha256(payload).hexdigest() != metadata["policy_sha256"]
        ):
            raise ValueError("Los pesos no corresponden al diagnóstico congelado")
        if metadata["case_seeds"] != spec["validation_seeds"]:
            raise ValueError("Los casos de validación no coinciden")
        # Revisar existencia y reproducción, sin abrir métricas/campos de validación.
        for seed in spec["validation_seeds"]:
            folder = source / f"case-{seed}"
            if not all(
                (folder / name).is_file()
                for name in ("report.json", "fields-private.npz", "reproduction.json")
            ):
                raise ValueError(f"Faltan archivos del caso {seed}")
            if not json.loads((folder / "reproduction.json").read_text())["match"]:
                raise ValueError("El diagnóstico de validación no reprodujo el control")
        seeds = [
            int(np.random.SeedSequence([training.seed, 0, i]).generate_state(1)[0])
            for i in range(settings["calibration_cases"])
        ]
        if set(seeds) & set(spec["validation_seeds"]):
            raise ValueError("Las semillas de calibración y validación deben diferir")
        if args.check:
            print(
                json.dumps(
                    {
                        "simulation_executed": False,
                        "neural_training_executed": False,
                        "new_training_partition_missions": len(seeds),
                        "validation_missions_simulated": 0,
                        "spatial_scales_m": scales,
                        "source": str(source),
                    },
                    indent=2,
                )
            )
            return
        torch.set_num_threads(training.cpu_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        policy, _ = load_policy(io.BytesIO(payload), device=args.device)
        output = (
            args.output
            or CACHE_DIRECTORY
            / "gp-spatial-experiments"
            / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        )
        output.mkdir(parents=True, exist_ok=False)
        _write_json(
            output / "config.json",
            {
                "settings": settings,
                "protocol": protocol,
                "training": asdict(training),
                "torch_version": str(torch.__version__),
                "original_gp": asdict(original),
                "calibration_seeds": seeds,
                "validation_seeds": spec["validation_seeds"],
                "policy_sha256": metadata["policy_sha256"],
                "diagnostic_source": str(source),
                "neural_training_executed": False,
                "selection_partition": "training",
                "code_sha256": _sha256(Path(__file__)),
            },
        )
        print(f"Salida: {output}", flush=True)
        calibration_rows, source_files = [], []
        for index, seed in enumerate(seeds):
            folder = output / "calibration" / f"case-{seed}"
            diagnose_case(train, policy, seed=seed, gamma=training.gamma, output=folder)
            frozen = load_frozen_case(folder, train.domain.grid, **train_bounds)
            for scale in scales:
                calibration_rows.append(
                    score_frozen_case(
                        train.domain.grid,
                        train.estimator_config,
                        frozen,
                        scale=scale,
                        scale_key="spatial_scale_m",
                        control=scale == original.x_length_scale_m,
                        output=folder / f"scale-{scale:g}",
                    )
                )
            print(f"Calibración {index + 1}/{len(seeds)}", flush=True)
        selection = select_scale(
            calibration_rows, scales, seeds, original.x_length_scale_m
        )
        selected = selection["selected_scale_m"]
        _write_json(output / "calibration.json", calibration_rows)
        # Congelar la elección antes de abrir campos o métricas de validación.
        _write_json(
            output / "selection.json",
            {
                **selection,
                "selected_gp": asdict(
                    gp_variant(original, selected, "spatial_scale_m")
                ),
                "selection_partition": "training",
            },
        )
        validation_rows = []
        variants = list(dict.fromkeys((original.x_length_scale_m, selected)))
        for index, seed in enumerate(spec["validation_seeds"]):
            folder = source / f"case-{seed}"
            frozen = load_frozen_case(folder, train.domain.grid, **validation_bounds)
            source_files.append(
                {
                    "case_seed": seed,
                    **{
                        name: _sha256(folder / name)
                        for name in ("report.json", "fields-private.npz")
                    },
                }
            )
            target = output / "validation" / f"case-{seed}"
            for scale in variants:
                validation_rows.append(
                    score_frozen_case(
                        train.domain.grid,
                        train.estimator_config,
                        frozen,
                        scale=scale,
                        scale_key="spatial_scale_m",
                        control=scale == original.x_length_scale_m,
                        output=target / f"scale-{scale:g}",
                    )
                )
            if index == 0:
                plot_pair(
                    train.domain.grid,
                    frozen,
                    target
                    / f"scale-{original.x_length_scale_m:g}/posterior-private.npz",
                    target / f"scale-{selected:g}/posterior-private.npz",
                    ticks=settings["map_ticks"],
                    output=target,
                )
            print(
                f"Validación congelada {index + 1}/{len(spec['validation_seeds'])}",
                flush=True,
            )
        _write_json(output / "validation.json", validation_rows)
        _write_json(output / "source-files.json", source_files)
        _write_json(
            output / "summary.json",
            {
                "selection": selection,
                **summarize_validation(
                    validation_rows,
                    original.x_length_scale_m,
                    selected,
                    spec["validation_seeds"],
                ),
                "scope": "Reconstrucción con entradas congeladas; no demuestra mejora de una política realimentada con el GP candidato.",
            },
        )
        print(f"Alcance seleccionado solo en entrenamiento: {selected:g} m", flush=True)
    finally:
        train.close()
        validation.close()


if __name__ == "__main__":
    main()
