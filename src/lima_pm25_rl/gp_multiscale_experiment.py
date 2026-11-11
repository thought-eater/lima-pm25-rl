"""Último contraste espacial: kernel 2000 m frente a suma 2000/500 m 50/50."""

import argparse
import hashlib
import inspect
import io
import json
import tarfile
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from lima_pm25.domain import GridDefinition
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionConfig,
    ReconstructionRequest,
)

from .arrival_experiment import _write_json
from .gp_diagnostics import spatial_metrics
from .gp_multiscale import multiscale_kernel
from .gp_spatial_experiment import partition_bounds, validate_frozen_case
from .hotspot_gp_experiment import (
    criteria,
    error_metrics,
    summarize,
    summarize_frames,
)
from .hotspot_gp_experiment import (
    load_source as load_hotspots,
)
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256

ARMS = ("original", "multiscale")
HISTORY_SECONDS = 21600


def _training_member(path):
    """Filtrar antes de leer el contenido: ningún archivo de validación se abre."""
    parts = path.parts
    if "validation" in parts:
        return False
    if path.name == "config.json":
        return True
    if "calibration" not in parts:
        return False
    tail = parts[parts.index("calibration") + 1 :]
    return bool(tail and tail[0].startswith("case-")) and (
        len(tail) == 2
        and tail[1] in {"report.json", "fields-private.npz"}
        or len(tail) == 3
        and tail[1] == "scale-2000"
        and tail[2] in {"report.json", "posterior-private.npz"}
    )


def _load_arrays(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def grid_from_fields(fields, crs):
    """Reconstruir únicamente el raster conservado; no cargar OSM ni otra grilla."""
    shape = tuple(int(v) for v in fields["grid_shape"])
    bounds = np.asarray(fields["grid_bounds_m"], dtype=float)
    cells = fields["active_cells"]
    if (
        len(shape) != 2
        or min(shape) < 1
        or bounds.shape != (4,)
        or cells.ndim != 2
        or cells.shape[1] != 2
        or cells.dtype.kind not in "iu"
        or (cells < 0).any()
        or (cells >= np.asarray(shape)).any()
        or not np.allclose(
            bounds[2:] - bounds[:2], np.asarray(shape[::-1]) * 200, rtol=0, atol=1e-6
        )
    ):
        raise ValueError("Grilla congelada incompatible con celdas de 200 m")
    mask = np.zeros(shape, dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = True
    if not np.array_equal(cells, np.argwhere(mask)):
        raise ValueError("Celdas activas duplicadas o desordenadas")
    return GridDefinition(
        crs=crs,
        domain_bounds_m=bounds,
        grid_bounds_m=bounds,
        cell_size_m=200,
        x_centers_m=bounds[0] + (np.arange(shape[1]) + 0.5) * 200,
        y_centers_m=bounds[1] + (np.arange(shape[0]) + 0.5) * 200,
        active_mask=mask,
    )


def load_training(source):
    if source.is_dir():
        raw = {"config.json": (source / "config.json").read_bytes()}
        seeds = json.loads(raw["config.json"])["calibration_seeds"]
        for seed in seeds:
            for name in (
                "report.json",
                "fields-private.npz",
                "scale-2000/report.json",
                "scale-2000/posterior-private.npz",
            ):
                path = f"calibration/case-{seed}/{name}"
                raw[path] = (source / path).read_bytes()
    else:
        raw = {}
        with tarfile.open(source, "r|zst") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if member.isfile() and _training_member(path):
                    if path.is_absolute() or ".." in path.parts or str(path) in raw:
                        raise ValueError("Ruta de entrada inválida o duplicada")
                    raw[str(path)] = archive.extractfile(member).read()
        roots = [
            PurePosixPath(p).parent
            for p in raw
            if PurePosixPath(p).name == "config.json"
        ]
        if len(roots) != 1:
            raise ValueError("gp3 debe contener una única ejecución")
        raw = {
            str(PurePosixPath(p).relative_to(roots[0])): value
            for p, value in raw.items()
        }
    config = json.loads(raw["config.json"])
    gp = ReconstructionConfig(**config["original_gp"])
    multiscale_kernel(gp)  # Validar el control sin ajustar ni predecir.
    seeds = config["calibration_seeds"]
    if (
        config["selection_partition"] != "training"
        or config["neural_training_executed"]
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != config["settings"]["calibration_cases"]
        or set(seeds) & set(config["validation_seeds"])
    ):
        raise ValueError(
            "Casos de entrenamiento incompletos, duplicados o fuera de partición"
        )
    bounds = partition_bounds(config["protocol"], "calibration")
    expected = {"config.json"}
    cases = {}
    grid_reference = None
    for seed in seeds:
        folder = f"calibration/case-{seed}"
        names = [
            f"{folder}/{name}"
            for name in (
                "report.json",
                "fields-private.npz",
                "scale-2000/report.json",
                "scale-2000/posterior-private.npz",
            )
        ]
        expected.update(names)
        report = json.loads(raw[names[0]])
        fields = _load_arrays(raw[names[1]])
        crs_values = {r["crs"] for r in report["measurements"]}
        if len(crs_values) != 1:
            raise ValueError("Las lecturas deben compartir un CRS métrico")
        grid = grid_from_fields(fields, crs_values.pop())
        if list(grid.shape) != config["protocol"]["grid_shape"]:
            raise ValueError("La grilla difiere del protocolo")
        if grid_reference is not None and (
            not np.array_equal(grid.active_mask, grid_reference.active_mask)
            or not np.array_equal(grid.grid_bounds_m, grid_reference.grid_bounds_m)
            or grid.crs != grid_reference.crs
        ):
            raise ValueError("Los casos de entrenamiento no comparten dominio")
        grid_reference = grid
        frozen = validate_frozen_case(report, fields, grid, **bounds)
        reference = json.loads(raw[names[2]])
        posterior = _load_arrays(raw[names[3]])
        if (
            report["case_seed"] != seed
            or reference["summary"]["case_seed"] != seed
            or reference["summary"]["spatial_scale_m"] != 2000
            or reference["reconstruction_config"] != asdict(gp)
            or reference["input_signatures"] != report["signatures"]
            or not reference["summary"]["control_reproduced"]
            or not np.array_equal(posterior["elapsed_s"], fields["elapsed_s"])
            or not np.array_equal(posterior["active_cells"], fields["active_cells"])
            or (fields["truth"] < 0).any()
            or (fields["std_after"] < 0).any()
        ):
            raise ValueError("Referencia original incompatible con el caso congelado")
        for name in ("mean", "std"):
            if posterior[name].shape != fields[
                f"{name}_after"
            ].shape or not np.allclose(
                posterior[name], fields[f"{name}_after"], rtol=1e-7, atol=1e-8
            ):
                raise ValueError("El posterior original guardado no coincide con gp3")
        cases[seed] = grid, frozen, reference
    if set(raw) != expected:
        raise ValueError(
            "La fuente incluye casos de calibración distintos de los declarados"
        )
    return config, cases, raw


def reconstruct(grid, config, records, origin, times, arm):
    if arm not in ARMS:
        raise ValueError("Brazo desconocido")
    kernel = multiscale_kernel(config) if arm == "multiscale" else None
    model = GaussianProcessFieldReconstructor(config, kernel=kernel)
    means, stds, counts = [], [], []
    for time in times:
        now = origin + timedelta(seconds=float(time))
        available = tuple(
            r
            for r in records
            if now - timedelta(seconds=HISTORY_SECONDS) <= r.timestamp_utc <= now
        )
        posterior = model.reconstruct(
            ReconstructionRequest(
                grid=grid, reconstruction_timestamp_utc=now, measurements=available
            )
        )
        means.append(posterior.mean_ug_m3[grid.active_mask])
        stds.append(posterior.standard_deviation_ug_m3[grid.active_mask])
        counts.append(posterior.observations.used_record_count)
    return {"mean": np.asarray(means), "std": np.asarray(stds)}, counts


def score_case(grid, config, frozen, output, *, suite, reference=None):
    report, fields, records, origin = frozen
    times = (
        fields["elapsed_s"]
        if suite == "training"
        else np.asarray([r["elapsed_s"] for r in report["frames"]])
    )
    gp = ReconstructionConfig(
        **(config["original_gp"] if suite == "training" else config["gp"])
    )
    rows, predictions = [], {}
    for arm in ARMS:
        arrays, counts = reconstruct(grid, gp, records, origin, times, arm)
        if arm == "original":
            for key in ("mean", "std"):
                baseline = (
                    fields[f"{key}_after"] if suite == "training" else fields[key]
                )
                if not np.allclose(arrays[key], baseline, rtol=1e-7, atol=1e-8):
                    raise ValueError(
                        "El GP original no reproduce las medias/desviaciones congeladas"
                    )
        frames = []
        for i, time in enumerate(times):
            truth, mean, std = fields["truth"][i], arrays["mean"][i], arrays["std"][i]
            if suite == "training":
                error = mean - truth
                values = {
                    **spatial_metrics(truth, mean, std),
                    "spatial_bias": float(error.mean()),
                    "spatial_bias_mse": float(error.mean() ** 2),
                    "centered_mse": float(np.mean((error - error.mean()) ** 2)),
                    "negative_area_km2": float(
                        (mean < 0).sum() * grid.cell_area_m2 / 1e6
                    ),
                }
            else:
                regions = {
                    k: fields[f"region_{k}"]
                    for k in ("core", "surroundings", "background")
                }
                threshold = (
                    config["settings"]["background_ug_m3"]
                    + config["settings"]["high_excess_threshold_ug_m3"]
                )
                values = {
                    "phase": report["frames"][i]["phase"],
                    **error_metrics(
                        truth, mean, std, regions, threshold, grid.cell_area_m2
                    ),
                }
            frames.append(
                {
                    "tick": i,
                    "elapsed_s": float(time),
                    "eligible_records": counts[i],
                    **values,
                }
            )
        if suite == "training":
            summary = {"case_seed": report["case_seed"], "arm": arm}
            for key in (
                "mse",
                "spatial_bias",
                "spatial_bias_mse",
                "centered_mse",
                "negative_area_km2",
            ):
                summary[f"temporal_{key}"] = float(
                    np.trapezoid([r[key] for r in frames], times) / times[-1]
                )
                summary[f"final_{key}"] = frames[-1][key]
            if arm == "original" and not np.allclose(
                [summary["temporal_mse"], summary["final_mse"]],
                [
                    reference["summary"]["temporal_mse"],
                    reference["summary"]["final_mse"],
                ],
                rtol=1e-7,
                atol=1e-8,
            ):
                raise ValueError("Las métricas originales no reproducen gp3")
        else:
            summary = {
                **{k: report[k] for k in ("regime", "plan", "sensor_seed")},
                "arm": arm,
                **summarize_frames(frames),
            }
            if arm == "original" and any(
                not np.isclose(summary[k], v, rtol=1e-7, atol=1e-8)
                for k, v in report["summary"].items()
                if k in summary and isinstance(v, (int, float))
            ):
                raise ValueError(
                    "Las métricas originales no reproducen el control de focos"
                )
        folder = output / arm
        folder.mkdir(parents=True, exist_ok=False)
        _write_json(
            folder / "report.json",
            {
                "summary": summary,
                "frames": frames,
                "original_gp": asdict(gp),
                "kernel": str(multiscale_kernel(gp))
                if arm == "multiscale"
                else "Matérn 2000/2000/3600, nu=1.5, varianza original",
                "history_duration_s": HISTORY_SECONDS,
                "control_reproduced": arm == "original",
                "input_signatures": report["signatures"],
            },
        )
        with (folder / "posterior-private.npz").open("xb") as file:
            np.savez_compressed(
                file, **arrays, elapsed_s=times, active_cells=fields["active_cells"]
            )
        rows.append(summary)
        predictions[arm] = arrays
    return rows, predictions


def training_summary(rows, seeds):
    indexed = {(r["case_seed"], r["arm"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {
        (s, a) for s in seeds for a in ARMS
    }:
        raise ValueError("Se requieren todos los pares de entrenamiento")
    groups = []
    metrics = [k for k in rows[0] if k not in {"arm", "case_seed"}]
    for arm in ARMS:
        group = {"arm": arm}
        for key in metrics:
            group[key] = float(np.mean([indexed[s, arm][key] for s in seeds]))
            if key.endswith("mse"):
                group[key[:-3] + "rmse"] = float(np.sqrt(group[key]))
        groups.append(group)
    a, b = groups
    return {
        "summaries": groups,
        "lower_temporal_mse": b["temporal_mse"] < a["temporal_mse"],
        "final_mse_not_higher": b["final_mse"] <= a["final_mse"],
        "temporal_cases_improved": sum(
            indexed[s, "multiscale"]["temporal_mse"]
            < indexed[s, "original"]["temporal_mse"]
            for s in seeds
        ),
        "final_cases_not_worse": sum(
            indexed[s, "multiscale"]["final_mse"] <= indexed[s, "original"]["final_mse"]
            for s in seeds
        ),
        "paired": [
            {
                "case_seed": s,
                "candidate_minus_original": {
                    k: indexed[s, "multiscale"][k] - indexed[s, "original"][k]
                    for k in metrics
                },
            }
            for s in seeds
        ],
    }


def hotspot_criteria(original, candidate):
    values = criteria(original, candidate)
    values["lower_false_high_area"] = (
        candidate["temporal_false_high_area_km2"]
        < original["temporal_false_high_area_km2"]
    )
    values["hotspot_pass"] = values["primary_pass"] and values["lower_false_high_area"]
    return values


def plot_case(grid, frozen, predictions, ticks, output, title):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report, fields, records, _ = frozen
    vmax = max(
        float(fields["truth"].max()),
        *[float(p["mean"].max()) for p in predictions.values()],
        1,
    )
    vmin = min(0, *[float(p["mean"].min()) for p in predictions.values()])
    stdmax = max(float(p["std"].max()) for p in predictions.values())
    errmax = max(
        float(abs(p["mean"] - fields["truth"]).max()) for p in predictions.values()
    )
    for tick in ticks:
        truth = fields["truth"][tick]
        a, b = (predictions[arm] for arm in ARMS)
        values = (
            truth,
            a["mean"][tick],
            b["mean"][tick],
            b["mean"][tick] - a["mean"][tick],
            a["std"][tick],
            b["std"][tick],
            abs(a["mean"][tick] - truth),
            abs(b["mean"][tick] - truth),
        )
        labels = (
            "Verdad privada",
            "GP 2000 m",
            "GP 2000+500 m",
            "Media candidato − original",
            "Desviación original",
            "Desviación candidato",
            "Error original",
            "Error candidato",
        )
        fig = Figure(figsize=(15, 7), layout="constrained")
        FigureCanvasAgg(fig)
        for i, (ax, value, label) in enumerate(
            zip(fig.subplots(2, 4).ravel(), values, labels, strict=True)
        ):
            raster = np.full(grid.shape, np.nan)
            raster[grid.active_mask] = value
            low, high = (
                (vmin, vmax)
                if i < 3
                else (-vmax, vmax)
                if i == 3
                else (0, stdmax)
                if i < 6
                else (0, max(errmax, 1e-12))
            )
            image = ax.imshow(
                raster,
                origin="lower",
                extent=grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
                vmin=low,
                vmax=high,
                cmap="coolwarm" if i == 3 else "viridis",
            )
            for vehicle, route in enumerate(report["routes_m"]):
                positions = (
                    np.asarray(
                        [route[0]]
                        + [r["positions_m"][vehicle] for r in report["ticks"][:tick]]
                    )
                    / 1000
                )
                ax.plot(positions[:, 0], positions[:, 1], "w-", linewidth=0.5)
            samples = [r for r in records if r.simulation_time_s <= tick * 300]
            ax.scatter(
                [r.x_m / 1000 for r in samples],
                [r.y_m / 1000 for r in samples],
                c="red",
                s=3,
            )
            ax.set(title=label, xlabel="Este (km)", ylabel="Norte (km)")
            fig.colorbar(image, ax=ax, shrink=0.7)
        fig.suptitle(f"{title} | tick {tick}; µg/m³")
        with (output / f"tick-{tick:02d}.png").open("xb") as file:
            fig.savefig(file, format="png", dpi=110)


def run_comparison(training, hotspots, sources, output, *, maps=True):
    train_config, train_cases, train_raw = training
    hot_config, hot_grid, hot_cases, hot_raw = hotspots
    if train_config["original_gp"] != hot_config["gp"]:
        raise ValueError("Ambas baterías deben usar el mismo GP original")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            "sources": {k: str(v) for k, v in sources.items()},
            "training_source_config": train_config,
            "hotspot_source_config": hot_config,
            "arms": ARMS,
            "spatial_scales_m": [2000, 500],
            "variance_fractions": [0.5, 0.5],
            "matern_nu": 1.5,
            "temporal_scale_s": 3600,
            "history_duration_s": HISTORY_SECONDS,
            "simulation_executed": False,
            "neural_training_executed": False,
            "validation_opened": False,
            "core_reconstruction_sha256": _sha256(
                Path(inspect.getfile(GaussianProcessFieldReconstructor))
            ),
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "gp_multiscale.py",
                    "gp_multiscale_experiment.py",
                    "gp_spatial_experiment.py",
                    "hotspot_gp_experiment.py",
                )
            },
            "criterion": "Menor MSE temporal de entrenamiento sin empeorar final; en cada régimen/plan de focos, menor error exterior y área falsamente alta sin aumentar núcleo ni pico temporales. Sin búsqueda posterior de repartos.",
        },
    )
    manifest = {}
    for suite, raw in (("training", train_raw), ("hotspots", hot_raw)):
        for name, data in raw.items():
            relative = f"{suite}/{name}"
            path = output / "input" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as file:
                file.write(data)
            manifest[relative] = hashlib.sha256(data).hexdigest()
    _write_json(output / "source-files.json", manifest)
    training_rows, hotspot_rows = [], []
    for seed, (grid, frozen, reference) in train_cases.items():
        folder = output / "training" / f"case-{seed}"
        rows, predictions = score_case(
            grid, train_config, frozen, folder, suite="training", reference=reference
        )
        training_rows.extend(rows)
        if maps and seed == train_config["calibration_seeds"][0]:
            plot_case(
                grid,
                frozen,
                predictions,
                train_config["settings"]["map_ticks"],
                folder,
                f"Entrenamiento {seed}",
            )
        print(f"Entrenamiento congelado {seed}: original/multiescala", flush=True)
    for (regime, seed, plan), frozen in hot_cases.items():
        folder = output / "hotspots" / regime / f"noise-{seed}" / plan
        rows, predictions = score_case(
            hot_grid, hot_config, frozen, folder, suite="hotspots"
        )
        hotspot_rows.extend(rows)
        if maps and seed == hot_config["settings"]["sensor_seeds"][0]:
            plot_case(
                hot_grid,
                frozen,
                predictions,
                hot_config["settings"]["map_ticks"],
                folder,
                f"{regime} | {plan} | ruido {seed}",
            )
        print(
            f"Foco {regime} | ruido {seed} | {plan}: original/multiescala", flush=True
        )
    _write_json(output / "training.json", training_rows)
    _write_json(output / "hotspots.json", hotspot_rows)
    training_result = training_summary(training_rows, train_config["calibration_seeds"])
    hotspot_result = summarize(
        hotspot_rows,
        hot_config["settings"]["sensor_seeds"],
        scales=ARMS,
        scale_key="arm",
        criteria_fn=hotspot_criteria,
    )
    result = {
        "training": training_result,
        "hotspots": hotspot_result,
        "advance_criterion_pass": training_result["lower_temporal_mse"]
        and training_result["final_mse_not_higher"]
        and all(
            r["hotspot_pass"] for r in hotspot_result["criteria_by_regime_and_plan"]
        ),
        "all_controls_reproduced": True,
        "scope": "Diagnóstico sobre entrenamiento simulado y control artificial ya examinados; no validación final, política realimentada ni adopción automática.",
    }
    _write_json(output / "summary.json", result)
    return result


def resolve_source(explicit, archive, cache, parser):
    if explicit is not None:
        return explicit
    path = PROJECT_ROOT / archive
    if path.is_file():
        return path
    candidates = sorted((CACHE_DIRECTORY / cache).glob("*/config.json"))
    if not candidates:
        parser.error(f"Falta {archive} o una ejecución en cache/{cache}/")
    return candidates[-1].parent


def main():
    parser = argparse.ArgumentParser(
        description="Comparar GP original y aditivo 2000/500 m sobre gp3 y focos"
    )
    parser.add_argument("--training-source", type=Path)
    parser.add_argument("--hotspot-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe salidas")
    sources = {
        "training": resolve_source(
            args.training_source, "gp3.tar.zst", "gp-spatial-experiments", parser
        ),
        "hotspots": resolve_source(
            args.hotspot_source, "hotspot.tar.zst", "hotspot-sampling", parser
        ),
    }
    training = load_training(sources["training"])
    hotspots = load_hotspots(sources["hotspots"])
    if training[0]["original_gp"] != hotspots[0]["gp"]:
        parser.error("Los GP originales de las dos fuentes deben coincidir")
    # También detecta un core desactualizado antes de iniciar la batería.
    gp = ReconstructionConfig(**training[0]["original_gp"])
    GaussianProcessFieldReconstructor(gp, kernel=multiscale_kernel(gp))
    if args.check:
        print(
            json.dumps(
                {
                    "sources": {k: str(v) for k, v in sources.items()},
                    "training_cases": len(training[1]),
                    "hotspot_missions": len(hotspots[2]),
                    "reconstructions": 2 * (len(training[1]) + len(hotspots[2])),
                    "spatial_scales_m": [2000, 500],
                    "variance_fractions": [0.5, 0.5],
                    "temporal_scale_s": 3600,
                    "history_duration_s": HISTORY_SECONDS,
                    "gp_reconstruction_executed": False,
                    "validation_opened": False,
                    "simulation_executed": False,
                    "training_executed": False,
                },
                indent=2,
            )
        )
        return
    torch.set_num_threads(1)
    output = (
        args.output
        or CACHE_DIRECTORY
        / "gp-multiscale-experiments"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    )
    print(f"Salida: {output}", flush=True)
    run_comparison(training, hotspots, sources, output)


if __name__ == "__main__":
    main()
