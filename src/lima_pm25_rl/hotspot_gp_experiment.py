"""Comparar 2000/500 m sobre lecturas congeladas del control de focos."""

import argparse
import hashlib
import io
import json
import tarfile
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from lima_pm25.domain import build_grid
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionConfig,
    ReconstructionRequest,
)
from lima_pm25.simulation import MeasurementRecord, SensorType
from shapely.geometry import box

from .arrival_experiment import _write_json
from .gp_spatial_experiment import gp_variant
from .hotspot_sampling import PLANS, REGIMES, frame_metrics, validate_settings
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256

SCALES = (2000, 500)
METRICS = (
    "global_mse",
    "core_mse",
    "surroundings_mse",
    "background_mse",
    "peak_abs_error",
    "peak_bias",
    "false_high_area_km2",
    "background_overestimation_mse",
    "background_underestimation_mse",
    "negative_area_km2",
)


def read_source_files(source, names):
    """Leer archivos seleccionados de una ejecución, sin extraer el tar al disco."""
    if source.is_dir():
        raw = {
            p.relative_to(source).as_posix(): p.read_bytes()
            for p in source.rglob("*")
            if p.is_file() and p.name in names
        }
    else:
        raw = {}
        with tarfile.open(source, "r|zst") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if member.isfile() and path.name in names:
                    if path.is_absolute() or ".." in path.parts or str(path) in raw:
                        raise ValueError(
                            "Ruta o entrada duplicada en el archivo fuente"
                        )
                    raw[str(path)] = archive.extractfile(member).read()
        roots = [PurePosixPath(p).parent for p in raw if p.endswith("config.json")]
        if roots:
            depth = min(len(root.parts) for root in roots)
            roots = [root for root in roots if len(root.parts) == depth]
        if len(roots) != 1:
            raise ValueError("El archivo debe contener exactamente una ejecución")
        raw = {str(PurePosixPath(p).relative_to(roots[0])): b for p, b in raw.items()}
    return raw


def load_source(source):
    names = {"config.json", "route-preview.json", "report.json", "fields-private.npz"}
    return load_frozen_inputs(read_source_files(source, names))


def load_frozen_inputs(raw):
    """Validar las entradas originales de muestreo, también dentro de hot2."""
    config = json.loads(raw["config.json"])
    spec = config["settings"]
    validate_settings(spec)
    gp = ReconstructionConfig(**config["gp"])
    if (
        config["purpose"] != "controlled_mechanism_not_lima_validation"
        or config["training_executed"]
        or gp.x_length_scale_m != 2000
        or gp.y_length_scale_m != 2000
        or gp.temporal_length_scale_s != 3600
        or config["history_duration_s"] != 21600
    ):
        raise ValueError("Se requiere el control de focos con GP 2000 m/3600 s/6 h")
    width = spec["grid_cells"] * spec["cell_size_m"]
    grid = build_grid(box(0, 0, width, width), "EPSG:32718", spec["cell_size_m"])
    cases = {}
    expected = {"config.json", "route-preview.json"}
    for regime in REGIMES:
        for seed in spec["sensor_seeds"]:
            for plan in PLANS:
                name = f"{regime}/noise-{seed}/{plan}"
                report_name, field_name = (
                    f"{name}/report.json",
                    f"{name}/fields-private.npz",
                )
                expected.update((report_name, field_name))
                report = json.loads(raw[report_name])
                if (report["regime"], report["sensor_seed"], report["plan"]) != (
                    regime,
                    seed,
                    plan,
                ):
                    raise ValueError("La identidad del informe no coincide con su ruta")
                with np.load(io.BytesIO(raw[field_name]), allow_pickle=False) as file:
                    fields = {key: file[key] for key in file.files}
                records, origin = validate_case(grid, spec, report, fields)
                cases[regime, seed, plan] = (report, fields, records, origin)
    if set(raw) != expected:
        raise ValueError("La fuente contiene casos adicionales o incompletos")
    validate_pairs(cases, spec, json.loads(raw["route-preview.json"]))
    return config, grid, cases, raw


def validate_case(grid, spec, report, fields):
    steps = spec["steps"]
    cells = np.argwhere(grid.active_mask)
    if (
        not np.array_equal(fields["active_cells"], cells)
        or not np.array_equal(fields["grid_shape"], grid.shape)
        or not np.array_equal(fields["grid_bounds_m"], grid.grid_bounds_m)
    ):
        raise ValueError("La grilla congelada no coincide con la configuración")
    for key in ("truth", "mean", "std"):
        if (
            fields[key].shape != (steps + 1, len(cells))
            or not np.isfinite(fields[key]).all()
        ):
            raise ValueError("Campos congelados incompletos o no finitos")
    if (fields["truth"] < 0).any() or (fields["std"] < 0).any():
        raise ValueError(
            "Verdad o desviación negativa; la media GP sí puede ser negativa"
        )
    radius = np.linalg.norm(
        (cells - spec["grid_cells"] // 2) * grid.cell_size_m, axis=1
    )
    regions = {
        "core": radius <= spec["core_radius_m"],
        "surroundings": (radius > spec["core_radius_m"])
        & (radius < spec["background_from_m"]),
        "background": radius >= spec["background_from_m"],
    }
    if any(not np.array_equal(fields[f"region_{k}"], v) for k, v in regions.items()):
        raise ValueError("Las máscaras regionales no coinciden")
    if (
        hashlib.sha256(fields["truth"].tobytes()).hexdigest()
        != report["signatures"]["field_sha256"]
    ):
        raise ValueError("El campo no coincide con su huella")
    if (
        [r["elapsed_s"] for r in report["frames"]]
        != list(range(0, steps * 300 + 1, 300))
        or [r["tick"] for r in report["ticks"]] != list(range(1, steps + 1))
        or report["signatures"]["horizon_s"] != steps * 300
    ):
        raise ValueError("Reloj u horizonte incompleto")
    records = tuple(
        MeasurementRecord(
            **{**r, "timestamp_utc": datetime.fromisoformat(r["timestamp_utc"])}
        )
        for r in report["measurements"]
    )
    origins = {
        r.timestamp_utc - timedelta(seconds=r.simulation_time_s) for r in records
    }
    if len(origins) != 1 or len(
        {(r.sensor_id, r.timestamp_utc) for r in records}
    ) != len(records):
        raise ValueError("Mediciones duplicadas o relojes incompatibles")
    expected = {
        (f"MOBILE_{v}", t * 300) for v in range(3) for t in range(1, steps + 1)
    } | {
        (f"CONTROL_FIXED_{v}", t)
        for v in range(4)
        for t in range(0, steps * 300 + 1, 1800)
    }
    if {(r.sensor_id, r.simulation_time_s) for r in records} != expected:
        raise ValueError("Cadencia o presupuesto de mediciones distinto del control")
    for record in records:
        if record.crs != grid.crs or not grid.is_active_xy(record.x_m, record.y_m):
            raise ValueError("Coordenadas de medición incompatibles")
        if record.sensor_id.startswith("MOBILE_"):
            tick = int(record.simulation_time_s / 300)
            vehicle = int(record.sensor_id.removeprefix("MOBILE_"))
            if (
                record.sensor_type is not SensorType.MOBILE
                or record.noise_stddev_ug_m3 != spec["mobile_noise_stddev_ug_m3"]
                or not np.array_equal(
                    (record.x_m, record.y_m),
                    report["ticks"][tick - 1]["positions_m"][vehicle],
                )
            ):
                raise ValueError("Lectura móvil incompatible con el recorrido o ruido")
        elif record.sensor_type is not SensorType.FIXED:
            raise ValueError("Tipo de estación fija incorrecto")
    distance = sum(sum(t["route_distance_m"]) for t in report["ticks"])
    if (
        distance != report["summary"]["distance_m"]
        or report["summary"]["mobile_records"] != 3 * steps
    ):
        raise ValueError("Presupuesto registrado incompatible con las trazas")
    return records, origins.pop()


def validate_pairs(cases, spec, previews):
    """Verificar campo, fijas, ruido efectivo y recorridos antes de reconstruir."""
    for regime in REGIMES:
        reference = cases[regime, spec["sensor_seeds"][0], PLANS[0]][1]["truth"]
        for seed in spec["sensor_seeds"]:
            pair = [cases[regime, seed, p] for p in PLANS]
            if pair[0][0]["signatures"] != pair[1][0]["signatures"]:
                raise ValueError("Campos, fijas, inicio u horizonte no emparejados")
            if [r for r in pair[0][2] if r.sensor_type is SensorType.FIXED] != [
                r for r in pair[1][2] if r.sensor_type is SensorType.FIXED
            ]:
                raise ValueError("Lecturas fijas no emparejadas")
            noises = []
            for plan, (report, fields, records, _) in zip(PLANS, pair, strict=True):
                if not np.array_equal(reference, fields["truth"]):
                    raise ValueError("El campo cambió con el plan o el ruido")
                if len(previews[plan]) != spec["steps"]:
                    raise ValueError("Vista previa de rutas incompleta")
                for tick, preview in zip(report["ticks"], previews[plan], strict=True):
                    for key in ("positions_m", "route_distance_m"):
                        if not np.array_equal(tick[key], preview[key]):
                            raise ValueError("El recorrido difiere de la vista previa")
                noise = {}
                for r in records:
                    if r.sensor_type is SensorType.MOBILE:
                        row, col = int(r.y_m // 200), int(r.x_m // 200)
                        truth = fields["truth"][
                            int(r.simulation_time_s / 300),
                            row * spec["grid_cells"] + col,
                        ]
                        noise[r.sensor_id, r.simulation_time_s] = r.value_ug_m3 - truth
                noises.append(noise)
            if any(
                not np.isclose(v, noises[1][k], rtol=0, atol=1e-10)
                for k, v in noises[0].items()
            ):
                raise ValueError("El ruido móvil efectivo no está emparejado")
            for a, b in zip(pair[0][0]["ticks"], pair[1][0]["ticks"], strict=True):
                for key in (
                    "route_distance_m",
                    "sensor_rng_before_sha256",
                    "sensor_rng_after_sha256",
                ):
                    if a[key] != b[key]:
                        raise ValueError(
                            "Distancia o generadores de ruido no emparejados"
                        )
            prefix = spec["initial_common_ticks"] + 1
            for key in ("truth", "mean", "std"):
                if not np.array_equal(
                    pair[0][1][key][:prefix], pair[1][1][key][:prefix]
                ):
                    raise ValueError("El prefijo común difiere entre planes")


def error_metrics(truth, mean, std, regions, threshold, area):
    values = frame_metrics(truth, mean, std, regions, threshold, area)
    error = mean[regions["background"]] - truth[regions["background"]]
    values["background_underestimation_mse"] = float(np.mean(np.minimum(error, 0) ** 2))
    values["negative_area_km2"] = float(np.sum(mean < 0) * area / 1e6)
    values["minimum_prediction"] = float(mean.min())
    values["peak_bias"] = values["predicted_peak"] - values["truth_peak"]
    return values


def score_case(
    grid, source_config, frozen, scale, output, *, scale_key="spatial_scale_m"
):
    report, fields, records, origin = frozen
    base = ReconstructionConfig(**source_config["gp"])
    gp = gp_variant(base, scale, scale_key)
    control = scale == (
        base.x_length_scale_m
        if scale_key == "spatial_scale_m"
        else base.temporal_length_scale_s
    )
    model = GaussianProcessFieldReconstructor(gp)
    history = timedelta(seconds=source_config["history_duration_s"])
    spec = source_config["settings"]
    regions = {k: fields[f"region_{k}"] for k in ("core", "surroundings", "background")}
    threshold = spec["background_ug_m3"] + spec["high_excess_threshold_ug_m3"]
    predictions, deviations, rows = [], [], []
    for tick, saved in enumerate(report["frames"]):
        now = origin + timedelta(seconds=saved["elapsed_s"])
        eligible = tuple(r for r in records if now - history <= r.timestamp_utc <= now)
        posterior = model.reconstruct(
            ReconstructionRequest(
                grid=grid, reconstruction_timestamp_utc=now, measurements=eligible
            )
        )
        mean = posterior.mean_ug_m3[grid.active_mask]
        std = posterior.standard_deviation_ug_m3[grid.active_mask]
        if control and (
            not np.allclose(mean, fields["mean"][tick], rtol=1e-7, atol=1e-8)
            or not np.allclose(std, fields["std"][tick], rtol=1e-7, atol=1e-8)
        ):
            raise ValueError("El GP de referencia no reproduce el posterior original")
        metrics = error_metrics(
            fields["truth"][tick], mean, std, regions, threshold, grid.cell_area_m2
        )
        if control and any(
            not np.isclose(metrics[k], saved[k], rtol=1e-7, atol=1e-8)
            for k in METRICS
            if k in saved
        ):
            raise ValueError(
                "Las métricas del control no reproducen el informe original"
            )
        rows.append(
            {
                "tick": tick,
                "elapsed_s": saved["elapsed_s"],
                "eligible_records": len(eligible),
                "phase": saved["phase"],
                **metrics,
            }
        )
        predictions.append(mean)
        deviations.append(std)
    summary = summarize_frames(rows)
    identity = {k: report[k] for k in ("regime", "sensor_seed", "plan")}
    result = {**identity, scale_key: scale, **summary}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "report.json",
        {
            "summary": result,
            "frames": rows,
            "gp": asdict(gp),
            "history_duration_s": source_config["history_duration_s"],
            "control_reproduced": control,
            "input_signatures": report["signatures"],
        },
    )
    with (output / "posterior-private.npz").open("xb") as file:
        np.savez_compressed(
            file,
            mean=predictions,
            std=deviations,
            elapsed_s=[r["elapsed_s"] for r in rows],
        )
    return result, np.asarray(predictions)


def summarize_frames(rows):
    """Métricas comunes a los GP fijos y las mezclas sobre el mismo reloj."""
    summary = {}
    times = [r["elapsed_s"] for r in rows]
    for key in METRICS:
        summary[f"temporal_{key}"] = float(
            np.trapezoid([r[key] for r in rows], times) / times[-1]
        )
        summary[f"final_{key}"] = rows[-1][key]
    for phase in ("steady", "rising", "declining"):
        selected = [r for r in rows[1:] if r["phase"] == phase]
        summary[f"{phase}_ticks"] = len(selected)
        for key in ("core_mse", "peak_abs_error", "peak_bias"):
            summary[f"{phase}_{key}"] = (
                float(np.mean([r[key] for r in selected])) if selected else None
            )
    for active, key in (
        (True, "active_detection_rate"),
        (False, "inactive_false_alarm_rate"),
    ):
        selected = [
            r["detected_focus"] for r in rows[1:] if r["active_focus"] == active
        ]
        summary[key] = float(np.mean(selected)) if selected else None
    summary["minimum_prediction"] = min(r["minimum_prediction"] for r in rows)
    return summary


def criteria(original, candidate):
    """Criterios separados: la caída del error total no sustituye la delimitación."""
    primary = {
        "lower_background_error": candidate["temporal_background_mse"]
        < original["temporal_background_mse"],
        "core_error_not_higher": candidate["temporal_core_mse"]
        <= original["temporal_core_mse"],
        "peak_error_not_higher": candidate["temporal_peak_abs_error"]
        <= original["temporal_peak_abs_error"],
    }
    mechanism = {
        "surroundings_error_not_higher": candidate["temporal_surroundings_mse"]
        <= original["temporal_surroundings_mse"],
        "false_high_area_not_higher": candidate["temporal_false_high_area_km2"]
        <= original["temporal_false_high_area_km2"],
        "lower_background_overestimation": candidate[
            "temporal_background_overestimation_mse"
        ]
        < original["temporal_background_overestimation_mse"],
        "lower_background_underestimation": candidate[
            "temporal_background_underestimation_mse"
        ]
        < original["temporal_background_underestimation_mse"],
        "negative_area_not_higher": candidate["temporal_negative_area_km2"]
        <= original["temporal_negative_area_km2"],
    }
    return {
        **primary,
        **mechanism,
        "primary_pass": all(primary.values()),
        "mechanism_pass": all(mechanism.values()),
    }


def summarize(
    rows, seeds, *, scales=SCALES, scale_key="spatial_scale_m", criteria_fn=criteria
):
    index = {(r["regime"], r["plan"], r["sensor_seed"], r[scale_key]): r for r in rows}
    expected = {
        (r, p, s, scale)
        for r in REGIMES
        for p in PLANS
        for s in seeds
        for scale in scales
    }
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError("La comparación requiere todos los pares, sin duplicados")
    keys = [k for k in rows[0] if k not in {"regime", "plan", "sensor_seed", scale_key}]
    groups, paired, decisions = [], [], []
    for regime in REGIMES:
        for plan in PLANS:
            arms = []
            for scale in scales:
                group = {"regime": regime, "plan": plan, scale_key: scale}
                for key in keys:
                    values = [index[regime, plan, seed, scale][key] for seed in seeds]
                    values = [v for v in values if v is not None]
                    group[key] = float(np.mean(values)) if values else None
                    if key == "minimum_prediction":
                        group[key] = min(values)
                    if key.endswith("_mse") and values:
                        group[key[:-3] + "rmse"] = float(np.sqrt(group[key]))
                arms.append(group)
                groups.append(group)
            decisions.append({"regime": regime, "plan": plan, **criteria_fn(*arms)})
            for seed in seeds:
                a, b = (index[regime, plan, seed, scale] for scale in scales)
                paired.append(
                    {
                        "regime": regime,
                        "plan": plan,
                        "sensor_seed": seed,
                        "criteria": criteria_fn(a, b),
                        "candidate_minus_control": {
                            k: b[k] - a[k]
                            if a[k] is not None and b[k] is not None
                            else None
                            for k in keys
                        },
                    }
                )
    return {
        "summaries": groups,
        "paired": paired,
        "criteria_by_regime_and_plan": decisions,
    }


def plot_pair(grid, spec, frozen, predictions, output, *, scales=SCALES, unit="m"):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report, fields, records, _ = frozen
    control, candidate = scales
    vmax = spec["background_ug_m3"] + 1.5 * spec["peak_excess_ug_m3"]
    for tick in spec["map_ticks"]:
        fig = Figure(figsize=(12, 7), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3).ravel()
        truth = fields["truth"][tick]
        arrays = (
            truth,
            predictions[control][tick],
            predictions[candidate][tick],
            predictions[control][tick] - truth,
            predictions[candidate][tick] - truth,
            predictions[candidate][tick] - predictions[control][tick],
        )
        titles = (
            "Verdad privada",
            f"GP {control} {unit}",
            f"GP {candidate} {unit}",
            f"Error {control} {unit}",
            f"Error {candidate} {unit}",
            f"Media {candidate} − {control} {unit}",
        )
        for i, (ax, values, title) in enumerate(zip(axes, arrays, titles, strict=True)):
            image = ax.imshow(
                values.reshape(grid.shape),
                origin="lower",
                extent=grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
                cmap="viridis" if i < 3 else "RdBu_r",
                vmin=-spec["peak_excess_ug_m3"],
                vmax=vmax if i < 3 else spec["peak_excess_ug_m3"],
            )
            for vehicle, route in enumerate(report["routes_m"]):
                # Extremos medidos hasta el tick, sin dibujar recorridos futuros.
                points = (
                    np.asarray(
                        [route[0]]
                        + [r["positions_m"][vehicle] for r in report["ticks"][:tick]]
                    )
                    / 1000
                )
                ax.plot(points[:, 0], points[:, 1], "k-", linewidth=0.5)
            for kind, color, marker in (
                (SensorType.FIXED, "red", "^"),
                (SensorType.MOBILE, "white", "."),
            ):
                samples = [
                    r
                    for r in records
                    if r.sensor_type is kind and r.simulation_time_s <= tick * 300
                ]
                ax.scatter(
                    [r.x_m / 1000 for r in samples],
                    [r.y_m / 1000 for r in samples],
                    c=color,
                    marker=marker,
                    s=5,
                )
            ax.set(
                title=title, xlabel="Este sintético (km)", ylabel="Norte sintético (km)"
            )
            fig.colorbar(image, ax=ax, shrink=0.75)
        fig.suptitle(
            f"{report['regime']} | {report['plan']} | ruido {report['sensor_seed']} | tick {tick}; µg/m³"
        )
        with (output / f"tick-{tick:02d}.png").open("xb") as file:
            fig.savefig(file, format="png", dpi=110)


def run_comparison(loaded, source, output, *, maps=True):
    config, grid, cases, raw = loaded
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            "source": str(source),
            "source_config": config,
            "spatial_scales_m": SCALES,
            "simulation_executed": False,
            "training_executed": False,
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "hotspot_gp_experiment.py",
                    "hotspot_sampling.py",
                    "gp_spatial_experiment.py",
                )
            },
            "criterion": "Comparar 500 frente a 2000 m dentro de cada régimen y plan: menor error temporal exterior sin empeorar núcleo ni pico; comprobar además delimitación y ambos signos del error.",
        },
    )
    manifest = {}
    for name, data in raw.items():
        path = output / "input" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    _write_json(output / "source-files.json", manifest)
    rows = []
    for (regime, seed, plan), frozen in cases.items():
        folder = output / regime / f"noise-{seed}" / plan
        predictions = {}
        for scale in SCALES:
            row, mean = score_case(
                grid, config, frozen, scale, folder / f"meters-{scale}"
            )
            rows.append(row)
            predictions[scale] = mean
        if maps and seed == config["settings"]["sensor_seeds"][0]:
            plot_pair(grid, config["settings"], frozen, predictions, folder)
        print(
            f"Reconstruido: {regime} | ruido {seed} | {plan} | 2000/500 m", flush=True
        )
    _write_json(output / "cases.json", rows)
    result = {
        **summarize(rows, config["settings"]["sensor_seeds"]),
        "all_controls_reproduced": True,
        "scope": "Control mecanístico artificial con lecturas y rutas congeladas. Sin selección sobre Lima, adopción automática ni evaluación de una política realimentada.",
    }
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comparar GP 2000/500 m sobre el control de focos guardado"
    )
    parser.add_argument(
        "--source", type=Path, help="hotspot.tar.zst o carpeta de una ejecución"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe resultados")
    source = args.source
    if source is None:
        source = PROJECT_ROOT / "hotspot.tar.zst"
        if not source.is_file():
            candidates = sorted(
                (CACHE_DIRECTORY / "hotspot-sampling").glob("*/config.json")
            )
            if not candidates:
                parser.error(
                    "Falta hotspot.tar.zst o una ejecución en cache/hotspot-sampling/"
                )
            source = candidates[-1].parent
    loaded = load_source(source)
    if args.check:
        print(
            json.dumps(
                {
                    "source": str(source),
                    "frozen_missions": len(loaded[2]),
                    "reconstructions": len(loaded[2]) * len(SCALES),
                    "spatial_scales_m": SCALES,
                    "inputs_verified": True,
                    "gp_reconstruction_executed": False,
                    "simulation_executed": False,
                    "training_executed": False,
                },
                indent=2,
            )
        )
        return
    torch.set_num_threads(1)
    output = args.output or CACHE_DIRECTORY / "hotspot-gp-experiments" / datetime.now(
        UTC
    ).strftime("%Y%m%d-%H%M%S-%f")
    print(f"Entrada: {source}\nSalida: {output}", flush=True)
    run_comparison(loaded, source, output)


if __name__ == "__main__":
    main()
