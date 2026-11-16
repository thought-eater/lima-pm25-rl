"""Reevaluación sintética sobre lecturas congeladas, sin política ni simulador."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import MeasurementRecord, SensorType

from .arrival_experiment import _write_json
from .gp_spatial_experiment import partition_bounds, validate_frozen_case
from .gp_temporal_experiment import check_control
from .hotspot_metrics import (
    aggregate_missions,
    cell_detection_events,
    compare_references,
    frame_metrics,
    summarize_mission,
)
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256, prepare_experiment

DEFAULT_CONFIG = PROJECT_ROOT / "config/hotspot-evaluation.json"


def source_files(path, names, *, calibration=False, cases=False):
    """Leer carpetas o tar sin extraer rutas ni cargar archivos ajenos."""
    result = {}

    def wanted(name):
        p = Path(name)
        return (
            p.name in names
            and (not calibration or "calibration" in p.parts)
            and (not cases or p.parent.name.startswith("case-"))
        )

    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_file() and wanted(str(file.relative_to(path))):
                result[str(file.relative_to(path))] = file.read_bytes()
    else:
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                if member.isfile() and wanted(member.name):
                    if member.name in result:
                        raise ValueError("Miembro duplicado en el archivo")
                    result[member.name] = archive.extractfile(member).read()
    if not result:
        raise ValueError(f"No se encontraron entradas en {path}")
    return result


def unique_file(files, name):
    matches = [value for path, value in files.items() if Path(path).name == name]
    if len(matches) != 1:
        raise ValueError(f"Se requiere exactamente un {name} por evaluación")
    return matches[0]


def frozen_cases(path, grid, protocol, *, training=False):
    files = source_files(
        path, {"report.json", "fields-private.npz"}, calibration=training, cases=True
    )
    result = {}
    for name, data in files.items():
        if Path(name).name != "report.json":
            continue
        report = json.loads(data)
        field_name = str(Path(name).with_name("fields-private.npz"))
        with np.load(io.BytesIO(files[field_name]), allow_pickle=False) as stored:
            fields = {key: stored[key] for key in stored.files}
        seed = report["case_seed"]
        if seed in result:
            raise ValueError("Casos congelados duplicados")
        result[seed] = validate_frozen_case(
            report,
            fields,
            grid,
            **partition_bounds(protocol, "calibration" if training else "validation"),
        )
    return result


def freeze_protocol(cases, *, quantile, expected_seeds, source, grid, protocol):
    if set(cases) != set(expected_seeds) or not 0 < quantile < 1:
        raise ValueError(
            "El umbral requiere todos los casos de entrenamiento prefijados"
        )
    shapes = {case[1]["truth"][1:].shape for case in cases.values()}
    if len(shapes) != 1:
        raise ValueError("El percentil requiere casos con igual dominio y horizonte")
    values = np.concatenate(
        [cases[seed][1]["truth"][1:].ravel() for seed in expected_seeds]
    )
    return {
        "version": 2,
        "purpose": "retrospective_development_not_final_historical_replay",
        "threshold_ug_m3": float(np.quantile(values, quantile, method="linear")),
        "threshold_rule": "pooled_training_cell_tick_quantile_excluding_t0; hot_if_greater_or_equal",
        "quantile": quantile,
        "calibration_case_seeds": expected_seeds,
        "calibration_field_sha256": {
            str(s): cases[s][0]["signatures"]["field_sha256"] for s in expected_seeds
        },
        "calibration_source": str(source),
        "protocol": protocol,
        "cell_area_m2": grid.cell_area_m2,
        "health_threshold": False,
        "aggregation": "trapezoid_in_time; ratios_of_cell_time_integrals; replicas_then_equal_cases; sqrt_after_MSE_average",
        "delay_rule": "per_cell_contiguous_hot_episodes; exclude_left_censored; separate_completed_undetected_and_undetected_right_censored; missed_fraction_among_completed_events",
        "original_target_percent": 20,
        "original_target_rule": "unchanged; retain original global indicators and references; no new local pass criterion",
    }


def parse_measurements(rows):
    records = []
    for row in rows:
        values = {
            key: value
            for key, value in row.items()
            if key not in {"case_seed", "strategy", "policy_seed"}
        }
        for key in ("simulation_time_s", "x_m", "y_m", "noise_stddev_ug_m3"):
            # Conservar int/float del CSV para reproducir también la huella JSON.
            values[key] = json.loads(values[key])
        values["value_ug_m3"] = (
            json.loads(values["value_ug_m3"]) if values["value_ug_m3"] else None
        )
        values["timestamp_utc"] = datetime.fromisoformat(values["timestamp_utc"])
        records.append(MeasurementRecord(**values))
    if len({(r.sensor_id, r.timestamp_utc) for r in records}) != len(records):
        raise ValueError("Mediciones duplicadas; no se incorporan dos veces")
    return tuple(records)


def measurement_digest(records):
    return hashlib.sha256(
        json.dumps([asdict(r) for r in records], default=str, sort_keys=True).encode()
    ).hexdigest()


def route_progress(routes, records, times, total_distance_m):
    """Reconstruir distancia vial por extremos observados; no usar distancia en grilla."""
    if not routes:
        if total_distance_m != 0:
            raise ValueError("Faltan trayectorias para el coste vial")
        return np.zeros(len(times))
    progress = np.zeros(len(times))
    distance = 0.0
    for vehicle, points in routes.items():
        points = np.asarray(points, dtype=float)
        cumulative = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        ]
        distance += cumulative[-1]
        cursor, previous = 0, 0.0
        mobile = {
            r.simulation_time_s: r
            for r in records
            if r.sensor_id == f"MOBILE_{vehicle}"
        }
        for index, time in enumerate(times):
            if time in mobile:
                r = mobile[time]
                candidates = np.flatnonzero(
                    np.all(
                        np.isclose(points[cursor:], [r.x_m, r.y_m], rtol=0, atol=1e-6),
                        axis=1,
                    )
                )
                if not len(candidates):
                    raise ValueError("La lectura móvil no coincide con su trayectoria")
                cursor += int(candidates[0])
                previous = cumulative[cursor]
            progress[index] += previous
        if not np.isclose(previous, cumulative[-1], rtol=1e-8, atol=1e-5):
            raise ValueError("No se pudo fechar el final de la trayectoria")
    if not np.isclose(distance, total_distance_m, rtol=1e-8, atol=1e-5):
        raise ValueError(
            "La distancia de las polilíneas no reproduce el coste guardado"
        )
    return progress


def reconstruct_mission(grid, updater, frozen, records, old, *, threshold, routes):
    report, fields, _, origin = frozen
    times = fields["elapsed_s"]
    if (
        old["field_sha256"] != report["signatures"]["field_sha256"]
        or old["fixed_measurements_sha256"]
        != report["signatures"]["fixed_measurements_sha256"]
        or old["initial_positions_sha256"]
        != report["signatures"]["initial_positions_sha256"]
        or datetime.fromisoformat(old["start_utc"]) != origin
        or float(old["elapsed_s"]) != times[-1]
    ):
        raise ValueError(
            "Campo, fijas, posiciones iniciales u horizonte no coinciden con la misión"
        )
    if any(
        r.timestamp_utc != origin + timedelta(seconds=r.simulation_time_s)
        or r.simulation_time_s > times[-1]
        for r in records
    ):
        raise ValueError("Lecturas fuera del reloj de la misión")
    fixed = tuple(r for r in records if r.sensor_type is SensorType.FIXED)
    if routes:
        starts = np.asarray([routes[vehicle][0] for vehicle in sorted(routes)])
        if (
            hashlib.sha256(starts.tobytes()).hexdigest()
            != old["initial_positions_sha256"]
        ):
            raise ValueError("Las posiciones iniciales no coinciden con su huella")
    if measurement_digest(fixed) != old["fixed_measurements_sha256"]:
        raise ValueError("Las lecturas fijas no coinciden con su huella")
    if len(fixed) != int(old["fixed_measurements"]) or len(records) - len(fixed) != int(
        old["mobile_measurements"]
    ):
        raise ValueError("Número de lecturas distinto del guardado")
    decision_end = float(old["decision_end_elapsed_s"])
    if any(
        r.sensor_type is SensorType.MOBILE and r.simulation_time_s > decision_end
        for r in records
    ):
        raise ValueError("Hay móviles después de la terminación")
    coordinates = np.asarray(
        [grid.cell_center(int(r), int(c)) for r, c in fields["active_cells"]]
    )
    model = GaussianProcessFieldReconstructor(updater.reconstruction_config)
    means, deviations, rows = [], [], []
    distances = route_progress(routes, records, times, float(old["distance_m"]))
    for index, time in enumerate(times):
        now = origin + timedelta(seconds=float(time))
        eligible = tuple(
            r
            for r in records
            if now - updater.history_duration <= r.timestamp_utc <= now
        )
        posterior = model.reconstruct(
            ReconstructionRequest(
                grid=grid, reconstruction_timestamp_utc=now, measurements=eligible
            )
        )
        mean, std = (
            posterior.mean_ug_m3[grid.active_mask],
            posterior.standard_deviation_ug_m3[grid.active_mask],
        )
        means.append(mean)
        deviations.append(std)
        rows.append(
            {
                "elapsed_s": float(time),
                "distance_m": float(distances[index]),
                **frame_metrics(
                    fields["truth"][index],
                    mean,
                    std,
                    threshold=threshold,
                    cell_area_m2=grid.cell_area_m2,
                    coordinates=coordinates,
                ),
            }
        )
    summary = summarize_mission(rows, times)
    for new, saved in (
        ("final_mse", "synthetic_mse"),
        ("final_mae_ug_m3", "synthetic_mae_ug_m3"),
        ("final_integrated_variance", "final_integrated_variance"),
        ("mean_integrated_variance", "mean_integrated_variance"),
    ):
        if not np.isclose(summary[new], float(old[saved]), rtol=1e-7, atol=1e-8):
            raise ValueError(
                f"La reconstrucción no reproduce {saved}: {summary[new]} != {old[saved]}"
            )
    if not np.isclose(
        rows[0]["integrated_variance"],
        float(old["initial_integrated_variance"]),
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError("No se reproduce la incertidumbre inicial")
    return (
        rows,
        np.asarray(means),
        np.asarray(deviations),
        {
            **summary,
            **cell_detection_events(
                fields["truth"], np.asarray(means), times, threshold
            ),
            "distance_m": float(old["distance_m"]),
            "sampling_coverage": float(old["sampling_coverage"]),
            "original_net_variance_reduction_percent": float(
                old["net_variance_reduction_percent"]
            ),
            "reproduction_passed": True,
        },
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved(
    path, frozen, grid, updater, protocol, output, *, smoke=False, map_ticks=()
):
    files = source_files(
        path, {"config.json", "cases.csv", "measurements.csv", "trajectories.csv"}
    )
    metadata = json.loads(unique_file(files, "config.json"))
    if metadata["protocol"] != protocol["protocol"]:
        raise ValueError("El protocolo original difiere del congelado")

    def csv_rows(name):
        return list(csv.DictReader(io.StringIO(unique_file(files, name).decode())))

    old_rows = csv_rows("cases.csv")
    seeds = metadata["case_seeds"]
    if set(frozen) != set(seeds):
        raise ValueError("La verdad debe cubrir exactamente los casos prefijados")
    measurements, trajectories = (
        defaultdict(list),
        defaultdict(lambda: defaultdict(list)),
    )

    def key(row):
        return int(row["case_seed"]), row["strategy"], row["policy_seed"]

    for row in csv_rows("measurements.csv"):
        measurements[key(row)].append(row)
    for row in csv_rows("trajectories.csv"):
        trajectories[key(row)][int(row["vehicle"])].append(
            (int(row["vertex"]), float(row["x_m"]), float(row["y_m"]))
        )
    keys = [key(r) for r in old_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Misiones duplicadas")
    for seed in seeds:
        signatures = {
            tuple(
                row[name]
                for name in (
                    "field_sha256",
                    "fixed_measurements_sha256",
                    "initial_positions_sha256",
                    "start_utc",
                    "elapsed_s",
                )
            )
            for row in old_rows
            if int(row["case_seed"]) == seed
        }
        if len(signatures) != 1:
            raise ValueError(
                "Las políticas no comparten campo, fijas, inicio y horizonte"
            )
    expected = {
        (s, strategy, policy_seed)
        for s in seeds
        for strategy, policy_seeds in (
            ("fixed", [""]),
            ("neural", [""]),
            ("greedy", [str(p) for p in metadata["policy_seeds"]]),
            ("random", [str(p) for p in metadata["policy_seeds"]]),
            ("sweep", [""]),
        )
        for policy_seed in policy_seeds
    }
    if set(keys) != expected:
        raise ValueError("Faltan políticas, casos o réplicas en la evaluación")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "source.json",
        {
            "source": str(path),
            "metadata": metadata,
            "smoke_only": smoke,
            "file_sha256": {
                name: hashlib.sha256(data).hexdigest() for name, data in files.items()
            },
        },
    )
    results, tick_rows = [], []
    for old in old_rows:
        seed, strategy, policy_seed = key(old)
        if smoke and (seed != seeds[0] or strategy not in {"fixed", "neural"}):
            continue
        if strategy == "fixed":
            candidates = parse_measurements(measurements[seed, "neural", ""])
            records = tuple(r for r in candidates if r.sensor_type is SensorType.FIXED)
            routes = {}
        else:
            records = parse_measurements(measurements[seed, strategy, policy_seed])
            routes = {
                vehicle: [(x, y) for _, x, y in sorted(vertices)]
                for vehicle, vertices in trajectories[
                    seed, strategy, policy_seed
                ].items()
            }
        identity = {"case_seed": seed, "strategy": strategy, "policy_seed": policy_seed}
        rows, mean, std, summary = reconstruct_mission(
            grid,
            updater,
            frozen[seed],
            records,
            old,
            threshold=protocol["threshold_ug_m3"],
            routes=routes,
        )
        result = {**identity, **summary}
        results.append(result)
        tick_rows.extend({**identity, **r} for r in rows)
        folder = (
            output / f"case-{seed}" / f"{strategy}-{policy_seed or 'deterministic'}"
        )
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / "posterior-private.npz").open("xb") as file:
            np.savez_compressed(
                file,
                mean=mean,
                std=std,
                elapsed_s=frozen[seed][1]["elapsed_s"],
                active_cells=frozen[seed][1]["active_cells"],
            )
        if seed == seeds[0] and (
            not policy_seed or policy_seed == str(metadata["policy_seeds"][0])
        ):
            from .hotspot_plots import save_maps

            save_maps(
                grid,
                frozen[seed][1],
                mean,
                std,
                records,
                routes,
                threshold=protocol["threshold_ug_m3"],
                ticks=map_ticks,
                output=folder,
            )
        print(
            f"Reconstruida: caso={seed}, política={strategy}, réplica={policy_seed or '-'}",
            flush=True,
        )
    cases, summaries, paired = aggregate_missions(results)
    references = compare_references(summaries)
    for name, rows in (
        ("missions.csv", results),
        ("ticks.csv", tick_rows),
        ("cases.csv", cases),
        ("summary.csv", summaries),
        ("paired.csv", paired),
        ("references.csv", references),
    ):
        write_csv(output / name, rows)
    _write_json(
        output / "summary.json",
        {
            "smoke_only": smoke,
            "strategies": summaries,
            "neural_vs_references": references,
            "reproduction_passed": True,
        },
    )
    from .hotspot_plots import save_curves

    save_curves(tick_rows, summaries, output)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reevaluar focos sobre lecturas guardadas, sin simular ni entrenar"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Protocolo previamente congelado; no reajusta el umbral",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        action="append",
        help="Carpeta o tar de evaluación; repetible",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Solo primer caso, fijas y neural; no es la batería completa",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe archivos")
    settings = json.loads(args.config.read_text())
    directory = args.config.resolve().parent
    spec_path = (directory / settings["experiment_config"]).resolve()
    spec = json.loads(spec_path.read_text())
    if (
        settings["training_cases"] != 12
        or not np.isfinite(settings["quantile"])
        or not 0 < settings["quantile"] < 1
        or settings["map_ticks"] != sorted(set(settings["map_ticks"]))
        or any(
            type(t) is not int or not 0 <= t <= spec["steps_per_episode"]
            for t in settings["map_ticks"]
        )
    ):
        parser.error(
            "Se requieren doce casos de entrenamiento, percentil válido y ticks ordenados dentro del horizonte"
        )
    torch.set_num_threads(1)
    train, validation, model_protocol = prepare_experiment(spec, spec_path.parent)
    model_protocol = json.loads(json.dumps(model_protocol))
    try:
        grid, updater = train.domain.grid, train.estimator_config
        check_control(grid, updater)
        training_source = (directory / settings["training_source"]).resolve()
        if args.protocol:
            protocol = json.loads(args.protocol.read_text())
            if (
                protocol["version"] != 2
                or protocol["protocol"] != model_protocol
                or not np.isfinite(protocol["threshold_ug_m3"])
            ):
                raise ValueError("El protocolo congelado no es compatible")
        else:
            calibration = frozen_cases(
                training_source, grid, model_protocol, training=True
            )
            expected = [
                int(np.random.SeedSequence([42, 0, i]).generate_state(1)[0])
                for i in range(settings["training_cases"])
            ]
            protocol = freeze_protocol(
                calibration,
                quantile=settings["quantile"],
                expected_seeds=expected,
                source=training_source,
                grid=grid,
                protocol=model_protocol,
            )
        if args.check:
            print(
                json.dumps(
                    {
                        "status": "checked",
                        "threshold_ug_m3": protocol["threshold_ug_m3"],
                        "training_executed": False,
                        "simulation_executed": False,
                        "reconstruction_executed": False,
                    },
                    indent=2,
                )
            )
            return
        output = args.output or CACHE_DIRECTORY / "hotspot-evaluation" / datetime.now(
            UTC
        ).strftime("%Y%m%d-%H%M%S-%f")
        output.mkdir(parents=True, exist_ok=False)
        # Persistir el umbral ANTES de abrir cualquier campo de validación.
        _write_json(output / "protocol.json", protocol)
        _write_json(
            output / "config.json",
            {
                "settings": settings,
                "smoke_only": args.smoke,
                "training_executed": False,
                "simulation_executed": False,
                "protocol_source": str(args.protocol) if args.protocol else None,
                "code_sha256": {
                    name: _sha256(Path(__file__).with_name(name))
                    for name in (
                        "hotspot_evaluation.py",
                        "hotspot_metrics.py",
                        "hotspot_plots.py",
                    )
                },
            },
        )
        print(f"Salida: {output}", flush=True)
        if args.freeze_only:
            return
        truth_source = (directory / settings["truth_source"]).resolve()
        frozen = frozen_cases(truth_source, grid, model_protocol)
        sources = args.evaluation or [
            (directory / name).resolve() for name in settings["evaluation_sources"]
        ]
        for index, path in enumerate(sources):
            evaluate_saved(
                path,
                frozen,
                grid,
                updater,
                protocol,
                output / f"evaluation-{index + 1}",
                smoke=args.smoke,
                map_ticks=settings["map_ticks"],
            )
        comparisons = []
        neural_cases = []
        for index, path in enumerate(sources):
            folder = output / f"evaluation-{index + 1}"
            source = json.loads((folder / "source.json").read_text())
            summary = json.loads((folder / "summary.json").read_text())
            comparisons.extend(
                {
                    "evaluation": index + 1,
                    "source": str(path),
                    "checkpoint_sha256": source["metadata"].get("checkpoint_sha256"),
                    **row,
                }
                for row in summary["strategies"]
            )
            with (folder / "cases.csv").open() as file:
                neural_cases.append(
                    {
                        int(row["case_seed"]): row
                        for row in csv.DictReader(file)
                        if row["strategy"] == "neural"
                    }
                )
        write_csv(output / "comparison.csv", comparisons)
        paired = []
        for index, cases in enumerate(neural_cases[1:], start=2):
            for seed, row in cases.items():
                reference = neural_cases[0][seed]
                pair = {
                    "evaluation": index,
                    "reference_evaluation": 1,
                    "case_seed": seed,
                }
                for key in row:
                    if key in {"strategy", "case_seed"}:
                        continue
                    pair[f"delta_{key}"] = (
                        float(row[key]) - float(reference[key])
                        if row[key] and reference[key]
                        else None
                    )
                paired.append(pair)
        write_csv(output / "paired-evaluations.csv", paired)
    finally:
        train.close()
        if validation is not None:
            validation.close()
