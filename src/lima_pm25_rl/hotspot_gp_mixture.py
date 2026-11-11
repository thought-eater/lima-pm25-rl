"""Control congelado de dos GP y mezclas fija/adaptativa; sin aprendizaje de rutas."""

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
    ReconstructionConfig,
    ReconstructionRequest,
    select_reconstruction_observations,
)

from .arrival_experiment import _write_json
from .gp_mixture import (
    GateConfig,
    LocalEvidenceGate,
    log_predictive_density,
    mix_marginals,
)
from .hotspot_gp_experiment import (
    error_metrics,
    read_source_files,
    summarize,
    summarize_frames,
)
from .hotspot_gp_time_experiment import load_spatial_inputs, temporal_criteria
from .hotspot_sampling import PLANS
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256

ARMS = ("slow", "fast", "fixed", "adaptive")
SCALES = {"slow": 3600, "fast": 900}
GATE_CONFIG = GateConfig()


def load_source(source):
    raw = read_source_files(
        source,
        {
            "config.json",
            "route-preview.json",
            "report.json",
            "fields-private.npz",
            "posterior-private.npz",
            "source-files.json",
            "summary.json",
        },
    )
    config = json.loads(raw["config.json"])
    if (
        config["temporal_scales_s"] != [3600, 900]
        or config["spatial_scale_m"] != 500
        or config["history_duration_s"] != 21600
        or config["simulation_executed"]
        or config["training_executed"]
        or not json.loads(raw["summary.json"])["all_controls_reproduced"]
    ):
        raise ValueError(
            "Se requiere hot3 completo: GP 500 m, tiempo 3600/900 s e historial 6 h"
        )
    inputs = {
        k.removeprefix("input/"): v for k, v in raw.items() if k.startswith("input/")
    }
    manifest = json.loads(raw["source-files.json"])
    if set(inputs) != set(manifest) or any(
        hashlib.sha256(v).hexdigest() != manifest[k] for k, v in inputs.items()
    ):
        raise ValueError("Las entradas de hot3 no coinciden con su manifiesto")
    effective, grid, cases, _ = load_spatial_inputs(inputs)
    if (
        config["source_config"] != json.loads(inputs["config.json"])
        or config["reference_gp"] != effective["gp"]
    ):
        raise ValueError("Configuración de referencia incompatible con hot2")
    references = {}
    for key, (report, fields, _, _) in cases.items():
        regime, seed, plan = key
        references[key] = {}
        times = [r["elapsed_s"] for r in report["frames"]]
        for arm, scale in SCALES.items():
            folder = f"{regime}/noise-{seed}/{plan}/seconds-{scale}"
            saved = json.loads(raw[f"{folder}/report.json"])
            with np.load(
                io.BytesIO(raw[f"{folder}/posterior-private.npz"]), allow_pickle=False
            ) as file:
                arrays = {k: file[k] for k in file.files}
            identity = saved["summary"]
            if (
                saved["gp"] != {**effective["gp"], "temporal_length_scale_s": scale}
                or saved["history_duration_s"] != 21600
                or saved["input_signatures"] != report["signatures"]
                or saved["control_reproduced"] != (arm == "slow")
                or (
                    identity["regime"],
                    identity["sensor_seed"],
                    identity["plan"],
                    identity["temporal_scale_s"],
                )
                != (*key, scale)
                or not np.array_equal(arrays["elapsed_s"], times)
                or [r["elapsed_s"] for r in saved["frames"]] != times
                or [r["phase"] for r in saved["frames"]]
                != [r["phase"] for r in report["frames"]]
            ):
                raise ValueError("Referencia temporal incompatible con el caso")
            for name in ("mean", "std"):
                if (
                    arrays[name].shape != fields[name].shape
                    or not np.isfinite(arrays[name]).all()
                ):
                    raise ValueError("Posterior temporal incompleto")
                if arm == "slow" and not np.allclose(
                    arrays[name], fields[name], rtol=1e-7, atol=1e-8
                ):
                    raise ValueError("El GP lento guardado no reproduce hot2")
            if (arrays["std"] < 0).any():
                raise ValueError("Desviación temporal negativa")
            references[key][arm] = saved, arrays
    return effective, grid, cases, raw, references


def reconstruct_sequence(
    grid, base, history_duration_s, records, origin, times, gate_config=GATE_CONFIG
):
    """Recibe solo lecturas y reloj; no recibe verdad, fases ni regiones del foco."""
    models = {
        arm: GaussianProcessFieldReconstructor(
            replace(base, temporal_length_scale_s=scale)
        )
        for arm, scale in SCALES.items()
    }
    gate = LocalEvidenceGate(gate_config, history_duration_s=history_duration_s)
    positions = np.asarray(
        [grid.cell_center(int(r), int(c)) for r, c in np.argwhere(grid.active_mask)]
    )
    predictions = {arm: {"mean": [], "std": []} for arm in ARMS}
    weights, before_weights, counts, trace = [], [], [], []
    history = timedelta(seconds=history_duration_s)
    for time in times:
        now = origin + timedelta(seconds=float(time))
        previous = tuple(r for r in records if now - history <= r.timestamp_utc < now)
        available = tuple(r for r in records if now - history <= r.timestamp_utc <= now)
        request = ReconstructionRequest(
            grid=grid, reconstruction_timestamp_utc=now, measurements=previous
        )
        batch_request = ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=now,
            measurements=tuple(r for r in records if r.timestamp_utc == now),
        )
        batch, _ = select_reconstruction_observations(batch_request, base)
        before_weights.append(gate.weights(time, positions)[0])
        point_forecasts, log_density = {}, {}
        if batch:
            xy = [(r.x_m, r.y_m) for r in batch]
            prior_weight, _ = gate.weights(time, xy)
            for arm, model in models.items():
                point = model.reconstruct_points(request, xy)
                point_forecasts[arm] = point
                log_density[arm] = log_predictive_density(
                    [r.value_ug_m3 for r in batch],
                    point.mean_ug_m3,
                    point.standard_deviation_ug_m3,
                    [r.noise_stddev_ug_m3 for r in batch],
                )
        else:
            prior_weight = np.empty(0)
            log_density = {arm: np.empty(0) for arm in SCALES}
        eligible, score = gate.observe(
            time, batch, log_density["slow"], log_density["fast"]
        )
        after_weight, count = gate.weights(time, positions)
        weights.append(after_weight)
        counts.append(count)
        readings = []
        for i, record in enumerate(batch):
            readings.append(
                {
                    "sensor_id": record.sensor_id,
                    "timestamp_utc": record.timestamp_utc.isoformat(),
                    "x_m": record.x_m,
                    "y_m": record.y_m,
                    "value_ug_m3": record.value_ug_m3,
                    "noise_stddev_ug_m3": record.noise_stddev_ug_m3,
                    "revisit_used": bool(eligible[i]),
                    "clipped_log_ratio": float(score[i]),
                    "fast_weight_before": float(prior_weight[i]),
                    "slow_mean_before": float(point_forecasts["slow"].mean_ug_m3[i]),
                    "fast_mean_before": float(point_forecasts["fast"].mean_ug_m3[i]),
                    "slow_std_before": float(
                        point_forecasts["slow"].standard_deviation_ug_m3[i]
                    ),
                    "fast_std_before": float(
                        point_forecasts["fast"].standard_deviation_ug_m3[i]
                    ),
                    "slow_log_density": float(log_density["slow"][i]),
                    "fast_log_density": float(log_density["fast"][i]),
                    "fixed_log_density_before": float(
                        np.logaddexp(log_density["slow"][i], log_density["fast"][i])
                        - np.log(2)
                    ),
                    "adaptive_log_density_before": float(
                        np.logaddexp(
                            np.log1p(-prior_weight[i]) + log_density["slow"][i],
                            np.log(prior_weight[i]) + log_density["fast"][i],
                        )
                    ),
                }
            )
        trace.append(
            {
                "elapsed_s": time,
                "previous_records": len(previous),
                "eligible_records": len(available),
                "readings": readings,
            }
        )
        post_request = ReconstructionRequest(
            grid=grid, reconstruction_timestamp_utc=now, measurements=available
        )
        for arm, model in models.items():
            posterior = model.reconstruct(post_request)
            predictions[arm]["mean"].append(posterior.mean_ug_m3[grid.active_mask])
            predictions[arm]["std"].append(
                posterior.standard_deviation_ug_m3[grid.active_mask]
            )
        args = [predictions[arm][key][-1] for arm in SCALES for key in ("mean", "std")]
        for arm, weight in (("fixed", 0.5), ("adaptive", after_weight)):
            mean, std = mix_marginals(*args, weight)
            predictions[arm]["mean"].append(mean)
            predictions[arm]["std"].append(std)
    return (
        {
            arm: {k: np.asarray(v) for k, v in arrays.items()}
            for arm, arrays in predictions.items()
        },
        {
            "fast_weight": np.asarray(weights),
            "fast_weight_before": np.asarray(before_weights),
            "local_batch_count": np.asarray(counts),
        },
        trace,
    )


def score_case(config, grid, frozen, references, output, *, maps=True):
    report, fields, records, origin = frozen
    times = [r["elapsed_s"] for r in report["frames"]]
    predictions, weights, trace = reconstruct_sequence(
        grid,
        ReconstructionConfig(**config["gp"]),
        config["history_duration_s"],
        records,
        origin,
        times,
    )
    for arm in SCALES:
        for key in ("mean", "std"):
            if not np.allclose(
                predictions[arm][key], references[arm][1][key], rtol=1e-7, atol=1e-8
            ):
                raise ValueError(f"El GP {arm} no reproduce la referencia de hot3")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "prequential.json", trace)
    with (output / "weights-private.npz").open("xb") as file:
        np.savez_compressed(
            file, **weights, elapsed_s=times, active_cells=fields["active_cells"]
        )
    regions = {k: fields[f"region_{k}"] for k in ("core", "surroundings", "background")}
    threshold = (
        config["settings"]["background_ug_m3"]
        + config["settings"]["high_excess_threshold_ug_m3"]
    )
    rows = []
    for arm, arrays in predictions.items():
        frames = [
            {
                "tick": i,
                "elapsed_s": time,
                "phase": report["frames"][i]["phase"],
                **error_metrics(
                    fields["truth"][i],
                    arrays["mean"][i],
                    arrays["std"][i],
                    regions,
                    threshold,
                    grid.cell_area_m2,
                ),
            }
            for i, time in enumerate(times)
        ]
        summary = {
            **{k: report[k] for k in ("regime", "plan", "sensor_seed")},
            "arm": arm,
            **summarize_frames(frames),
        }
        if arm in SCALES:
            for key, value in references[arm][0]["summary"].items():
                if (
                    key in summary
                    and isinstance(value, (int, float))
                    and not np.isclose(value, summary[key], rtol=1e-7, atol=1e-8)
                ):
                    raise ValueError("Las métricas de referencia difieren de hot3")
        folder = output / arm
        folder.mkdir()
        _write_json(
            folder / "report.json",
            {
                "summary": summary,
                "frames": frames,
                "input_signatures": report["signatures"],
                "reference_reproduced": arm in SCALES,
                "intervals": "Media ± 1.96 desviaciones; en mezclas es aproximación por momentos, no cuantiles exactos de mezcla",
            },
        )
        with (folder / "posterior-private.npz").open("xb") as file:
            np.savez_compressed(
                file, **arrays, elapsed_s=times, active_cells=fields["active_cells"]
            )
        rows.append(summary)
    if maps:
        plot_case(grid, config["settings"], frozen, predictions, weights, output)
    return rows


def mixture_criteria(original, candidate):
    result = temporal_criteria(original, candidate)
    result["strict_temporal_gain"] = any(
        candidate[k] < original[k]
        for k in ("temporal_core_mse", "temporal_peak_abs_error")
    )
    return result


def summarize_mixture(rows, seeds):
    if any(r["arm"] not in ARMS for r in rows):
        raise ValueError("Brazo desconocido en la comparación")
    comparisons = {}
    groups = {}
    for reference in ("slow", "fast", "fixed"):
        selected = [r for r in rows if r["arm"] in (reference, "adaptive")]
        comparison = summarize(
            selected,
            seeds,
            scales=(reference, "adaptive"),
            scale_key="arm",
            criteria_fn=mixture_criteria,
        )
        for group in comparison["summaries"]:
            groups[group["regime"], group["plan"], group["arm"]] = group
        comparisons[reference] = comparison
    joint = []
    for plan in PLANS:
        checks = [
            r
            for reference in ("slow", "fixed")
            for r in comparisons[reference]["criteria_by_regime_and_plan"]
            if r["plan"] == plan
        ]
        joint.append(
            {
                "plan": plan,
                "primary_pass_vs_slow_and_fixed_both_regimes": all(
                    r["primary_pass"] for r in checks
                ),
                "strict_gain_vs_fixed_both_regimes": all(
                    r["strict_temporal_gain"]
                    for r in comparisons["fixed"]["criteria_by_regime_and_plan"]
                    if r["plan"] == plan
                ),
                "spatial_preserved_vs_slow_and_fixed": all(
                    r["spatial_preserved"] for r in checks
                ),
                "paired_by_noise": [
                    {
                        "sensor_seed": seed,
                        "primary_pass_vs_slow_and_fixed_both_regimes": all(
                            r["criteria"]["primary_pass"]
                            for reference in ("slow", "fixed")
                            for r in comparisons[reference]["paired"]
                            if r["plan"] == plan and r["sensor_seed"] == seed
                        ),
                    }
                    for seed in seeds
                ],
            }
        )
    return {
        "summaries": list(groups.values()),
        "comparisons": comparisons,
        "criteria_by_plan": joint,
    }


def plot_case(grid, spec, frozen, predictions, weights, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report, fields, records, _ = frozen
    names = {
        "slow": "Lento 3600 s",
        "fast": "Rápido 900 s",
        "fixed": "Mezcla 50/50",
        "adaptive": "Mezcla adaptativa",
    }
    for tick in spec["map_ticks"]:
        fig = Figure(figsize=(12, 7), layout="constrained")
        FigureCanvasAgg(fig)
        values = [
            fields["truth"][tick],
            *[predictions[a]["mean"][tick] for a in ARMS],
            weights["fast_weight"][tick],
        ]
        titles = ["Verdad privada", *names.values(), "Peso rápido local"]
        for i, (ax, array, title) in enumerate(
            zip(fig.subplots(2, 3).ravel(), values, titles, strict=True)
        ):
            image = ax.imshow(
                array.reshape(grid.shape),
                origin="lower",
                extent=grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
                vmin=0 if i == 5 else -spec["peak_excess_ug_m3"],
                vmax=1
                if i == 5
                else spec["background_ug_m3"] + 1.5 * spec["peak_excess_ug_m3"],
            )
            for vehicle, route in enumerate(report["routes_m"]):
                points = (
                    np.asarray(
                        [route[0]]
                        + [r["positions_m"][vehicle] for r in report["ticks"][:tick]]
                    )
                    / 1000
                )
                ax.plot(points[:, 0], points[:, 1], "w-", linewidth=0.5)
            samples = [r for r in records if r.simulation_time_s <= tick * 300]
            ax.scatter(
                [r.x_m / 1000 for r in samples],
                [r.y_m / 1000 for r in samples],
                s=4,
                c="red",
            )
            ax.set(
                title=title, xlabel="Este sintético (km)", ylabel="Norte sintético (km)"
            )
            fig.colorbar(image, ax=ax, shrink=0.7)
        fig.suptitle(
            f"{report['regime']} | {report['plan']} | ruido {report['sensor_seed']} | tick {tick}"
        )
        with (output / f"tick-{tick:02d}.png").open("xb") as file:
            fig.savefig(file, format="png", dpi=110)
    core = fields["region_core"]
    times = np.arange(len(fields["truth"])) * 5
    fig = Figure(figsize=(9, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    axes[0].plot(times, fields["truth"][:, core].max(axis=1), "k--", label="Verdad")
    for arm in ARMS:
        axes[0].plot(
            times, predictions[arm]["mean"][:, core].max(axis=1), label=names[arm]
        )
        error = predictions[arm]["mean"][:, core] - fields["truth"][:, core]
        axes[1].plot(times, np.sqrt(np.mean(error**2, axis=1)), label=names[arm])
    for region in ("core", "surroundings", "background"):
        axes[2].plot(
            times,
            weights["fast_weight"][:, fields[f"region_{region}"]].mean(axis=1),
            label=region,
        )
    axes[0].set_ylabel("Pico (µg/m³)")
    axes[1].set_ylabel("RMSE núcleo (µg/m³)")
    axes[2].set(ylabel="Peso rápido medio", xlabel="Minutos", ylim=(0, 1))
    for ax in axes:
        ax.legend()
    with (output / "tracking.png").open("xb") as file:
        fig.savefig(file, format="png", dpi=110)


def run_comparison(loaded, source, output, *, maps=True):
    config, grid, cases, raw, references = loaded
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            "source": str(source),
            "source_config": json.loads(raw["config.json"]),
            "gp": config["gp"],
            "history_duration_s": config["history_duration_s"],
            "arms": ARMS,
            "expert_temporal_scales_s": SCALES,
            "gate": asdict(GATE_CONFIG),
            "initial_fast_weight": 0.5,
            "simulation_executed": False,
            "training_executed": False,
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "gp_mixture.py",
                    "hotspot_gp_mixture.py",
                    "hotspot_gp_experiment.py",
                    "hotspot_gp_time_experiment.py",
                    "hotspot_sampling.py",
                )
            },
            "criterion": "Mejorar descensos frente al lento conservando núcleo/pico persistentes y demostrar ganancia frente a mezcla 50/50. Informar brecha con rápido, conservación espacial y consistencia por semilla.",
            "uncertainty": "Momentos marginales de mezcla local; no un GP conjunto ni incertidumbre calibrada automáticamente. No se usa en recompensa.",
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
    for key, frozen in cases.items():
        regime, seed, plan = key
        rows.extend(
            score_case(
                config,
                grid,
                frozen,
                references[key],
                output / regime / f"noise-{seed}" / plan,
                maps=maps and seed == config["settings"]["sensor_seeds"][0],
            )
        )
        print(
            f"Reconstruido: {regime} | ruido {seed} | {plan} | cuatro brazos",
            flush=True,
        )
    _write_json(output / "cases.json", rows)
    result = {
        **summarize_mixture(rows, config["settings"]["sensor_seeds"]),
        "all_references_reproduced": True,
        "scope": "Control artificial con lecturas congeladas. Pesos heurísticos locales, no persistencia física aprendida ni política realimentada.",
    }
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comparar GP lento/rápido y mezclas fija/adaptativa sobre hot3"
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe resultados")
    source = args.source or PROJECT_ROOT / "hot3.tar.zst"
    if args.source is None and not source.is_file():
        candidates = sorted(
            (CACHE_DIRECTORY / "hotspot-gp-time-experiments").glob("*/config.json")
        )
        if not candidates:
            parser.error("Falta hot3.tar.zst o una comparación temporal guardada")
        source = candidates[-1].parent
    loaded = load_source(source)
    if args.check:
        print(
            json.dumps(
                {
                    "source": str(source),
                    "frozen_missions": len(loaded[2]),
                    "evaluated_sequences": len(loaded[2]) * 4,
                    "gp_sequences": len(loaded[2]) * 2,
                    "arms": ARMS,
                    "gate": asdict(GATE_CONFIG),
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
    output = args.output or CACHE_DIRECTORY / "hotspot-gp-mixtures" / datetime.now(
        UTC
    ).strftime("%Y%m%d-%H%M%S-%f")
    print(f"Entrada: {source}\nSalida: {output}", flush=True)
    run_comparison(loaded, source, output)


if __name__ == "__main__":
    main()
