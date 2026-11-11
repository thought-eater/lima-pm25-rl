"""Comparar 3600/900 s a 500 m con las lecturas y referencias guardadas en hot2."""

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .arrival_experiment import _write_json
from .hotspot_gp_experiment import (
    load_frozen_inputs,
    plot_pair,
    read_source_files,
    score_case,
    summarize,
)
from .hotspot_sampling import PLANS
from .train_cli import CACHE_DIRECTORY, PROJECT_ROOT, _sha256

SCALES = (3600, 900)
SCALE_KEY = "temporal_scale_s"


def load_source(source):
    """Validar hot2 y usar sus posteriores de 500 m/3600 s como referencia."""
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
        config["spatial_scales_m"] != [2000, 500]
        or config["simulation_executed"]
        or config["training_executed"]
        or not json.loads(raw["summary.json"])["all_controls_reproduced"]
    ):
        raise ValueError("Se requiere una comparación espacial 2000/500 m completa")
    inputs = {
        name.removeprefix("input/"): b
        for name, b in raw.items()
        if name.startswith("input/")
    }
    manifest = json.loads(raw["source-files.json"])
    if set(inputs) != set(manifest) or any(
        hashlib.sha256(data).hexdigest() != manifest[name]
        for name, data in inputs.items()
    ):
        raise ValueError("Las entradas de hot2 no coinciden con su manifiesto")
    original, grid, cases, _ = load_frozen_inputs(inputs)
    if original != config["source_config"]:
        raise ValueError("La configuración fuente difiere de la entrada congelada")
    effective = {
        "settings": original["settings"],
        "history_duration_s": original["history_duration_s"],
        "gp": {**original["gp"], "x_length_scale_m": 500, "y_length_scale_m": 500},
    }
    references = {}
    for (regime, seed, plan), frozen in cases.items():
        report, fields, records, origin = frozen
        name = f"{regime}/noise-{seed}/{plan}"
        for scale in (2000, 500):
            folder = f"{name}/meters-{scale}"
            reference = json.loads(raw[f"{folder}/report.json"])
            with np.load(
                io.BytesIO(raw[f"{folder}/posterior-private.npz"]), allow_pickle=False
            ) as file:
                posterior = {key: file[key] for key in file.files}
            expected_gp = {
                **original["gp"],
                "x_length_scale_m": scale,
                "y_length_scale_m": scale,
            }
            identity = reference["summary"]
            if (
                reference["gp"] != expected_gp
                or reference["history_duration_s"] != 21600
                or reference["input_signatures"] != report["signatures"]
                or reference["control_reproduced"] != (scale == 2000)
                or (
                    identity["regime"],
                    identity["sensor_seed"],
                    identity["plan"],
                    identity["spatial_scale_m"],
                )
                != (regime, seed, plan, scale)
            ):
                raise ValueError("Referencia espacial incompatible con el caso")
            times = np.asarray([r["elapsed_s"] for r in report["frames"]])
            if (
                not np.array_equal(posterior["elapsed_s"], times)
                or [r["elapsed_s"] for r in reference["frames"]] != times.tolist()
                or [r["phase"] for r in reference["frames"]]
                != [r["phase"] for r in report["frames"]]
            ):
                raise ValueError("Reloj o fases de la referencia incompatibles")
            for key in ("mean", "std"):
                if (
                    posterior[key].shape != fields[key].shape
                    or not np.isfinite(posterior[key]).all()
                ):
                    raise ValueError("Posteriores de referencia incompletos")
                if scale == 2000 and not np.allclose(
                    posterior[key], fields[key], rtol=1e-7, atol=1e-8
                ):
                    raise ValueError(
                        "El control espacial guardado no reproduce el original"
                    )
            if (posterior["std"] < 0).any():
                raise ValueError("Desviación de referencia negativa")
            if scale == 500:
                # Copias de los diccionarios: no modificar los archivos fuente ni
                # reinterpretar las lecturas/verdad originales como datos nuevos.
                references[regime, seed, plan] = (
                    {**report, "frames": reference["frames"]},
                    {**fields, "mean": posterior["mean"], "std": posterior["std"]},
                    records,
                    origin,
                )
    return effective, grid, references, raw


def temporal_criteria(original, candidate):
    primary = {
        "core_temporal_error_not_higher": candidate["temporal_core_mse"]
        <= original["temporal_core_mse"],
        "peak_temporal_error_not_higher": candidate["temporal_peak_abs_error"]
        <= original["temporal_peak_abs_error"],
    }
    if original["regime"] == "pulsed":
        if any(
            row["declining_peak_abs_error"] is None for row in (original, candidate)
        ):
            raise ValueError("El régimen pulsante debe incluir descensos")
        primary["lower_declining_peak_error"] = (
            candidate["declining_peak_abs_error"] < original["declining_peak_abs_error"]
        )
    spatial = {
        f"{name}_not_higher": candidate[f"temporal_{name}"]
        <= original[f"temporal_{name}"]
        for name in (
            "background_mse",
            "surroundings_mse",
            "false_high_area_km2",
            "negative_area_km2",
        )
    }
    return {
        **primary,
        **spatial,
        "primary_pass": all(primary.values()),
        "spatial_preserved": all(spatial.values()),
    }


def summarize_time(rows, seeds):
    result = summarize(
        rows, seeds, scales=SCALES, scale_key=SCALE_KEY, criteria_fn=temporal_criteria
    )
    decisions = result["criteria_by_regime_and_plan"]
    paired = result["paired"]
    joint = []
    for plan in PLANS:
        group = [r for r in decisions if r["plan"] == plan]
        noise_pairs = []
        for seed in seeds:
            group_seed = [
                r["criteria"]
                for r in paired
                if r["plan"] == plan and r["sensor_seed"] == seed
            ]
            noise_pairs.append(
                {
                    "sensor_seed": seed,
                    "primary_pass_both_regimes": all(
                        r["primary_pass"] for r in group_seed
                    ),
                    "spatial_preserved_both_regimes": all(
                        r["spatial_preserved"] for r in group_seed
                    ),
                }
            )
        joint.append(
            {
                "plan": plan,
                "primary_pass_both_regimes": all(r["primary_pass"] for r in group),
                "spatial_preserved_both_regimes": all(
                    r["spatial_preserved"] for r in group
                ),
                "paired_by_noise": noise_pairs,
            }
        )
    result["criteria_by_plan"] = joint
    return result


def plot_tracking(frozen, predictions, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report, fields, _, _ = frozen
    core = fields["region_core"]
    truth = fields["truth"][:, core]
    times = np.asarray([r["elapsed_s"] for r in report["frames"]]) / 60
    fig = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    axes[0].plot(times, truth.max(axis=1), "k--", label="Pico verdadero")
    for scale in SCALES:
        mean = predictions[scale][:, core]
        axes[0].plot(times, mean.max(axis=1), label=f"GP {scale} s")
        axes[1].plot(times, mean.max(axis=1) - truth.max(axis=1), label=f"GP {scale} s")
        axes[2].plot(
            times, np.sqrt(np.mean((mean - truth) ** 2, axis=1)), label=f"GP {scale} s"
        )
    axes[0].set_ylabel("Pico (µg/m³)")
    axes[1].set_ylabel("Sesgo del pico (µg/m³)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[2].set(ylabel="RMSE núcleo (µg/m³)", xlabel="Minutos")
    for axis in axes:
        axis.legend()
    fig.suptitle(
        f"{report['regime']} | {report['plan']} | ruido {report['sensor_seed']} | espacio 500 m"
    )
    with (output / "peak-tracking.png").open("xb") as file:
        fig.savefig(file, format="png", dpi=110)


def run_comparison(loaded, source, output, *, maps=True):
    config, grid, cases, raw = loaded
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "config.json",
        {
            "source": str(source),
            "source_config": json.loads(raw["config.json"]),
            "reference_gp": config["gp"],
            "history_duration_s": config["history_duration_s"],
            "temporal_scales_s": SCALES,
            "spatial_scale_m": 500,
            "simulation_executed": False,
            "training_executed": False,
            "code_sha256": {
                name: _sha256(Path(__file__).parent / name)
                for name in (
                    "hotspot_gp_time_experiment.py",
                    "hotspot_gp_experiment.py",
                    "hotspot_sampling.py",
                    "gp_spatial_experiment.py",
                )
            },
            "criterion": "Por plan: reducir error del pico en descensos pulsantes sin aumentar error temporal del núcleo ni del pico en ninguno de los dos regímenes. Informar además conservación espacial y consistencia por ruido.",
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
                grid,
                config,
                frozen,
                scale,
                folder / f"seconds-{scale}",
                scale_key=SCALE_KEY,
            )
            rows.append(row)
            predictions[scale] = mean
        if maps and seed == config["settings"]["sensor_seeds"][0]:
            plot_pair(
                grid,
                config["settings"],
                frozen,
                predictions,
                folder,
                scales=SCALES,
                unit="s",
            )
            plot_tracking(frozen, predictions, folder)
        print(
            f"Reconstruido: {regime} | ruido {seed} | {plan} | 500 m; 3600/900 s",
            flush=True,
        )
    _write_json(output / "cases.json", rows)
    result = {
        **summarize_time(rows, config["settings"]["sensor_seeds"]),
        "all_controls_reproduced": True,
        "scope": "Cambio de escala temporal global con espacio 500 m y datos congelados; no adaptación local ni adopción automática para Lima.",
    }
    _write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comparar GP 3600/900 s a 500 m sobre hot2"
    )
    parser.add_argument(
        "--source", type=Path, help="hot2.tar.zst o carpeta de la comparación espacial"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check no escribe resultados")
    source = args.source
    if source is None:
        source = PROJECT_ROOT / "hot2.tar.zst"
        if not source.is_file():
            candidates = sorted(
                (CACHE_DIRECTORY / "hotspot-gp-experiments").glob("*/config.json")
            )
            if not candidates:
                parser.error(
                    "Falta hot2.tar.zst o una comparación en cache/hotspot-gp-experiments/"
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
                    "spatial_scale_m": 500,
                    "temporal_scales_s": SCALES,
                    "history_duration_s": loaded[0]["history_duration_s"],
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
    output = (
        args.output
        or CACHE_DIRECTORY
        / "hotspot-gp-time-experiments"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    )
    print(f"Entrada: {source}\nSalida: {output}", flush=True)
    run_comparison(loaded, source, output)


if __name__ == "__main__":
    main()
