"""Contrastes descriptivos del fijo contra los dos adaptativos en casos de selección."""

import csv
import json
from datetime import datetime, timedelta
from itertools import chain, pairwise
from itertools import product as cartesian

import numpy as np
from matplotlib.figure import Figure

from .files import product, write_csv

# Campo a nivel de caso, etiqueta en unidades de presentación y factor de escala.
METRICS = (
    ("temporal_mse", "RMSE global promedio (µg/m³)", 1),
    ("final_mse", "RMSE global final (µg/m³)", 1),
    ("temporal_mae_ug_m3", "MAE global promedio (µg/m³)", 1),
    ("final_mae_ug_m3", "MAE global final (µg/m³)", 1),
    ("mean_integrated_variance", "Varianza integrada promedio (10⁹)", 1e-9),
    ("final_integrated_variance", "Varianza integrada final (10⁹)", 1e-9),
    ("sampling_coverage", "Cobertura de muestreo móvil (%)", 100),
    ("distance_m", "Recorrido de flota (km)", 0.001),
    ("temporal_hot_mse", "RMSE en focos promedio (µg/m³)", 1),
    ("final_hot_mse", "RMSE en focos final (µg/m³)", 1),
    ("hot_cell_time_recall", "Sensibilidad en focos (%)", 100),
    ("mean_false_high_area_km2", "Área media de falsas alarmas (km²)", 1),
    ("cell_event_completed_missed_fraction", "Eventos completados omitidos (%)", 100),
)


def stratified_indices(groups):
    """Enumerar las muestras ordenadas equiprobables, dos extracciones por estrato."""
    if len(groups) != 6 or any(len(g) != 2 for g in groups):
        raise ValueError("Se requieren seis estratos de dos casos")
    return np.array(
        [
            list(chain.from_iterable(draw))
            for draw in cartesian(*(cartesian(group, repeat=2) for group in groups))
        ]
    )


def contrast(a, b, indices, *, root):
    """a y b son vectores alineados por caso. La raíz siempre sigue al promedio."""
    transform = np.sqrt if root else np.asarray
    x, y = transform(a.mean()), transform(b.mean())
    xs, ys = transform(a[indices].mean(axis=1)), transform(b[indices].mean(axis=1))
    diffs = transform(a) - transform(b)
    lo, hi = np.quantile(xs - ys, [0.025, 0.975])
    result = {
        "fixed": x,
        "adaptive_value": y,
        "delta": x - y,
        "low": lo,
        "high": hi,
        "lower": int(sum(diffs < 0)),
        "tied": int(sum(diffs == 0)),
        "higher": int(sum(diffs > 0)),
        "sd_case_delta": diffs.std(ddof=1),
    }
    if y > 0 and np.all(ys > 0):
        plo, phi = np.quantile(100 * (1 - xs / ys), [0.025, 0.975])
        result.update(
            reduction_percent=100 * (1 - x / y), reduction_low=plo, reduction_high=phi
        )
    else:
        result.update(reduction_percent=None, reduction_low=None, reduction_high=None)
    return result, diffs


def _neural(folder, name):
    with (folder / name).open(newline="") as src:
        return [r for r in csv.DictReader(src) if r["strategy"] == "neural"]


def analyze(spec, output):
    output.mkdir()
    strata = spec["strata"]
    seeds = list(chain.from_iterable(strata.values()))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Los estratos deben ser disjuntos")
    indices = stratified_indices(
        [[seeds.index(s) for s in group] for group in strata.values()]
    )
    identities, values, curves, references = [], [], [], []
    for role in ("fixed", "adaptive_final", "adaptive_best"):
        label, mission_key, metric_key = spec[role]
        missions, metrics = product(mission_key), product(metric_key)
        metadata = json.loads((missions / "config.json").read_text())
        summary = json.loads((metrics / "summary.json").read_text())
        if (
            metadata != json.loads((metrics / "source.json").read_text())["metadata"]
            or set(metadata["case_seeds"]) != set(seeds)
            or metadata["partition"] != "validation"
            or summary["smoke_only"]
            or not summary["reproduction_passed"]
        ):
            raise ValueError(
                "Las misiones y métricas no son la misma validación completa"
            )
        signature = (
            "start_utc",
            "elapsed_s",
            "field_sha256",
            "fixed_measurements_sha256",
            "initial_positions_sha256",
        )
        rows = _neural(missions, "cases.csv")
        identity = {int(r["case_seed"]): tuple(r[k] for k in signature) for r in rows}
        if (
            set(identity) != set(seeds)
            or len(rows) != len(seeds)
            or (identities and identity != identities)
        ):
            raise ValueError("Distinto campo, fijas, inicio, horizonte o casos")
        identities = identity
        rows = _neural(metrics, "cases.csv")
        cases = {int(r["case_seed"]): r for r in rows}
        if len(rows) != len(seeds) or set(cases) != set(seeds):
            raise ValueError("Métricas incompletas o duplicadas")
        scores = {
            k: np.array([float(cases[s][k]) for s in seeds]) * scale
            for k, _, scale in METRICS
        }
        neural = next(s for s in summary["strategies"] if s["strategy"] == "neural")
        for k, _, scale in METRICS:
            if not np.isclose(
                scores[k].mean(), neural[k] * scale, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"El promedio por caso no reproduce {k}")
        values.append(scores)
        ticks = np.full((len(seeds), 37), np.nan)
        for r in _neural(metrics, "ticks.csv"):
            t = float(r["elapsed_s"])
            i, j = seeds.index(int(r["case_seed"])), int(t / 300)
            if not 0 <= j <= 36 or t != j * 300 or not np.isnan(ticks[i, j]):
                raise ValueError("Reloj de misión distinto o repetido")
            ticks[i, j] = float(r["mse"])
        if (
            not np.isfinite(ticks).all()
            or not np.allclose(ticks[:, -1], scores["final_mse"])
            or not np.allclose(np.trapezoid(ticks, axis=1) / 36, scores["temporal_mse"])
        ):
            raise ValueError("La evolución guardada no reproduce los RMSE de cada caso")
        curves.append(np.sqrt(ticks.mean(axis=0)))
        refs = summary["neural_vs_references"]
        mobile = [r for r in refs if r["reference"] != "fixed"]
        if {r["reference"] for r in refs} != {
            "random",
            "sweep",
            "greedy",
            "fixed",
        } or len(refs) != 4:
            raise ValueError("Faltan referencias propias del estimador")
        references.extend({"label": label, **r} for r in refs)
        references.append(
            {
                "label": label,
                **{
                    k: "promedio tres móviles"
                    if k == "reference"
                    else np.mean([r[k] for r in mobile])
                    for k in mobile[0]
                },
            }
        )

    windows = []
    for group, pair in strata.items():
        for s in pair:
            start, horizon, *_ = identities[s]
            begin = datetime.fromisoformat(start)
            if float(horizon) != 10800:
                raise ValueError("Se requieren misiones de tres horas")
            windows.append(
                {
                    "case_seed": s,
                    "stratum": group,
                    "start_utc": start,
                    "end_utc": (begin + timedelta(seconds=float(horizon))).isoformat(),
                }
            )
    chronological = sorted(windows, key=lambda r: r["start_utc"])
    if any(
        a["end_utc"] >= b["start_utc"] for a, b in pairwise(chronological)
    ):
        raise ValueError(
            "Ventanas compartidas: remuestrear por bloques, no por escenario"
        )
    write_csv(output / "windows.csv", windows)
    summaries, pairs = [], []
    labels = [spec[k][0] for k in ("fixed", "adaptive_final", "adaptive_best")]
    for i, other in enumerate(values[1:], 1):
        for key, name, _ in METRICS:
            res, diff = contrast(
                values[0][key], other[key], indices, root=key.endswith("mse")
            )
            summaries.append({"adaptive": labels[i], "metric": name, **res})
            pairs.extend(
                {"adaptive": labels[i], "metric": name, "case_seed": s, "delta": d}
                for s, d in zip(seeds, diff, strict=True)
            )
    write_csv(output / "intervals.csv", summaries)
    write_csv(output / "paired.csv", pairs)
    write_csv(output / "references.csv", references)
    write_csv(
        output / "clock.csv",
        [
            dict(
                minutes=j * 5,
                **{label: c[j] for label, c in zip(labels, curves, strict=True)},
            )
            for j in range(37)
        ],
    )
    figure = Figure(figsize=(11, 7), layout="constrained")
    for ax, (key, name, _) in zip(figure.subplots(2, 1), METRICS[:2], strict=True):
        for i, other in enumerate(values[1:], 1):
            ax.plot(
                range(len(seeds)),
                np.sqrt(values[0][key]) - np.sqrt(other[key]),
                "o",
                fillstyle="none" if i == 1 else "full",
                label=f"Fijo menos {labels[i]}",
            )
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(range(len(seeds)), seeds, rotation=30)
        ax.set(ylabel=name, xlabel="Semilla de caso (dos por estrato)")
        ax.legend()
    figure.suptitle(
        "Diferencias por escenario de validación · valores negativos favorecen al fijo"
    )
    clock = Figure(figsize=(9, 4), layout="constrained")
    ax = clock.subplots()
    for label, curve in zip(labels, curves, strict=True):
        ax.plot(np.arange(37) * 5, curve, label=label)
    ax.set(xlabel="Minutos de misión", ylabel="RMSE global por instante (µg/m³)")
    ax.legend()
    return summaries, references, figure, clock
