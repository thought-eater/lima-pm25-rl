"""Informe offline de entrenamiento y evaluación, sin reconstruir ni simular."""

import csv
import io
import json
from dataclasses import asdict
from html import escape
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .files import PROJECT_ROOT, product, source_files, unique_file, write_csv
from .hotspot_evaluation import field_protocol
from .training import TrainingConfig, learning_protocol

DEFAULT_CONFIG = PROJECT_ROOT / "config/informe.json"
VALIDATION = {
    "validation_reward": "reward",
    "validation_variance": "final_integrated_variance",
    "validation_mean_variance": "mean_integrated_variance",
}
STRATEGIES = {
    "fixed": "Solo fijas",
    "random": "Aleatoria",
    "greedy": "Greedy",
    "sweep": "Barrido",
}


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


def trailing_mean(values, window):
    """Ventana retrospectiva; ausencias siguen ausentes en su propio instante."""
    if type(window) is not int or window < 1:
        raise ValueError("La ventana debe ser un entero positivo")
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(finite, values, 0))]
    counts = np.r_[0, np.cumsum(finite)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    result = np.full(len(values), np.nan)
    valid = finite & (counts[end] > counts[start])
    result[valid] = (sums[end[valid]] - sums[start[valid]]) / (
        counts[end[valid]] - counts[start[valid]]
    )
    return result


def load_training(paths, label, *, end_episodes=None):
    """Unir tramos explícitos; una cola descartada nunca modifica el CSV fuente."""
    ends = [None] * len(paths) if end_episodes is None else end_episodes
    if not isinstance(ends, list) or len(ends) != len(paths):
        raise ValueError("end_episodes debe indicar un límite por carpeta fuente")
    rows, validation, sources, boundaries = [], [], [], []
    expected = None
    for path, end in zip(paths, ends, strict=True):
        files = source_files(
            path,
            {
                "config.json",
                "metrics.csv",
                "validation-initial.json",
                "validation-resume.json",
            },
        )
        config = json.loads(unique_file(files, "config.json"))
        parameters = asdict(TrainingConfig(**config["training"]))
        parameters = {
            k: v for k, v in parameters.items() if k not in {"device", "cpu_threads"}
        }
        identity = (
            parameters,
            learning_protocol(config["protocol"]),
            config["validation_seeds"],
        )
        if expected is not None and identity != expected:
            raise ValueError(
                "No se pueden unir entrenamientos de distinto protocolo, gamma o semilla"
            )
        expected = identity
        start = config["start_episode"]
        if start != (int(rows[-1]["episode"]) if rows else 0):
            raise ValueError(
                "Los tramos deben comenzar en cero y continuar sin huecos ni solapamientos"
            )
        raw = read_csv(unique_file(files, "metrics.csv"))
        if [int(r["episode"]) for r in raw] != list(
            range(start + 1, start + len(raw) + 1)
        ) or start + len(raw) > config["target_episodes"]:
            raise ValueError(
                "El CSV tiene huecos, solapamientos o excede el presupuesto declarado"
            )
        if not raw:
            raise ValueError("Tramo sin episodios")
        recorded_end = int(raw[-1]["episode"])
        if end is not None:
            if type(end) is not int or not start < end <= recorded_end:
                raise ValueError("El final efectivo debe existir dentro del tramo")
            raw = raw[: end - start]
        if rows:
            if not config["resumed_from"]:
                raise ValueError("El tramo posterior no declara reanudación")
            for name, metric in (
                ("start_physical_steps", "physical_steps"),
                ("start_optimizer_updates", "optimizer_updates"),
                ("start_target_updates", "target_updates"),
            ):
                if config[name] != int(rows[-1][metric]):
                    raise ValueError(
                        f"El contador {name} no continúa el tramo anterior"
                    )
            boundaries.append(start)
            if "validation-resume.json" in files:
                # El corte inicial del tramo puede aportar RMSE/MAE que los CSV
                # antiguos no registraban. Conservar las mediciones originales.
                measurement = json.loads(files["validation-resume.json"])
                if not validation or validation[-1]["episode"] != start:
                    validation.append({"episode": start})
                for key in (
                    *VALIDATION.values(),
                    "synthetic_mse",
                    "synthetic_mae_ug_m3",
                ):
                    if key in measurement:
                        validation[-1].setdefault(key, measurement[key])
        for r in raw:
            row = {key: float(value) if value else None for key, value in r.items()}
            if any(
                value is not None and not np.isfinite(value) for value in row.values()
            ):
                raise ValueError("El entrenamiento contiene valores no finitos")
            for field in (
                "episode",
                "scenario_seed",
                "physical_steps",
                "optimizer_updates",
                "target_updates",
            ):
                if row[field] is None or row[field] != int(row[field]):
                    raise ValueError(f"Contador inválido: {field}")
                row[field] = int(row[field])
            previous_steps = rows[-1]["physical_steps"] if rows else 0
            if row["physical_steps"] != previous_steps + row["episode_steps"]:
                raise ValueError("Los ticks acumulados no coinciden con los episodios")
            if rows and any(
                row[k] < rows[-1][k] for k in ("optimizer_updates", "target_updates")
            ):
                raise ValueError("Contadores de optimización decrecientes")
            seed = int(
                np.random.SeedSequence(
                    [parameters["seed"], 0, row["episode"] - 1]
                ).generate_state(1)[0]
            )
            if row["scenario_seed"] != seed:
                raise ValueError(
                    "La secuencia de escenarios no corresponde a la semilla"
                )
            rows.append(row)
            if any(row[k] is not None for k in VALIDATION):
                if any(row[k] is None for k in VALIDATION):
                    raise ValueError("Validación incompleta")
                validation.append(
                    {
                        "episode": row["episode"],
                        **{out: row[k] for k, out in VALIDATION.items()},
                        **{
                            out: row[k]
                            for k, out in (
                                ("validation_mse", "synthetic_mse"),
                                ("validation_mae_ug_m3", "synthetic_mae_ug_m3"),
                            )
                            if row.get(k) is not None
                        },
                    }
                )
        initial = [
            data
            for name, data in files.items()
            if Path(name).name == "validation-initial.json"
        ]
        if initial:
            if start != 0 or len(initial) != 1:
                raise ValueError("La validación inicial pertenece solo al primer tramo")
            values = json.loads(initial[0])
            validation.insert(
                0,
                {
                    "episode": 0,
                    **{k: values[k] for k in VALIDATION.values()},
                    **{
                        k: values[k]
                        for k in ("synthetic_mse", "synthetic_mae_ug_m3")
                        if k in values
                    },
                },
            )
        sources.append(
            {
                "path": str(path),
                "config": config,
                "effective_end_episode": int(raw[-1]["episode"]),
                "recorded_end_episode": recorded_end,
                "excluded_tail_episodes": recorded_end - int(raw[-1]["episode"]),
            }
        )
    if not rows:
        raise ValueError("Se requiere al menos un tramo de entrenamiento")
    # Nuevas columnas de seguimiento pueden aparecer al reanudar una corrida.
    columns = dict.fromkeys(key for row in rows for key in row)
    rows = [{key: row.get(key) for key in columns} for row in rows]
    validation_columns = dict.fromkeys(key for row in validation for key in row)
    validation = [
        {key: row.get(key) for key in validation_columns} for row in validation
    ]
    return {
        "label": label,
        "rows": rows,
        "validation": validation,
        "boundaries": boundaries,
        "sources": sources,
        "parameters": expected[0],
        "protocol": expected[1],
        "target_episodes": config["target_episodes"],
    }


def block_summary(run, size):
    result = []
    for start in range(0, len(run["rows"]), size):
        rows = run["rows"][start : start + size]
        val = [
            r
            for r in run["validation"]
            if rows[0]["episode"] <= r["episode"] <= rows[-1]["episode"]
        ]
        item = {
            "run": run["label"],
            "first_episode": rows[0]["episode"],
            "last_episode": rows[-1]["episode"],
            "episodes": len(rows),
            "validations": len(val),
        }
        for key in (
            "reward",
            "loss",
            "information_gain",
            "distance_cost",
            "distance_m",
            "final_integrated_variance",
            "mean_integrated_variance",
        ):
            values = [r[key] for r in rows if r[key] is not None]
            item[f"training_mean_{key}"] = float(np.mean(values)) if values else None
        for key in VALIDATION.values():
            item[f"validation_mean_{key}"] = (
                float(np.mean([r[key] for r in val])) if val else None
            )
        item["physical_steps"] = sum(r["episode_steps"] for r in rows)
        item["optimizer_updates"] = rows[-1]["optimizer_updates"] - (
            run["rows"][start - 1]["optimizer_updates"] if start else 0
        )
        result.append(item)
    return result


def save_figure(figure, output, name):
    for extension in ("png", "svg"):
        with (output / f"{name}.{extension}").open("xb") as file:
            figure.savefig(file, format=extension, dpi=130)
    figure.clear()
    return f"{name}.png"


def learning_figures(runs, output, window):
    panels = (
        (
            "learning",
            "Aprendizaje: datos originales y tendencia",
            (
                ("reward", "Recompensa conjunta por episodio", 1),
                ("loss", "Pérdida Double DQN (escala logarítmica*)", 1),
                ("information_gain", "Reducción firmada de varianza acumulada", 1),
                ("distance_cost", "Coste vial acumulado", 1),
            ),
        ),
        (
            "operation",
            "Recorrido e incertidumbre durante el entrenamiento",
            (
                ("distance_m", "Distancia vial de flota (km)", 0.001),
                ("final_integrated_variance", "Varianza final integrada (10⁹)", 1e-9),
                ("mean_integrated_variance", "Varianza media temporal (10⁹)", 1e-9),
                ("episode_steps", "Ticks físicos por episodio", 1),
            ),
        ),
        (
            "schedule",
            "Exploración y trabajo efectivo: sin suavizado",
            (
                ("epsilon_end", "Epsilon al terminar el episodio", 1),
                ("physical_steps", "Ticks físicos acumulados", 1),
                ("optimizer_updates", "Actualizaciones Adam acumuladas", 1),
                ("target_updates", "Actualizaciones target acumuladas", 1),
            ),
        ),
    )
    figures = []
    for name, title, metrics in panels:
        figure = Figure(figsize=(13, 8), layout="constrained")
        FigureCanvasAgg(figure)
        for axis, (key, ylabel, scale) in zip(
            figure.subplots(2, 2).ravel(), metrics, strict=True
        ):
            for i, run in enumerate(runs):
                episodes = [r["episode"] for r in run["rows"]]
                values = (
                    np.array(
                        [r[key] if r[key] is not None else np.nan for r in run["rows"]]
                    )
                    * scale
                )
                color = f"C{i % 10}"
                smooth = name != "schedule"
                axis.plot(
                    episodes,
                    values,
                    color=color,
                    alpha=0.22 if smooth else 1,
                    linewidth=0.7 if smooth else 1.4,
                    label=None if smooth else run["label"],
                )
                if smooth:
                    axis.plot(
                        episodes,
                        trailing_mean(values, window),
                        color=color,
                        linewidth=1.5,
                        label=run["label"],
                    )
                for boundary in run["boundaries"]:
                    axis.axvline(
                        boundary, color=color, linestyle="--", linewidth=0.8, alpha=0.7
                    )
            axis.set(
                xlabel="Episodio acumulado",
                ylabel=ylabel,
                xlim=(0, max(len(r["rows"]) for r in runs)),
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            if key == "loss":
                axis.set_yscale("symlog", linthresh=1e-8)
                losses = [
                    r["loss"]
                    for run in runs
                    for r in run["rows"]
                    if r["loss"] is not None
                ]
                if losses:
                    axis.set_ylim(
                        bottom=max(0, min(losses) * 0.8),
                        top=max(max(losses) * 1.2, 1e-8),
                    )
        figure.suptitle(
            title
            + (
                f"\nMedia retrospectiva de {window} episodios; línea tenue = original; discontinua = reanudación"
                if name != "schedule"
                else "\nLínea discontinua = reanudación; no reinicia contadores"
            )
        )
        figures.append(save_figure(figure, output, name))
    figure = Figure(figsize=(15, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, (key, title, scale) in zip(
        figure.subplots(1, 3),
        (
            ("reward", "Recompensa de validación", 1),
            ("final_integrated_variance", "Varianza final integrada (10⁹)", 1e-9),
            ("mean_integrated_variance", "Varianza media temporal (10⁹)", 1e-9),
        ),
        strict=True,
    ):
        for i, run in enumerate(runs):
            rows = run["validation"]
            axis.plot(
                [r["episode"] for r in rows],
                [r[key] * scale for r in rows],
                ".-",
                markersize=4,
                linewidth=0.9,
                label=run["label"],
                color=f"C{i % 10}",
            )
            for boundary in run["boundaries"]:
                axis.axvline(
                    boundary, color=f"C{i % 10}", linestyle="--", linewidth=0.8
                )
        axis.set(xlabel="Episodio acumulado", ylabel=title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Validación sobre casos fijos: puntos realmente medidos\nLas líneas unen mediciones; no interpolan datos ni representan incertidumbre estadística"
    )
    figures.append(save_figure(figure, output, "validation"))
    measured = [
        run
        for run in runs
        if any(r.get("synthetic_mse") is not None for r in run["validation"])
    ]
    if measured:
        figure = Figure(figsize=(12, 4), layout="constrained")
        FigureCanvasAgg(figure)
        for axis, key, label in zip(
            figure.subplots(1, 2),
            ("synthetic_mse", "synthetic_mae_ug_m3"),
            ("RMSE final (µg/m³)", "MAE final (µg/m³)"),
            strict=True,
        ):
            for run in measured:
                rows = [r for r in run["validation"] if r.get(key) is not None]
                values = [
                    np.sqrt(r[key]) if key == "synthetic_mse" else r[key] for r in rows
                ]
                axis.plot(
                    [r["episode"] for r in rows], values, ".-", label=run["label"]
                )
            axis.set(xlabel="Episodio acumulado", ylabel=label)
            axis.legend()
            axis.grid(alpha=0.2)
        figures.append(save_figure(figure, output, "validation-errors"))
    return figures


def table_html(rows, columns):
    header = "".join(f"<th>{escape(title)}</th>" for _, title in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            text = (
                "—"
                if value is None
                else (f"{value:.4g}" if isinstance(value, float) else str(value))
            )
            cells.append(f"<td>{escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='scroll'><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def comparison_section(spec, output, number):
    # Solo PNG prefijados y tablas existentes; no se leen NPZ ni se ajusta el GP.
    names = {
        "source.json",
        "summary.json",
        "cases.csv",
        "references.csv",
        "protocol.json",
        "curves.png",
        "tick-01.png",
        "tick-12.png",
        "tick-24.png",
        "tick-36.png",
    }
    protocol = json.loads(
        (product(spec["protocol_product"]) / "protocol.json").read_text()
    )
    if protocol["version"] != 2:
        raise ValueError("El informe requiere la evaluación corregida versión 2")
    selected, baselines, manifest, references, maps, cases, mission_curves = (
        [],
        {},
        [],
        [],
        [],
        [],
        [],
    )
    folder = output / f"comparison-{number}"
    folder.mkdir()
    for evaluation, label in spec["products"].items():
        subset = source_files(product(evaluation), names)
        source = json.loads(unique_file(subset, "source.json"))
        summary = json.loads(unique_file(subset, "summary.json"))
        if (
            source["smoke_only"]
            or summary["smoke_only"]
            or not summary["reproduction_passed"]
        ):
            raise ValueError(
                "No se presentan comprobaciones smoke como resultados completos"
            )
        metadata = source["metadata"]
        if field_protocol(metadata["protocol"]) != field_protocol(protocol["protocol"]):
            raise ValueError("Los resultados no corresponden al protocolo congelado")
        strategies = {row["strategy"]: row for row in summary["strategies"]}
        if set(strategies) != {"neural", *STRATEGIES} or any(
            row["cases"] != len(metadata["case_seeds"]) for row in strategies.values()
        ):
            raise ValueError("Faltan políticas o casos en la comparación")
        actual_baselines = {k: strategies[k] for k in STRATEGIES}
        method = json.dumps(
            {
                k: v
                for k, v in metadata["protocol"].items()
                if k not in field_protocol(metadata["protocol"])
                and k != "validation_every"
            },
            sort_keys=True,
        )
        if method in baselines and actual_baselines != baselines[method][1]:
            raise ValueError(
                "Las referencias no coinciden entre las políticas comparadas"
            )
        baselines.setdefault(method, (label, actual_baselines))
        case_rows = [
            row
            for row in read_csv(unique_file(subset, "cases.csv"))
            if row["strategy"] == "neural"
        ]
        if len(case_rows) != len(metadata["case_seeds"]) or {
            int(r["case_seed"]) for r in case_rows
        } != set(metadata["case_seeds"]):
            raise ValueError(
                "Los puntos pareados deben incluir todos los casos prefijados"
            )
        cases.append({int(row["case_seed"]): row for row in case_rows})
        if set(cases[-1]) != set(cases[0]):
            raise ValueError("Las evaluaciones deben compartir casos")
        selected.append({"label": label, **strategies["neural"]})
        references.extend(
            {"label": label, **row} for row in summary["neural_vs_references"]
        )
        manifest.append({"product": evaluation, "label": label, "source": source})
        curves = unique_file(subset, "curves.png")
        (folder / f"{evaluation}-curves.png").write_bytes(curves)
        mission_curves.append(
            f"<h3>{escape(label)}</h3><img loading='lazy' src='comparison-{number}/{evaluation}-curves.png' alt='Evolución temporal de {escape(label, quote=True)}'>"
        )
        case = metadata["representative_case_seed"]
        for name, data in sorted(subset.items()):
            p = Path(name)
            if (
                p.suffix != ".png"
                or f"case-{case}" not in p.parts
                or p.parent.name != "neural-deterministic"
            ):
                continue
            target = f"{evaluation}-{p.name}"
            (folder / target).write_bytes(data)
            maps.append(
                f"<figure><img loading='lazy' src='comparison-{number}/{target}' alt='{escape(label, quote=True)}'><figcaption>{escape(label)} — caso {case}, tick {int(p.stem.split('-')[-1])}</figcaption></figure>"
            )
    if not selected:
        raise ValueError("Indicar las evaluaciones que se van a presentar")
    changes, paired = [], []
    for index, candidate in enumerate(selected[1:], start=1):
        for key, value in candidate.items():
            if key in {"label", "strategy", "cases"} or key.endswith("_cases"):
                continue
            reference = selected[0][key]
            changes.append(
                {
                    "candidate": candidate["label"],
                    "reference": selected[0]["label"],
                    "metric": key,
                    "reference_value": reference,
                    "candidate_value": value,
                    "delta": value - reference
                    if value is not None and reference is not None
                    else None,
                    "change_percent": 100 * (value / reference - 1)
                    if value is not None and reference
                    else None,
                }
            )
        for seed, row in cases[index].items():
            pair = {
                "candidate": candidate["label"],
                "reference": selected[0]["label"],
                "case_seed": seed,
            }
            for key, value in row.items():
                if key in {"strategy", "case_seed"}:
                    continue
                reference = cases[0][seed][key]
                pair[f"delta_{key}"] = (
                    float(value) - float(reference) if value and reference else None
                )
            paired.append(pair)
    write_csv(folder / "changes.csv", changes)
    write_csv(folder / "paired.csv", paired)
    for label, group in baselines.values():
        selected.extend(
            {
                "label": f"{STRATEGIES[key]} — {label}"
                if len(baselines) > 1
                else STRATEGIES[key],
                **row,
            }
            for key, row in group.items()
        )
    write_csv(folder / "summary.csv", selected)
    write_csv(folder / "references.csv", references)
    metrics = (
        ("final_integrated_variance", "Incertidumbre final (10⁹) ↓", 1e-9),
        ("temporal_rmse_ug_m3", "RMSE global temporal (µg/m³) ↓", 1),
        ("final_rmse_ug_m3", "RMSE global final (µg/m³) ↓", 1),
        ("temporal_hot_rmse_ug_m3", "RMSE en focos temporal (µg/m³) ↓", 1),
        ("temporal_hot_deficit_ug_m3", "Déficit local temporal (µg/m³) ↓", 1),
        ("hot_cell_time_recall", "Sensibilidad temporal (%) ↑", 100),
        ("mean_false_high_area_km2", "Falsa área alta media (km²) ↓", 1),
        ("distance_m", "Recorrido de flota (km) ↓", 0.001),
    )
    figure = Figure(figsize=(16, 8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, (key, title, scale) in zip(
        figure.subplots(2, 4).ravel(), metrics, strict=True
    ):
        values = [
            row[key] * scale if row[key] is not None else np.nan for row in selected
        ]
        axis.bar(
            range(len(selected)),
            values,
            color=[
                f"C{i}" if i < len(spec["products"]) else "#999999"
                for i in range(len(selected))
            ],
        )
        axis.set_xticks(
            range(len(selected)),
            [row["label"] for row in selected],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle(
        spec["title"]
        + "\nPromedios descriptivos: réplicas dentro de caso, después casos; sin intervalos de confianza"
    )
    save_figure(figure, folder, "metrics")
    paired_figures = []
    labels = list(spec["products"].values())
    for i, candidate in enumerate(cases[1:], start=1):
        figure = Figure(figsize=(11, 8), layout="constrained")
        FigureCanvasAgg(figure)
        for axis, (key, title, scale) in zip(
            figure.subplots(2, 2).ravel(),
            (
                ("final_mse", "RMSE final (µg/m³): menor es mejor", 1),
                (
                    "temporal_hot_deficit_ug_m3",
                    "Déficit local temporal (µg/m³): menor es mejor",
                    1,
                ),
                (
                    "hot_cell_time_recall",
                    "Sensibilidad temporal (%): mayor es mejor",
                    100,
                ),
                ("distance_m", "Distancia de flota (km): menor es mejor", 0.001),
            ),
            strict=True,
        ):
            points = []
            for seed in sorted(cases[0]):
                a, b = cases[0][seed][key], candidate[seed][key]
                if not a or not b:
                    continue
                a, b = float(a), float(b)
                if key == "final_mse":
                    a, b = np.sqrt(a), np.sqrt(b)
                a, b = a * scale, b * scale
                points.append((a, b))
                axis.scatter(a, b, s=24)
                axis.annotate(
                    str(seed),
                    (a, b),
                    fontsize=6,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
            if points:
                low, high = np.min(points), np.max(points)
                axis.plot([low, high], [low, high], "--", color="#555", linewidth=0.8)
            axis.set(title=title, xlabel=labels[0], ylabel=labels[i])
            axis.margins(0.12)
            axis.grid(alpha=0.2)
        figure.suptitle(
            f"{labels[i]} frente a {labels[0]} — cada punto es un caso\nDiagonal = igualdad; todos los casos definidos, sin selección por resultado"
        )
        paired_figures.append(save_figure(figure, folder, f"paired-{i}"))
    columns = [
        ("label", "Política"),
        ("final_rmse_ug_m3", "RMSE final"),
        ("temporal_hot_deficit_ug_m3", "Déficit temporal"),
        ("hot_cell_time_recall", "Sensibilidad (fracción)"),
        ("mean_false_high_area_km2", "Falsa área (km²)"),
        ("distance_m", "Recorrido (m)"),
        (
            "cell_event_completed_missed_fraction",
            "Eventos completados omitidos (fracción)",
        ),
    ]
    html = f"<h2>{escape(spec['title'])}</h2><p>Umbral experimental: {protocol['threshold_ug_m3']:.6f} µg/m³. No es sanitario.</p>"
    if len(baselines) > 1:
        html += "<p>Cada método conserva sus propias referencias y su estimador. La varianza es una predicción del GP; reducirla al cambiar la amplitud no demuestra menor error. El contraste combina estimador, recompensa y entrenamiento.</p>"
    html += f"<img src='comparison-{number}/metrics.png' alt='Comparación de métricas'>"
    html += table_html(selected, columns)
    html += "<h3>Variación entre casos</h3>" + "".join(
        f"<img src='comparison-{number}/{name}' alt='Comparación pareada por caso'>"
        for name in paired_figures
    )
    html += "<h3>Porcentajes frente a las referencias y meta del 20 %</h3>"
    html += table_html(
        references,
        [
            ("label", "Política"),
            ("reference", "Referencia"),
            ("target_percent", "Meta (%)"),
            ("variance_reduction_percent", "Reducción varianza (%)"),
            ("rmse_reduction_percent", "Reducción RMSE (%)"),
            ("distance_reduction_percent", "Reducción recorrido (%)"),
        ],
    )
    html += "<p>Porcentajes calculados después de agregar las métricas. Reducir recorrido un 20 % no equivale a cumplir el indicador general. Las falsas alarmas deben acompañar sensibilidad y déficit. El retraso solo entre detectados no resume los eventos omitidos o censurados.</p>"
    html += f"<p><a href='comparison-{number}/summary.csv'>Tabla completa CSV</a> · <a href='comparison-{number}/references.csv'>Referencias CSV</a> · <a href='comparison-{number}/metrics.svg'>Figura SVG</a></p>"
    if changes:
        html += f"<p><a href='comparison-{number}/changes.csv'>Cambios agregados</a> · <a href='comparison-{number}/paired.csv'>Diferencias por caso</a>. Cambios = candidato menos referencia; el signo favorable depende de la métrica.</p>"
    html += (
        "<details><summary>Evolución durante la misión y calidad frente al recorrido</summary>"
        + "".join(mission_curves)
        + "</details>"
    )
    html += (
        "<details><summary>Mapas guardados del caso prefijado: concentración, incertidumbre, muestras y trayectoria final</summary>"
        + "".join(maps)
        + "</details>"
    )
    return html, {
        "protocol": protocol,
        "evaluations": manifest,
    }


def generate_report(settings, output):
    window, block_size = settings["smoothing_episodes"], settings["block_episodes"]
    if any(type(value) is not int or value < 1 for value in (window, block_size)):
        raise ValueError("Ventanas y bloques deben ser enteros positivos")
    specs = settings["training_runs"]
    if not specs or len({s["label"] for s in specs}) != len(specs):
        raise ValueError("Se requieren entrenamientos con etiquetas distintas")
    runs = [
        load_training(
            [product(p) for p in s["sources"]],
            s["label"],
            end_episodes=s.get("end_episodes"),
        )
        for s in specs
    ]
    # Las configuraciones del método pueden variar, los escenarios deben coincidir.
    reference = runs[0]
    for run in runs[1:]:
        if field_protocol(run["protocol"]) != field_protocol(reference["protocol"]):
            raise ValueError("Las curvas comparadas deben compartir los escenarios")
    output.mkdir(parents=True, exist_ok=False)
    blocks = []
    for i, run in enumerate(runs, start=1):
        write_csv(output / f"training-{i}.csv", run["rows"])
        write_csv(output / f"validation-{i}.csv", run["validation"])
        blocks.extend(block_summary(run, block_size))
    write_csv(output / "blocks.csv", blocks)
    groups = {}
    for run in runs:
        key = json.dumps(
            {
                k: v
                for k, v in run["protocol"].items()
                if k not in field_protocol(run["protocol"]) and k != "validation_every"
            },
            sort_keys=True,
        )
        groups.setdefault(key, []).append(run)
    figures = []
    for index, group in enumerate(groups.values(), start=1):
        folder = output if len(groups) == 1 else output / f"method-{index}"
        if folder != output:
            folder.mkdir()
        figures.extend(
            str((folder / figure).relative_to(output))
            for figure in learning_figures(group, folder, window)
        )
    notes = (
        "La curva de entrenamiento mezcla escenarios distintos y exploración; la validación usa los mismos casos. "
        "Una pérdida menor no implica una política mejor, y pérdidas/Q con distinto gamma no tienen la misma escala objetivo. "
        "*La pérdida usa escala logarítmica, lineal cerca de cero (±10⁻⁸), sin ocultar ceros. "
        "El suavizado es retrospectivo, usa solo valores disponibles y conserva los datos originales; no es un intervalo de confianza. "
        "Las líneas de reanudación marcan continuidad del mismo entrenamiento, no una semilla nueva. "
        "Las colas excluidas mediante end_episodes se conservan en los CSV originales; no entran en las curvas ni en los bloques. "
        "Una meseta visual no prueba convergencia. Comparar también focos, errores y recorrido, incluyendo deterioros. "
        "Son escenarios sintéticos y una semilla por brazo; no demuestran desempeño físico ni beneficio sanitario. "
        "El replay histórico final sigue pendiente y no se usa aquí."
        " Recompensas y varianzas de estimadores distintos se grafican por separado: sus valores nativos no son una escala común de calidad."
    )
    html = "<!doctype html><html lang='es'><meta charset='utf-8'><title>Entrenamiento y evaluación O3</title><style>body{font:16px system-ui;max-width:1500px;margin:2em auto;padding:0 1em;color:#222}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:.55em;border:1px solid #ddd;text-align:right}th{background:#eee}.scroll{overflow:auto}summary{cursor:pointer;font-weight:bold}figure{margin:1em 0}</style><body>"
    html += "<h1>Entrenamiento y evaluación O3</h1><p>" + escape(notes) + "</p>"
    html += "<h2>Ejecuciones incluidas</h2>" + table_html(
        [
            {
                "label": r["label"],
                "gamma": r["parameters"]["gamma"],
                "episodes": len(r["rows"]),
                "target": r["target_episodes"],
                "physical_steps": r["rows"][-1]["physical_steps"],
                "updates": r["rows"][-1]["optimizer_updates"],
                "invalid": sum(row["invalid_actions"] for row in r["rows"]),
                "resumes": ", ".join(map(str, r["boundaries"])) or "—",
                "excluded": sum(s["excluded_tail_episodes"] for s in r["sources"]),
            }
            for r in runs
        ],
        [
            ("label", "Ejecución"),
            ("gamma", "Gamma"),
            ("episodes", "Episodios"),
            ("target", "Objetivo declarado"),
            ("physical_steps", "Ticks"),
            ("updates", "Adam"),
            ("invalid", "Acciones inválidas"),
            ("resumes", "Reanudaciones"),
            ("excluded", "Episodios de colas excluidas"),
        ],
    )
    for figure in figures:
        html += f"<figure><img src='{figure}' alt='{figure}'><figcaption><a href='{Path(figure).with_suffix('.svg')}'>Versión SVG</a></figcaption></figure>"
    html += f"<h2>Bloques de {block_size} episodios</h2><p>No confundir medias de entrenamiento con validación. <a href='blocks.csv'>Tabla completa CSV</a>.</p>"
    html += table_html(
        blocks,
        [
            ("run", "Ejecución"),
            ("first_episode", "Desde"),
            ("last_episode", "Hasta"),
            ("training_mean_reward", "Recompensa entrenamiento"),
            ("training_mean_loss", "Pérdida"),
            ("validation_mean_final_integrated_variance", "Varianza final validación"),
            ("validations", "Validaciones"),
            ("training_mean_distance_m", "Recorrido entrenamiento (m)"),
        ],
    )
    comparisons = []
    for i, spec in enumerate(settings["comparisons"], start=1):
        section, manifest = comparison_section(spec, output, i)
        html += section
        comparisons.append(manifest)
    html += (
        "<h2>Pendientes</h2><p>"
        + escape(settings["pending_note"])
        + "</p></body></html>"
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "training": [
                    {k: v for k, v in r.items() if k not in {"rows", "validation"}}
                    for r in runs
                ],
                "comparisons": comparisons,
                "training_executed": False,
                "simulation_executed": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return output / "index.html"


def run(args):
    settings = json.loads(args.config.read_text())
    output = args.output
    print(generate_report(settings, output), flush=True)
