"""Figuras humanas del evaluador; nunca son entradas del agente."""

from collections import defaultdict

import numpy as np


def save_maps(
    grid, fields, means, deviations, records, routes, *, threshold, ticks, output
):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    concentration_max = max(threshold, float(fields["truth"].max()))
    for tick in ticks:
        truth, mean, std = fields["truth"][tick], means[tick], deviations[tick]
        hot, detected = truth >= threshold, mean >= threshold
        classes = np.zeros(len(truth))
        classes[hot & detected] = 1
        classes[hot & ~detected] = 2
        classes[~hot & detected] = 3
        figure = Figure(figsize=(13, 4), layout="constrained")
        FigureCanvasAgg(figure)
        for index, (axis, values, title) in enumerate(
            zip(
                figure.subplots(1, 4),
                (truth, mean, std, classes),
                (
                    "Verdad sintética privada",
                    "Media GP (sin recorte numérico)",
                    "Desviación GP",
                    "0: fondo; 1: acierto; 2: omisión; 3: falsa alarma",
                ),
                strict=True,
            )
        ):
            raster = np.full(grid.shape, np.nan)
            raster[grid.active_mask] = values
            image = axis.imshow(
                raster,
                origin="lower",
                extent=grid.grid_bounds_m[[0, 2, 1, 3]] / 1000,
                vmin=0,
                vmax=3 if index == 3 else (50 if index == 2 else concentration_max),
                cmap=ListedColormap(["#eeeeee", "#229944", "#cc2222", "#e5a832"])
                if index == 3
                else "viridis",
            )
            eligible = [
                r for r in records if r.simulation_time_s <= fields["elapsed_s"][tick]
            ]
            for sensor_type, marker, color in (
                ("fixed", "^", "black"),
                ("mobile", ".", "white"),
            ):
                selected = [r for r in eligible if r.sensor_type == sensor_type]
                axis.scatter(
                    [r.x_m / 1000 for r in selected],
                    [r.y_m / 1000 for r in selected],
                    marker=marker,
                    c=color,
                    s=8,
                )
            if tick == len(means) - 1:
                for points in routes.values():
                    xy = np.asarray(points) / 1000
                    axis.plot(xy[:, 0], xy[:, 1], linewidth=0.4, color="white")
            axis.set(title=title, xlabel="Este UTM (km)", ylabel="Norte UTM (km)")
            figure.colorbar(
                image, ax=axis, shrink=0.65, extend="max" if index != 3 else "neither"
            )
        figure.suptitle(
            f"t={fields['elapsed_s'][tick] / 60:g} min | umbral experimental {threshold:.3f} µg/m³ | escalas comunes; saturación solo visual"
        )
        with (output / f"tick-{tick:02d}.png").open("xb") as file:
            figure.savefig(file, format="png", dpi=110)


def save_curves(rows, summaries, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Promedio de réplicas dentro de cada caso antes de agregar casos.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"], row["case_seed"], row["elapsed_s"]].append(row)
    curves = defaultdict(list)
    for (strategy, _, time), replicas in grouped.items():
        curves[strategy, time].append(
            {
                key: float(np.mean([r[key] for r in replicas]))
                for key in (
                    "distance_m",
                    "integrated_variance",
                    "hot_deficit_sum_ug_m3",
                    "hot_cells",
                    "true_positive_cells",
                    "false_high_area_km2",
                )
            }
        )
    figure = Figure(figsize=(13, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3).ravel()
    for strategy in dict.fromkeys(r["strategy"] for r in rows):
        times = sorted(time for name, time in curves if name == strategy)
        values = []
        for time in times:
            cases = curves[strategy, time]
            hot = [r for r in cases if r["hot_cells"] > 0]
            values.append(
                [
                    np.mean([r["integrated_variance"] for r in cases]),
                    np.mean([r["true_positive_cells"] / r["hot_cells"] for r in hot])
                    if hot
                    else np.nan,
                    np.mean([r["hot_deficit_sum_ug_m3"] / r["hot_cells"] for r in hot])
                    if hot
                    else np.nan,
                    np.mean([r["false_high_area_km2"] for r in cases]),
                    np.mean([r["distance_m"] for r in cases]) / 1000,
                ]
            )
        values = np.asarray(values)
        for index in range(4):
            axes[index].plot(np.asarray(times) / 60, values[:, index], label=strategy)
        axes[4].plot(values[:, 4], values[:, 0], label=strategy)
        summary = next(r for r in summaries if r["strategy"] == strategy)
        if summary["temporal_hot_deficit_ug_m3"] is not None:
            axes[5].scatter(
                summary["distance_m"] / 1000,
                summary["temporal_hot_deficit_ug_m3"],
                label=strategy,
            )
    titles = (
        "Varianza integrada",
        "Sensibilidad por tick",
        "Déficit local (µg/m³)",
        "Falsa área alta (km²)",
        "Incertidumbre frente a recorrido",
        "Déficit temporal frente a recorrido total",
    )
    for index, axis in enumerate(axes):
        axis.set(
            title=titles[index],
            xlabel="Tiempo (min)" if index < 4 else "Distancia vial (km)",
        )
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    with (output / "curves.png").open("xb") as file:
        figure.savefig(file, format="png", dpi=110)
