"""Métricas privadas del evaluador: focos, incertidumbre y esfuerzo vial."""

from collections import defaultdict

import numpy as np

from .comparison import _reduction


def frame_metrics(truth, mean, std, *, threshold, cell_area_m2, coordinates):
    truth, mean, std = (np.asarray(a, dtype=float) for a in (truth, mean, std))
    coordinates = np.asarray(coordinates, dtype=float)
    if (
        truth.ndim != 1
        or not truth.size
        or mean.shape != truth.shape
        or std.shape != truth.shape
        or coordinates.shape != (truth.size, 2)
        or not all(np.isfinite(a).all() for a in (truth, mean, std, coordinates))
        or np.any(std < 0)
        or not np.isfinite(threshold)
        or not np.isfinite(cell_area_m2)
        or cell_area_m2 <= 0
    ):
        raise ValueError("Campos, coordenadas y parámetros del evaluador inválidos")
    hot, detected = truth >= threshold, mean >= threshold
    error = mean - truth
    squared = error**2
    deficit = np.maximum(-error, 0)
    maximum = truth.max()
    # Los empates del máximo verdadero se promedian, sin escoger un índice favorable.
    peak_error = mean[truth == maximum] - maximum
    true_peaks, predicted_peaks = (
        coordinates[truth == maximum],
        coordinates[mean == mean.max()],
    )
    distances = np.concatenate(
        [
            np.linalg.norm(
                predicted_peaks[start : start + 256, None] - true_peaks[None, :], axis=2
            ).min(axis=1)
            for start in range(0, len(predicted_peaks), 256)
        ]
    )
    return {
        "mse": float(squared.mean()),
        "mae_ug_m3": float(np.abs(error).mean()),
        "integrated_variance": float(np.sum(std**2) * cell_area_m2),
        "hot_cells": int(hot.sum()),
        "nonhot_cells": int((~hot).sum()),
        "true_positive_cells": int(np.sum(hot & detected)),
        "false_negative_cells": int(np.sum(hot & ~detected)),
        "hot_squared_error_sum": float(squared[hot].sum()),
        "nonhot_squared_error_sum": float(squared[~hot].sum()),
        "hot_deficit_sum_ug_m3": float(deficit[hot].sum()),
        "false_high_area_km2": float(np.sum(~hot & detected) * cell_area_m2 / 1e6),
        "hot_area_km2": float(hot.sum() * cell_area_m2 / 1e6),
        "peak_signed_error_ug_m3": float(peak_error.mean()),
        "peak_absolute_error_ug_m3": float(np.abs(peak_error).mean()),
        "peak_location_distance_m": float(distances.mean()),
    }


def cell_detection_events(truth, mean, times, threshold):
    """Eventos por celda; no asume identidad ni seguimiento de objetos espaciales."""
    hot, detected = truth >= threshold, mean >= threshold
    delays, missed, completed_detected, left_censored, right_censored = [], 0, 0, 0, 0
    for cell in range(hot.shape[1]):
        starts = np.flatnonzero(hot[:, cell] & ~np.r_[False, hot[:-1, cell]])
        for start in starts:
            if start == 0:
                left_censored += 1
                continue
            end = start
            while end + 1 < len(times) and hot[end + 1, cell]:
                end += 1
            hits = np.flatnonzero(detected[start : end + 1, cell])
            if len(hits):
                delays.append(float(times[start + hits[0]] - times[start]))
                completed_detected += int(end < len(times) - 1)
            elif end == len(times) - 1:
                right_censored += 1
            else:
                missed += 1
    return {
        "cell_events_detected": len(delays),
        "cell_events_completed_detected": completed_detected,
        "cell_events_completed_undetected": missed,
        "cell_events_left_censored": left_censored,
        "cell_events_undetected_right_censored": right_censored,
        "cell_event_delay_detected_mean_s": float(np.mean(delays)) if delays else None,
        "cell_event_completed_missed_fraction": missed / (missed + completed_detected)
        if missed + completed_detected
        else None,
    }


def summarize_mission(rows, times):
    times = np.asarray(times, dtype=float)
    if (
        len(rows) != len(times)
        or len(times) < 2
        or times[0] != 0
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError(
            "Se requiere una misión completa con reloj creciente desde cero"
        )

    def integral(key):
        return float(np.trapezoid([r[key] for r in rows], times))

    def ratio(numerator, denominator):
        total = integral(denominator)
        return integral(numerator) / total if total > 0 else None

    result = {
        "temporal_mse": integral("mse") / times[-1],
        "temporal_mae_ug_m3": integral("mae_ug_m3") / times[-1],
        "mean_integrated_variance": integral("integrated_variance") / times[-1],
        "temporal_hot_mse": ratio("hot_squared_error_sum", "hot_cells"),
        "temporal_nonhot_mse": ratio("nonhot_squared_error_sum", "nonhot_cells"),
        "temporal_hot_deficit_ug_m3": ratio("hot_deficit_sum_ug_m3", "hot_cells"),
        "hot_cell_time_recall": ratio("true_positive_cells", "hot_cells"),
        "hot_cell_time_missed_fraction": ratio("false_negative_cells", "hot_cells"),
        "mean_false_high_area_km2": integral("false_high_area_km2") / times[-1],
        "hot_cell_seconds": integral("hot_cells"),
        "mean_peak_absolute_error_ug_m3": integral("peak_absolute_error_ug_m3")
        / times[-1],
        "mean_peak_location_distance_m": integral("peak_location_distance_m")
        / times[-1],
    }
    final = rows[-1]
    result.update(
        final_mse=final["mse"],
        final_mae_ug_m3=final["mae_ug_m3"],
        final_integrated_variance=final["integrated_variance"],
        final_hot_mse=final["hot_squared_error_sum"] / final["hot_cells"]
        if final["hot_cells"]
        else None,
        final_hot_deficit_ug_m3=final["hot_deficit_sum_ug_m3"] / final["hot_cells"]
        if final["hot_cells"]
        else None,
        final_hot_recall=final["true_positive_cells"] / final["hot_cells"]
        if final["hot_cells"]
        else None,
        final_false_high_area_km2=final["false_high_area_km2"],
    )
    return result


def aggregate_missions(rows):
    """Igual peso por caso tras promediar réplicas. Ausencias explícitas."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["strategy"], row["case_seed"]].append(row)
    excluded = {"strategy", "case_seed", "policy_seed", "reproduction_passed"}
    keys = [key for key in rows[0] if key not in excluded]

    def average(items, key):
        values = [r[key] for r in items if r[key] is not None]
        return float(np.mean(values)) if values else None

    cases = [
        {
            "strategy": strategy,
            "case_seed": seed,
            **{k: average(items, k) for k in keys},
        }
        for (strategy, seed), items in groups.items()
    ]
    summaries = []
    for strategy in dict.fromkeys(row["strategy"] for row in rows):
        selected = [r for r in cases if r["strategy"] == strategy]
        summary = {"strategy": strategy, "cases": len(selected)}
        for key in keys:
            value = average(selected, key)
            summary[key] = value
            if key.endswith("mse"):
                summary[key.replace("mse", "rmse_ug_m3")] = (
                    float(np.sqrt(value)) if value is not None else None
                )
            summary[f"{key}_cases"] = sum(r[key] is not None for r in selected)
        summaries.append(summary)
    paired = []
    index = {(r["strategy"], r["case_seed"]): r for r in cases}
    for (strategy, seed), candidate in index.items():
        if strategy != "neural":
            continue
        for reference in (r for r in summaries if r["strategy"] != "neural"):
            other = index[reference["strategy"], seed]
            pair = {"case_seed": seed, "reference": reference["strategy"]}
            for key in keys:
                a, b = candidate[key], other[key]
                pair[f"delta_{key}"] = (
                    a - b if a is not None and b is not None else None
                )
            paired.append(pair)
    return cases, summaries, paired


def compare_references(summaries):
    """Porcentajes sobre métricas agregadas, como en la evaluación original."""
    neural = next(row for row in summaries if row["strategy"] == "neural")
    return [
        {
            "reference": reference["strategy"],
            "target_percent": 20,
            **{
                name: _reduction(neural[metric], reference[metric])
                for name, metric in (
                    ("variance_reduction_percent", "final_integrated_variance"),
                    ("mean_variance_reduction_percent", "mean_integrated_variance"),
                    ("rmse_reduction_percent", "final_rmse_ug_m3"),
                    ("temporal_rmse_reduction_percent", "temporal_rmse_ug_m3"),
                    ("mae_reduction_percent", "final_mae_ug_m3"),
                    ("distance_reduction_percent", "distance_m"),
                )
            },
        }
        for reference in summaries
        if reference["strategy"] != "neural"
    ]
