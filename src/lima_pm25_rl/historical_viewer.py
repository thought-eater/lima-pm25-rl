"""Convertir la ventana ilustrativa prefijada en un visor sin verdad espacial histórica."""

import json
from pathlib import Path

import numpy as np

from .hotspot_evaluation import measurement_digest, parse_measurements, route_progress
from .map_viewer import _csv, route_frames


def load_replay(replay, grid, latlon):
    proto = json.loads((replay / "protocol.json").read_text())
    i = proto["illustrative_window"]
    folder = replay / f"window-{i + 1:02d}"
    selected = json.loads((folder / "viewer.json").read_text())
    expected_digest = json.loads((folder / "window.json").read_text())["fixed_digest"]
    if {(m["study"], m["strategy"]) for m in selected} != {
        (s, st)
        for s in proto["studies"]
        for st in ("neural", "random", "sweep", "greedy", "fixed")
    } or len(selected) != 10:
        raise ValueError("Selección histórica incompleta")
    missions = []
    for m in selected:
        stem = m["stem"]
        if (
            m["start_utc"] != proto["start_utc"][i]
            or Path(stem).name != stem
            or Path(m["id"]).name != m["id"]
        ):
            raise ValueError("Misión ajena a la ventana prefijada")
        times = np.array([r["elapsed_s"] for r in m["frames"]])
        with np.load(folder / (m["id"] + ".npz"), allow_pickle=False) as data:
            mean, std = data["mean"], data["std"]
            if not np.array_equal(data["cells"], np.argwhere(grid.active_mask)):
                raise ValueError("Grilla de replay incompatible")
        if (
            not np.array_equal(times, np.arange(37) * 300)
            or mean.shape != std.shape
            or mean.shape != (37, int(grid.active_mask.sum()))
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std < 0)
            or not np.allclose(
                np.sum(std**2, axis=1) * grid.cell_area_m2,
                [r["integrated_variance"] for r in m["frames"]],
            )
        ):
            raise ValueError("Reloj, varianza o mapas incompletos")
        recs = parse_measurements(_csv(folder / f"{stem}-samples.csv"))
        fixed = tuple(r for r in recs if r.sensor_type == "fixed")
        if measurement_digest(fixed) != expected_digest or any(
            "PM2_5_SOURCE=senamhi_observed" not in r.provenance for r in fixed
        ):
            raise ValueError("Serie fija histórica distinta de la evaluación")
        routes = {}
        actions = []
        if m["strategy"] == "fixed":
            recs = [r for r in recs if r.sensor_type == "fixed"]
        else:
            for r in _csv(folder / f"{stem}-routes.csv"):
                routes.setdefault(int(r["vehicle"]), []).append(
                    (float(r["x"]), float(r["y"]))
                )
            for r in _csv(folder / f"{stem}-actions.csv"):
                actions.append(
                    {
                        "time": float(r["elapsed_s"]),
                        "vehicle": int(r["vehicle"]),
                        "direction": int(r["direction"]) if r["direction"] else None,
                        "destination": latlon([(float(r["to_x"]), float(r["to_y"]))])[
                            0
                        ],
                    }
                )
        samples = [
            {
                "id": r.sensor_id,
                "type": str(r.sensor_type),
                "time": r.simulation_time_s,
                "position": latlon([(r.x_m, r.y_m)])[0],
                "value": r.value_ug_m3,
            }
            for r in recs
        ]
        trace = m.get("q_trace", [])
        chosen = [
            (r["time"], r["vehicle"] * 8 + r["direction"])
            for r in actions
            if r["direction"] is not None
        ]
        if "q_trace" in m:
            if [(r["time"], r["chosen"]) for r in trace] != chosen:
                raise ValueError(
                    "La traza Q no coincide con las asignaciones ejecutadas"
                )
            for r in trace:
                q, valid = np.asarray(r["q"]), np.asarray(r["valid"], dtype=bool)
                if (
                    q.shape != (3, 8)
                    or valid.shape != q.shape
                    or not np.isfinite(q).all()
                    or not valid.any()
                    or np.argmax(np.where(valid, q, -np.inf)) != r["chosen"]
                ):
                    raise ValueError("Máximo Q admisible incompatible con la decisión")
        route_data = route_frames(routes, recs, times)
        reported = [float(f["distance_m"]) for f in m["frames"]]
        if not np.allclose(
            route_progress(routes, recs, times, reported[-1]),
            reported,
            rtol=1e-8,
            atol=1e-5,
        ):
            raise ValueError("Los recorridos no reproducen las distancias guardadas")
        for r in route_data:
            r["points"] = latlon(r["points"])
        hourly = m["hourly"]
        targets = {
            (r.simulation_time_s, r.sensor_id): r.value_ug_m3
            for r in fixed
            if r.simulation_time_s > 0
        }
        if (
            any(r["last_prediction_s"] != r["elapsed_s"] - 300 for r in hourly)
            or len(hourly) != 9
            or {(r["elapsed_s"], r["station"]): r["observed"] for r in hourly}
            != targets
            or any(
                not np.isclose(r["error"], r["predicted"] - r["observed"])
                for r in hourly
            )
        ):
            raise ValueError("Comparaciones horarias no causales o incompletas")
        metrics = []
        for f in m["frames"]:
            err = np.array(
                [r["error"] for r in hourly if r["elapsed_s"] <= f["elapsed_s"]]
            )
            metrics.append(
                {
                    "distance_km": float(f["distance_m"]) / 1000,
                    "variance": f["integrated_variance"],
                    "mae": float(np.abs(err).mean()) if len(err) else None,
                    "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else None,
                }
            )
        missions.append(
            {
                "id": m["id"],
                "study": m["study"],
                "estimator": "fijo" if m["study"] == "fijo" else "adaptativo",
                "strategy": m["strategy"],
                "strategy_id": m["strategy"],
                "episode": proto["models"][m["study"]][3]
                if m["strategy"] == "neural"
                else None,
                "case": i + 1,
                "historical": True,
                "start": m["start_utc"],
                "decision_end": m["end"],
                "times": times,
                "mean": mean,
                "std": std,
                "routes": route_data,
                "samples": samples,
                "actions": actions,
                "q_trace": trace,
                "hourly": hourly,
                "metrics": metrics,
                "source": {"group": m["id"], "window_start": m["start_utc"]},
            }
        )
    return missions
