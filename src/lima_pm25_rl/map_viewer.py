"""Visor Folium de misiones conservadas: no entrena, simula ni reajusta el GP."""

import base64
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import folium
import numpy as np
import osmnx as ox
import torch
from branca.element import Element, Figure, MacroElement, Template
from lima_pm25.domain import load_grid, load_study_environment_config
from matplotlib import colormaps
from pyproj import Transformer
from shapely.geometry import LineString

from .experiments.gp_spatial_experiment import partition_bounds, validate_frozen_case
from .files import PROJECT_ROOT, product
from .hotspot_evaluation import (
    field_protocol,
    measurement_digest,
    parse_measurements,
    route_progress,
    validate_fixed_reference,
)
from .routing import DIRECTIONS

STATIC = Path(__file__).with_name("static") / "viewer"
DEFAULT_CONFIG = PROJECT_ROOT / "config/visualizacion.json"


def _csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def _sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _mission_key(row):
    return int(row["case_seed"]), row["strategy"], str(row["policy_seed"] or "")


def raster_projection(grid, width):
    """Muestreo vecino más próximo en Web Mercator, desde la grilla UTM original.

    Transformar solo las esquinas de una imagen UTM deformaría su colocación.
    Cada píxel de salida se transforma inversamente y toma su celda original.
    Este muestreo es exclusivamente visual; métricas y consultas usan los arrays.
    """
    if type(width) is not int or not 100 <= width <= 1600:
        raise ValueError("raster_width debe ser un entero entre 100 y 1600")
    to_mercator = Transformer.from_crs(grid.crs, 3857, always_xy=True)
    from_mercator = Transformer.from_crs(3857, grid.crs, always_xy=True)
    to_latlon = Transformer.from_crs(3857, 4326, always_xy=True)
    x0, y0, x1, y1 = grid.grid_bounds_m
    t = np.linspace(0, 1, 33)
    x = np.r_[x0 + (x1 - x0) * t, x0 + (x1 - x0) * t, np.full(33, x0), np.full(33, x1)]
    y = np.r_[np.full(33, y0), np.full(33, y1), y0 + (y1 - y0) * t, y0 + (y1 - y0) * t]
    mx, my = to_mercator.transform(x, y)
    west, east, south, north = min(mx), max(mx), min(my), max(my)
    height = max(1, round(width * (north - south) / (east - west)))
    xx, yy = np.meshgrid(
        west + (np.arange(width) + 0.5) * (east - west) / width,
        north - (np.arange(height) + 0.5) * (north - south) / height,
    )
    gx, gy = from_mercator.transform(xx, yy)
    columns = np.floor((gx - x0) / grid.cell_size_m).astype(int)
    rows = np.floor((gy - y0) / grid.cell_size_m).astype(int)
    inside = (
        (rows >= 0)
        & (rows < grid.shape[0])
        & (columns >= 0)
        & (columns < grid.shape[1])
    )
    lookup = np.full(grid.shape, -1, dtype=int)
    lookup[grid.active_mask] = np.arange(int(grid.active_mask.sum()))
    indices = np.full((height, width), -1, dtype=int)
    indices[inside] = lookup[rows[inside], columns[inside]]
    lon0, lat0 = to_latlon.transform(west, south)
    lon1, lat1 = to_latlon.transform(east, north)
    return indices, [[lat0, lon0], [lat1, lon1]]


def colorize(values, indices, limits, palette):
    low, high = limits
    colors = colormaps[palette](
        np.clip((values - low) / (high - low), 0, 1), bytes=True
    )
    image = np.zeros((*indices.shape, 4), dtype=np.uint8)
    visible = indices >= 0
    image[visible] = colors[indices[visible]]
    return image


def route_frames(routes, records, times):
    """Posiciones y prefijos viales por tick, sin mostrar recorrido futuro."""
    frames = []
    for vehicle, points in sorted(routes.items()):
        points = np.asarray(points, dtype=float)
        mobile = {
            r.simulation_time_s: r
            for r in records
            if r.sensor_id == f"MOBILE_{vehicle}"
        }
        cursor, ends = 0, []
        for time in times:
            if time in mobile:
                r = mobile[time]
                matches = np.flatnonzero(
                    np.all(
                        np.isclose(points[cursor:], [r.x_m, r.y_m], rtol=0, atol=1e-6),
                        axis=1,
                    )
                )
                if not len(matches):
                    raise ValueError(
                        "La medición móvil no pertenece al recorrido guardado"
                    )
                cursor += int(matches[0])
            ends.append(cursor)
        if cursor != len(points) - 1:
            raise ValueError("No se pudo fechar todo el recorrido")
        frames.append({"vehicle": vehicle, "points": points, "ends": ends})
    return frames


def _geometry(root):
    study = load_study_environment_config(root / "config/study_environment.json")
    directory = root / study.domain.cache_directory
    graph = ox.io.load_graphml(directory / "road_graph.graphml")
    grid = load_grid(
        directory / "grid.npz", study.domain.fingerprint, graph.graph["crs"]
    )
    transform = Transformer.from_crs(grid.crs, 4326, always_xy=True)

    def latlon(points):
        points = np.asarray(points, dtype=float)
        lon, lat = transform.transform(points[:, 0], points[:, 1])
        return np.column_stack((lat, lon)).round(7).tolist()

    roads, seen = [], set()
    for u, v, data in graph.edges(data=True):
        line = data.get(
            "geometry",
            LineString(
                [
                    (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                    (graph.nodes[v]["x"], graph.nodes[v]["y"]),
                ]
            ),
        )
        key = line.normalize().wkb
        if key not in seen:
            roads.append(latlon(line.coords))
            seen.add(key)
    cells = []
    half = grid.cell_size_m / 2
    for row, column in np.argwhere(grid.active_mask):
        x, y = grid.cell_center(int(row), int(column))
        cells.append(
            {
                "row": int(row),
                "column": int(column),
                "polygon": latlon(
                    [
                        (x - half, y - half),
                        (x + half, y - half),
                        (x + half, y + half),
                        (x - half, y + half),
                    ]
                ),
            }
        )
    left, bottom, right, top = study.domain.bbox
    return grid, latlon, roads, cells, [[bottom, left], [top, right]]


def load_missions(settings, directory, grid, latlon):
    """Comprobar identidad física y correspondencia numérica antes de presentar."""
    threshold_protocol = json.loads(
        (product(settings["protocol_product"]) / "protocol.json").read_text()
    )
    threshold = threshold_protocol["threshold_ug_m3"]
    missions = []
    for study in settings["studies"]:
        source = product(study["missions_product"])
        posterior_root = product(study["posteriors_product"])
        metadata = json.loads((source / "config.json").read_text())
        posterior_source = json.loads((posterior_root / "source.json").read_text())
        if metadata != posterior_source["metadata"]:
            raise ValueError(
                "Las mediciones y posteriores proceden de evaluaciones distintas"
            )
        if field_protocol(metadata["protocol"]) != field_protocol(
            threshold_protocol["protocol"]
        ):
            raise ValueError("Las misiones no comparten el protocolo físico del umbral")
        if metadata["protocol"]["kind"] != "generated":
            raise ValueError(
                "El visor de verdad completa requiere escenarios sintéticos"
            )
        policy_path = product(study["policy_product"]) / study["policy_file"]
        state = torch.load(policy_path, map_location="cpu", weights_only=True)
        if (
            state["completed_episodes"] != study["episodes"]
            or state["protocol"] != metadata["protocol"]
        ):
            raise ValueError(
                "La política no corresponde al episodio o protocolo declarado"
            )
        if _sha(policy_path) != metadata["checkpoint_sha256"]:
            raise ValueError(
                "Los pesos seleccionados no corresponden a esta evaluación"
            )
        del state
        cases = {_mission_key(row): row for row in _csv(source / "cases.csv")}
        measurements, routes, ticks = (
            defaultdict(list),
            defaultdict(lambda: defaultdict(list)),
            defaultdict(list),
        )
        for row in _csv(source / "measurements.csv"):
            measurements[_mission_key(row)].append(row)
        for row in _csv(source / "trajectories.csv"):
            routes[_mission_key(row)][int(row["vehicle"])].append(
                (int(row["vertex"]), float(row["x_m"]), float(row["y_m"]))
            )
        for row in _csv(posterior_root / "ticks.csv"):
            ticks[_mission_key(row)].append(row)
        for seed in settings["case_seeds"]:
            truth_dir = product(settings["truth_product"]) / f"case-{seed}"
            report = json.loads((truth_dir / "report.json").read_text())
            fields = _npz(truth_dir / "fields-private.npz")
            _, _, reference_records, _ = validate_frozen_case(
                report,
                fields,
                grid,
                **partition_bounds(metadata["protocol"], "validation"),
            )
            times, truth = fields["elapsed_s"], fields["truth"]
            if (
                not np.array_equal(
                    fields["active_cells"], np.argwhere(grid.active_mask)
                )
                or not np.allclose(
                    fields["grid_bounds_m"], grid.grid_bounds_m, rtol=0, atol=1e-6
                )
                or not np.isfinite(truth).all()
                or hashlib.sha256(truth.tobytes()).hexdigest()
                != report["signatures"]["field_sha256"]
            ):
                raise ValueError(
                    "La verdad sintética no corresponde a la grilla o su registro"
                )
            for strategy in settings["strategies"]:
                replica = (
                    ""
                    if strategy["policy_seed"] is None
                    else str(strategy["policy_seed"])
                )
                key = seed, strategy["id"], replica
                case = cases[key]
                for signature in (
                    "field_sha256",
                    "initial_positions_sha256",
                ):
                    if case[signature] != report["signatures"][signature]:
                        raise ValueError(
                            "La misión y la verdad no comparten escenario, fijas o inicio"
                        )
                folder = (
                    posterior_root
                    / f"case-{seed}"
                    / f"{strategy['id']}-{replica or 'deterministic'}"
                )
                posterior = _npz(folder / "posterior-private.npz")
                mean, std = posterior["mean"], posterior["std"]
                if (
                    not np.array_equal(times, posterior["elapsed_s"])
                    or not np.array_equal(
                        fields["active_cells"], posterior["active_cells"]
                    )
                    or mean.shape != truth.shape
                    or std.shape != truth.shape
                    or not np.isfinite(mean).all()
                    or not np.isfinite(std).all()
                    or np.any(std < 0)
                ):
                    raise ValueError("Posteriors incompletos o con otro reloj/grilla")
                records = parse_measurements(
                    measurements[(seed, "neural", "")]
                    if strategy["id"] == "fixed"
                    else measurements[key]
                )
                if strategy["id"] == "fixed":
                    records = tuple(r for r in records if r.sensor_type == "fixed")
                fixed = tuple(r for r in records if r.sensor_type == "fixed")
                validate_fixed_reference(fixed, reference_records)
                if measurement_digest(fixed) != case["fixed_measurements_sha256"]:
                    raise ValueError("Las lecturas fijas difieren de las de la misión")
                if any(
                    r.sensor_type == "mobile"
                    and r.simulation_time_s > float(case["decision_end_elapsed_s"])
                    for r in records
                ):
                    raise ValueError("Mediciones móviles posteriores a la terminación")
                mission_routes = {
                    vehicle: [(x, y) for _, x, y in sorted(vertices)]
                    for vehicle, vertices in routes[key].items()
                }
                distances = route_progress(
                    mission_routes, records, times, float(case["distance_m"])
                )
                frame_rows = sorted(ticks[key], key=lambda r: float(r["elapsed_s"]))
                if len(frame_rows) != len(times) or not np.array_equal(
                    times, [float(r["elapsed_s"]) for r in frame_rows]
                ):
                    raise ValueError("Faltan métricas por tick")
                mse = np.mean((truth - mean) ** 2, axis=1)
                hot = truth >= threshold
                hot_count = hot.sum(axis=1)
                hot_mse = np.divide(
                    np.sum((truth - mean) ** 2 * hot, axis=1),
                    hot_count,
                    out=np.full(len(times), np.nan),
                    where=hot_count > 0,
                )
                if (
                    not np.allclose(
                        mse, [float(r["mse"]) for r in frame_rows], rtol=1e-7, atol=1e-8
                    )
                    or not np.allclose(
                        distances,
                        [float(r["distance_m"]) for r in frame_rows],
                        rtol=1e-8,
                        atol=1e-5,
                    )
                    or not np.isclose(
                        mse[-1], float(case["synthetic_mse"]), rtol=1e-7, atol=1e-8
                    )
                ):
                    raise ValueError("Los mapas no reproducen las métricas guardadas")
                route_data = route_frames(mission_routes, records, times)
                executed = []
                for route in route_data:
                    reached = [
                        grid.xy_to_cell(*route["points"][j]) for j in route["ends"]
                    ]
                    for i in range(1, len(reached)):
                        if times[i] > float(case["decision_end_elapsed_s"]):
                            break
                        a, b = reached[i - 1], reached[i]
                        direction = (
                            None
                            if a == b
                            else DIRECTIONS.index((b[0] - a[0], b[1] - a[1]))
                        )
                        executed.append(
                            {
                                "time": times[i],
                                "vehicle": route["vehicle"],
                                "direction": direction,
                            }
                        )
                    route["points"] = latlon(route["points"])
                samples = []
                for r in records:
                    samples.append(
                        {
                            "id": r.sensor_id,
                            "type": str(r.sensor_type),
                            "time": r.simulation_time_s,
                            "position": latlon([(r.x_m, r.y_m)])[0],
                            "value": r.value_ug_m3,
                        }
                    )
                missions.append(
                    {
                        "id": f"{study['id']}-{seed}-{strategy['id']}-{replica or 'det'}",
                        "study": study["label"],
                        "estimator": "adaptativo"
                        if metadata["protocol"].get("adaptive_gp")
                        else "fijo",
                        "case": seed,
                        "strategy": strategy["label"],
                        "strategy_id": strategy["id"],
                        "episode": study["episodes"]
                        if strategy["id"] == "neural"
                        else None,
                        "start": case["start_utc"],
                        "decision_end": float(case["decision_end_elapsed_s"]),
                        "times": times,
                        "truth": truth,
                        "mean": mean,
                        "std": std,
                        "routes": route_data,
                        "samples": samples,
                        "executed": executed,
                        "metrics": [
                            {
                                "rmse": float(np.sqrt(mse[i])),
                                "hot_rmse": float(np.sqrt(hot_mse[i]))
                                if hot_count[i]
                                else None,
                                "distance_km": float(distances[i] / 1000),
                                "recall": float(frame_rows[i]["true_positive_cells"])
                                / int(hot_count[i])
                                if hot_count[i]
                                else None,
                            }
                            for i in range(len(times))
                        ],
                        "source": {
                            "missions_product": study["missions_product"],
                            "posteriors_product": study["posteriors_product"],
                            "policy_file": str(policy_path.relative_to(PROJECT_ROOT)),
                            "checkpoint_sha256": metadata["checkpoint_sha256"],
                            "field_sha256": case["field_sha256"],
                        },
                    }
                )
    if not missions:
        raise ValueError("Se requiere al menos una misión declarada")
    return missions, threshold


def generate_viewer(settings, directory, output, *, replay=None):
    root = (directory / settings["core_root"]).resolve()
    grid, latlon, roads, cells, domain_bounds = _geometry(root)
    missions, threshold = load_missions(settings, directory, grid, latlon)
    if replay is not None or settings.get("replay_product"):
        from .historical_viewer import load_replay

        missions.extend(
            load_replay(replay or product(settings["replay_product"]), grid, latlon)
        )
    indices, bounds = raster_projection(grid, settings["raster_width"])
    concentrations = [m["mean"] for m in missions]
    errors = []
    for m in missions:
        if m.get("historical"):
            concentrations.append(
                np.array([s["value"] for s in m["samples"] if s["type"] == "fixed"])
            )
            errors.extend(abs(r["observed"] - r["predicted"]) for r in m["hourly"])
        else:
            concentrations.append(m["truth"])
            errors.append(float(np.abs(m["truth"] - m["mean"]).max()))
    max_error = max(errors)
    limits = {
        "concentration": [
            min(0.0, min(float(c.min()) for c in concentrations)),
            max(float(c.max()) for c in concentrations),
        ],
        "std": [0.0, max(float(m["std"].max()) for m in missions)],
        "error": [
            -max_error,
            max_error,
        ],
    }
    for name, (low, high) in limits.items():
        if low == high:
            limits[name] = [-1.0, 1.0] if name == "error" else [low, low + 1.0]
    output.mkdir(parents=True, exist_ok=False)
    public = []
    blobs = {}
    seen = set()

    def pack(raw):
        return base64.b64encode(gzip.compress(raw, compresslevel=3, mtime=0)).decode(
            "ascii"
        )

    for mission in missions:
        key = (
            mission.get("historical", False),
            mission["case"],
            mission.get("estimator", mission["study"]),
            mission["strategy_id"],
        )
        if mission["strategy_id"] != "neural" and key in seen:
            continue  # Las dos políticas adaptativas comparten las referencias.
        seen.add(key)
        fields = {}
        for k in ("truth", "mean", "std"):
            if k not in mission:
                continue
            # Una copia por campo; misma precisión visual de cinco decimales.
            v = np.rint(mission[k] * 100000)
            if np.abs(v).max() > 2**31 - 1:
                raise ValueError("Campo fuera del rango de representación")
            raw = v.astype("<i4").tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest not in blobs:
                blobs[digest] = pack(raw)
            fields[k] = digest
        public.append(
            {
                **{
                    k: v
                    for k, v in mission.items()
                    if k not in {"truth", "mean", "std", "times"}
                },
                "times": mission["times"].tolist(),
                "fields": fields,
            }
        )
    data = {
        "fields": blobs,
        "raster": {
            "width": indices.shape[1],
            "height": indices.shape[0],
            "index": pack(indices.astype("<i4").tobytes()),
        },
        "missions": public,
        "roads": roads,
        "cells": cells,
        "bounds": bounds,
        "domain_bounds": domain_bounds,
        "limits": limits,
        "threshold": threshold,
        "cell_size_m": grid.cell_size_m,
    }
    (output / "index.html").write_text(viewer_html(data), encoding="utf-8")
    return output / "index.html"


def viewer_html(data):
    data = {
        **data,
        "palettes": {
            palette: [
                f"#{r:02x}{g:02x}{b:02x}"
                for r, g, b, _ in colormaps[palette](np.linspace(0, 1, 256), bytes=True)
            ]
            for palette in ("viridis", "magma", "RdBu_r")
        },
    }
    figure = Figure(title="Muestreo de PM2.5")
    figure.header.add_child(
        Element("<style>" + (STATIC / "viewer.css").read_text() + "</style>")
    )
    figure.html.add_child(Element((STATIC / "layout.html").read_text()))
    figure.html.add_child(
        Element(
            '<script type="application/json" id="viewer-data">'
            + json.dumps(
                data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).replace("<", "\\u003c")
            + "</script>"
        )
    )
    maps = []
    for _ in range(4):
        map_ = folium.Map(
            location=np.mean(data["bounds"], axis=0).tolist(),
            zoom_start=12,
            zoom_snap=0.1,
            zoom_delta=0.5,
            width="100%",
            height="100%",
            position="relative",
            tiles=None,
            prefer_canvas=True,
            control_scale=True,
        )
        map_.add_to(figure)
        maps.append(map_)
    controller = MacroElement()
    controller._template = Template(
        "{% macro script(this, kwargs) %}\n"
        + "window.missionMaps = ["
        + ",".join(m.get_name() for m in maps)
        + "];\n"
        + (STATIC / "viewer.js").read_text()
        + "\n{% endmacro %}"
    )
    figure.add_child(controller)
    return figure.render()


def run():
    page = product("visor") / "index.html"
    if not page.is_file():
        raise FileNotFoundError(page)
    print(page, flush=True)
