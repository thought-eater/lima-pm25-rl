"""Auditoría de decisiones y conectividad, sin modificar rutas ni entrenar."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from lima_pm25.simulation import MeasurementBatch, SensorType

from .evaluate_cli import _compatible
from .evaluation import _spatial_errors
from .routing import DIRECTION_NAMES
from .train_cli import CACHE_DIRECTORY, DEFAULT_CONFIG, PROJECT_ROOT, prepare_experiment
from .training import TrainingConfig, load_policy


def archive_files(path: Path, names: set[str]) -> dict[str, bytes]:
    """Leer únicamente los miembros solicitados en memoria; nunca extraer rutas."""
    result = {}
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in names:
                continue
            if name in result:
                raise ValueError(f"Miembro ambiguo en {path}: {name}")
            file = archive.extractfile(member)
            assert file is not None
            result[name] = file.read()
    if missing := names - result.keys():
        raise ValueError(f"Faltan miembros en {path}: {sorted(missing)}")
    return result


def compare_fixed_measurements(actual, reference):
    """Comparación semántica: una reproyección puede cambiar el último bit de x/y."""
    tolerances = {
        "x_m": 1e-6,
        "y_m": 1e-6,
        "simulation_time_s": 1e-8,
        "value_ug_m3": 1e-8,
        "noise_stddev_ug_m3": 1e-8,
    }

    def key(row):
        return row["sensor_id"], row["timestamp_utc"]

    expected = {key(row): row for row in reference}
    differences = dict.fromkeys(tolerances, 0.0)
    matches = len(actual) == len(reference) == len(expected)
    matches = matches and len({key(row) for row in actual}) == len(actual)
    for row in actual:
        other = expected.get(key(row))
        if other is None:
            matches = False
            continue
        for name, value in row.items():
            if name in tolerances:
                difference = abs(float(value) - float(other[name]))
                differences[name] = max(differences[name], difference)
                matches = (
                    matches
                    and np.isfinite(difference)
                    and difference <= tolerances[name]
                )
            elif value != other.get(name):
                matches = False
    return {
        "match": bool(matches),
        "max_absolute_differences": differences,
        "absolute_tolerances": tolerances,
    }


def _node(router, node):
    cell = router.cell(node)
    return {
        "node": repr(node),
        "position_m": list(router.position(node)),
        "cell": list(cell) if cell is not None else None,
    }


def continuation_directions(router, node) -> list[str]:
    """Continuidad estructural; no bloquea posiciones/reservas de otros vehículos."""
    return [
        name
        for action, name in enumerate(DIRECTION_NAMES)
        if router.plan(node, action) is not None
    ]


def destination_connectivity(domain, router, node):
    goals = set(router._goals.values())
    reached = (nx.descendants(router.graph, node) | {node}) - goals
    cells = {router.cell(n) for n in reached} - {None}
    result = {
        **_node(router, node),
        "continuation_directions": continuation_directions(router, node),
        "reachable_physical_nodes": len(reached),
        "reachable_active_cells": len(cells),
        "small_component_nodes": [_node(router, n) for n in sorted(reached, key=repr)]
        if len(reached) <= 20
        else None,
    }
    original = domain.road_graph
    if isinstance(node, tuple) and node[0] == "rl-stop":
        # La numeración de paradas corresponde al orden de las aristas originales.
        u, v, key, data = list(original.edges(keys=True, data=True))[node[1]]
        endpoint = v
        result["original_edge"] = {
            "u": repr(u),
            "v": repr(v),
            "key": repr(key),
            "name": data.get("name"),
            "oneway": data.get("oneway"),
            "osmid": data.get("osmid"),
        }
    else:
        endpoint = node
    original_reached = nx.descendants(original, endpoint) | {endpoint}
    result["original_forward_endpoint"] = repr(endpoint)
    result["forward_endpoint_reachable_in_decoder"] = endpoint in reached
    result["original_reachable_nodes"] = len(original_reached)
    if len(original_reached) <= 20:
        from shapely.geometry import Point

        result["original_component"] = [
            {
                **_node(router, n),
                "inside_study_polygon": domain.boundary.covers(
                    Point(router.position(n))
                ),
                "distance_to_study_boundary_m": float(
                    domain.boundary.boundary.distance(Point(router.position(n)))
                ),
                "inside_permitted_buffer": router._permitted.covers(
                    Point(router.position(n))
                ),
                "outgoing": [
                    {
                        "destination": repr(v),
                        "name": d.get("name"),
                        "oneway": d.get("oneway"),
                    }
                    for _, v, d in original.out_edges(n, data=True)
                ],
            }
            for n in sorted(original_reached, key=repr)
        ]
    return result


def audit_arrival(domain, router, origin, direction):
    """Comparar las paradas de la MISMA celda, conservando la elección vigente."""
    chosen = router.plan(origin, direction)
    if chosen is None:
        raise ValueError("La dirección auditada debe tener una ruta")
    lengths = nx.single_source_dijkstra_path_length(
        router.graph, origin, weight="length"
    )
    candidates = []
    for node in router.graph.predecessors(router._goals[chosen.cell]):
        if node in lengths:
            candidates.append(
                {
                    **_node(router, node),
                    "distance_m": float(lengths[node]),
                    "continuation_directions": continuation_directions(router, node),
                }
            )
    candidates.sort(key=lambda row: (row["distance_m"], row["node"]))
    alternative = next(
        (row for row in candidates if row["continuation_directions"]), None
    )
    alternative_coordinates = None
    if alternative is not None:
        # Obtener el objeto del nodo sin interpretar ni evaluar su representación.
        endpoint = next(
            node
            for node in router.graph.predecessors(router._goals[chosen.cell])
            if repr(node) == alternative["node"]
        )
        path = nx.shortest_path(router.graph, origin, endpoint, weight="length")
        alternative_coordinates = [router.position(origin)]
        for a, b in pairwise(path):
            alternative_coordinates.extend(router.graph.edges[a, b]["coordinates"][1:])
    return {
        "origin": _node(router, origin),
        "direction": DIRECTION_NAMES[direction],
        "target_cell": list(chosen.cell),
        "chosen_distance_m": float(chosen.distance_m),
        "chosen_coordinates_m": chosen.coordinates,
        "chosen_connectivity": destination_connectivity(
            domain, router, chosen.destination
        ),
        "same_cell_candidates": candidates,
        "nearest_continuing_candidate": alternative,
        "additional_distance_m": float(alternative["distance_m"] - chosen.distance_m)
        if alternative is not None
        else None,
        "alternative_coordinates_m": alternative_coordinates,
        "initial_directions": [
            {
                "direction": name,
                "distance_m": float(route.distance_m),
                "destination": _node(router, route.destination),
                "continuation_directions": continuation_directions(
                    router, route.destination
                ),
            }
            for action, name in enumerate(DIRECTION_NAMES)
            if (route := router.plan(origin, action)) is not None
        ],
        "scope": "Conectividad del grafo conservado; no certifica calles reales fuera del recorte. Sin modificar el decodificador.",
    }


def trace_episode(
    env, policy, *, case_seed: int, vehicle: int, complete_horizon: bool = False
):
    """Capturar las salidas reales de forward; no reimplementar el controlador."""
    observation, _ = env.reset(seed=case_seed)
    initial_positions = np.asarray(env.positions)
    field_hash = hashlib.sha256()

    def record_field():
        field_hash.update(
            env._simulation.state.concentration_ug_m3[
                env.domain.grid.active_mask
            ].tobytes()
        )

    record_field()
    policy.reset()
    trace, arrivals = [], []
    reward_sum = 0.0
    variances = [env._info()["integrated_variance"]]
    first_audit = None
    captured = []
    handle = policy.network.register_forward_hook(
        lambda module, arguments, output: captured.append(
            output.detach().cpu().tolist()
        )
    )
    try:
        while True:
            captured.clear()
            origins = env._nodes.copy()
            positions = env.positions
            actions, decisions = policy.plan(env, observation)
            if len(captured) != len(decisions):
                raise RuntimeError(
                    "El diagnóstico necesita inferencia determinista sin exploración"
                )
            stages = []
            for scores, decision in zip(captured, decisions, strict=True):
                route = env.route(decision.vehicle, decision.action)
                assert route is not None
                stages.append(
                    {
                        "vehicle": decision.vehicle,
                        "direction": DIRECTION_NAMES[decision.action],
                        "q_values": scores,
                        "valid_masks": decision.mask.tolist(),
                        "pending": decision.pending,
                        "reserved_cells": decision.reserved_cells,
                        "assigned_positions_normalized": decision.assigned_positions,
                        "destination": _node(env.router, route.destination),
                        "route_distance_m": float(route.distance_m),
                        "route_coordinates_m": route.coordinates,
                    }
                )
                if not trace and decision.vehicle == vehicle:
                    first_audit = audit_arrival(
                        env.domain, env.router, origins[vehicle], decision.action
                    )
                    first_audit["executed_arrival"] = {
                        **_node(env.router, route.destination),
                        "distance_m": float(route.distance_m),
                        "strategy": env.arrival_strategy,
                    }
            observation, reward, terminated, truncated, info = env.step(actions)
            record_field()
            reward_sum += reward
            variances.append(info["integrated_variance"])
            arrivals.append(np.asarray(env.positions))
            trace.append(
                {
                    "tick": env._steps,
                    "positions_before_m": positions,
                    "assignments": actions.tolist(),
                    "stages": stages,
                    "positions_after_m": env.positions,
                    "nodes_after": [repr(node) for node in env._nodes],
                    "masks_after": observation["action_mask"].tolist(),
                    "audited_vehicle_structural_directions": continuation_directions(
                        env.router, env._nodes[vehicle]
                    ),
                    "reward": reward,
                    "information_gain": info["uncertainty_gain"],
                    "distance_cost": float(info["distance_cost"]),
                    "route_distance_m": info["route_distance_m"].tolist(),
                    "held_vehicles": info["held_vehicles"].tolist(),
                    "invalid_actions": info["invalid_actions"].tolist(),
                    "integrated_variance": info["integrated_variance"],
                    **_spatial_errors(env, env._posterior),
                }
            )
            if terminated or truncated:
                break
    finally:
        handle.remove()
    decision_end = env._info()["elapsed_s"]
    if complete_horizon:
        while env._simulation.state.simulation_time_s < env._simulation.duration_s:
            for batch in env._simulation.advance(env.positions):
                fixed = tuple(
                    r for r in batch.records if r.sensor_type is SensorType.FIXED
                )
                if fixed:
                    env._assimilate(
                        MeasurementBatch(
                            batch.simulation_time_s, batch.timestamp_utc, fixed
                        )
                    )
            env._posterior = env._predict()
            variances.append(env._integrated_variance(env._posterior))
            record_field()
    return {
        "first_arrival_audit": first_audit,
        "ticks": trace,
        "arrivals_m": np.stack(arrivals).tolist(),
        "held_vehicle_ticks": sum(sum(t["held_vehicles"]) for t in trace),
        "fixed_measurements": [
            {**asdict(r), "timestamp_utc": r.timestamp_utc.isoformat()}
            for r in env._records
            if r.sensor_type is SensorType.FIXED
        ],
        "signatures": {
            "field_sha256": field_hash.hexdigest(),
            "initial_positions_sha256": hashlib.sha256(
                initial_positions.tobytes()
            ).hexdigest(),
            "fixed_measurements_sha256": hashlib.sha256(
                json.dumps(
                    [
                        asdict(r)
                        for r in env._records
                        if r.sensor_type is SensorType.FIXED
                    ],
                    default=str,
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        },
        "metrics": {
            "reward": reward_sum,
            "final_integrated_variance": env._info()["integrated_variance"],
            "mean_integrated_variance": float(
                np.trapezoid(variances) / (len(variances) - 1)
            ),
            "distance_m": float(env._info()["distance_m"].sum()),
            "elapsed_s": env._info()["elapsed_s"],
            "decision_end_elapsed_s": decision_end,
            **_spatial_errors(env, env._posterior),
        },
    }


def plot_arrival(env, audit, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from shapely.geometry import LineString, box

    chosen = np.asarray(audit["chosen_coordinates_m"])
    alternative = audit["alternative_coordinates_m"]
    points = chosen if alternative is None else np.vstack((chosen, alternative))
    low, high = points.min(axis=0) - 100, points.max(axis=0) + 100
    window = box(*low, *high)
    figure = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    graph = env.domain.road_graph
    for u, v, data in graph.edges(data=True):
        line = data.get(
            "geometry", LineString((env.router.position(u), env.router.position(v)))
        )
        if line.intersects(window):
            xy = np.asarray(line.coords)
            axis.plot(xy[:, 0], xy[:, 1], color="0.8", linewidth=0.7)
    for boundary, label, style in (
        (env.domain.boundary.boundary, "Límite de estudio", "--"),
        (env.router._permitted.boundary, "Margen vial permitido", ":"),
    ):
        xy = np.asarray(boundary.coords)
        axis.plot(xy[:, 0], xy[:, 1], style, color="black", label=label)
    row, column = audit["target_cell"]
    x0, y0 = env.domain.grid.grid_bounds_m[:2]
    size = env.domain.grid.cell_size_m
    axis.add_patch(
        Rectangle(
            (x0 + column * size, y0 + row * size),
            size,
            size,
            fill=False,
            edgecolor="blue",
            label="Misma celda destino",
        )
    )
    axis.plot(chosen[:, 0], chosen[:, 1], color="red", label="Ruta elegida")
    axis.scatter(*chosen[-1], color="red", marker="x", s=80)
    if alternative is not None:
        xy = np.asarray(alternative)
        axis.plot(xy[:, 0], xy[:, 1], color="green", label="Parada con continuidad")
        axis.scatter(*xy[-1], color="green", marker="o", s=50)
    axis.scatter(*chosen[0], color="black", marker="*", s=100, label="Inicio")
    axis.set(
        xlim=(low[0], high[0]),
        ylim=(low[1], high[1]),
        xlabel="Este UTM (m)",
        ylabel="Norte UTM (m)",
        title="Auditoría geométrica; alternativa no ejecutada",
    )
    axis.set_aspect("equal")
    axis.ticklabel_format(useOffset=False, style="plain")
    axis.legend(fontsize=8)
    with path.open("xb") as file:
        figure.savefig(file, format="png", dpi=120)


def main():
    parser = argparse.ArgumentParser(
        description="Auditar un caso de validación sin entrenar ni cambiar las rutas"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--training-archive", type=Path, default=PROJECT_ROOT / "training.tar.zst"
    )
    parser.add_argument(
        "--evaluation-archive", type=Path, default=PROJECT_ROOT / "evaluation.tar.zst"
    )
    parser.add_argument("--case", type=int, default=100002)
    parser.add_argument("--vehicle", type=int, choices=range(3), default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        CACHE_DIRECTORY
        / "mobility-diagnostics"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    )
    if output.exists():
        parser.error("La salida debe ser nueva")
    spec = json.loads(args.config.read_text())
    if spec["kind"] != "generated" or args.case not in spec["validation_seeds"]:
        parser.error("El caso debe pertenecer a la validación configurada")
    saved = archive_files(args.training_archive, {"policy-best.pt"})
    reference = archive_files(
        args.evaluation_archive, {"config.json", "cases.csv", "measurements.csv"}
    )
    reference_config = json.loads(reference["config.json"])
    payload = saved["policy-best.pt"]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference_config["checkpoint_sha256"]:
        parser.error("La evaluación no corresponde a la política best del archivo")
    config = TrainingConfig(**spec.get("training", {}))
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    training_env, env, protocol = prepare_experiment(spec, args.config.resolve().parent)
    training_env.close()
    assert env is not None
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not _compatible(state, config, protocol) or reference_config[
            "protocol"
        ] != json.loads(json.dumps(protocol)):
            raise ValueError(
                "El protocolo local no coincide con los archivos recibidos"
            )
        del state
        policy, _ = load_policy(io.BytesIO(payload), device=args.device)
        report = trace_episode(env, policy, case_seed=args.case, vehicle=args.vehicle)
        expected = next(
            row
            for row in csv.DictReader(io.StringIO(reference["cases.csv"].decode()))
            if row["strategy"] == "neural" and int(row["case_seed"]) == args.case
        )
        measurements = [
            row
            for row in csv.DictReader(
                io.StringIO(reference["measurements.csv"].decode())
            )
            if row["strategy"] == "neural"
            and row["sensor_type"] == "mobile"
            and int(row["case_seed"]) == args.case
        ]
        fixed_reference = [
            row
            for row in csv.DictReader(
                io.StringIO(reference["measurements.csv"].decode())
            )
            if row["strategy"] == "neural"
            and row["sensor_type"] == "fixed"
            and int(row["case_seed"]) == args.case
        ]
        fixed_comparison = compare_fixed_measurements(
            report["fixed_measurements"], fixed_reference
        )
        measurements.sort(
            key=lambda row: (float(row["simulation_time_s"]), row["sensor_id"])
        )
        positions = np.asarray(
            [[float(row["x_m"]), float(row["y_m"])] for row in measurements]
        )
        actual = np.asarray(report["arrivals_m"]).reshape(-1, 2)
        position_match = positions.shape == actual.shape and np.allclose(
            positions, actual, atol=1e-6, rtol=0
        )
        matches = {
            key: bool(np.isclose(value, float(expected[key]), rtol=1e-7, atol=1e-8))
            for key, value in report["metrics"].items()
        }
        report.update(
            case_seed=args.case,
            vehicle=args.vehicle,
            training_executed=False,
            decoder_modified=False,
            policy_sha256=digest,
            device=str(next(policy.network.parameters()).device),
            torch_version=str(torch.__version__),
            protocol=protocol,
            code_sha256={
                name: hashlib.sha256(
                    (Path(__file__).parent / name).read_bytes()
                ).hexdigest()
                for name in (
                    "routing.py",
                    "env.py",
                    "network.py",
                    "policies.py",
                    "mobility_diagnostics.py",
                )
            },
            reproduction={
                "positions_match": bool(position_match),
                "metrics_match": matches,
                "fixed_measurements": fixed_comparison,
                "signatures_match": {
                    key: value == expected[key]
                    for key, value in report["signatures"].items()
                },
            },
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if not (
            position_match
            and all(matches.values())
            and fixed_comparison["match"]
            and report["signatures"]["field_sha256"] == expected["field_sha256"]
            and report["signatures"]["initial_positions_sha256"]
            == expected["initial_positions_sha256"]
        ):
            raise RuntimeError(
                f"El caso no reproduce el archivo; revisar {output / 'report.json'}"
            )
        if report["first_arrival_audit"] is not None:
            plot_arrival(env, report["first_arrival_audit"], output / "arrival.png")
        print(f"Salida: {output}")
        print(json.dumps(report["reproduction"], indent=2))
        audit = report["first_arrival_audit"]
        if audit is not None:
            print(
                json.dumps(
                    {
                        "chosen_distance_m": audit["chosen_distance_m"],
                        "continuation": audit["chosen_connectivity"][
                            "continuation_directions"
                        ],
                        "nearest_continuing_candidate": audit[
                            "nearest_continuing_candidate"
                        ],
                        "additional_distance_m": audit["additional_distance_m"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
