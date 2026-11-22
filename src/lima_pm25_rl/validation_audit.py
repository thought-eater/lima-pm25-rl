"""Calibración del campo latente y velocidad requerida en misiones ya guardadas."""

import csv
import hashlib
import json
from collections import defaultdict
from itertools import chain

import numpy as np

from .files import PROJECT_ROOT, _write_json, product, write_csv
from .hotspot_evaluation import field_protocol, parse_measurements
from .map_viewer import route_frames
from .paired_validation import contrast, stratified_indices
from .probabilistic_metrics import Z95, scores

ROLES = ("fixed", "adaptive_final", "adaptive_best")


def segments(routes, records, times, decision_end, total):
    """Cada fila pertenece a un vehículo y al tick que termina en elapsed_s."""
    if (
        set(routes) != {0, 1, 2}
        or decision_end not in times
        or not np.all(np.diff(times) == 300)
    ):
        raise ValueError("Flota o reloj incompleto")
    rows = []
    for frame in route_frames(routes, records, times):
        vehicle = frame["vehicle"]
        measured = {
            r.simulation_time_s for r in records if r.sensor_id == f"MOBILE_{vehicle}"
        }
        if measured != set(times[(times > 0) & (times <= decision_end)]):
            raise ValueError("Lecturas móviles incompatibles con la terminación")
        points = frame["points"]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        for tick, distance in enumerate(np.diff(arc[frame["ends"]]), 1):
            ended = times[tick] > decision_end
            rows.append(
                {
                    "vehicle": vehicle,
                    "tick": tick,
                    "elapsed_s": times[tick],
                    "state": "ended" if ended else "moving" if distance > 0 else "held",
                    "distance_m": distance,
                    "required_kmh": None if ended else distance * 3.6 / 300,
                }
            )
    if not np.isclose(sum(r["distance_m"] for r in rows), total, rtol=1e-8, atol=1e-5):
        raise ValueError("La suma de tramos no reproduce la distancia de la misión")
    return rows


def speed_summary(rows):
    moving = [r for r in rows if r["state"] == "moving"]
    speeds = [r["required_kmh"] for r in moving]
    held = sum(r["state"] == "held" for r in rows)
    peak = max(moving, key=lambda r: r["required_kmh"], default={})
    return {
        "moving_vehicle_ticks": len(moving),
        "held_vehicle_ticks": held,
        "ended_vehicle_ticks": len(rows) - len(moving) - held,
        "held_pct_active": 100 * held / (len(moving) + held)
        if moving or held
        else None,
        "moving_median_kmh": np.median(speeds) if speeds else None,
        "moving_p95_kmh": np.quantile(speeds, 0.95) if speeds else None,
        "moving_max_kmh": peak.get("required_kmh"),
        **{
            f"max_{k}": peak.get(k)
            for k in ("case_seed", "policy_seed", "vehicle", "tick")
        },
    }


def _rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _key(r):
    return int(r["case_seed"]), r["strategy"], r["policy_seed"]


def _groups(path):
    groups = defaultdict(list)
    for r in _rows(path):
        groups[_key(r)].append(r)
    return groups


def _average(rows, keys):
    return {k: np.mean([r[k] for r in rows]) for k in keys}


def run(output):
    config = PROJECT_ROOT / "config"
    spec = json.loads((config / "informe.json").read_text())["paired_validation"]
    truth_key = json.loads((config / "evaluacion.json").read_text())["truth_product"]
    src = product(truth_key)
    seeds = list(chain.from_iterable(spec["strata"].values()))
    indices = stratified_indices(
        [[seeds.index(s) for s in g] for g in spec["strata"].values()]
    )
    if len(set(seeds)) != 12:
        raise ValueError("Estratos de validación inválidos")
    truth_proto = json.loads((src / "config.json").read_text())["protocol"]
    fields = {}
    for seed in seeds:
        folder = src / f"case-{seed}"
        report = json.loads((folder / "report.json").read_text())
        with np.load(folder / "fields-private.npz", allow_pickle=False) as f:
            y, cells, times = f["truth"], f["active_cells"], f["elapsed_s"]
        if (
            report["case_seed"] != seed
            or y.shape != (len(times), len(cells))
            or not np.array_equal(times, np.arange(37) * 300)
            or hashlib.sha256(y.tobytes()).hexdigest()
            != report["signatures"]["field_sha256"]
        ):
            raise ValueError("La verdad no corresponde al caso o reloj congelado")
        fields[seed] = (y, cells, times, report["signatures"])

    forecasts, legs, duplicate_signatures, identities = [], [], {}, {}
    duplicates = 0
    manifest = {}
    for role in ROLES:
        label, mkey, pkey = spec[role]
        m, p = product(mkey), product(pkey)
        meta = json.loads((m / "config.json").read_text())
        status = json.loads((p / "summary.json").read_text())
        if (
            meta != json.loads((p / "source.json").read_text())["metadata"]
            or meta["partition"] != "validation"
            or set(meta["case_seeds"]) != set(seeds)
            or field_protocol(meta["protocol"]) != field_protocol(truth_proto)
            or status["smoke_only"]
            or not status["reproduction_passed"]
        ):
            raise ValueError(
                "Las misiones, los posteriores y la verdad no comparten protocolo"
            )
        manifest[role] = {
            "label": label,
            "missions": str(m.relative_to(PROJECT_ROOT)),
            "posteriors": str(p.relative_to(PROJECT_ROOT)),
        }
        missions = _rows(m / "cases.csv")
        expected = {
            (s, st, rep)
            for s in seeds
            for st in ("fixed", "neural", "sweep", "random", "greedy")
            for rep in (
                ["0", "1", "2", "3", "4"] if st in ("random", "greedy") else [""]
            )
        }
        if len(missions) != len(expected) or set(map(_key, missions)) != expected:
            raise ValueError("Misiones o réplicas incompletas")
        trajectories, measurements = (
            _groups(m / "trajectories.csv"),
            _groups(m / "measurements.csv"),
        )
        ticks = _groups(p / "ticks.csv")
        for row in missions:
            key = _key(row)
            seed, strategy, rep = key
            y, cells, times, signatures = fields[seed]
            identity = tuple(
                row[k]
                for k in (
                    "start_utc",
                    "elapsed_s",
                    "field_sha256",
                    "initial_positions_sha256",
                    "fixed_measurements_sha256",
                )
            )
            if (
                identities.setdefault(seed, identity) != identity
                or float(row["elapsed_s"]) != times[-1]
                or any(
                    row[k] != signatures[k]
                    for k in ("field_sha256", "initial_positions_sha256")
                )
            ):
                raise ValueError("La misión no corresponde al caso pareado")
            source = (
                p
                / f"case-{seed}"
                / f"{strategy}-{rep or 'deterministic'}"
                / "posterior-private.npz"
            )
            with np.load(source, allow_pickle=False) as f:
                if not np.array_equal(f["elapsed_s"], times) or not np.array_equal(
                    f["active_cells"], cells
                ):
                    raise ValueError("El posterior no coincide en reloj o grilla")
                mu, sd = f["mean"], f["std"]
            if strategy != "neural" and role != "fixed":
                # Las referencias adaptativas proceden de dos evaluaciones: verificar
                # igualdad de entradas, recorrido y distribuciones antes de contar una.
                signature = (
                    row,
                    measurements[key],
                    trajectories[key],
                    hashlib.sha256(mu.tobytes()).digest(),
                    hashlib.sha256(sd.tobytes()).digest(),
                )
                if role == "adaptive_final":
                    duplicate_signatures[key] = signature
                elif signature == duplicate_signatures[key]:
                    duplicates += 1
                    continue
                else:
                    raise ValueError(f"Referencia adaptativa no reproducida: {key}")
            ident = {
                "study": role
                if strategy == "neural" or role == "fixed"
                else "adaptive_refs",
                "strategy": strategy,
                "case_seed": seed,
                "policy_seed": rep,
            }
            per_tick = scores(y, mu, sd)
            observed = sorted(ticks[key], key=lambda r: float(r["elapsed_s"]))
            if not np.array_equal(
                times, [float(r["elapsed_s"]) for r in observed]
            ) or not np.allclose(
                ((y - mu) ** 2).mean(axis=1),
                [float(r["mse"]) for r in observed],
                rtol=1e-8,
                atol=1e-10,
            ):
                raise ValueError("El posterior no reproduce el MSE por instante")
            values = {
                f"{scope}_{k}": val
                for k, v in per_tick.items()
                for scope, val in (
                    ("mean", np.trapezoid(v, times) / times[-1]),
                    ("final", v[-1]),
                )
            }
            forecasts.append({**ident, **values})
            if strategy == "fixed":
                if float(row["distance_m"]) != 0 or trajectories[key]:
                    raise ValueError("Solo fijas con recorrido móvil")
                continue
            routes = defaultdict(list)
            for r in sorted(trajectories[key], key=lambda r: int(r["vertex"])):
                points = routes[int(r["vehicle"])]
                if int(r["vertex"]) != len(points):
                    raise ValueError("Vértices viales incompletos")
                points.append((float(r["x_m"]), float(r["y_m"])))
            stops = segments(
                routes,
                parse_measurements(measurements[key]),
                times,
                float(row["decision_end_elapsed_s"]),
                float(row["distance_m"]),
            )
            # Reproducir la distancia acumulada publicada para cada tick, no solo al cierre.
            distances = np.array([r["distance_m"] for r in stops]).reshape(3, -1)
            if not np.allclose(
                np.r_[0.0, distances.sum(axis=0).cumsum()],
                [float(r["distance_m"]) for r in observed],
                rtol=1e-8,
                atol=1e-5,
            ):
                raise ValueError("La distancia no reproduce los ticks guardados")
            legs.extend({**ident, **r} for r in stops)

    metrics = list(values)
    by_case = defaultdict(list)
    for r in forecasts:
        by_case[r["study"], r["strategy"], r["case_seed"]].append(r)
    cases = [
        {
            "study": st,
            "strategy": pol,
            "case_seed": s,
            "replicas": len(rs),
            **_average(rs, metrics),
        }
        for (st, pol, s), rs in by_case.items()
    ]
    by_strategy = defaultdict(dict)
    for r in cases:
        by_strategy[r["study"], r["strategy"]][r["case_seed"]] = r
    summary = [
        {
            "study": st,
            "strategy": pol,
            "cases": len(rs),
            **_average(rs.values(), metrics),
        }
        for (st, pol), rs in by_strategy.items()
    ]
    pairs = [(("fixed", "neural"), (role, "neural")) for role in ROLES[1:]]
    pairs.extend(
        ((role, "neural"), ("fixed" if role == "fixed" else "adaptive_refs", pol))
        for role in ROLES
        for pol in ("random", "greedy", "sweep", "fixed")
    )
    intervals = []
    for a, b in pairs:
        for metric in metrics:
            va, vb = (
                np.array([by_strategy[k][s][metric] for s in seeds]) for k in (a, b)
            )
            result, _ = contrast(va, vb, indices, root=False)
            intervals.append(
                {
                    "candidate": "/".join(a),
                    "reference": "/".join(b),
                    "metric": metric,
                    **{
                        k: result[k]
                        for k in (
                            "delta",
                            "low",
                            "high",
                            "sd_case_delta",
                            "lower",
                            "tied",
                            "higher",
                        )
                    },
                }
            )
    speeds = []
    for grouping in (
        ("study", "strategy", "case_seed", "policy_seed"),
        ("study", "strategy", "case_seed"),
        ("study", "strategy"),
    ):
        groups = defaultdict(list)
        for r in legs:
            groups[tuple(r[k] for k in grouping)].append(r)
        speed_rows = [
            {**dict(zip(grouping, k, strict=True)), **speed_summary(rs)}
            for k, rs in groups.items()
        ]
        speeds.append(speed_rows)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("missions", forecasts),
        ("cases", cases),
        ("summary", summary),
        ("intervals", intervals),
        ("segments", legs),
        ("speeds-missions", speeds[0]),
        ("speeds-cases", speeds[1]),
        ("speeds", speeds[2]),
    ):
        write_csv(output / f"{name}.csv", rows)
    _write_json(
        output / "manifest.json",
        {
            "partition": "synthetic_validation",
            "sources": manifest,
            "truth": truth_key,
            "strata": spec["strata"],
            "z95": Z95,
            "missions": len(forecasts),
            "duplicate_adaptive_references": duplicates,
            "vehicle_ticks": len(legs),
            "bootstrap": "4096 paired stratified ordered draws; percentile 95%",
            "scores": "latent cell mean; trapezoidal time mean incl. t=0 or last instant; replicas then equal cases",
            "speed": "per vehicle and 300 s tick; pooled moving ticks for percentiles; held fraction excludes ended ticks",
        },
    )
    print(output, flush=True)
