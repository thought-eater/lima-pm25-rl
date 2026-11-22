"""Prueba sintética con políticas congeladas; el evaluador conserva la verdad aparte."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from itertools import product as combinations

import numpy as np
import torch
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from threadpoolctl import threadpool_limits

from .comparison import SIGNATURES
from .evaluate import _compatible
from .evaluation import evaluate_episode
from .files import PROJECT_ROOT, _write_json, product, write_csv
from .hotspot_evaluation import parse_measurements
from .hotspot_metrics import cell_detection_events, frame_metrics, summarize_mission
from .independent_statistics import summarize
from .policies import BaselinePolicy
from .probabilistic_metrics import scores
from .study import _sha256, _utc, prepare_experiment
from .training import TrainingConfig, load_policy, resolve_device

CLOCK = np.arange(37) * 300


def validate_calendar(cfg, spec):
    cases = cfg["cases"]
    expected = set(
        combinations(
            ("control", "event"), ("historical", "historical_perturbed", "synthetic")
        )
    )
    strata = Counter((family, mode) for _, _, family, mode in cases)
    if set(strata) != expected or set(strata.values()) != {4}:
        raise ValueError(
            "Se requieren cuatro ventanas por cada uno de los seis estratos"
        )
    seeds = [s for s, *_ in cases]
    if len(set(seeds)) != 24 or set(seeds) & set(spec["validation_seeds"]):
        raise ValueError("Semillas repetidas o de selección")
    forbidden = list(map(_utc, cfg["excluded_starts_utc"]))
    core = (PROJECT_ROOT / "config" / spec["core_root"]).resolve()
    partition = json.loads((core / spec["gp_partition_manifest"]).read_text())
    for _, date, _, _ in cases:
        start = _utc(date)
        end = start + timedelta(hours=3)
        if (
            start < _utc(spec["gp_fit_end_utc"]) + timedelta(days=1)
            or end + timedelta(days=1) > _utc(spec["reserved_final_start_utc"])
            or any(abs(start - v) < timedelta(hours=27) for v in forbidden)
            or any(
                start - timedelta(days=1) <= _utc(r["end_utc"])
                and _utc(r["context_start_utc"]) <= end
                for r in partition["calibration_windows"] + partition["replay_windows"]
            )
        ):
            raise ValueError(f"Ventana no elegible: {date}")
        forbidden.append(start)


def check_request(request, case):
    _, date, family, mode = case
    meta = json.loads(request.episode_parameters_json)
    event = any(s.source_id == "alfonso-ugarte-event" for s in request.sources)
    if (
        request.config.start_utc != _utc(date)
        or request.config.duration_s != 10800
        or meta["partition"] != "evaluation"
        or meta["weather_mode"] != mode
        or event != (family == "event")
    ):
        raise ValueError(f"Fecha, miembro o meteorología no corresponde al caso {case}")


def preflight(cfg, *, synthetic_calendar=True):
    """Comprobar todos los pesos y protocolos antes de ejecutar la primera misión."""
    directory = PROJECT_ROOT / "config"
    envs, policies = {}, {}
    for label, (profile, model, filename, episode, sha) in cfg["models"].items():
        spec = json.loads((directory / profile).read_text())
        if synthetic_calendar:
            validate_calendar(cfg, spec)
        conf = TrainingConfig(**spec["training"])
        if (
            conf.device != "cuda"
            or cfg["policy_seeds"] != spec["evaluation_policy_seeds"]
            or cfg["policy_seeds"] != list(range(5))
        ):
            raise ValueError(
                "Se requiere el perfil de servidor y las cinco réplicas prefijadas"
            )
        device = resolve_device(conf.device)
        torch.set_num_threads(conf.cpu_threads)
        if profile not in envs:
            train, env, protocol = prepare_experiment(spec, directory)
            train.close()
            envs[profile] = (env, protocol)
        env, protocol = envs[profile]
        path = product(model) / filename
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            _sha256(path) != sha
            or state["completed_episodes"] != episode
            or not _compatible(state, conf, protocol)
        ):
            raise ValueError(f"Pesos incompatibles: {path}")
        policies[label] = (env, load_policy(path, device=str(device))[0])
    return policies


def mission(env, policy, case, threshold, coords, *, fixed=False, save_truth=False):
    """Una ruta; opcionalmente puntuar solo fijas en el mismo reloj sin otra misión."""
    tracks = defaultdict(list)
    means = defaultdict(list)
    truths = []
    mask = env.domain.grid.active_mask

    def observe(e, posterior, _distance):
        sim = e._simulation
        time = sim.state.simulation_time_s
        if time == 0:
            check_request(sim.request, case)
        y = sim.state.concentration_ug_m3[mask]
        truths.append(y)

        def score(tag, post):
            mu = post.mean_ug_m3[mask]
            sd = post.standard_deviation_ug_m3[mask]
            means[tag].append(mu)
            frame = frame_metrics(
                y,
                mu,
                sd,
                threshold=threshold,
                coordinates=coords,
                cell_area_m2=e.domain.grid.cell_area_m2,
            )
            prob = scores(y[None, :], mu[None, :], sd[None, :])
            tracks[tag].append(
                {
                    "elapsed_s": time,
                    **frame,
                    **{k: float(v[0]) for k, v in prob.items()},
                }
            )

        score("mobile", posterior)
        if fixed:
            score("fixed", e.predict_fixed())

    raw = evaluate_episode(
        env, policy, seed=case[0], comparison=True, observe_tick=observe
    )
    if raw["invalid_actions"] or not np.array_equal(
        [r["elapsed_s"] for r in tracks["mobile"]], CLOCK
    ):
        raise ValueError("Acciones inválidas u horizonte incompleto")
    truth = np.stack(truths)
    result = {}
    for tag, frames in tracks.items():
        metrics = summarize_mission(frames, CLOCK)
        metrics.update(
            cell_detection_events(truth, np.array(means[tag]), CLOCK, threshold)
        )
        for k in ("coverage95_pct", "width95_ug_m3", "crps_ug_m3"):
            metrics[f"mean_{k}"] = float(
                np.trapezoid([r[k] for r in frames], CLOCK) / CLOCK[-1]
            )
            metrics[f"final_{k}"] = frames[-1][k]
        metrics.update(
            distance_m=raw["distance_m"] if tag == "mobile" else 0.0,
            sampling_coverage=raw["sampling_coverage"] if tag == "mobile" else 0.0,
            mobile_measurements=raw["mobile_measurements"] if tag == "mobile" else 0,
            held_vehicle_ticks=raw["held_vehicle_ticks"] if tag == "mobile" else 0,
            decision_end_s=raw["decision_end_elapsed_s"]
            if tag == "mobile"
            else raw["elapsed_s"],
            early_termination=int(tag == "mobile" and raw["early_termination"]),
        )
        prefix = "" if tag == "mobile" else "fixed_"
        variance_key = (
            "final_integrated_variance"
            if tag == "mobile"
            else "fixed_integrated_variance"
        )
        if not np.allclose(
            [metrics["final_mse"], metrics["final_integrated_variance"]],
            [raw[prefix + "synthetic_mse"], raw[variance_key]],
            rtol=1e-8,
            atol=1e-9,
        ):
            raise ValueError("El error o la varianza al cierre no reproduce la misión")
        result[tag] = metrics
    return raw, result, tracks, truth if save_truth else None


def reconstruct_scales(env, folder, expected, origin):
    """Volver a leer la misión neural fija conservada, sin volver a decidir rutas."""
    with (folder / "fijo-neural-measurements.csv").open() as stream:
        records = parse_measurements(list(csv.DictReader(stream)))
    with np.load(folder / "truth-private.npz", allow_pickle=False) as f:
        truth = f["truth"]
        if not np.array_equal(f["elapsed_s"], CLOCK) or not np.array_equal(
            f["active_cells"], np.argwhere(env.domain.grid.active_mask)
        ):
            raise ValueError("Reloj/grilla de reconstrucción incompatible")
    config = env.estimator_config.reconstruction_config
    if config.x_length_scale_m != 2000 or config.y_length_scale_m != 2000:
        raise ValueError("El control de reconstrucción debe ser 2 km")
    out = {}
    for scale in (2000, 4000):
        gp = GaussianProcessFieldReconstructor(
            replace(config, x_length_scale_m=scale, y_length_scale_m=scale)
        )
        ticks = []
        for i, t in enumerate(CLOCK):
            now = origin + timedelta(seconds=int(t))
            eligible = tuple(
                r
                for r in records
                if now - env.estimator_config.history_duration <= r.timestamp_utc <= now
            )
            post = gp.reconstruct(
                ReconstructionRequest(
                    grid=env.domain.grid,
                    reconstruction_timestamp_utc=now,
                    measurements=eligible,
                )
            )
            err = post.mean_ug_m3[env.domain.grid.active_mask] - truth[i]
            ticks.append([np.mean(err**2), np.mean(np.abs(err))])
        mse, mae = np.array(ticks).T
        if scale == 2000 and not np.allclose(
            ticks, [[r["mse"], r["mae_ug_m3"]] for r in expected], atol=1e-8, rtol=1e-7
        ):
            raise ValueError("2 km no reproduce el error por instante de la política")
        out[scale] = {
            "temporal_mse": np.trapezoid(mse, CLOCK) / CLOCK[-1],
            "final_mse": mse[-1],
            "temporal_mae_ug_m3": np.trapezoid(mae, CLOCK) / CLOCK[-1],
            "final_mae_ug_m3": mae[-1],
        }
    return out


def run_case(case, cfg, policies, folder):
    seed = case[0]
    folder.mkdir()
    signature = None
    rows = []
    all_ticks = []
    coordinates = np.array(
        [
            policies["fijo"][0].domain.grid.cell_center(int(r), int(c))
            for r, c in np.argwhere(policies["fijo"][0].domain.grid.active_mask)
        ]
    )
    scale_rows = []
    for study in ("fijo", "adaptativo-final", "adaptativo-mejor"):
        env, neural = policies[study]
        strategies = (
            ("neural",)
            if study == "adaptativo-mejor"
            else ("neural", "random", "sweep", "greedy")
        )
        for strategy in strategies:
            for rep in (
                cfg["policy_seeds"] if strategy in ("random", "greedy") else [None]
            ):
                policy = (
                    neural
                    if strategy == "neural"
                    else BaselinePolicy(strategy, seed=rep or 0)
                )
                retain = study == "fijo" and strategy == "neural"
                raw, metrics, tracks, truth = mission(
                    env,
                    policy,
                    case,
                    cfg["threshold_ug_m3"],
                    coordinates,
                    fixed=strategy == "neural" and study != "adaptativo-mejor",
                    save_truth=retain,
                )
                sig = {k: raw[k] for k in SIGNATURES}
                if signature is not None and signature != sig:
                    raise ValueError(
                        f"Campo, fijas, inicio u horizonte distintos en {seed}"
                    )
                signature = sig
                for tag, values in metrics.items():
                    st = strategy if tag == "mobile" else "fixed"
                    ident = {
                        "case_seed": seed,
                        "study": study,
                        "strategy": st,
                        "policy_seed": rep,
                    }
                    rows.append({**ident, **values})
                    all_ticks.extend({**ident, **r} for r in tracks[tag])
                # Guardar las rutas y lecturas móviles y fijas una vez por misión.
                stem = f"{study}-{strategy}" + (f"-{rep}" if rep is not None else "")
                write_csv(
                    folder / f"{stem}-measurements.csv",
                    [
                        {**asdict(r), "timestamp_utc": r.timestamp_utc.isoformat()}
                        for r in env._records
                    ],
                )
                write_csv(
                    folder / f"{stem}-routes.csv",
                    [
                        {"vehicle": i, "vertex": j, "x_m": x, "y_m": y}
                        for i, route in enumerate(env._routes)
                        for j, (x, y) in enumerate(route)
                    ],
                )
                if retain:
                    if (
                        hashlib.sha256(truth.tobytes()).hexdigest()
                        != sig["field_sha256"]
                    ):
                        raise ValueError("La verdad no reproduce la firma")
                    with (folder / "truth-private.npz").open("xb") as stream:
                        np.savez_compressed(
                            stream,
                            truth=truth,
                            elapsed_s=CLOCK,
                            active_cells=np.argwhere(env.domain.grid.active_mask),
                        )
                    for scale, r in reconstruct_scales(
                        env,
                        folder,
                        tracks["mobile"],
                        datetime.fromisoformat(sig["start_utc"]),
                    ).items():
                        scale_rows.append({"case_seed": seed, "scale_m": scale, **r})
    write_csv(folder / "missions.csv", rows)
    write_csv(folder / "ticks.csv", all_ticks)
    write_csv(folder / "scales.csv", scale_rows)
    _write_json(folder / "identity.json", {"case": case, **signature})
    return rows, scale_rows


def run(config, output):
    cfg = json.loads(config.read_text())
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policies = preflight(cfg)
    try:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "protocol.json", cfg)
        print(f"Salida: {output}", flush=True)
        rows, scales = [], []
        with threadpool_limits(limits=1, user_api="blas"):
            for i, case in enumerate(cfg["cases"], 1):
                r, s = run_case(case, cfg, policies, output / f"case-{case[0]}")
                rows.extend(r)
                scales.extend(s)
                print(f"[{i}/24] caso {case[0]} completo", flush=True)
        summarize(rows, scales, cfg, output)
    finally:
        for env in {pair[0] for pair in policies.values()}:
            env.close()
