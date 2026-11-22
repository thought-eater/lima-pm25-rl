"""Replay reservado: decisiones con fijas observadas y móviles sobre un campo modelado."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import timedelta
from itertools import pairwise

import numpy as np
import torch
from lima_pm25.domain import (
    AmbientBaseline,
    RegionalBoundaryForcing,
    create_lima_pm25_station_catalog,
    load_study_environment_config,
)
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
    load_historical_pm25_csv,
)
from lima_pm25.simulation import (
    RegionalBoundarySchedule,
    RoadLineSource,
    create_synthetic_study_family_requests,
    load_historical_weather_csv,
)
from threadpoolctl import threadpool_limits

from .episodes import HistoricalReplay
from .files import PROJECT_ROOT, _write_json, write_csv
from .hotspot_evaluation import measurement_digest
from .independent_evaluation import preflight
from .policies import BaselinePolicy
from .replay_statistics import summarize_replay
from .study import _utc

HOUR = timedelta(hours=1)
STUDIES = ("fijo", "adaptativo-final")
STRATEGIES = ("neural", "random", "sweep", "greedy")


def check_windows(cfg, spec, part):
    starts = list(map(_utc, cfg["start_utc"]))
    if (
        not starts
        or len(set(starts)) != len(starts)
        or any(b - a < 72 * HOUR for a, b in pairwise(starts))
    ):
        raise ValueError("Las ventanas deben estar separadas al menos 72 horas")
    for t in starts:
        if (
            t - 24 * HOUR < _utc(spec["reserved_final_start_utc"])
            or t + 3 * HOUR >= _utc(spec["reserved_final_end_utc"])
            or any(
                t - 24 * HOUR <= _utc(w["end_utc"])
                and _utc(w["context_start_utc"]) <= t + 3 * HOUR
                for w in part["calibration_windows"] + part["replay_windows"]
            )
        ):
            raise ValueError("Ventana fuera de reserva o solapada con O2")
    return starts


def make_replay(template, start, weather, pm, cfg, spec, scale, seed):
    sites = cfg["stations"]
    context = pm.observed_measurements(
        grid=template.model.grid,
        replay_start_utc=start - 24 * HOUR,
        end_utc=start,
        station_ids=sites,
        noise_stddev_ug_m3=0.0,
    )
    if {r.sensor_id for r in context} != set(sites) or any(
        r.timestamp_utc >= start for r in context
    ):
        raise ValueError("El fondo requiere observaciones anteriores al inicio")
    ambient = float(np.median([r.value_ug_m3 for r in context]))
    provenance = "Mediana fija del contexto observado anterior a la misión (24 h)"
    initial = template.model.initialize_uniform(
        AmbientBaseline("replay-context", ambient, provenance)
    )
    if not all(isinstance(s, RoadLineSource) for s in template.sources):
        raise ValueError("El control de replay requiere fuentes viales rutinarias")
    request = replace(
        template,
        initial_state=initial,
        sources=tuple(
            RoadLineSource(
                source_id=s.source_id,
                line=s.line,
                role=s.role,
                emission_rate_ug_m_s=s.emission_rate_ug_m_s * scale,
                starts_at_s=0.0,
                ends_at_s=10800.0,
                provenance="Tasa rutinaria hipotética con escala congelada O2",
            )
            for s in template.sources
        ),
        regional_boundary=RegionalBoundarySchedule.constant(
            RegionalBoundaryForcing("replay-context", ambient, provenance)
        ),
        mobile_position_providers=(),
        sensor_configs=(),
    )
    episode = HistoricalReplay(
        request,
        weather,
        pm,
        start,
        _utc(spec["gp_fit_end_utc"]),
        _utc(spec["reserved_final_start_utc"]),
        include_closing_stamp=True,
    )(seed)
    stamps = Counter(
        (r.sensor_id, r.simulation_time_s) for r in episode.historical_fixed
    )
    if stamps != Counter((s, h * 3600.0) for s in sites for h in range(4)) or any(
        "PM2_5_SOURCE=senamhi_observed" not in r.provenance
        for r in episode.historical_fixed
    ):
        raise ValueError("Las fijas no corresponden al soporte prefijado de SENAMHI")
    if not all(
        r.wind_speed_provenance.source == "senamhi_observed"
        and r.precipitation_provenance.source == "senamhi_observed"
        and r.wind_direction_provenance.source in {"senamhi_observed", "not_applicable"}
        for r in episode.request.weather.records
    ):
        raise ValueError("Forzamiento meteorológico distinto del protocolo")
    return episode, ambient


def point_means(env, now, points, fixed):
    """GP vigente en el instante de emisión: coordenadas instrumentales, no centro de celda."""
    records = tuple(
        r
        for r in env._records
        if now - env.estimator_config.history_duration <= r.timestamp_utc <= now
        and (not fixed or r.sensor_type == "fixed")
    )
    kernel = None
    if hasattr(env, "_adaptive"):
        updater = env._fixed_adaptive if fixed else env._adaptive
        if updater.timestamp and updater.timestamp > now:
            raise ValueError("Parámetros GP posteriores a la emisión")
        kernel = updater.parameters.kernel()
    return (
        GaussianProcessFieldReconstructor(
            env.estimator_config.reconstruction_config, kernel=kernel
        )
        .reconstruct_points(
            ReconstructionRequest(
                grid=env.domain.grid,
                reconstruction_timestamp_utc=now,
                measurements=records,
            ),
            points,
        )
        .mean_ug_m3
    )


def mission(
    env, policy, episode, seed, site_ids, *, fixed_control, illustrate, q_trace=None
):
    env._episode_factory = lambda _: episode
    obs, _ = env.reset(seed=seed, options={"scenario_seed": seed})
    sim = env._simulation
    if not sim.is_historical_replay:
        raise ValueError("El replay requiere lecturas observadas")
    sites = {s.site_id: s for s in sim.request.station_catalog.sites}
    points = [(sites[s].x_m, sites[s].y_m) for s in site_ids]
    if any(
        not np.allclose(
            points[site_ids.index(r.sensor_id)], (r.x_m, r.y_m), atol=1e-6, rtol=0
        )
        for r in episode.historical_fixed
    ):
        raise ValueError("Coordenadas instrumentales distintas del catálogo")
    policy.reset()
    tags = ("mobile", "fixed") if fixed_control else ("mobile",)
    forecasts = {k: [] for k in tags}
    sealed = {}
    field_digest = hashlib.sha256()
    hours = {k: [] for k in tags}
    frames = {k: [] for k in tags}
    maps = {k: [] for k in tags}
    moves = []
    start = sim.request.config.start_utc
    active = True
    decision_end = 0
    initial = env.positions
    for tick in range(37):
        seconds = sim.state.simulation_time_s
        if seconds != tick * 300:
            raise ValueError("Reloj del replay incompatible")
        now = start + timedelta(seconds=seconds)
        # Verificación de emparejamiento del evaluador; nunca puntuar el campo como observado.
        field_digest.update(
            sim.state.concentration_ug_m3[env.domain.grid.active_mask].tobytes()
        )
        # Al cierre ya se puede puntuar la media SELLADA en el tick anterior.
        if tick and tick % 12 == 0:
            actual = {
                r.sensor_id: r.value_ug_m3
                for r in env._records
                if r.sensor_type == "fixed" and r.timestamp_utc == now
            }
            for tag in tags:
                issued, pred = sealed.pop(tag)
                if issued != seconds - 300 or actual.keys() != set(site_ids):
                    raise ValueError("Hora prequential incompleta")
                for station, x in zip(site_ids, pred, strict=True):
                    hours[tag].append(
                        {
                            "elapsed_s": seconds,
                            "last_prediction_s": issued,
                            "station": station,
                            "observed": actual[station],
                            "predicted": float(x),
                            "error": float(x - actual[station]),
                        }
                    )
        for tag in tags:
            fixed = tag == "fixed"
            post = env.predict_fixed() if fixed else env._posterior
            if tick < 36:
                forecasts[tag].append(point_means(env, now, points, fixed).copy())
                if tick % 12 == 11:
                    if len(forecasts[tag]) != 12:
                        raise ValueError(
                            "No se completaron doce estimaciones previas al cierre"
                        )
                    sealed[tag] = (seconds, np.mean(forecasts[tag], axis=0))
                    forecasts[tag].clear()
            frame = {
                "elapsed_s": seconds,
                "distance_m": 0.0 if fixed else float(env._distance.sum()),
                "integrated_variance": env._integrated_variance(post),
            }
            frames[tag].append(frame)
            if illustrate:
                mask = env.domain.grid.active_mask
                maps[tag].append(
                    (post.mean_ug_m3[mask], post.standard_deviation_ug_m3[mask])
                )
        if tick == 36:
            break
        if active:
            action = (
                policy.act(env, obs)
                if q_trace is None
                else policy.plan(env, obs, q_trace=q_trace)[0]
            )
            prev = env.positions
            obs, _, terminated, truncated, info = env.step(action)
            if info["invalid_actions"].any():
                raise ValueError("Acción inválida en replay")
            decision_end = seconds + 300
            for order, code in enumerate(action):
                v, d = divmod(int(code), 8)
                moves.append(
                    {
                        "elapsed_s": seconds,
                        "order": order,
                        "vehicle": v,
                        "direction": None if info["held_vehicles"][v] else d,
                        "from_x": prev[v][0],
                        "from_y": prev[v][1],
                        "to_x": env.positions[v][0],
                        "to_y": env.positions[v][1],
                    }
                )
            active = not (terminated or truncated)
        else:
            for batch in sim.advance(env.positions, include_mobile=False):
                env._assimilate(batch)
            env._posterior = env._predict()
    fixed_records = tuple(r for r in env._records if r.sensor_type == "fixed")
    if measurement_digest(fixed_records) != measurement_digest(
        tuple(
            sorted(
                episode.historical_fixed, key=lambda r: (r.timestamp_utc, r.sensor_id)
            )
        )
    ):
        raise ValueError("No se incorporaron exactamente las lecturas reales")
    metrics = {}
    for tag in tags:
        errors = np.array([r["error"] for r in hours[tag]])
        cells = {
            env.domain.grid.xy_to_cell(r.x_m, r.y_m)
            for r in env._records
            if tag == "mobile" and r.sensor_type == "mobile"
        }
        variances = [r["integrated_variance"] for r in frames[tag]]
        metrics[tag] = {
            "hourly_mae": float(np.abs(errors).mean()),
            "hourly_mse": float(np.mean(errors**2)),
            "hourly_bias": float(errors.mean()),
            "final_integrated_variance": variances[-1],
            "mean_integrated_variance": float(np.trapezoid(variances) / 36),
            "distance_km": frames[tag][-1]["distance_m"] / 1000,
            "sampling_coverage": len(cells) / int(env.domain.grid.active_mask.sum()),
        }
    return (
        hours,
        metrics,
        frames,
        maps,
        moves,
        decision_end,
        initial,
        field_digest.hexdigest(),
    )


def run_window(
    start, seed, policies, templates, weather, pm, cfg, spec, scale, folder, illustrate
):
    folder.mkdir()
    rows, observations, show = [], [], []
    signatures = []
    for study in STUDIES:
        env, neural = policies[study]
        episode, ambient = make_replay(
            templates[study], start, weather, pm, cfg, spec, scale, seed
        )
        for st in STRATEGIES:
            for rep in cfg["policy_seeds"] if st in ("random", "greedy") else [None]:
                demo = illustrate and rep in (None, cfg["illustrative_replica"])
                trace = [] if demo and st == "neural" else None
                out = mission(
                    env,
                    neural if st == "neural" else BaselinePolicy(st, seed=rep or 0),
                    episode,
                    seed,
                    cfg["stations"],
                    fixed_control=st == "neural",
                    illustrate=demo,
                    q_trace=trace,
                )
                hours, metrics, frames, maps, actions, end, initial, digest = out
                signature = (
                    initial,
                    measurement_digest(
                        tuple(r for r in env._records if r.sensor_type == "fixed")
                    ),
                    ambient,
                    digest,
                )
                if signatures and signature != signatures[0]:
                    raise ValueError(
                        "Inicio, estaciones o fondo no coinciden entre estrategias"
                    )
                signatures.append(signature)
                stem = f"{study}-{st}-{rep if rep is not None else 'det'}"
                samples = [
                    {**asdict(r), "timestamp_utc": r.timestamp_utc.isoformat()}
                    for r in env._records
                ]
                write_csv(folder / f"{stem}-samples.csv", samples)
                write_csv(folder / f"{stem}-actions.csv", actions)
                write_csv(
                    folder / f"{stem}-routes.csv",
                    [
                        {"vehicle": v, "vertex": j, "x": x, "y": y}
                        for v, line in enumerate(env._routes)
                        for j, (x, y) in enumerate(line)
                    ],
                )
                for tag, values in metrics.items():
                    strategy = "fixed" if tag == "fixed" else st
                    identity = {
                        "start_utc": start.isoformat(),
                        "study": study,
                        "strategy": strategy,
                        "replica": rep,
                    }
                    rows.append({**identity, **values})
                    observations.extend({**identity, **r} for r in hours[tag])
                    if demo:
                        name = f"{study}-{strategy}"
                        with (folder / f"{name}.npz").open("xb") as f:
                            mean, std = np.array(maps[tag]).transpose(1, 0, 2)
                            np.savez_compressed(
                                f,
                                mean=mean,
                                std=std,
                                cells=np.argwhere(env.domain.grid.active_mask),
                            )
                        show.append(
                            {
                                **identity,
                                "id": name,
                                "stem": stem,
                                "end": end if tag == "mobile" else 10800,
                                "frames": frames[tag],
                                "hourly": hours[tag],
                                **(
                                    {"q_trace": trace}
                                    if trace is not None and tag == "mobile"
                                    else {}
                                ),
                            }
                        )
    write_csv(folder / "missions.csv", rows)
    write_csv(folder / "station-hours.csv", observations)
    if illustrate:
        _write_json(folder / "viewer.json", show)
    _write_json(
        folder / "window.json",
        {
            "seed": seed,
            "start_utc": start.isoformat(),
            "ambient_ug_m3": ambient,
            "initial_positions": signatures[0][0],
            "fixed_digest": signatures[0][1],
            "simulated_mobile_field_sha256": signatures[0][3],
        },
    )
    return rows


def run(config, output):
    cfg = json.loads(config.read_text())
    directory = PROJECT_ROOT / "config"
    spec = json.loads((directory / "entrenamiento.json").read_text())
    root = (directory / spec["core_root"]).resolve()
    part = json.loads((root / spec["gp_partition_manifest"]).read_text())
    starts = check_windows(cfg, spec, part)
    frozen = json.loads((directory / cfg["models_protocol"]).read_text())
    frozen["models"] = {s: frozen["models"][s] for s in STUDIES}
    if (
        tuple(cfg["studies"]) != STUDIES
        or cfg["policy_seeds"] != frozen["policy_seeds"]
        or cfg["illustrative_window"] != 0
    ):
        raise ValueError("Selección de modelos o réplicas no corresponde al protocolo")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policies = preflight(frozen, synthetic_calendar=False)
    try:
        weather = load_historical_weather_csv(
            root / "data" / part["weather_resource_id"],
            station_id=cfg["weather_station"],
        )
        pm = load_historical_pm25_csv(root / "data" / part["pm25_resource_id"])
        if (
            pm.resource_sha256 != part["pm25_resource_sha256"]
            or weather.resource_sha256 != part["weather_resource_sha256"]
        ):
            raise ValueError("Datos históricos distintos del manifiesto")
        scale = json.loads((root / "config/estimation_evaluation.json").read_text())[
            "routine_source_scale"
        ]
        study_config = load_study_environment_config(
            root / "config/study_environment.json"
        )
        templates = {
            s: create_synthetic_study_family_requests(
                e.domain,
                station_catalog=create_lima_pm25_station_catalog(e.domain),
                study_config=study_config,
            )["control"]
            for s, (e, _) in policies.items()
        }
        output.mkdir(parents=True, exist_ok=False)
        _write_json(
            output / "protocol.json",
            {
                **cfg,
                "models": frozen["models"],
                "source_scale": scale,
                "pm25_sha256": pm.resource_sha256,
                "weather_sha256": weather.resource_sha256,
            },
        )
        print(f"Replay: {output}", flush=True)
        rows = []
        with threadpool_limits(limits=1, user_api="blas"):
            for i, t in enumerate(starts):
                rows.extend(
                    run_window(
                        t,
                        cfg["first_seed"] + i,
                        policies,
                        templates,
                        weather,
                        pm,
                        cfg,
                        spec,
                        scale,
                        output / f"window-{i + 1:02d}",
                        i == cfg["illustrative_window"],
                    )
                )
                print(f"[{i + 1}/{len(starts)}] {t.isoformat()}", flush=True)
        result = summarize_replay(rows, cfg, output)
        _write_json(output / "completed.json", result)
        print(f"Replay completo: {output}", flush=True)
    finally:
        for env, _ in policies.values():
            env.close()
