"""Visor local de inferencia: cada estrategia parte del mismo escenario y reloj."""

import base64
import io
import json
import secrets
from copy import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

from .files import PROJECT_ROOT, product
from .independent_evaluation import preflight
from .map_viewer import (
    DEFAULT_CONFIG,
    _geometry,
    colorize,
    raster_projection,
)
from .network import NeuralController
from .policies import BaselinePolicy

LABELS = {
    "neural": "BiLG-D3QN",
    "random": "Aleatoria",
    "sweep": "Barrido",
    "greedy": "Greedy",
    "fixed": "Solo fijas",
}
STUDIES = {
    "fijo": "GP fijo · final 30 000",
    "adaptativo-final": "Adaptativo · final 30 000",
    "adaptativo-mejor": "Adaptativo · mejor 28 400",
}


class LiveBranch:
    """Episodio, controlador e historial independientes; solo los pesos y el dominio se comparten."""

    def __init__(self, prototype, controller, strategy, seed):
        self.env = copy(prototype)
        self.strategy = strategy
        self.policy = (
            NeuralController(
                controller.network,
                history_length=controller.history_length,
                history_mode=controller.history_mode,
            )
            if strategy == "neural"
            else BaselinePolicy(strategy, seed=0)
            if strategy != "fixed"
            else None
        )
        self.obs, _ = self.env.reset(seed=seed)
        self.finished = False
        self.decision_end = self.env._simulation.duration_s
        self.action = None
        self.frames = []

    def advance(self):
        e = self.env
        if self.strategy == "fixed" or self.finished:
            for batch in e._simulation.advance(e.positions, include_mobile=False):
                e._assimilate(batch)
            e._posterior = e._predict()
        else:
            self.obs, _, term, trunc, info = e.step(self.action)
            if info["invalid_actions"].any():
                raise ValueError("Movimiento inválido en la sesión")
            self.finished = term or trunc
            if self.finished:
                self.decision_end = float(e._simulation.state.simulation_time_s)

    def snapshot(self, app):
        e = self.env
        t = float(e._simulation.state.simulation_time_s)
        trace, actions = [], []
        if (
            self.policy is not None
            and not self.finished
            and t < e._simulation.duration_s
        ):
            if self.strategy == "neural":
                self.action, decisions = self.policy.plan(e, self.obs, q_trace=trace)
                for decision in decisions:
                    route = e.route(decision.vehicle, decision.action)
                    actions.append(
                        {
                            "time": t,
                            "vehicle": decision.vehicle,
                            "direction": decision.action,
                            "destination": app.latlon([route.coordinates[-1]])[0],
                        }
                    )
            else:
                self.action = self.policy.act(e, self.obs)
                reserved = set()
                for assignment in self.action:
                    vehicle, direction = divmod(int(assignment), 8)
                    if e.action_masks(reserved)[vehicle, direction]:
                        route = e.route(vehicle, direction)
                        reserved.add(route.cell)
                        destination = route.coordinates[-1]
                    else:
                        direction = None
                        destination = e.positions[vehicle]
                    actions.append(
                        {
                            "time": t,
                            "vehicle": vehicle,
                            "direction": direction,
                            "destination": app.latlon([destination])[0],
                        }
                    )
        mask = e.domain.grid.active_mask
        truth = e._simulation.state.concentration_ug_m3[mask]
        mean = e._posterior.mean_ug_m3[mask]
        std = e._posterior.standard_deviation_ug_m3[mask]
        error = truth - mean
        hot = truth >= app.threshold
        values = {"mean": mean, "std": std, "truth": truth, "error": error}
        images = {}
        for k, v in values.items():
            scale = "concentration" if k in ("mean", "truth") else k
            rgba = colorize(
                v,
                app.indices,
                app.limits[scale],
                {"std": "magma", "error": "RdBu_r"}.get(k, "viridis"),
            )
            buf = io.BytesIO()
            Image.fromarray(rgba).save(buf, format="PNG", compress_level=1)
            images[k] = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
        samples = [
            {
                "id": r.sensor_id,
                "time": r.simulation_time_s,
                "type": str(r.sensor_type),
                "value": r.value_ug_m3,
                "position": app.latlon([(r.x_m, r.y_m)])[0],
            }
            for r in e._records
        ]
        routes = (
            []
            if self.strategy == "fixed"
            else [
                {"vehicle": v, "points": app.latlon(line)}
                for v, line in enumerate(e._routes)
            ]
        )
        return {
            "start": e._simulation.request.config.start_utc.isoformat(),
            "decision_end": self.decision_end,
            "values": {
                k: v.round(5).tolist() for k, v in values.items() if k != "error"
            },
            "images": images,
            "samples": samples,
            "routes": routes,
            "actions": actions,
            "q_trace": trace,
            "metrics": {
                "rmse": np.sqrt(np.mean(error**2)),
                "distance_km": float(e._distance.sum() / 1000),
                "hot_rmse": np.sqrt(np.mean(error[hot] ** 2)) if hot.any() else None,
                "recall": float(np.mean(mean[hot] >= app.threshold))
                if hot.any()
                else None,
            },
        }


class LiveViewer:
    def __init__(self, policies, settings, threshold):
        self.policies = policies
        grid, self.latlon, _, _, _ = _geometry(
            (DEFAULT_CONFIG.parent / settings["core_root"]).resolve()
        )
        self.indices, _ = raster_projection(grid, settings["raster_width"])
        self.limits = settings["live_limits"]
        self.threshold = threshold
        self.known = set()
        self.seed = None
        self.branches = {}

    def new(self):
        seed = secrets.randbits(32)
        while seed in self.known:
            seed = secrets.randbits(32)
        self.known.add(seed)
        return [
            {
                "case": seed,
                "live": True,
                "limits": self.limits,
                "study": label,
                "study_id": study,
                "estimator": "fijo" if study == "fijo" else "adaptativo",
                "strategy": name,
                "strategy_id": st,
                "episode": (28400 if study == "adaptativo-mejor" else 30000)
                if st == "neural"
                else None,
                "times": list(range(0, 10801, 300)),
            }
            for study, label in STUDIES.items()
            for st, name in LABELS.items()
            if study != "adaptativo-mejor" or st == "neural"
        ]

    def frame(self, seed, study, st, tick):
        if (
            seed not in self.known
            or study not in self.policies
            or st not in LABELS
            or not 0 <= tick <= 36
            or (study == "adaptativo-mejor" and st != "neural")
        ):
            raise ValueError("Selección de misión no válida")
        if self.seed != seed:
            self.branches.clear()
            self.seed = seed
        key = study, st
        if key not in self.branches:
            self.branches[key] = LiveBranch(*self.policies[study], st, seed)
        b = self.branches[key]
        while len(b.frames) <= tick:
            if b.frames:
                b.advance()
            b.frames.append(
                json.dumps(
                    b.snapshot(self), allow_nan=False, separators=(",", ":")
                ).encode()
            )
        return b.frames[tick]


def run():
    cfg = json.loads((PROJECT_ROOT / "config/prueba-independiente.json").read_text())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    policies = preflight(cfg, synthetic_calendar=False)
    app = LiveViewer(
        policies, json.loads(DEFAULT_CONFIG.read_text()), cfg["threshold_ug_m3"]
    )
    page = (
        (product("visor") / "index.html")
        .read_bytes()
        .replace(b"</head>", b'<meta name="pm25-live" content="1"></head>', 1)
    )
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError, ConnectionResetError:
                pass

        def do_GET(self):
            u = urlsplit(self.path)
            if u.path == "/":
                self.send(page, "text/html; charset=utf-8")
            elif u.path == "/frame":
                try:
                    q = parse_qs(u.query, strict_parsing=True)
                    with lock:
                        body = app.frame(
                            int(q["case"][0]),
                            q["model"][0],
                            q["strategy"][0],
                            int(q["tick"][0]),
                        )
                except ValueError, KeyError:
                    self.send_error(400)
                    return
                self.send(body, "application/json")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/scenario":
                self.send_error(404)
                return
            with lock:
                choices = app.new()
            self.send(json.dumps(choices).encode(), "application/json")

    with (
        ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server,
        threadpool_limits(1, user_api="blas"),
    ):
        print(
            f"Visor en vivo: http://127.0.0.1:{server.server_port}/  (Ctrl+C para cerrar)",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            for env in {e for e, _ in policies.values()}:
                env.close()
