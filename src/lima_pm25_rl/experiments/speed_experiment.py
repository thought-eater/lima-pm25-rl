"""Sensibilidad de evaluación: filtrar rutas antes de decidir, solo en este ensayo."""

import csv
import json
from collections import defaultdict
from itertools import groupby

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..comparison import SIGNATURES
from ..evaluate import _compatible
from ..evaluation import evaluate_episode
from ..files import PROJECT_ROOT, _write_json, product, write_csv
from ..policies import BaselinePolicy
from ..simulation import STEP_SECONDS
from ..study import _sha256, prepare_experiment
from ..training import TrainingConfig, load_policy, resolve_device

METRICS = (
    "temporal_mse",
    "final_mse",
    "mean_integrated_variance",
    "final_integrated_variance",
    "distance_m",
    "sampling_coverage",
)


class LimitedRoutes:
    """Vista local del router: máscara, reservas y ejecución consultan el mismo plan."""

    def __init__(self, router, limit_m):
        if not np.isfinite(limit_m) or limit_m <= 0:
            raise ValueError("El máximo por tick debe ser positivo y finito")
        self.router = router
        self.limit_m = limit_m

    def __getattr__(self, name):
        return getattr(self.router, name)

    def plan(self, *args, **kwargs):
        route = self.router.plan(*args, **kwargs)
        return route if route is not None and route.distance_m <= self.limit_m else None


def indexed(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    result = {(int(r["case_seed"]), r["strategy"], r["policy_seed"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Misiones originales duplicadas")
    return result


def aggregate(rows):
    """Cada fila ya representa la media de réplicas de un caso (MSE sin raíz)."""
    return {
        k.replace("mse", "rmse_ug_m3"): float(
            np.sqrt(np.mean([r[k] for r in rows]))
            if k.endswith("mse")
            else np.mean([r[k] for r in rows])
        )
        for k in METRICS
    }


def summarize(rows, labels, output):
    cases = []
    key = lambda r: (r["study"], r["strategy"], r["case_seed"], r["variant"])
    for (study, strategy, seed, variant), items in groupby(
        sorted(rows, key=key), key=key
    ):
        items = list(items)
        cases.append(
            {
                "study": study,
                "strategy": strategy,
                "case_seed": seed,
                "variant": variant,
                "replicas": len(items),
                **{k: np.mean([r[k] for r in items]) for k in METRICS},
            }
        )
    write_csv(output / "cases.csv", cases)
    changes, advantages = [], []
    for scope in ["all", *sorted({r["case_seed"] for r in cases})]:
        group = defaultdict(list)
        for r in cases:
            if scope == "all" or r["case_seed"] == scope:
                group[r["study"], r["strategy"], r["variant"]].append(r)
        stats = {k: aggregate(v) for k, v in group.items()}
        for study, strategy in sorted({k[:2] for k in stats}):
            for metric, a in stats[study, strategy, "original"].items():
                b = stats[study, strategy, "limited"][metric]
                changes.append(
                    {
                        "case_seed": scope,
                        "study": study,
                        "strategy": strategy,
                        "metric": metric,
                        "original": a,
                        "limited": b,
                        "delta": b - a,
                    }
                )
        for study in labels:
            base = "adaptativo-final" if study == "adaptativo-mejor" else study
            for strategy in ("random", "greedy", "sweep", "fixed"):
                for metric in (
                    "temporal_rmse_ug_m3",
                    "final_rmse_ug_m3",
                    "final_integrated_variance",
                ):
                    a, b = [
                        100
                        * (
                            1
                            - stats[study, "neural", v][metric]
                            / stats[base, strategy, v][metric]
                        )
                        for v in ("original", "limited")
                    ]
                    advantages.append(
                        {
                            "case_seed": scope,
                            "study": study,
                            "reference": strategy,
                            "metric": metric,
                            "original_reduction_pct": a,
                            "limited_reduction_pct": b,
                            "change_pp": b - a,
                        }
                    )
    write_csv(output / "changes.csv", changes)
    write_csv(output / "advantages.csv", advantages)


def run(args):
    cfg = json.loads(args.config.read_text())
    output = args.output
    limit = cfg["max_kmh"] * STEP_SECONDS / 3.6
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "protocol.json",
        {
            **cfg,
            "max_m_per_vehicle_tick": limit,
            "partition": "synthetic_validation",
            "decision_rule": "mask before each reservation; same cap in execution; forced holds as usual",
            "control": "retained unrestricted missions; fixed-only unchanged; shared adaptive references",
            "training_executed": False,
        },
    )
    rows = []
    adaptive_reference = None
    with (
        (output / "missions.csv").open("x", newline="") as f,
        (output / "ticks.csv").open("x", newline="") as t,
        threadpool_limits(limits=1, user_api="blas"),
    ):
        writer = None
        ticks = csv.writer(t)
        ticks.writerow(
            [
                "study",
                "case_seed",
                "strategy",
                "policy_seed",
                "elapsed_s",
                "mse",
                "m0",
                "m1",
                "m2",
            ]
        )
        for study, profile, model_key, file, mkey, pkey, ep in cfg["studies"]:
            spec = json.loads((PROJECT_ROOT / "config" / profile).read_text())
            meta = json.loads((product(mkey) / "config.json").read_text())
            baseline = indexed(product(mkey) / "cases.csv")
            original = indexed(product(pkey) / "missions.csv")
            if (
                json.loads((product(pkey) / "source.json").read_text())["metadata"]
                != meta
                or meta["partition"] != "validation"
                or meta["case_seeds"] != spec["validation_seeds"]
                or meta["policy_seeds"] != spec["evaluation_policy_seeds"]
                or spec.get("arrival_strategy", "shortest") != "shortest"
            ):
                raise ValueError("Control incompatible con la sensibilidad")
            expected = {
                (s, st, i)
                for s in meta["case_seeds"]
                for st in ("neural", "fixed", "sweep", "greedy", "random")
                for i in (
                    [str(p) for p in meta["policy_seeds"]]
                    if st in ("greedy", "random")
                    else [""]
                )
            }
            if set(original) != expected or set(baseline) != expected:
                raise ValueError("Faltan misiones originales")
            if study.startswith("adaptativo"):
                refs = {k: r for k, r in original.items() if k[1] != "neural"}
                if adaptive_reference is not None and refs != adaptive_reference:
                    raise ValueError(
                        "Los adaptativos no comparten sus referencias originales"
                    )
                adaptive_reference = refs
            conf = TrainingConfig(**spec["training"])
            device = resolve_device(conf.device)
            torch.set_num_threads(conf.cpu_threads)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            train, env, protocol = prepare_experiment(spec, PROJECT_ROOT / "config")
            train.close()
            path = product(model_key) / file
            state = torch.load(path, map_location="cpu", weights_only=True)
            if (
                _sha256(path) != meta["checkpoint_sha256"]
                or state["completed_episodes"] != ep
                or not _compatible(state, conf, protocol)
            ):
                raise ValueError("Pesos distintos del control original")
            del state
            neural, _ = load_policy(path, device=str(device))
            env.router = LimitedRoutes(env.router, limit)
            try:
                for seed, strategy, rep in original:
                    if study == "adaptativo-mejor" and strategy != "neural":
                        continue
                    ident = {
                        "study": study,
                        "case_seed": seed,
                        "strategy": strategy,
                        "policy_seed": rep,
                    }
                    key = seed, strategy, rep
                    before = {k: float(original[key][k]) for k in METRICS}
                    original_held = int(baseline[key]["held_vehicle_ticks"])
                    end = float(baseline[key]["decision_end_elapsed_s"])
                    first = {
                        **ident,
                        "variant": "original",
                        **before,
                        "held_vehicle_ticks": original_held,
                        "decision_end_s": end,
                    }
                    if strategy == "fixed":
                        second = {**first, "variant": "limited"}
                    else:
                        series = []

                        def observe(
                            e, posterior, distance, *, ident=ident, series=series
                        ):
                            time = e._simulation.state.simulation_time_s
                            mask = e.domain.grid.active_mask
                            error = (
                                posterior.mean_ug_m3
                                - e._simulation.state.concentration_ug_m3
                            )[mask]
                            mse = float(np.mean(error**2))
                            if distance is not None and np.any(distance > limit):
                                raise ValueError(
                                    "Movimiento ejecutado fuera del límite"
                                )
                            ticks.writerow(
                                [
                                    *ident.values(),
                                    time,
                                    mse,
                                    *(distance if distance is not None else [""] * 3),
                                ]
                            )
                            series.append((time, mse))

                        policy = (
                            neural
                            if strategy == "neural"
                            else BaselinePolicy(strategy, seed=int(rep or 0))
                        )
                        r = evaluate_episode(
                            env,
                            policy,
                            seed=seed,
                            comparison=True,
                            observe_tick=observe,
                        )
                        if r["invalid_actions"] or any(
                            str(r[k]) != baseline[key][k] for k in SIGNATURES
                        ):
                            raise ValueError(f"Acción inválida o caso distinto: {key}")
                        control = original[seed, "fixed", ""]
                        if not np.allclose(
                            [r["fixed_synthetic_mse"], r["fixed_integrated_variance"]],
                            [
                                float(control["final_mse"]),
                                float(control["final_integrated_variance"]),
                            ],
                            rtol=1e-8,
                        ):
                            raise ValueError("La referencia solo fijas cambió")
                        times, mses = np.array(series).T
                        if not np.array_equal(
                            times, np.arange(37) * STEP_SECONDS
                        ) or not np.isclose(mses[-1], r["synthetic_mse"]):
                            raise ValueError("Horizonte o MSE incompleto")
                        after = {
                            **before,
                            **{k: r[k] for k in METRICS if not k.endswith("mse")},
                            "temporal_mse": np.trapezoid(mses, times) / times[-1],
                            "final_mse": r["synthetic_mse"],
                        }
                        second = {
                            **ident,
                            "variant": "limited",
                            **after,
                            "held_vehicle_ticks": r["held_vehicle_ticks"],
                            "decision_end_s": r["decision_end_elapsed_s"],
                        }
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(first))
                        writer.writeheader()
                    writer.writerows([first, second])
                    rows.extend([first, second])
                    f.flush()
                    if strategy == "sweep" or study == "adaptativo-mejor":
                        print(study, seed, flush=True)
            finally:
                env.close()
    summarize(rows, [r[0] for r in cfg["studies"]], output)
    print(output, flush=True)
