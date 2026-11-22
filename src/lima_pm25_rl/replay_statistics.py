"""Contrastes pareados sobre ventanas históricas, conservando el emparejamiento."""

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from .files import write_csv


def summarize_replay(rows, cfg, output):
    starts = cfg["start_utc"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["study"], r["strategy"], r["start_utc"]].append(r)
    metrics = (
        "hourly_mse",
        "hourly_mae",
        "hourly_bias",
        "final_integrated_variance",
        "mean_integrated_variance",
        "distance_km",
        "sampling_coverage",
    )
    cases, summary, contrasts = [], [], []
    sides = {}
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(
        len(starts), size=(cfg["bootstrap_draws"], len(starts))
    )
    for study in cfg["studies"]:
        for st in ("neural", "random", "greedy", "sweep", "fixed"):
            values = []
            for start in starts:
                rs = groups.pop((study, st, start), [])
                reps = cfg["policy_seeds"] if st in ("random", "greedy") else [None]
                if Counter(r["replica"] for r in rs) != Counter(reps):
                    raise ValueError("Ventana o réplica incompleta")
                values.append([np.mean([r[k] for r in rs]) for k in metrics])
                cases.append(
                    {
                        "study": study,
                        "strategy": st,
                        "start_utc": start,
                        **dict(zip(metrics, values[-1], strict=True)),
                    }
                )
            for i, k in enumerate(metrics):
                vals = np.array(values)[:, i]
                transform = np.sqrt if k == "hourly_mse" else np.asarray
                point = float(transform(vals.mean()))
                boot = transform(vals[draws].mean(axis=1))
                name = k.replace("mse", "rmse")
                sides[study, st, name] = point, boot, transform(vals)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                summary.append(
                    {
                        "study": study,
                        "strategy": st,
                        "metric": name,
                        "value": point,
                        "sd_windows": transform(vals).std(ddof=1),
                        "lo95": lo,
                        "hi95": hi,
                    }
                )
    if groups:
        raise ValueError("Grupos no previstos en el replay")
    for a, b in [
        *(
            ((s, "neural"), (s, st))
            for s in cfg["studies"]
            for st in ("random", "greedy", "sweep", "fixed")
        ),
        (("fijo", "neural"), ("adaptativo-final", "neural")),
    ]:
        for k in [x.replace("mse", "rmse") for x in metrics]:
            x, xx, xw = sides[*a, k]
            y, yy, yw = sides[*b, k]
            lo, hi = np.quantile(xx - yy, [0.025, 0.975])
            pct = k != "hourly_bias" and y > 0 and np.all(yy > 0)
            pl, ph = (
                np.quantile(100 * (1 - xx / yy), [0.025, 0.975])
                if pct
                else (None, None)
            )
            contrasts.append(
                {
                    "candidate": "/".join(a),
                    "reference": "/".join(b),
                    "metric": k,
                    "delta": x - y,
                    "lo95": lo,
                    "hi95": hi,
                    "lower": int(sum(xw < yw)),
                    "tied": int(sum(xw == yw)),
                    "higher": int(sum(xw > yw)),
                    "reduction_pct": 100 * (1 - x / y) if pct else None,
                    "pct_lo95": pl,
                    "pct_hi95": ph,
                }
            )
    primary = next(
        r
        for r in contrasts
        if (r["candidate"], r["reference"], r["metric"])
        == ("fijo/neural", "fijo/greedy", "hourly_mae")
    )
    n = primary["lower"] + primary["higher"]
    mobile = []
    for study in cfg["studies"]:
        for k in ("hourly_mae", "hourly_rmse", "final_integrated_variance"):
            x, xx, _ = sides[study, "neural", k]
            refs = [sides[study, st, k] for st in ("random", "sweep", "greedy")]
            defined = all(y > 0 and np.all(yy > 0) for y, yy, _ in refs)
            pl, ph = (
                np.quantile(
                    np.mean([100 * (1 - xx / yy) for _, yy, _ in refs], axis=0),
                    [0.025, 0.975],
                )
                if defined
                else (None, None)
            )
            mobile.append(
                {
                    "study": study,
                    "metric": k,
                    "mean_reduction_pct": np.mean(
                        [100 * (1 - x / y) for y, _, _ in refs]
                    )
                    if defined
                    else None,
                    "lo95": pl,
                    "hi95": ph,
                }
            )
    for name, data in (
        ("cases", cases),
        ("summary", summary),
        ("contrasts", contrasts),
        ("mobile-references", mobile),
    ):
        write_csv(output / f"{name}.csv", data)
    return {
        "windows": len(starts),
        "mobile_missions": sum(r["strategy"] != "fixed" for r in rows),
        "fixed_only_references": 2 * len(starts),
        "distinct_station_hours": 9 * len(starts),
        "primary": primary,
        "one_sided_sign_p": float(
            binomtest(primary["lower"], n, 0.5, alternative="greater").pvalue
        )
        if n
        else None,
        "physical_deployment": False,
        "spatial_ground_truth_observed": False,
        "training_executed": False,
    }
