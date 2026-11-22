"""Agregados e intervalos por ventana, no por celda ni por réplica de política."""

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from .files import _write_json, write_csv

MOBILE = ("random", "sweep", "greedy")
ERRORS = ("temporal_mse", "final_mse", "temporal_mae_ug_m3", "final_mae_ug_m3")


def draws(cases, seed, count):
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for i, (_, _, family, mode) in enumerate(cases):
        strata[family, mode].append(i)
    return np.concatenate(
        [
            rng.choice(group, (count, len(group)), replace=True)
            for group in strata.values()
        ],
        axis=1,
    )


def summarize(rows, scales, cfg, output):
    cases = cfg["cases"]
    seeds = [c[0] for c in cases]
    ix = draws(cases, cfg["bootstrap_seed"], cfg["bootstrap_draws"])
    groups = defaultdict(list)
    for r in rows:
        groups[r["study"], r["strategy"], r["case_seed"]].append(r)
    expected = {
        (study, st, s)
        for s in seeds
        for study in ("fijo", "adaptativo-final", "adaptativo-mejor")
        for st in (
            ("neural",) if study == "adaptativo-mejor" else ("neural", *MOBILE, "fixed")
        )
    }
    if set(groups) != expected:
        raise ValueError("Faltan estrategias o casos completos")
    means = []
    series = defaultdict(dict)

    def add(label, seed, measures):
        for k, v in measures.items():
            series[label, k][seed] = v

    for (study, st, s), rs in groups.items():
        expected_reps = cfg["policy_seeds"] if st in ("greedy", "random") else [None]
        if Counter(r["policy_seed"] for r in rs) != Counter(expected_reps):
            raise ValueError("Réplicas faltantes o repetidas")
        measures = {}
        for k in rs[0]:
            if k in ("study", "strategy", "case_seed", "policy_seed"):
                continue
            vals = [r[k] for r in rs if r[k] is not None]
            measures[k] = float(np.mean(vals)) if vals else None
        means.append({"study": study, "strategy": st, "case_seed": s, **measures})
        add(f"{study}/{st}", s, measures)
    write_csv(output / "cases.csv", means)
    if Counter((r["case_seed"], r["scale_m"]) for r in scales) != Counter(
        (s, m) for s in seeds for m in (2000, 4000)
    ):
        raise ValueError("El contraste de escalas requiere todos los pares")
    write_csv(output / "scale-cases.csv", scales)
    for r in scales:
        add(str(r["scale_m"]), r["case_seed"], {k: r[k] for k in ERRORS})

    point, boot, raw = {}, {}, {}
    stats = []
    for key, by_case in series.items():
        group, k = key
        v = np.array([np.nan if by_case[s] is None else by_case[s] for s in seeds])
        transform = np.sqrt if k.endswith("mse") else np.asarray
        raw[key] = transform(v)
        finite = np.isfinite(v)
        point[key] = float(transform(v[finite].mean())) if finite.any() else None
        # No inventar observaciones ni intervalos donde falta el denominador de focos.
        boot[key] = transform(v[ix].mean(axis=1)) if finite.all() else None
        lo, hi = (
            np.quantile(boot[key], [0.025, 0.975]) if finite.all() else (None, None)
        )
        stats.append(
            {
                "group": group,
                "metric": k.replace("mse", "rmse_ug_m3"),
                "value": point[key],
                "defined_cases": int(finite.sum()),
                "sd_across_cases": float(raw[key][finite].std(ddof=1))
                if finite.sum() > 1
                else None,
                "lo95": lo,
                "hi95": hi,
            }
        )
    write_csv(output / "summary.csv", stats)
    pairs = [
        ("fijo/neural", f"{study}/neural")
        for study in ("adaptativo-final", "adaptativo-mejor")
    ]
    for study in ("fijo", "adaptativo-final", "adaptativo-mejor"):
        base = "fijo" if study == "fijo" else "adaptativo-final"
        pairs.extend((f"{study}/neural", f"{base}/{s}") for s in (*MOBILE, "fixed"))
    pairs.append(("4000", "2000"))
    changes = []
    for a, b in pairs:
        for k in (
            ERRORS
            if a == "4000"
            else (
                *ERRORS,
                "final_integrated_variance",
                "mean_integrated_variance",
                "sampling_coverage",
                "distance_m",
                "hot_cell_time_recall",
                "mean_false_high_area_km2",
                "cell_event_completed_missed_fraction",
                "mean_crps_ug_m3",
            )
        ):
            x, y = point[a, k], point[b, k]
            xs, ys = boot[a, k], boot[b, k]
            ds = raw[a, k] - raw[b, k]
            complete = xs is not None and ys is not None
            lo, hi = np.quantile(xs - ys, [0.025, 0.975]) if complete else (None, None)
            pl, ph = (
                np.quantile(100 * (1 - xs / ys), [0.025, 0.975])
                if complete and np.all(ys > 0)
                else (None, None)
            )
            changes.append(
                {
                    "candidate": a,
                    "reference": b,
                    "metric": k.replace("mse", "rmse_ug_m3"),
                    "delta": x - y if complete else None,
                    "lo95": lo,
                    "hi95": hi,
                    "reduction_pct": 100 * (1 - x / y) if complete and y else None,
                    "pct_lo95": pl,
                    "pct_hi95": ph,
                    "lower_cases": int((ds < 0).sum()),
                    "ties": int((ds == 0).sum()),
                    "higher_cases": int((ds > 0).sum()),
                }
            )
    write_csv(output / "contrasts.csv", changes)
    mobile = []
    for study in ("fijo", "adaptativo-final", "adaptativo-mejor"):
        base = "fijo" if study == "fijo" else "adaptativo-final"
        for k in ("final_integrated_variance", *ERRORS):
            x = study + "/neural", k
            refs = [(base + "/" + r, k) for r in MOBILE]
            v = float(np.mean([100 * (1 - point[x] / point[y]) for y in refs]))
            lo, hi = np.quantile(
                np.mean([100 * (1 - boot[x] / boot[y]) for y in refs], axis=0),
                [0.025, 0.975],
            )
            mobile.append(
                {
                    "study": study,
                    "metric": k.replace("mse", "rmse_ug_m3"),
                    "mean_reduction_pct": v,
                    "lo95": lo,
                    "hi95": hi,
                }
            )
    write_csv(output / "mobile-references.csv", mobile)
    k = "final_integrated_variance"
    delta = raw["fijo/neural", k] - raw["fijo/greedy", k]
    wins = int((delta < 0).sum())
    n = int((delta != 0).sum())
    _write_json(
        output / "completed.json",
        {
            "cases": len(seeds),
            "mobile_missions": sum(r["strategy"] != "fixed" for r in rows),
            "fixed_only_references": sum(r["strategy"] == "fixed" for r in rows),
            "scale_pairs": len(scales) // 2,
            "primary": mobile[0],
            "sign_test_fixed_vs_greedy": {
                "wins": wins,
                "non_ties": n,
                "ties": len(seeds) - n,
                "one_sided_p": float(
                    binomtest(wins, n, 0.5, alternative="greater").pvalue
                )
                if n
                else None,
            },
            "training_executed": False,
            "historical_replay_executed": False,
        },
    )
