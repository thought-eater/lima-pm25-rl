import csv
import json

import numpy as np
import pytest

from lima_pm25_rl.files import PROJECT_ROOT
from lima_pm25_rl.independent_evaluation import validate_calendar
from lima_pm25_rl.independent_statistics import ERRORS, MOBILE, draws, summarize


def test_calendar_rejects_a_prior_window_and_reused_validation_seed():
    settings = PROJECT_ROOT / "config"
    spec = json.loads((settings / "entrenamiento.json").read_text())
    cfg = json.loads((settings / "prueba-independiente.json").read_text())
    validate_calendar(cfg, spec)
    cfg["cases"][0][1] = cfg["excluded_starts_utc"][0]
    with pytest.raises(ValueError, match="Ventana"):
        validate_calendar(cfg, spec)
    cfg["cases"][0][0] = spec["validation_seeds"][0]
    with pytest.raises(ValueError, match="Semillas"):
        validate_calendar(cfg, spec)


def test_independent_statistics_pools_mse_first_and_pairs_the_bootstrap(tmp_path):
    cfg = json.loads((PROJECT_ROOT / "config/prueba-independiente.json").read_text())
    cfg["bootstrap_draws"] = 100
    cases = cfg["cases"]
    ix = draws(cases, 11, 10)
    for row in ix:
        assert sorted(tuple(cases[i][2:]) for i in row) == sorted(
            tuple(c[2:]) for c in cases
        )
    rows, scales = [], []
    for i, (seed, *_) in enumerate(cases):
        for study in cfg["models"]:
            for strategy in (
                ("neural",)
                if study == "adaptativo-mejor"
                else ("fixed", "neural", *MOBILE)
            ):
                for rep in range(5) if strategy in ("random", "greedy") else [None]:
                    mse = (i + 1) ** 2
                    if strategy == "neural":
                        mse /= 4
                    stats = dict.fromkeys(
                        (
                            *ERRORS,
                            "mean_integrated_variance",
                            "final_integrated_variance",
                            "sampling_coverage",
                            "distance_m",
                            "mean_crps_ug_m3",
                            "cell_event_completed_missed_fraction",
                            "hot_cell_time_recall",
                            "mean_false_high_area_km2",
                        ),
                        mse,
                    )
                    rows.append(
                        {
                            "study": study,
                            "strategy": strategy,
                            "policy_seed": rep,
                            "case_seed": seed,
                            **stats,
                        }
                    )
        for scale in (2000, 4000):
            scales.append(
                {
                    "scale_m": scale,
                    "case_seed": seed,
                    **dict.fromkeys(ERRORS, (i + 1) ** 2),
                }
            )
    summarize(rows, scales, cfg, tmp_path)
    result = json.loads((tmp_path / "completed.json").read_text())
    assert result["mobile_missions"] == 600 and result["fixed_only_references"] == 48
    assert result["primary"]["mean_reduction_pct"] == pytest.approx(75.0)
    assert result["sign_test_fixed_vs_greedy"]["one_sided_p"] == pytest.approx(2**-24)
    with (tmp_path / "contrasts.csv").open() as f:
        contrasts = list(csv.DictReader(f))
    r = next(
        r
        for r in contrasts
        if (r["candidate"], r["reference"], r["metric"])
        == ("fijo/neural", "fijo/random", "final_rmse_ug_m3")
    )
    assert float(r["delta"]) == pytest.approx(
        -0.5 * np.sqrt(np.mean(np.arange(1, 25) ** 2))
    )
    assert float(r["pct_lo95"]) == pytest.approx(50.0)
    assert float(r["pct_hi95"]) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="Réplicas"):
        summarize([*rows, rows[-1]], scales, cfg, tmp_path)
