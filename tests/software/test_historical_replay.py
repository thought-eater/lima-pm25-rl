import csv
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from lima_pm25.estimation import GPFitConfig
from lima_pm25.simulation import MeasurementRecord, SensorType

from lima_pm25_rl.adaptive_env import AdaptiveGPEnv, ModelChangeReward
from lima_pm25_rl.episodes import Episode
from lima_pm25_rl.historical_replay import mission
from lima_pm25_rl.policies import BaselinePolicy
from lima_pm25_rl.replay_statistics import summarize_replay


@pytest.mark.parametrize("adaptive", [False, True])
def test_changing_first_closing_target_cannot_change_its_prediction_or_prior_actions(
    adaptive, template, make_env, domain, estimator_config
):
    request = replace(
        template,
        weather=replace(template.weather, duration_s=10800),
        config=replace(template.config, duration_s=10800),
    )
    env = (
        AdaptiveGPEnv(
            domain=domain,
            episode_factory=lambda _: None,
            estimator_config=estimator_config,
            gp_fit=GPFitConfig(),
            model_change=ModelChangeReward(),
        )
        if adaptive
        else make_env()
    )

    def simulate(value):
        rec = tuple(
            MeasurementRecord(
                sensor_id="FIXED",
                sensor_type=SensorType.FIXED,
                timestamp_utc=request.config.start_utc + timedelta(hours=i),
                simulation_time_s=3600.0 * i,
                x_m=700.0,
                y_m=700.0,
                crs=domain.grid.crs,
                value_ug_m3=value if i == 1 else 12.0,
                quality_status="accepted",
                noise_stddev_ug_m3=0.0,
                position_source_id="fixture",
                provenance="fixture",
            )
            for i in range(4)
        )
        return mission(
            env,
            BaselinePolicy("sweep"),
            Episode(request, rec),
            128,
            ["FIXED"],
            fixed_control=True,
            illustrate=False,
        )

    a, b = simulate(8.0), simulate(48.0)
    for tag in ("mobile", "fixed"):
        assert a[0][tag][0]["predicted"] == pytest.approx(b[0][tag][0]["predicted"])
        assert a[0][tag][0]["error"] - b[0][tag][0]["error"] == pytest.approx(40.0)
        assert a[0][tag][1]["predicted"] != pytest.approx(b[0][tag][1]["predicted"])
    assert [r for r in a[4] if r["elapsed_s"] < 3600] == [
        r for r in b[4] if r["elapsed_s"] < 3600
    ]
    assert a[-1] == b[-1]  # El campo móvil no depende de concentraciones futuras.


def test_summary_preserves_station_window_denominators_and_replicas(tmp_path):
    cfg = {
        "studies": ["fijo", "adaptativo-final"],
        "start_utc": ["a", "b", "c"],
        "policy_seeds": [0, 1],
        "bootstrap_draws": 128,
        "bootstrap_seed": 27,
    }
    rows = []
    for study in cfg["studies"]:
        for i, date in enumerate(cfg["start_utc"], 1):
            for st in ("neural", "greedy", "sweep", "random", "fixed"):
                for rep in [0, 1] if st in ("greedy", "random") else [None]:
                    err = i if st == "neural" else 2 * i
                    rows.append(
                        {
                            "study": study,
                            "start_utc": date,
                            "strategy": st,
                            "replica": rep,
                            "hourly_mae": err,
                            "hourly_bias": err,
                            "hourly_mse": err**2,
                            "distance_km": 0 if st == "fixed" else 1,
                            "sampling_coverage": 0.1,
                            "final_integrated_variance": 2.0,
                            "mean_integrated_variance": 3.0,
                        }
                    )
    done = summarize_replay(rows, cfg, tmp_path)
    assert done["one_sided_sign_p"] == 0.125
    with (tmp_path / "contrasts.csv").open() as f:
        r = next(
            r
            for r in csv.DictReader(f)
            if (r["candidate"], r["reference"], r["metric"])
            == ("fijo/neural", "fijo/greedy", "hourly_rmse")
        )
    assert float(r["delta"]) == pytest.approx(-np.sqrt(14 / 3))
    assert float(r["pct_lo95"]) == pytest.approx(50)
    with pytest.raises(ValueError, match="incompleta"):
        summarize_replay(rows[:-1], cfg, tmp_path)
