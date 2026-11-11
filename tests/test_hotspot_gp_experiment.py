import copy
import hashlib
import json
import sys
import tarfile
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from lima_pm25_rl import hotspot_gp_experiment as experiment
from lima_pm25_rl import hotspot_sampling


@pytest.fixture
def frozen(tmp_path, estimator_config):
    spec = json.loads(hotspot_sampling.DEFAULT_CONFIG.read_text())
    spec.update(
        grid_cells=9,
        steps=6,
        initial_common_ticks=2,
        excursion_cells=3,
        focus_sigma_m=200,
        core_radius_m=250,
        background_from_m=600,
        pulse_on_ticks=3,
        pulse_period_ticks=6,
        sensor_seeds=[0],
        map_ticks=[6],
    )
    gp = replace(
        estimator_config,
        reconstruction_config=replace(
            estimator_config.reconstruction_config,
            x_length_scale_m=2000,
            y_length_scale_m=2000,
        ),
    )
    source = tmp_path / "source"
    hotspot_sampling.run_experiment(spec, gp, source, maps=False)
    return source, experiment.load_source(source)


def test_reconstruction_reproduces_control_and_changes_only_space(frozen, tmp_path):
    source, loaded = frozen
    output = tmp_path / "comparison"
    result = experiment.run_comparison(loaded, source, output, maps=True)
    assert result["all_controls_reproduced"]
    assert len(result["summaries"]) == 8 and len(result["paired"]) == 4
    manifest = json.loads((output / "source-files.json").read_text())
    for name, checksum in manifest.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == checksum
        assert (source / name).read_bytes() == (output / "input" / name).read_bytes()
    for (regime, seed, plan), (_, fields, _, _) in loaded[2].items():
        folder = output / regime / f"noise-{seed}" / plan
        assert (folder / "tick-06.png").is_file()
        for scale in experiment.SCALES:
            report = json.loads((folder / f"meters-{scale}/report.json").read_text())
            assert report["history_duration_s"] == 21600
            expected = {
                **loaded[0]["gp"],
                "x_length_scale_m": scale,
                "y_length_scale_m": scale,
            }
            assert report["gp"] == expected
            assert [r["eligible_records"] for r in report["frames"]] == [
                4,
                7,
                10,
                13,
                16,
                19,
                26,
            ]
            assert report["control_reproduced"] == (scale == 2000)
            with np.load(folder / f"meters-{scale}/posterior-private.npz") as saved:
                if scale == 2000:
                    np.testing.assert_allclose(saved["mean"], fields["mean"], atol=1e-8)
                else:
                    assert not np.allclose(saved["mean"][1:], fields["mean"][1:])
                # El error evaluado es el posterior bruto, sin recortar negativos.
                error = saved["mean"] - fields["truth"]
                mse = np.mean(error**2, axis=1)
                assert report["summary"]["temporal_global_mse"] == pytest.approx(
                    np.trapezoid(mse, dx=300) / 1800
                )
    with pytest.raises(FileExistsError):
        experiment.run_comparison(loaded, source, output, maps=False)


def test_archive_and_directory_match_and_check_does_not_reconstruct(
    frozen, tmp_path, monkeypatch, capsys
):
    source, loaded = frozen
    archive_path = tmp_path / "hotspot.tar.zst"
    with tarfile.open(archive_path, "w:zst") as archive:
        archive.add(source, arcname="cache/hotspot-sampling/example")
    archived = experiment.load_source(archive_path)
    assert loaded[3] == archived[3]
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(
        experiment,
        "GaussianProcessFieldReconstructor",
        lambda *a: pytest.fail("No reconstruir"),
    )
    monkeypatch.setattr(
        hotspot_sampling, "build_episode", lambda *a: pytest.fail("No simular")
    )
    monkeypatch.setattr(
        sys, "argv", ["compare", "--source", str(archive_path), "--check"]
    )
    capsys.readouterr()
    experiment.main()
    report = json.loads(capsys.readouterr().out)
    assert report["frozen_missions"] == 4 and report["reconstructions"] == 8
    assert not report["gp_reconstruction_executed"]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    "change, message",
    [
        ("duplicate", "duplicadas"),
        ("clock", "relojes"),
        ("truth", "huella"),
        ("region", "máscaras"),
    ],
)
def test_rejects_inconsistent_frozen_inputs(frozen, change, message):
    _, (config, grid, cases, _) = frozen
    report, fields, _, _ = copy.deepcopy(next(iter(cases.values())))
    if change == "duplicate":
        report["measurements"].append(report["measurements"][-1])
    elif change == "clock":
        report["measurements"][-1]["simulation_time_s"] += 1
    elif change == "truth":
        fields["truth"][0, 0] += 1
    else:
        fields["region_core"][0] = True
    with pytest.raises(ValueError, match=message):
        experiment.validate_case(grid, config["settings"], report, fields)


def test_control_detects_incompatible_predictions_and_noise(frozen, tmp_path):
    _, loaded = frozen
    config, grid, cases, raw = loaded
    damaged = copy.deepcopy(next(iter(cases.values())))
    damaged[1]["mean"][0, 0] += 1
    with pytest.raises(ValueError, match="posterior original"):
        experiment.score_case(grid, config, damaged, 2000, tmp_path / "bad")
    damaged_cases = copy.deepcopy(cases)
    r, f, records, origin = damaged_cases["persistent", 0, "center"]
    records = list(records)
    records[-1] = replace(records[-1], value_ug_m3=records[-1].value_ug_m3 + 1)
    damaged_cases["persistent", 0, "center"] = (r, f, tuple(records), origin)
    with pytest.raises(ValueError, match="ruido móvil efectivo"):
        experiment.validate_pairs(
            damaged_cases, config["settings"], json.loads(raw["route-preview.json"])
        )


def test_causal_window_excludes_future_and_expired_records(
    frozen, tmp_path, monkeypatch
):
    _, (config, grid, cases, _) = frozen
    report, fields, records, origin = next(iter(cases.values()))
    # Una lectura anterior a seis horas no puede entrar en ningún posterior.
    old = replace(
        records[0],
        sensor_id="EXPIRED",
        timestamp_utc=origin - timedelta(hours=7),
        simulation_time_s=0,
    )
    records = tuple(
        replace(r, simulation_time_s=r.simulation_time_s + 25200) for r in records
    )
    calls = []
    original = experiment.GaussianProcessFieldReconstructor

    class CheckedReconstructor(original):
        def reconstruct(self, request):
            now = request.reconstruction_timestamp_utc
            assert all(
                now - timedelta(hours=6) <= r.timestamp_utc <= now
                for r in request.measurements
            )
            assert all(r.sensor_id != "EXPIRED" for r in request.measurements)
            calls.append(len(request.measurements))
            return super().reconstruct(request)

    monkeypatch.setattr(
        experiment, "GaussianProcessFieldReconstructor", CheckedReconstructor
    )
    experiment.score_case(
        grid,
        config,
        (report, fields, (*records, old), origin),
        2000,
        tmp_path / "causal",
    )
    assert calls == [4, 7, 10, 13, 16, 19, 26]


def test_signed_error_and_criteria_do_not_hide_a_worse_core():
    regions = {
        "core": np.array([True, False, False, False]),
        "surroundings": np.array([False, True, False, False]),
        "background": np.array([False, False, True, True]),
    }
    metrics = experiment.error_metrics(
        np.array([80, 30, 20, 20]),
        np.array([80, 50, -10, 30]),
        np.ones(4),
        regions,
        40,
        40000,
    )
    assert metrics["background_underestimation_mse"] == 450
    assert metrics["background_overestimation_mse"] == 50
    assert metrics["background_mse"] == 500
    assert metrics["negative_area_km2"] == 0.04
    original = {f"temporal_{k}": 2.0 for k in experiment.METRICS}
    candidate = {f"temporal_{k}": 1.0 for k in experiment.METRICS}
    assert experiment.criteria(original, candidate)["primary_pass"]
    candidate["temporal_core_mse"] = 3
    assert not experiment.criteria(original, candidate)["primary_pass"]


def test_aggregation_pools_mse_and_keeps_regimes_and_plans_separate():
    rows = []
    for regime in experiment.REGIMES:
        for plan in experiment.PLANS:
            for seed, mse in ((0, 1), (1, 9)):
                for scale in experiment.SCALES:
                    rows.append(
                        {
                            "regime": regime,
                            "plan": plan,
                            "sensor_seed": seed,
                            "spatial_scale_m": scale,
                            **{f"temporal_{k}": mse for k in experiment.METRICS},
                        }
                    )
    result = experiment.summarize(rows, [0, 1])
    assert len(result["summaries"]) == 8
    assert all(
        r["temporal_core_rmse"] == pytest.approx(np.sqrt(5))
        for r in result["summaries"]
    )
    with pytest.raises(ValueError, match="pares"):
        experiment.summarize(rows[:-1], [0, 1])
