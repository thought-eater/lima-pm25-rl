import copy
import hashlib
import json
import sys
import tarfile
from dataclasses import replace

import numpy as np
import pytest
from lima_pm25.estimation import ReconstructionConfig

from lima_pm25_rl import hotspot_gp_experiment as spatial
from lima_pm25_rl import hotspot_gp_mixture as mixture
from lima_pm25_rl import hotspot_gp_time_experiment as temporal
from lima_pm25_rl import hotspot_sampling
from lima_pm25_rl.gp_mixture import mix_marginals


@pytest.fixture
def source(tmp_path, estimator_config):
    spec = json.loads(hotspot_sampling.DEFAULT_CONFIG.read_text())
    spec.update(
        grid_cells=9,
        steps=10,
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
    sampling = tmp_path / "sampling"
    hotspot_sampling.run_experiment(spec, gp, sampling, maps=False)
    spatial_dir = tmp_path / "spatial"
    spatial.run_comparison(
        spatial.load_source(sampling), sampling, spatial_dir, maps=False
    )
    temporal_dir = tmp_path / "temporal"
    temporal.run_comparison(
        temporal.load_source(spatial_dir), spatial_dir, temporal_dir, maps=False
    )
    return temporal_dir


def test_four_arms_reproduce_references_and_save_weights_and_likelihoods(
    source, tmp_path
):
    loaded = mixture.load_source(source)
    output = tmp_path / "mixture"
    result = mixture.run_comparison(loaded, source, output)
    assert result["all_references_reproduced"] and len(result["summaries"]) == 16
    manifest = json.loads((output / "source-files.json").read_text())
    for name, checksum in manifest.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == checksum
        assert (source / name).read_bytes() == (output / "input" / name).read_bytes()
    for key, (_, fields, _, _) in loaded[2].items():
        regime, seed, plan = key
        folder = output / regime / f"noise-{seed}" / plan
        assert (folder / "tracking.png").is_file() and (
            folder / "tick-06.png"
        ).is_file()
        with np.load(folder / "weights-private.npz") as file:
            weights = file["fast_weight"]
            assert weights.shape == (11, 81)
            assert (weights >= 0.05).all() and (weights <= 0.95).all()
            assert (weights[0] == 0.5).all() and (
                file["fast_weight_before"][0] == 0.5
            ).all()
            assert (weights != 0.5).any()
            assert (file["local_batch_count"] <= 3).all()
        predictions = {}
        for arm in mixture.ARMS:
            with np.load(folder / arm / "posterior-private.npz") as file:
                predictions[arm] = {k: file[k] for k in ("mean", "std")}
            report = json.loads((folder / arm / "report.json").read_text())
            mse = ((predictions[arm]["mean"] - fields["truth"]) ** 2).mean(axis=1)
            assert report["summary"]["temporal_global_mse"] == pytest.approx(
                np.trapezoid(mse, dx=300) / 3000
            )
            if arm in mixture.SCALES:
                for name in ("mean", "std"):
                    np.testing.assert_allclose(
                        predictions[arm][name], loaded[4][key][arm][1][name], atol=1e-8
                    )
        args = [predictions[a][k] for a in ("slow", "fast") for k in ("mean", "std")]
        for arm, weight in (("fixed", 0.5), ("adaptive", weights)):
            expected = mix_marginals(*args, weight)
            np.testing.assert_allclose(predictions[arm]["mean"], expected[0])
            np.testing.assert_allclose(predictions[arm]["std"], expected[1])
        trace = json.loads((folder / "prequential.json").read_text())
        assert trace[0]["previous_records"] == 0
        assert trace[-1]["eligible_records"] == 38
        assert all(not r["revisit_used"] for r in trace[0]["readings"])
        assert any(r["revisit_used"] for tick in trace for r in tick["readings"])
        for tick in trace:
            for row in tick["readings"]:
                w = row["fast_weight_before"]
                expected = np.logaddexp(
                    np.log(w) + row["fast_log_density"],
                    np.log1p(-w) + row["slow_log_density"],
                )
                assert row["adaptive_log_density_before"] == pytest.approx(expected)
    with pytest.raises(FileExistsError):
        mixture.run_comparison(loaded, source, output, maps=False)


def test_forecasts_exclude_current_batch_and_future_changes_preserve_past(
    source, monkeypatch
):
    config, grid, cases, _, _ = mixture.load_source(source)
    report, _, records, origin = cases["pulsed", 0, "center"]
    times = [r["elapsed_s"] for r in report["frames"]]
    original = mixture.GaussianProcessFieldReconstructor
    point_calls = []

    class CheckedReconstructor(original):
        def reconstruct_points(self, request, positions_m):
            assert all(
                r.timestamp_utc < request.reconstruction_timestamp_utc
                for r in request.measurements
            )
            batch = [
                r
                for r in records
                if r.timestamp_utc == request.reconstruction_timestamp_utc
            ]
            assert set(positions_m) == {(r.x_m, r.y_m) for r in batch}
            point_calls.append(len(request.measurements))
            return super().reconstruct_points(request, positions_m)

        def reconstruct(self, request):
            assert all(
                r.timestamp_utc <= request.reconstruction_timestamp_utc
                for r in request.measurements
            )
            return super().reconstruct(request)

    monkeypatch.setattr(
        mixture, "GaussianProcessFieldReconstructor", CheckedReconstructor
    )
    base = ReconstructionConfig(**config["gp"])
    first = mixture.reconstruct_sequence(grid, base, 21600, records, origin, times)
    # Cambiar una lectura futura no modifica ningún peso ni posterior anterior.
    changed = tuple(
        replace(r, value_ug_m3=r.value_ug_m3 + 100)
        if r.simulation_time_s == times[-1]
        else r
        for r in records
    )
    second = mixture.reconstruct_sequence(grid, base, 21600, changed, origin, times)
    assert point_calls[:2] == [0, 0]
    for arm in mixture.ARMS:
        for name in ("mean", "std"):
            np.testing.assert_array_equal(
                first[0][arm][name][:-1], second[0][arm][name][:-1]
            )
    np.testing.assert_array_equal(
        first[1]["fast_weight"][:-1], second[1]["fast_weight"][:-1]
    )
    for a, b in zip(first[2][-1]["readings"], second[2][-1]["readings"], strict=True):
        assert a["slow_mean_before"] == b["slow_mean_before"]
        assert a["fast_mean_before"] == b["fast_mean_before"]
        assert a["fast_weight_before"] == b["fast_weight_before"]
    third = mixture.reconstruct_sequence(
        grid, base, 21600, tuple(reversed(records)), origin, times
    )
    np.testing.assert_array_equal(first[1]["fast_weight"], third[1]["fast_weight"])
    np.testing.assert_array_equal(
        first[0]["adaptive"]["mean"], third[0]["adaptive"]["mean"]
    )


def test_hot3_archive_check_does_not_reconstruct_or_write(
    source, tmp_path, monkeypatch, capsys
):
    archive_path = tmp_path / "hot3.tar.zst"
    with tarfile.open(archive_path, "w:zst") as archive:
        archive.add(source, arcname="cache/hotspot-gp-time-experiments/example")
    assert mixture.load_source(archive_path)[3] == mixture.load_source(source)[3]
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(
        mixture, "reconstruct_sequence", lambda *a: pytest.fail("No reconstruir")
    )
    monkeypatch.setattr(mixture, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["mixture", "--check"])
    capsys.readouterr()
    mixture.main()
    report = json.loads(capsys.readouterr().out)
    assert report["evaluated_sequences"] == 16 and report["gp_sequences"] == 8
    assert not report["gp_reconstruction_executed"]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_manifest_and_fast_reference_are_verified(source, tmp_path):
    loaded = mixture.load_source(source)
    key = ("pulsed", 0, "center")
    bad = copy.deepcopy(loaded[4][key])
    bad["fast"][1]["mean"][0, 0] += 1
    with pytest.raises(ValueError, match="fast.*hot3"):
        mixture.score_case(
            loaded[0], loaded[1], loaded[2][key], bad, tmp_path / "bad", maps=False
        )
    path = source / "input/config.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifiesto"):
        mixture.load_source(source)


def test_equal_fixed_and_adaptive_results_are_not_evidence_for_adaptation():
    rows = []
    for regime in hotspot_sampling.REGIMES:
        for plan in hotspot_sampling.PLANS:
            for arm in mixture.ARMS:
                rows.append(
                    {
                        "regime": regime,
                        "plan": plan,
                        "sensor_seed": 0,
                        "arm": arm,
                        **{f"temporal_{k}": 2.0 for k in spatial.METRICS},
                        "declining_peak_abs_error": 2.0 if regime == "pulsed" else None,
                    }
                )
    result = mixture.summarize_mixture(rows, [0])
    assert len(result["summaries"]) == 16
    assert all(
        not r["strict_gain_vs_fixed_both_regimes"] for r in result["criteria_by_plan"]
    )
    assert all(
        not r["primary_pass_vs_slow_and_fixed_both_regimes"]
        for r in result["criteria_by_plan"]
    )
    with pytest.raises(ValueError):
        mixture.summarize_mixture(rows[:-1], [0])
