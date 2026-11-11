import copy
import hashlib
import json
import sys
import tarfile
from dataclasses import replace

import numpy as np
import pytest

from lima_pm25_rl import hotspot_gp_experiment as spatial
from lima_pm25_rl import hotspot_gp_time_experiment as temporal
from lima_pm25_rl import hotspot_sampling


@pytest.fixture
def spatial_source(tmp_path, estimator_config):
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
    sampling = tmp_path / "sampling"
    hotspot_sampling.run_experiment(spec, gp, sampling, maps=False)
    source = tmp_path / "spatial"
    spatial.run_comparison(spatial.load_source(sampling), sampling, source, maps=False)
    return source


def test_time_comparison_reproduces_500m_control_and_keeps_reading_history(
    spatial_source, tmp_path
):
    loaded = temporal.load_source(spatial_source)
    config, _, references, raw = loaded
    output = tmp_path / "temporal"
    result = temporal.run_comparison(loaded, spatial_source, output)
    assert result["all_controls_reproduced"]
    assert len(result["summaries"]) == 8
    assert len(result["criteria_by_plan"]) == 2
    manifest = json.loads((output / "source-files.json").read_text())
    for name, checksum in manifest.items():
        assert hashlib.sha256(raw[name]).hexdigest() == checksum
        assert raw[name] == (spatial_source / name).read_bytes()
        assert raw[name] == (output / "input" / name).read_bytes()
    for (regime, seed, plan), (_, fields, _, _) in references.items():
        folder = output / regime / f"noise-{seed}" / plan
        assert (folder / "peak-tracking.png").is_file()
        assert (folder / "tick-06.png").is_file()
        for scale in temporal.SCALES:
            report = json.loads((folder / f"seconds-{scale}/report.json").read_text())
            assert report["gp"] == {**config["gp"], "temporal_length_scale_s": scale}
            assert report["gp"]["x_length_scale_m"] == 500
            assert report["gp"]["y_length_scale_m"] == 500
            assert report["history_duration_s"] == 21600
            assert report["control_reproduced"] == (scale == 3600)
            # A 900 s se retienen también las lecturas con más de 900 s de edad.
            assert [r["eligible_records"] for r in report["frames"]] == [
                4,
                7,
                10,
                13,
                16,
                19,
                26,
            ]
            with np.load(folder / f"seconds-{scale}/posterior-private.npz") as saved:
                if scale == 3600:
                    np.testing.assert_allclose(saved["mean"], fields["mean"], atol=1e-8)
                    np.testing.assert_allclose(saved["std"], fields["std"], atol=1e-8)
                else:
                    assert not np.allclose(saved["mean"][1:], fields["mean"][1:])
                peak_error = saved["mean"][:, fields["region_core"]].max(
                    axis=1
                ) - fields["truth"][:, fields["region_core"]].max(axis=1)
                assert report["summary"]["temporal_peak_abs_error"] == pytest.approx(
                    np.trapezoid(abs(peak_error), dx=300) / 1800
                )
                if regime == "pulsed":
                    assert report["summary"]["declining_ticks"] == 3
                    assert report["summary"][
                        "declining_peak_abs_error"
                    ] == pytest.approx(np.mean(abs(peak_error[4:])))
                    assert report["summary"]["declining_peak_bias"] == pytest.approx(
                        np.mean(peak_error[4:])
                    )
                else:
                    assert report["summary"]["declining_ticks"] == 0
                    assert report["summary"]["declining_peak_abs_error"] is None
    with pytest.raises(FileExistsError):
        temporal.run_comparison(loaded, spatial_source, output, maps=False)


def test_nested_archive_matches_directory_and_check_is_read_only(
    spatial_source, tmp_path, monkeypatch, capsys
):
    archive_path = tmp_path / "hot2.tar.zst"
    with tarfile.open(archive_path, "w:zst") as archive:
        archive.add(spatial_source, arcname="cache/hotspot-gp-experiments/example")
    directory = temporal.load_source(spatial_source)
    archived = temporal.load_source(archive_path)
    assert directory[3] == archived[3]
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(
        spatial,
        "GaussianProcessFieldReconstructor",
        lambda *a: pytest.fail("No reconstruir en --check"),
    )
    monkeypatch.setattr(
        hotspot_sampling, "build_episode", lambda *a: pytest.fail("No simular")
    )
    # Verificar también la resolución predeterminada de hot2.tar.zst.
    monkeypatch.setattr(temporal, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare-time", "--check"])
    capsys.readouterr()
    temporal.main()
    report = json.loads(capsys.readouterr().out)
    assert report["frozen_missions"] == 4 and report["reconstructions"] == 8
    assert report["temporal_scales_s"] == [3600, 900]
    assert report["spatial_scale_m"] == 500
    assert not report["gp_reconstruction_executed"]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("change", ["manifest", "gp", "clock", "control"])
def test_rejects_incompatible_or_modified_spatial_sources(spatial_source, change):
    folder = spatial_source / "persistent/noise-0/center"
    if change == "manifest":
        path = spatial_source / "input/config.json"
        path.write_bytes(path.read_bytes() + b"\n")
        message = "manifiesto"
    elif change == "gp":
        path = folder / "meters-500/report.json"
        report = json.loads(path.read_text())
        report["gp"]["temporal_length_scale_s"] = 900
        path.write_text(json.dumps(report))
        message = "Referencia espacial"
    else:
        scale = 500 if change == "clock" else 2000
        path = folder / f"meters-{scale}/posterior-private.npz"
        with np.load(path) as file:
            fields = {k: file[k] for k in file.files}
        if change == "clock":
            fields["elapsed_s"][-1] += 1
            message = "Reloj"
        else:
            fields["mean"][0, 0] += 1
            message = "control espacial"
        np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match=message):
        temporal.load_source(spatial_source)


def test_time_control_must_reproduce_archived_500m_posterior(spatial_source, tmp_path):
    config, grid, cases, _ = temporal.load_source(spatial_source)
    frozen = copy.deepcopy(next(iter(cases.values())))
    frozen[1]["mean"][0, 0] += 1
    with pytest.raises(ValueError, match="posterior original"):
        temporal.score_case(
            grid, config, frozen, 3600, tmp_path / "bad", scale_key=temporal.SCALE_KEY
        )


def test_temporal_decision_cannot_hide_persistent_damage_or_noise_inconsistency():
    rows = []
    for regime in hotspot_sampling.REGIMES:
        for plan in hotspot_sampling.PLANS:
            for seed in (0, 1):
                for scale in temporal.SCALES:
                    candidate = scale == 900
                    values = {f"temporal_{k}": 2.0 for k in spatial.METRICS}
                    values["temporal_core_mse"] = (
                        (3.0 if regime == "persistent" else 1.0) if candidate else 2.0
                    )
                    rows.append(
                        {
                            "regime": regime,
                            "plan": plan,
                            "sensor_seed": seed,
                            "temporal_scale_s": scale,
                            **values,
                            "declining_peak_abs_error": (1.0 if candidate else 10.0)
                            if regime == "pulsed"
                            else None,
                        }
                    )
    result = temporal.summarize_time(rows, [0, 1])
    assert all(not r["primary_pass_both_regimes"] for r in result["criteria_by_plan"])
    assert all(
        r["primary_pass"]
        for r in result["criteria_by_regime_and_plan"]
        if r["regime"] == "pulsed"
    )
    # La media puede pasar y aun así debe mostrarse el deterioro de una semilla.
    for row in rows:
        if (
            row["regime"] == "persistent"
            and row["temporal_scale_s"] == 900
            and row["sensor_seed"] == 1
        ):
            row["temporal_core_mse"] = 0.5
    result = temporal.summarize_time(rows, [0, 1])
    for decision in result["criteria_by_plan"]:
        assert decision["primary_pass_both_regimes"]
        assert not decision["paired_by_noise"][0]["primary_pass_both_regimes"]
        assert decision["paired_by_noise"][1]["primary_pass_both_regimes"]
    with pytest.raises(ValueError, match="pares"):
        temporal.summarize_time(rows[:-1], [0, 1])
