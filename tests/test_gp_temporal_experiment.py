import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import MeasurementRecord

from lima_pm25_rl import gp_temporal_experiment as experiment
from lima_pm25_rl.gp_spatial_experiment import (
    gp_variant,
    load_frozen_case,
    score_frozen_case,
    select_scale,
    temporal_thirds,
)
from lima_pm25_rl.training import TrainingConfig


@pytest.fixture
def updater(estimator_config):
    return replace(
        estimator_config,
        reconstruction_config=replace(
            estimator_config.reconstruction_config,
            x_length_scale_m=2000,
            y_length_scale_m=2000,
        ),
    )


def frozen_case(folder, grid, updater, seed, month):
    """Datos minúsculos para verificar reconstrucción; no simula una misión."""
    folder.mkdir(parents=True)
    start = datetime(2025, month, 1, tzinfo=UTC)
    times = np.array([0.0, 300.0, 600.0])
    records = tuple(
        MeasurementRecord(
            sensor_id="FIXED",
            sensor_type="fixed",
            timestamp_utc=start + timedelta(seconds=t),
            simulation_time_s=t,
            x_m=100,
            y_m=100,
            crs=grid.crs,
            value_ug_m3=value,
            quality_status="accepted",
            noise_stddev_ug_m3=1,
            position_source_id="fixture",
            provenance="Prueba de software",
        )
        for t, value in zip(times, (10, 20, 8), strict=True)
    )
    truth = np.stack([np.full(grid.active_mask.sum(), v) for v in (10.0, 15.0, 9.0)])
    model = GaussianProcessFieldReconstructor(updater.reconstruction_config)
    fields = [
        model.reconstruct(
            ReconstructionRequest(
                grid=grid,
                reconstruction_timestamp_utc=start + timedelta(seconds=t),
                measurements=tuple(r for r in records if r.simulation_time_s <= t),
            )
        )
        for t in times
    ]
    report = {
        "case_seed": seed,
        "measurements": [
            {**asdict(r), "timestamp_utc": r.timestamp_utc.isoformat()} for r in records
        ],
        "signatures": {
            "field_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
            "horizon_s": 600,
        },
    }
    (folder / "report.json").write_text(json.dumps(report))
    np.savez_compressed(
        folder / "fields-private.npz",
        truth=truth,
        elapsed_s=times,
        active_cells=np.argwhere(grid.active_mask),
        grid_shape=grid.shape,
        grid_bounds_m=grid.grid_bounds_m,
        mean_after=[r.mean_ug_m3[grid.active_mask] for r in fields],
        std_after=[r.standard_deviation_ug_m3[grid.active_mask] for r in fields],
    )


def test_temporal_variant_changes_only_time_and_keeps_all_readings(
    domain, updater, tmp_path
):
    variant = gp_variant(updater.reconstruction_config, 900, "temporal_scale_s")
    expected = asdict(updater.reconstruction_config)
    expected["temporal_length_scale_s"] = 900
    assert asdict(variant) == expected
    assert updater.history_duration.total_seconds() == 21600
    frozen_case(tmp_path / "input", domain.grid, updater, 1, 1)
    frozen = load_frozen_case(
        tmp_path / "input",
        domain.grid,
        start_utc=datetime(2025, 1, 1, tzinfo=UTC),
        end_utc=datetime(2025, 2, 1, tzinfo=UTC),
    )
    for scale in (900, 3600):
        score_frozen_case(
            domain.grid,
            updater,
            frozen,
            scale=scale,
            scale_key="temporal_scale_s",
            output=tmp_path / str(scale),
            control=scale == 3600,
        )
    reports = [
        json.loads((tmp_path / str(s) / "report.json").read_text()) for s in (900, 3600)
    ]
    assert [r["eligible_records"] for r in reports[0]["ticks"]] == [1, 2, 3]
    assert [r["eligible_records"] for r in reports[1]["ticks"]] == [1, 2, 3]
    assert reports[0]["summary"]["temporal_mse"] != pytest.approx(
        reports[1]["summary"]["temporal_mse"]
    )
    with (
        np.load(tmp_path / "900/posterior-private.npz") as a,
        np.load(tmp_path / "3600/posterior-private.npz") as b,
    ):
        np.testing.assert_allclose(a["mean"][0], b["mean"][0])
        np.testing.assert_allclose(a["std"][0], b["std"][0])
    with pytest.raises(ValueError, match="2000"):
        experiment.check_control(
            domain.grid,
            replace(
                updater,
                reconstruction_config=gp_variant(
                    updater.reconstruction_config, 4000, "spatial_scale_m"
                ),
            ),
        )


def test_temporal_thirds_preserve_total_integral_and_selection_units():
    times = np.array([0, 300, 600])
    mse = np.array([0, 4, 1])
    thirds = temporal_thirds(mse, times)
    assert np.mean(list(thirds.values())) == pytest.approx(
        np.trapezoid(mse, times) / 600
    )
    rows = [
        {"case_seed": c, "temporal_scale_s": s, "temporal_mse": value}
        for s, value in ((900, 2), (3600, 2))
        for c in (1, 2)
    ]
    result = select_scale(rows, [900, 3600], [1, 2], 3600, scale_key="temporal_scale_s")
    assert result["selected_scale_s"] == 3600
    assert "selected_scale_m" not in result


def test_cli_reuses_both_partitions_without_simulation_and_freezes_selection(
    domain, updater, tmp_path, monkeypatch, capsys
):
    parameters = TrainingConfig(device="cpu")
    seed = int(np.random.SeedSequence([42, 0, 0]).generate_state(1)[0])
    protocol = {
        "scenarios": [
            {
                "configuration": {
                    "training_start_utc": "2025-01-01T00:00:00+00:00",
                    "training_end_utc": "2025-02-01T00:00:00+00:00",
                    "evaluation_start_utc": "2025-02-01T00:00:00+00:00",
                    "evaluation_end_utc": "2025-03-01T00:00:00+00:00",
                }
            }
        ],
        "gp_fit_end_utc": "2025-02-01T00:00:00+00:00",
        "reserved_final_start_utc": "2025-03-01T00:00:00+00:00",
    }
    source, diagnostic = tmp_path / "spatial", tmp_path / "diagnostic"
    frozen_case(source / "calibration" / f"case-{seed}", domain.grid, updater, seed, 1)
    frozen_case(diagnostic / "case-100", domain.grid, updater, 100, 2)
    metadata = {
        "protocol": protocol,
        "original_gp": asdict(updater.reconstruction_config),
        "training": asdict(parameters),
        "calibration_seeds": [seed],
        "validation_seeds": [100],
        "selection_partition": "training",
        "diagnostic_source": str(diagnostic),
        "policy_sha256": "fixture",
    }
    (source / "config.json").write_text(json.dumps(metadata))
    (source / "summary.json").write_text("{}")
    (source / "source-files.json").write_text(
        json.dumps(
            [{"case_seed": 100, **experiment.frozen_files(diagnostic / "case-100")}]
        )
    )
    (diagnostic / "config.json").write_text(
        json.dumps(
            {"protocol": protocol, "policy_sha256": "fixture", "case_seeds": [100]}
        )
    )
    spec_path = tmp_path / "experiment.json"
    spec_path.write_text(
        json.dumps(
            {
                "kind": "generated",
                "episodes": 200,
                "steps_per_episode": 2,
                "training": asdict(parameters),
                "validation_seeds": [100],
            }
        )
    )
    settings = {
        "experiment_config": spec_path.name,
        "calibration_cases": 1,
        "temporal_scales_s": [900, 3600],
        "selection_metric": "mean_case_temporal_mse",
        "map_ticks": [1],
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings))
    env = SimpleNamespace(
        domain=domain,
        estimator_config=updater,
        close=lambda: None,
        reset=lambda *a, **k: pytest.fail("No simular"),
        step=lambda *a: pytest.fail("No simular"),
    )
    monkeypatch.setattr(
        experiment, "prepare_experiment", lambda *a: (env, env, protocol)
    )
    args = [
        "compare-gp-time",
        "--config",
        str(settings_path),
        "--spatial-experiment",
        str(source),
    ]
    loader = experiment.load_frozen_case
    monkeypatch.setattr(
        experiment,
        "load_frozen_case",
        lambda *a, **k: pytest.fail("check no carga campos"),
    )
    monkeypatch.setattr(sys, "argv", [*args, "--check"])
    experiment.main()
    assert json.loads(capsys.readouterr().out)["new_missions"] == 0
    output = tmp_path / "output"
    validation_reads = []

    def guarded(path, *a, **k):
        if path == diagnostic / "case-100":
            assert (
                json.loads((output / "selection.json").read_text())[
                    "selection_partition"
                ]
                == "training"
            )
            validation_reads.append(path)
        return loader(path, *a, **k)

    monkeypatch.setattr(experiment, "load_frozen_case", guarded)
    monkeypatch.setattr(sys, "argv", [*args, "--output", str(output)])
    experiment.main()
    assert len(validation_reads) == 1
    assert (output / "validation/case-100/tick-01.png").is_file()
    result = json.loads((output / "summary.json").read_text())
    assert all("last_third_rmse" in r for r in result["variants"])
    (source / "source-files.json").write_text(
        json.dumps(
            [
                {
                    "case_seed": 100,
                    "report.json": "incorrect",
                    "fields-private.npz": "incorrect",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="difieren"):
        experiment.run_comparison(
            domain.grid,
            updater,
            settings=settings,
            source=source,
            diagnostic=diagnostic,
            metadata=metadata,
            output=tmp_path / "mismatch",
        )
