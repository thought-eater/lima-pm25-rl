import json
import sys
import tarfile
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import MeasurementRecord
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from lima_pm25_rl import gp_multiscale_experiment as experiment
from lima_pm25_rl import hotspot_sampling
from lima_pm25_rl.gp_multiscale import multiscale_kernel


def test_kernel_preserves_variance_and_time_covariance_and_has_no_free_parameters(
    estimator_config,
):
    config = replace(
        estimator_config.reconstruction_config,
        x_length_scale_m=2000,
        y_length_scale_m=2000,
    )
    kernel = multiscale_kernel(config)
    variance = config.prior_stddev_ug_m3**2
    regional = ConstantKernel(variance, constant_value_bounds="fixed") * Matern(
        [2000, 2000, 3600], length_scale_bounds="fixed", nu=1.5
    )
    points = np.array([[0, 0, 0], [0, 0, 300], [500, 500, -600], [2000, 0, 0]])
    np.testing.assert_allclose(kernel.diag(points), variance)
    np.testing.assert_allclose(kernel(points[:2]), regional(points[:2]))
    assert np.linalg.eigvalsh(kernel(points)).min() > 0
    assert kernel.n_dims == 0
    assert kernel(points)[0, 3] < regional(points)[0, 3]
    with pytest.raises(ValueError):
        multiscale_kernel(replace(config, temporal_length_scale_s=900))


def test_joint_gp_is_not_an_average_of_separately_fitted_maps(domain, estimator_config):
    config = replace(
        estimator_config.reconstruction_config,
        x_length_scale_m=2000,
        y_length_scale_m=2000,
    )
    now = datetime(2025, 1, 1, tzinfo=UTC)
    records = tuple(
        MeasurementRecord(
            sensor_id=f"S{i}",
            sensor_type="mobile",
            timestamp_utc=now,
            simulation_time_s=0,
            x_m=x,
            y_m=100,
            crs=domain.grid.crs,
            value_ug_m3=value,
            noise_stddev_ug_m3=1,
            quality_status="accepted",
            position_source_id="test",
            provenance="Control numérico artificial",
        )
        for i, (x, value) in enumerate(((100, 80), (700, 20)))
    )
    request = ReconstructionRequest(
        grid=domain.grid, reconstruction_timestamp_utc=now, measurements=records
    )
    joint = GaussianProcessFieldReconstructor(config, kernel=multiscale_kernel(config))
    original = GaussianProcessFieldReconstructor(config).reconstruct(request)
    local = GaussianProcessFieldReconstructor(
        replace(config, x_length_scale_m=500, y_length_scale_m=500)
    ).reconstruct(request)
    combined = joint.reconstruct(request)
    assert not np.allclose(
        combined.mean_ug_m3, (original.mean_ug_m3 + local.mean_ug_m3) / 2
    )
    prior = joint.reconstruct(replace(request, measurements=()))
    np.testing.assert_allclose(
        prior.standard_deviation_ug_m3[domain.grid.active_mask],
        config.prior_stddev_ug_m3,
    )
    np.testing.assert_allclose(
        prior.mean_ug_m3[domain.grid.active_mask], config.prior_mean_ug_m3
    )


@pytest.fixture
def sources(tmp_path, estimator_config):
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
    hotspots = tmp_path / "hotspots"
    hotspot_sampling.run_experiment(spec, gp, hotspots, maps=False)
    config = json.loads((hotspots / "config.json").read_text())
    training = tmp_path / "training"
    training.mkdir()
    protocol = {
        "grid_shape": [9, 9],
        "gp_fit_end_utc": "2025-01-02T00:00:00+00:00",
        "scenarios": [
            {
                "configuration": {
                    "training_start_utc": "2025-01-01T00:00:00+00:00",
                    "training_end_utc": "2025-01-02T00:00:00+00:00",
                }
            }
        ],
    }
    (training / "config.json").write_text(
        json.dumps(
            {
                "original_gp": config["gp"],
                "calibration_seeds": [42, 43],
                "validation_seeds": [100000],
                "selection_partition": "training",
                "neural_training_executed": False,
                "settings": {"calibration_cases": 2, "map_ticks": [1, 6]},
                "protocol": protocol,
            }
        )
    )
    for seed, regime in ((42, "persistent"), (43, "pulsed")):
        source = hotspots / regime / "noise-0/center"
        report = json.loads((source / "report.json").read_text())
        with np.load(source / "fields-private.npz") as file:
            fields = {k: file[k] for k in file.files}
        folder = training / "calibration" / f"case-{seed}"
        (folder / "scale-2000").mkdir(parents=True)
        report["case_seed"] = seed
        (folder / "report.json").write_text(json.dumps(report))
        times = np.arange(7) * 300
        np.savez_compressed(
            folder / "fields-private.npz",
            **fields,
            mean_after=fields["mean"],
            std_after=fields["std"],
            elapsed_s=times,
        )
        np.savez_compressed(
            folder / "scale-2000/posterior-private.npz",
            mean=fields["mean"],
            std=fields["std"],
            active_cells=fields["active_cells"],
            elapsed_s=times,
        )
        (folder / "scale-2000/report.json").write_text(
            json.dumps(
                {
                    "reconstruction_config": config["gp"],
                    "input_signatures": report["signatures"],
                    "summary": {
                        "case_seed": seed,
                        "spatial_scale_m": 2000,
                        "control_reproduced": True,
                        "temporal_mse": report["summary"]["temporal_global_mse"],
                        "final_mse": report["summary"]["final_global_mse"],
                    },
                }
            )
        )
    # La batería nueva debe ignorar estos bytes incluso dentro de un tar.
    validation = training / "validation/case-100000"
    validation.mkdir(parents=True)
    (validation / "fields-private.npz").write_bytes(b"NO ABRIR VALIDACION")
    return {"training": training, "hotspots": hotspots}


def test_both_suites_reproduce_control_and_keep_inputs(sources, tmp_path):
    training = experiment.load_training(sources["training"])
    hotspots = experiment.load_hotspots(sources["hotspots"])
    output = tmp_path / "comparison"
    result = experiment.run_comparison(training, hotspots, sources, output)
    assert result["all_controls_reproduced"]
    assert len(result["training"]["paired"]) == 2
    assert len(result["hotspots"]["paired"]) == 4
    assert (output / "training/case-42/tick-01.png").is_file()
    assert (output / "hotspots/pulsed/noise-0/center/tick-06.png").is_file()
    assert not (output / "input/training/validation").exists()
    for suite, raw in (("training", training[2]), ("hotspots", hotspots[3])):
        for name, data in raw.items():
            assert (output / "input" / suite / name).read_bytes() == data
    for seed, (_, frozen, _) in training[1].items():
        fields = frozen[1]
        for arm in experiment.ARMS:
            folder = output / "training" / f"case-{seed}" / arm
            report = json.loads((folder / "report.json").read_text())
            with np.load(folder / "posterior-private.npz") as file:
                if arm == "original":
                    np.testing.assert_allclose(
                        file["mean"], fields["mean_after"], atol=1e-8
                    )
                else:
                    assert not np.allclose(file["mean"][1:], fields["mean_after"][1:])
                mse = ((file["mean"] - fields["truth"]) ** 2).mean(axis=1)
                assert report["summary"]["temporal_mse"] == pytest.approx(
                    np.trapezoid(mse, dx=300) / 1800
                )
            assert all(
                f["mse"] == pytest.approx(f["spatial_bias_mse"] + f["centered_mse"])
                for f in report["frames"]
            )
            assert [f["eligible_records"] for f in report["frames"]] == [
                4,
                7,
                10,
                13,
                16,
                19,
                26,
            ]
    with pytest.raises(FileExistsError):
        experiment.run_comparison(training, hotspots, sources, output, maps=False)


def test_archive_loader_and_check_never_open_validation_or_reconstruct(
    sources, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "gp3.tar.zst"
    with tarfile.open(path, "w:zst") as file:
        file.add(sources["training"], arcname="cache/gp-spatial-experiments/example")
    original = tarfile.TarFile.extractfile

    def checked_extract(self, member):
        assert "validation" not in member.name.split("/")
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", checked_extract)
    assert (
        experiment.load_training(path)[2]
        == experiment.load_training(sources["training"])[2]
    )
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(
        experiment.GaussianProcessFieldReconstructor,
        "reconstruct",
        lambda *a: pytest.fail("No reconstruir"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multiscale",
            "--training-source",
            str(path),
            "--hotspot-source",
            str(sources["hotspots"]),
            "--check",
        ],
    )
    capsys.readouterr()
    experiment.main()
    report = json.loads(capsys.readouterr().out)
    assert report["training_cases"] == 2 and report["reconstructions"] == 12
    assert not report["validation_opened"] and not report["gp_reconstruction_executed"]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("change", ["partition", "duplicate", "reference"])
def test_training_loader_rejects_wrong_partition_or_reference(sources, change):
    path = sources["training"] / "config.json"
    config = json.loads(path.read_text())
    if change == "partition":
        config["protocol"]["scenarios"][0]["configuration"]["training_end_utc"] = (
            "2025-01-01T00:01:00+00:00"
        )
    elif change == "duplicate":
        config["calibration_seeds"] = [42, 42]
    else:
        reference = sources["training"] / "calibration/case-42/scale-2000/report.json"
        report = json.loads(reference.read_text())
        report["reconstruction_config"]["temporal_length_scale_s"] = 900
        reference.write_text(json.dumps(report))
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        experiment.load_training(sources["training"])


def test_training_decision_reports_final_regression_and_case_consistency():
    rows = [
        {
            "case_seed": seed,
            "arm": arm,
            "temporal_mse": value if arm == "original" else value / 2,
            "final_mse": value if arm == "original" else value * 2,
        }
        for seed, value in ((42, 1), (43, 9))
        for arm in experiment.ARMS
    ]
    result = experiment.training_summary(rows, [42, 43])
    assert result["lower_temporal_mse"] and not result["final_mse_not_higher"]
    assert (
        result["temporal_cases_improved"] == 2 and result["final_cases_not_worse"] == 0
    )
    assert result["summaries"][0]["temporal_rmse"] == pytest.approx(np.sqrt(5))
    with pytest.raises(ValueError):
        experiment.training_summary(rows[:-1], [42, 43])
