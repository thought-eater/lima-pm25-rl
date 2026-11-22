import io
import json
import tarfile
from dataclasses import asdict, replace
from datetime import timedelta

import numpy as np
import pytest

from lima_pm25_rl.comparison import run_comparison
from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.experiments.gp_diagnostics import diagnose_case
from lima_pm25_rl.experiments.gp_spatial_experiment import load_frozen_case
from lima_pm25_rl.hotspot_evaluation import (
    evaluate_saved,
    freeze_protocol,
    measurement_digest,
    parse_measurements,
    reconstruct_mission,
    route_progress,
    source_files,
    validate_fixed_reference,
)
from lima_pm25_rl.policies import BaselinePolicy


def test_archive_reader_does_not_extract_or_read_other_members(tmp_path):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as file:
        for name, data in (("root/config.json", b"{}"), ("../unwanted.txt", b"no")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            file.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="carpeta"):
        source_files(archive, {"config.json"})
    assert not (tmp_path / "unwanted.txt").exists()


def test_freeze_uses_only_prefixed_training_cases_and_excludes_t0(domain):
    truth = np.array([[1000.0, 1000.0], [1.0, 3.0], [5.0, 7.0]])
    frozen = ({"signatures": {"field_sha256": "test"}}, {"truth": truth}, (), None)
    result = freeze_protocol(
        {42: frozen},
        quantile=0.5,
        expected_seeds=[42],
        source="training",
        grid=domain.grid,
        protocol={},
    )
    assert result["threshold_ug_m3"] == 4
    assert result["health_threshold"] is False
    with pytest.raises(ValueError):
        freeze_protocol(
            {42: frozen},
            quantile=0.5,
            expected_seeds=[43],
            source="training",
            grid=domain.grid,
            protocol={},
        )


def test_route_progress_follows_polyline_not_grid_displacement():
    from types import SimpleNamespace

    records = [
        SimpleNamespace(sensor_id="MOBILE_0", simulation_time_s=300.0, x_m=1.0, y_m=1.0)
    ]
    routes = {0: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]}
    np.testing.assert_array_equal(
        route_progress(routes, records, [0, 300, 600], 2), [0, 2, 2]
    )
    with pytest.raises(ValueError, match="coste guardado"):
        route_progress(routes, records, [0, 300], 3)


def make_frozen(env, tmp_path, seed=42):
    report = diagnose_case(
        env,
        BaselinePolicy("random", seed=7),
        seed=seed,
        gamma=0.99,
        output=tmp_path / "frozen",
    )
    origin = env._simulation.request.config.start_utc
    frozen = load_frozen_case(
        tmp_path / "frozen",
        env.domain.grid,
        start_utc=origin,
        end_utc=origin + timedelta(days=1),
    )
    return report, frozen


def test_reconstruction_reproduces_global_metrics_and_is_causal(make_env, tmp_path):
    env = make_env()
    report, frozen = make_frozen(env, tmp_path)
    old = evaluate_episode(
        env, BaselinePolicy("random", seed=7), seed=42, comparison=True
    )
    records = tuple(env._records)
    _rows, mean, std, summary = reconstruct_mission(
        env.domain.grid,
        env.estimator_config,
        frozen,
        records,
        old,
        threshold=20.0,
        routes=dict(enumerate(report["routes_m"])),
    )
    assert summary["reproduction_passed"]
    np.testing.assert_allclose(mean, frozen[1]["mean_after"], atol=1e-8)
    np.testing.assert_allclose(std, frozen[1]["std_after"], atol=1e-8)
    # Cambiar una lectura final no puede alterar los posteriores anteriores.
    from dataclasses import replace

    altered = (
        *records[:-1],
        replace(records[-1], value_ug_m3=records[-1].value_ug_m3 + 5),
    )
    from lima_pm25.estimation import (
        GaussianProcessFieldReconstructor,
        ReconstructionRequest,
    )

    now = frozen[3]
    model = GaussianProcessFieldReconstructor(
        env.estimator_config.reconstruction_config
    )
    first = model.reconstruct(
        ReconstructionRequest(
            grid=env.domain.grid,
            reconstruction_timestamp_utc=now,
            measurements=tuple(r for r in altered if r.timestamp_utc <= now),
        )
    )
    np.testing.assert_allclose(first.mean_ug_m3[env.domain.grid.active_mask], mean[0])
    bad = {**old, "synthetic_mse": old["synthetic_mse"] + 1}
    with pytest.raises(ValueError, match="no reproduce"):
        reconstruct_mission(
            env.domain.grid,
            env.estimator_config,
            frozen,
            records,
            bad,
            threshold=20.0,
            routes=dict(enumerate(report["routes_m"])),
        )


def test_measurement_csv_roundtrip_preserves_fixed_hash_and_rejects_duplicates(
    make_env,
):
    env = make_env()
    evaluate_episode(env, BaselinePolicy("random"), seed=42, comparison=True)
    rows = [
        {
            key: str(value) if value is not None else ""
            for key, value in asdict(r).items()
        }
        for r in env._records
    ]
    records = parse_measurements(rows)
    assert measurement_digest(records) == measurement_digest(tuple(env._records))
    with pytest.raises(ValueError, match="duplicadas"):
        parse_measurements([rows[0], rows[0]])


def test_fixed_reference_tolerates_coordinate_roundoff_only(make_env):
    env = make_env()
    env.reset(seed=42)
    fixed = tuple(r for r in env._records if r.sensor_type == "fixed")
    assert fixed
    rounded = (replace(fixed[0], x_m=fixed[0].x_m + 1e-10), *fixed[1:])
    validate_fixed_reference(rounded, fixed)
    for changed in (
        replace(fixed[0], x_m=fixed[0].x_m + 0.001),
        replace(fixed[0], value_ug_m3=fixed[0].value_ug_m3 + 1e-6),
        replace(fixed[0], noise_stddev_ug_m3=fixed[0].noise_stddev_ug_m3 + 0.01),
        replace(fixed[0], timestamp_utc=fixed[0].timestamp_utc + timedelta(seconds=1)),
    ):
        with pytest.raises(ValueError, match="fijas"):
            validate_fixed_reference((changed, *fixed[1:]), fixed)


@pytest.mark.parametrize("smoke", [True, False])
def test_saved_evaluation_end_to_end_including_fixed_only(make_env, tmp_path, smoke):
    env = make_env()
    _, frozen = make_frozen(env, tmp_path)
    source = tmp_path / "evaluation"
    original_summary = run_comparison(
        env,
        BaselinePolicy("random", seed=7),
        case_seeds=[42],
        policy_seeds=[0],
        output=source,
        metadata={"protocol": {}},
        maps=False,
    )
    output = tmp_path / "result"
    results = evaluate_saved(
        source,
        {42: frozen},
        env.domain.grid,
        env.estimator_config,
        {"protocol": {}, "threshold_ug_m3": 20.0},
        output,
        smoke=smoke,
        map_ticks=[1, 3],
    )
    assert {r["strategy"] for r in results} == (
        {"fixed", "neural"}
        if smoke
        else {"fixed", "neural", "random", "greedy", "sweep"}
    )
    assert all(r["reproduction_passed"] for r in results)
    assert (output / "curves.png").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["smoke_only"] == smoke
    assert (output / "references.csv").is_file()
    for row in summary["neural_vs_references"]:
        expected = next(
            r
            for r in original_summary["neural_vs_references"]
            if r["reference"] == row["reference"]
        )
        for key in ("rmse_reduction_percent", "variance_reduction_percent"):
            assert row[key] == pytest.approx(expected[key], abs=1e-8)
