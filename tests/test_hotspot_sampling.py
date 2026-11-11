import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from lima_pm25_rl import hotspot_sampling as experiment
from lima_pm25_rl.routing import RoadRouter
from lima_pm25_rl.simulation import Simulation


@pytest.fixture
def spec():
    value = json.loads(experiment.DEFAULT_CONFIG.read_text())
    value.update(
        grid_cells=15,
        steps=12,
        initial_common_ticks=2,
        excursion_cells=3,
        focus_sigma_m=200,
        core_radius_m=250,
        background_from_m=600,
        pulse_on_ticks=3,
        pulse_period_ticks=6,
        sensor_seeds=[0, 1],
        map_ticks=[6],
    )
    return value


def test_plans_match_road_distance_and_do_not_consult_field_parameters(spec):
    domain = experiment.build_domain(spec)
    router = RoadRouter(domain)
    plans = [experiment.prescribed_actions(spec, p) for p in experiment.PLANS]
    previews = [
        experiment.preview_plan(router, experiment.initial_nodes(spec), p)
        for p in plans
    ]
    np.testing.assert_array_equal(plans[0][:2], plans[1][:2])
    np.testing.assert_array_equal(plans[0][:, 0], plans[1][:, 0])
    np.testing.assert_allclose(
        [r["route_distance_m"] for r in previews[0]],
        [r["route_distance_m"] for r in previews[1]],
    )
    assert sum(sum(r["route_distance_m"]) for r in previews[0]) == 3 * (150 + 11 * 200)
    changed = {**spec, "peak_excess_ug_m3": 1000, "focus_sigma_m": 100}
    np.testing.assert_array_equal(
        experiment.prescribed_actions(changed, "center_and_surroundings"), plans[1]
    )
    for preview in previews:
        for frame in preview:
            cells = [domain.grid.xy_to_cell(*p) for p in frame["positions_m"]]
            assert len(set(cells)) == 3


def test_core_transport_keeps_equilibrium_and_generates_pulses_with_same_shape(spec):
    domain = experiment.build_domain(spec)
    results = {}
    for regime in experiment.REGIMES:
        episode = experiment.build_episode(domain, spec, regime, 0)
        positions = tuple(
            domain.grid.cell_center(*c) for c in experiment.initial_nodes(spec)
        )
        simulation = Simulation(
            episode, seed=0, positions=positions, noise_stddev_ug_m3=1
        )
        initial = simulation.state.concentration_ug_m3.copy()
        simulation.initial_batch()
        amplitude = spec["peak_excess_ug_m3"]
        background = spec["background_ug_m3"]
        shape = (initial - background) / amplitude
        decay = np.exp(-300 / spec["relaxation_time_s"])
        fields = [initial]
        for tick in range(1, spec["steps"] + 1):
            on = (
                regime == "persistent"
                or (tick - 1) % spec["pulse_period_ticks"] < spec["pulse_on_ticks"]
            )
            amplitude = amplitude * decay + spec["peak_excess_ug_m3"] * (1 - decay) * on
            simulation.advance(positions)
            actual = simulation.state.concentration_ug_m3
            np.testing.assert_allclose(
                actual, background + amplitude * shape, rtol=0, atol=1e-10
            )
            fields.append(actual.copy())
        results[regime] = np.stack(fields)
    np.testing.assert_allclose(
        results["persistent"],
        np.repeat(results["persistent"][:1], 13, axis=0),
        rtol=0,
        atol=1e-10,
    )
    assert np.max(np.ptp(results["pulsed"], axis=0)) > 30
    np.testing.assert_array_equal(results["persistent"][0], results["pulsed"][0])


def test_metrics_expose_improved_focus_and_false_exterior_at_same_time():
    regions = {
        "core": np.array([True, False, False]),
        "surroundings": np.array([False, True, False]),
        "background": np.array([False, False, True]),
    }
    truth = np.array([80, 30, 20])
    a = experiment.frame_metrics(
        truth, np.array([20, 20, 20]), np.ones(3), regions, 40, 40000
    )
    b = experiment.frame_metrics(
        truth, np.array([80, 30, 50]), np.ones(3), regions, 40, 40000
    )
    assert b["core_mse"] < a["core_mse"] and b["background_mse"] > a["background_mse"]
    assert b["false_high_area_km2"] == pytest.approx(0.04)
    assert b["detected_focus"] and not a["detected_focus"]


def test_small_experiment_pairs_fields_noise_budget_and_keeps_covariance_fixed(
    spec, estimator_config, tmp_path
):
    result = experiment.run_experiment(
        spec, estimator_config, tmp_path / "run", maps=True
    )
    assert len(result["paired"]) == 4 and all(r["paired"] for r in result["paired"])
    assert result["max_std_difference_between_regimes"] == pytest.approx(0, abs=1e-10)
    previews = json.loads((tmp_path / "run/route-preview.json").read_text())
    for regime in experiment.REGIMES:
        for plan in experiment.PLANS:
            folder = tmp_path / "run" / regime / "noise-0" / plan
            report = json.loads((folder / "report.json").read_text())
            assert len(report["ticks"]) == 12
            assert report["summary"]["mobile_records"] == 36
            assert report["summary"]["distance_m"] == 7050
            np.testing.assert_allclose(
                [r["positions_m"] for r in report["ticks"]],
                [r["positions_m"] for r in previews[plan]],
            )
            assert (folder / "tick-06.png").is_file()
            with np.load(folder / "fields-private.npz") as fields:
                assert fields["truth"].shape == (13, 225)
                assert np.all(
                    fields["region_core"]
                    | fields["region_surroundings"]
                    | fields["region_background"]
                )
    with pytest.raises(FileExistsError):
        experiment.run_experiment(spec, estimator_config, tmp_path / "run", maps=False)


def test_check_has_no_simulation_or_gp_updates(
    spec, estimator_config, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec))
    gp = replace(
        estimator_config,
        reconstruction_config=replace(
            estimator_config.reconstruction_config,
            x_length_scale_m=2000,
            y_length_scale_m=2000,
        ),
    )
    monkeypatch.setattr(experiment, "load_gp", lambda root: gp)
    monkeypatch.setattr(
        experiment, "build_episode", lambda *a: pytest.fail("No generar episodio")
    )
    monkeypatch.setattr(
        experiment, "run_experiment", lambda *a: pytest.fail("No simular")
    )
    monkeypatch.setattr(sys, "argv", ["hotspot", "--config", str(path), "--check"])
    experiment.main()
    report = json.loads(capsys.readouterr().out)
    assert report["simulation_executed"] is False
    assert report["missions"] == 8 and report["road_distance_per_mission_m"] == 7050
