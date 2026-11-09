import json
from dataclasses import replace

import numpy as np
import pytest
from lima_pm25.domain import AmbientBaseline
from lima_pm25.simulation import GaussianKernelSource

from lima_pm25_rl.diagnostics import (
    field_diagnostics,
    run_diagnostics,
    simulate_diagnostic,
)
from lima_pm25_rl.episodes import Episode
from lima_pm25_rl.policies import BaselinePolicy


def generated(template, *, value=8.0, partition="training"):
    initial = template.model.initialize_uniform(
        AmbientBaseline("test", value, "Prueba sintética")
    )
    return Episode(
        replace(
            template,
            initial_state=initial,
            episode_parameters_json=json.dumps(
                {"partition": partition, "weather_mode": "synthetic"}
            ),
        )
    )


def test_uniform_but_changing_field_is_not_spatial_contrast():
    fields = np.array([np.full((2, 2), value) for value in (8, 10, 12)])
    active = np.ones((2, 2), dtype=bool)
    ticks, summary, temporal, changes = field_diagnostics(fields, active, active)
    assert all(row["active_p99_minus_p50"] == 0 for row in ticks)
    assert summary["mean_spatial_contrast_ug_m3"] == 0
    np.testing.assert_allclose(temporal, np.std([8, 10, 12]))
    np.testing.assert_allclose(changes, 2)
    assert summary["initial_field_persistence_rmse_ug_m3"] == pytest.approx(np.sqrt(10))


def test_stationary_plume_has_spatial_contrast_but_no_temporal_change():
    field = np.array([[8, 8], [8, 40]])
    active = np.ones((2, 2), dtype=bool)
    roads = np.array([[True, True], [True, False]])
    ticks, summary, temporal, changes = field_diagnostics(
        np.stack([field] * 3), active, roads
    )
    assert summary["mean_spatial_contrast_ug_m3"] > 20
    assert all(row["road_support_p99_minus_p50"] == 0 for row in ticks)
    assert summary["mean_temporal_std_ug_m3"] == 0
    assert summary["initial_field_persistence_rmse_ug_m3"] == 0
    np.testing.assert_array_equal(temporal, 0)
    np.testing.assert_array_equal(changes, 0)


def test_inactive_cells_do_not_pollute_metrics_and_empty_roads_are_explicit():
    fields = np.array([[[8, np.nan]], [[9, np.inf]]])
    active = np.array([[True, False]])
    roads = np.zeros_like(active)
    ticks, summary, temporal, _ = field_diagnostics(fields, active, roads)
    assert summary["active_cells"] == 1 and summary["road_support_cells"] == 0
    assert ticks[0]["road_support_mean"] is None
    assert np.isnan(temporal[0, 1])
    fields[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finito"):
        field_diagnostics(fields, active, roads)


def test_core_diagnostic_preserves_rl_field_and_event_schedule(make_env, template):
    pulse = GaussianKernelSource(
        source_id="pulse",
        x_m=400,
        y_m=400,
        emission_rate_ug_s=1_000_000,
        sigma_m=100,
        starts_at_s=125,
        ends_at_s=175,
        provenance="Pulso de prueba",
    )
    episode = generated(replace(template, sources=(pulse,)))
    fields, records, _ = simulate_diagnostic(episode)
    env = make_env(episode_factory=lambda seed: episode)
    observation, _ = env.reset(seed=1)
    policy = BaselinePolicy("random", seed=2)
    np.testing.assert_array_equal(fields[0], env._simulation.state.concentration_ug_m3)
    for tick in range(1, 4):
        observation, *_ = env.step(policy.act(env, observation))
        np.testing.assert_allclose(
            fields[tick], env._simulation.state.concentration_ug_m3
        )
    assert fields[-1].max() > fields[0].max()
    assert all(record.sensor_type == "fixed" for record in records)


@pytest.mark.parametrize("partition", ["evaluation", "replay"])
def test_non_training_cases_are_rejected_before_writing(
    make_env, template, tmp_path, partition
):
    episode = generated(template, partition=partition)
    env = make_env(episode_factory=lambda seed: episode)
    with pytest.raises(ValueError, match="entrenamiento"):
        run_diagnostics(
            env, seeds=[1, 2], fit_cases=1, output=tmp_path / "audit", metadata={}
        )
    assert not (tmp_path / "audit").exists()
    with pytest.raises(ValueError, match="entrenamiento"):
        simulate_diagnostic(episode)


def test_constant_is_fitted_only_on_separate_cases_and_does_not_initialize_agent(
    make_env,
    template,
    tmp_path,
):
    env = make_env(
        episode_factory=lambda seed: generated(template, value=8 if seed == 1 else 20)
    )
    output = tmp_path / "audit"
    report = run_diagnostics(env, seeds=[1, 2], fit_cases=1, output=output, metadata={})
    assert report["constant_value_ug_m3"] == 8
    assert report["constant_diagnostic_rmse_ug_m3"] == pytest.approx(12)
    assert env._simulation is None
    assert not hasattr(env, "_updater")
    assert (output / "summary.json").is_file()
    assert (output / "ticks.csv").is_file()
    with np.load(output / "fields/case-01.npz") as fields:
        np.testing.assert_array_equal(fields["concentration_ug_m3"], 20)
        np.testing.assert_array_equal(fields["elapsed_s"], [0, 300, 600, 900])
    with pytest.raises(FileExistsError):
        run_diagnostics(env, seeds=[1, 2], fit_cases=1, output=output, metadata={})


def test_cli_uses_first_training_seeds_without_creating_a_trainer(
    make_env,
    template,
    tmp_path,
    monkeypatch,
    capsys,
):
    from lima_pm25_rl import diagnose_cli

    env = make_env(episode_factory=lambda seed: generated(template))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"kind": "generated", "steps_per_episode": 3, "training": {"seed": 42}}
        )
    )
    monkeypatch.setattr(
        diagnose_cli, "prepare_experiment", lambda *args: (env, None, {})
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "lima-pm25-diagnose",
            "--config",
            str(config),
            "--cases",
            "2",
            "--output",
            str(tmp_path / "output"),
        ],
    )
    diagnose_cli.main()
    payload = json.loads((tmp_path / "output/config.json").read_text())
    expected = [
        int(np.random.SeedSequence([42, 0, index]).generate_state(1)[0])
        for index in range(2)
    ]
    assert payload["seeds"] == expected
    assert payload["training_executed"] is False
    assert "Diagnóstico CPU" in capsys.readouterr().out
