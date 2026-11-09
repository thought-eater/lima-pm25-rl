import socket
from dataclasses import replace

import gymnasium as gym
import networkx as nx
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import (
    GaussianKernelSource,
    MeasurementQuality,
    MeasurementRecord,
    SensorType,
    run_simulation,
)

from lima_pm25_rl import Episode
from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.policies import BaselinePolicy, vehicle_input


def test_gymnasium_contract_and_registration(env, domain, template, estimator_config):
    check_env(env, skip_render_check=True)
    wrapped = gym.make(
        "lima_pm25_rl/LimaPM25-v0",
        domain=domain,
        episode_factory=lambda seed: Episode(template),
        estimator_config=estimator_config,
        initial_nodes=((0, 0), (0, 3), (3, 0)),
    )
    observation, _ = wrapped.reset(seed=42)
    assert wrapped.observation_space.contains(observation)
    wrapped.close()


def test_three_moves_consume_one_tick_and_one_joint_mobile_batch(env):
    observation, _ = env.reset(seed=42)
    action = BaselinePolicy("greedy").act(env, observation)
    observation, reward, terminated, truncated, info = env.step(action)
    assert info["elapsed_s"] == 300
    assert info["step_count"] == 1
    assert observation["remaining"][0] == pytest.approx(2 / 3)
    assert info["invalid_actions"].sum() == 0
    assert len({env.domain.grid.xy_to_cell(*p) for p in info["positions_m"]}) == 3
    mobile = env._simulation.mobile.history.records
    assert len(mobile) == 3
    assert {record.simulation_time_s for record in mobile} == {300}
    assert env._updater.current_snapshot.sequence_number == 2
    assert reward > 0
    assert not terminated and not truncated


def test_reservations_do_not_advance_time_change_truth_or_assimilate(env):
    observation, _ = env.reset(seed=7)
    state = env._simulation.state
    count = len(env._records)
    action = np.flatnonzero(observation["action_mask"][0])[0]
    target = env.route(0, int(action)).cell
    masks = env.action_masks((target,))
    view, _ = vehicle_input(observation, 1, (target,))
    assert view[6, *target] == 1
    assert env._simulation.state is state
    assert len(env._records) == count
    assert env._simulation.state.simulation_time_s == 0
    for vehicle in range(3):
        for direction in np.flatnonzero(masks[vehicle]):
            assert env.route(vehicle, int(direction)).cell != target


def test_invalid_direction_is_not_executed_and_is_penalized(env):
    observation, _ = env.reset(seed=1)
    before = env.positions[0]
    action = BaselinePolicy("random").act(env, observation)
    assert not observation["action_mask"][0, 6]  # Oeste desde la columna cero.
    action[0] = 6
    _, reward, _, _, info = env.step(action)
    assert env.positions[0] == before
    assert info["invalid_actions"][0] == 1
    assert reward == pytest.approx(
        info["uncertainty_gain"] - info["invalid_actions"].sum()
    )


@pytest.mark.parametrize(
    "action", [np.array([0, 1]), np.array([0.0, 1.0, 2.0]), np.array([0, 1, 8])]
)
def test_malformed_action_does_not_mutate_the_episode(env, action):
    env.reset(seed=1)
    state, positions = env._simulation.state, env.positions
    with pytest.raises(ValueError):
        env.step(action)
    assert env._simulation.state is state
    assert env.positions == positions


def test_horizon_and_reset_clear_all_episode_state(env):
    observation, _ = env.reset(seed=42)
    policy = BaselinePolicy("random", seed=2)
    for _ in range(3):
        observation, _, _, truncated, info = env.step(policy.act(env, observation))
    assert truncated
    assert observation["remaining"][0] == 0
    assert info["elapsed_s"] == 900
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(3, dtype=int))
    observation, info = env.reset(seed=42)
    assert info["elapsed_s"] == 0
    assert info["distance_m"].sum() == 0
    assert len(env._simulation.mobile.history.records) == 0
    assert len(env._records) == 1
    assert observation["remaining"][0] == 1


def test_observations_and_info_are_fresh_and_never_contain_truth(env):
    observation, info = env.reset(seed=42)
    saved = {key: value.copy() for key, value in observation.items()}
    positions = info["positions_m"].copy()
    next_observation, _, _, _, next_info = env.step(
        BaselinePolicy("random").act(env, observation)
    )
    for key in observation:
        np.testing.assert_array_equal(observation[key], saved[key])
        assert not np.shares_memory(observation[key], next_observation[key])
    np.testing.assert_array_equal(info["positions_m"], positions)
    assert set(next_info) == {
        "elapsed_s",
        "positions_m",
        "distance_m",
        "integrated_variance",
        "measurement_count",
        "step_count",
        "uncertainty_gain",
        "invalid_actions",
        "held_vehicles",
        "route_distance_m",
    }


def test_random_streams_and_fields_are_isolated_between_policies_and_sensor_noise(
    make_env,
):
    first, second = make_env(), make_env(mobile_noise_stddev_ug_m3=1)
    obs_a, _ = first.reset(seed=42)
    obs_b, _ = second.reset(seed=42)
    a, b = BaselinePolicy("random", seed=1), BaselinePolicy("greedy", seed=2)
    for _ in range(3):
        obs_a, *_ = first.step(a.act(first, obs_a))
        obs_b, *_ = second.step(b.act(second, obs_b))
        np.testing.assert_array_equal(
            first._simulation.state.concentration_ug_m3,
            second._simulation.state.concentration_ug_m3,
        )
    assert first._simulation.model is not second._simulation.model
    assert {
        record.noise_stddev_ug_m3
        for record in second._simulation.mobile.history.records
    } == {1}


def test_seed_replays_observations_measurements_and_reward(env):
    outcomes = []
    for _ in range(2):
        observation, _ = env.reset(seed=42)
        policy = BaselinePolicy("random", seed=7)
        observation, reward, _, _, info = env.step(policy.act(env, observation))
        outcomes.append(
            (observation, reward, info, env._simulation.mobile.history.records)
        )
    for key in outcomes[0][0]:
        np.testing.assert_array_equal(outcomes[0][0][key], outcomes[1][0][key])
    assert outcomes[0][1] == outcomes[1][1]
    assert outcomes[0][3] == outcomes[1][3]


def test_reward_excludes_new_fixed_measurements_and_compares_the_same_time(env):
    observation, _ = env.reset(seed=42)
    policy = BaselinePolicy("greedy")
    observation, *_ = env.step(policy.act(env, observation))
    history = tuple(env._records)
    _, reward, _, _, info = env.step(policy.act(env, observation))
    time = env._simulation.clock.utc_at(600)
    new_fixed = tuple(
        r for r in env._records[len(history) :] if r.sensor_type is SensorType.FIXED
    )
    assert new_fixed
    before = GaussianProcessFieldReconstructor(
        env.estimator_config.reconstruction_config
    ).reconstruct(
        ReconstructionRequest(
            grid=env.domain.grid,
            reconstruction_timestamp_utc=time,
            measurements=(*history, *new_fixed),
        )
    )
    grid = env.domain.grid
    scale = (
        grid.active_mask.sum()
        * grid.cell_area_m2
        * env.estimator_config.reconstruction_config.prior_stddev_ug_m3**2
    )
    expected = (env._integrated_variance(before) - info["integrated_variance"]) / scale
    assert reward == pytest.approx(expected, abs=1e-9)


def test_physical_events_and_fixed_cadence_are_not_replaced_by_agent_ticks(
    make_env, template
):
    sensor = replace(
        template.sensor_configs[0], sampling_interval_s=150, first_due_time_s=0
    )
    request = replace(template, sensor_configs=(sensor,))
    env = make_env(episode_factory=lambda seed: Episode(request))
    observation, _ = env.reset(seed=42)
    env.step(BaselinePolicy("greedy").act(env, observation))
    assert {r.simulation_time_s for r in env._simulation.fixed.history.records} == {
        0,
        150,
        300,
    }
    assert {r.simulation_time_s for r in env._simulation.mobile.history.records} == {
        300
    }
    result = run_simulation(request)
    np.testing.assert_allclose(
        env._simulation.state.concentration_ug_m3, result.frames[1].concentration_ug_m3
    )


def test_replay_never_generates_missing_historical_fixed_readings(make_env, template):
    record = MeasurementRecord(
        sensor_id="OBSERVED",
        sensor_type=SensorType.FIXED,
        timestamp_utc=template.config.start_utc,
        simulation_time_s=0,
        x_m=700,
        y_m=700,
        crs=template.model.grid.crs,
        value_ug_m3=30,
        quality_status=MeasurementQuality.ACCEPTED,
        noise_stddev_ug_m3=0,
        position_source_id="fixture",
        provenance="Observación ficticia exclusivamente para la prueba de replay",
    )
    env = make_env(episode_factory=lambda seed: Episode(template, (record,)))
    observation, _ = env.reset(seed=42)
    env.step(BaselinePolicy("greedy").act(env, observation))
    fixed = [r for r in env._records if r.sensor_type is SensorType.FIXED]
    assert fixed == [record]
    assert env._simulation.fixed.history.records == ()
    metrics = evaluate_episode(env, BaselinePolicy("greedy"), seed=42)
    assert metrics["synthetic_rmse_ug_m3"] is None


@pytest.mark.parametrize("strategy", ["random", "greedy", "sweep"])
def test_baselines_use_valid_distinct_destinations_and_report_metrics(env, strategy):
    metrics = evaluate_episode(env, BaselinePolicy(strategy, seed=5), seed=42)
    assert metrics["steps"] == 3
    assert metrics["invalid_actions"] == 0
    assert metrics["distance_m"] > 0
    assert metrics["synthetic_rmse_ug_m3"] is not None


def test_render_shows_estimates_and_runs_without_network_or_domain_mutation(
    make_env, domain, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("No se debe abrir una conexión de red")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "socket", forbidden)
    original = domain.road_graph.copy()
    env = make_env(render_mode="rgb_array")
    observation, _ = env.reset(seed=42)
    env.step(BaselinePolicy("greedy").act(env, observation))
    image = env.render()
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    assert list(tmp_path.iterdir()) == []
    assert nx.utils.graphs_equal(domain.road_graph, original)
    env.close()
    env.close()


def test_irregular_source_events_match_the_core_transport(make_env, template):
    source = GaussianKernelSource(
        source_id="pulse",
        x_m=400,
        y_m=400,
        emission_rate_ug_s=1_000_000,
        sigma_m=100,
        starts_at_s=125,
        ends_at_s=175,
        provenance="Pulso sintético exclusivamente para la prueba numérica",
    )
    request = replace(template, sources=(source,))
    env = make_env(episode_factory=lambda seed: Episode(request))
    observation, _ = env.reset(seed=42)
    env.step(BaselinePolicy("greedy").act(env, observation))
    reference = run_simulation(request)
    np.testing.assert_array_equal(
        env._simulation.state.concentration_ug_m3,
        reference.frames[1].concentration_ug_m3,
    )
    assert env._simulation.state.concentration_ug_m3.max() > 8


def test_history_window_expires_records_without_exposing_future_data(
    domain, template, estimator_config
):
    from lima_pm25_rl import LimaPM25Env

    env = LimaPM25Env(
        domain=domain,
        episode_factory=lambda seed: Episode(template),
        estimator_config=replace(
            estimator_config, history_duration=estimator_config.history_duration / 72
        ),
        initial_nodes=((0, 0), (0, 3), (3, 0)),
    )
    observation, _ = env.reset(seed=42)
    policy = BaselinePolicy("greedy")
    for _ in range(2):
        observation, *_ = env.step(policy.act(env, observation))
    keys = env._posterior.observations.used_measurement_keys
    assert all(
        300 <= (time - template.config.start_utc).total_seconds() <= 600
        for _, time in keys
    )
