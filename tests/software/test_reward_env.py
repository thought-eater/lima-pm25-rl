from dataclasses import replace

import numpy as np
import pytest
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
)
from lima_pm25.simulation import MeasurementBatch, SensorType

from lima_pm25_rl import Episode
from lima_pm25_rl.policies import BaselinePolicy
from lima_pm25_rl.reward_env import error_environment, error_reward_protocol
from lima_pm25_rl.training import Trainer, TrainingConfig, load_policy


@pytest.mark.parametrize("invalid", [False, True])
def test_only_reward_changes_and_error_gain_excludes_fixed_and_aging(make_env, invalid):
    base = make_env(
        mobile_noise_stddev_ug_m3=1, concentration_priority=1, distance_cost_per_km=0.01
    )
    candidate = error_environment(base)
    observation, _ = base.reset(seed=42)
    other, _ = candidate.reset(seed=42)
    for tick in range(2):
        for key in observation:
            np.testing.assert_array_equal(observation[key], other[key])
        mask = candidate.domain.grid.active_mask
        gp = candidate.estimator_config.reconstruction_config
        mean = np.maximum(candidate._posterior.mean_ug_m3[mask], 0)
        weights = 1 + mean / (max(gp.prior_mean_ug_m3, gp.prior_stddev_ug_m3) + mean)
        history = tuple(candidate._records)
        actions = BaselinePolicy("greedy", seed=tick).act(base, observation)
        if invalid and tick == 0:
            actions[0] = 6  # Oeste desde la columna cero: penalización idéntica.
        observation, _, _, _, base_info = base.step(actions)
        other, reward, _, _, info = candidate.step(actions)
        assert set(info) == set(base_info)
        for key in info:
            np.testing.assert_allclose(
                info[key], base_info[key], rtol=1e-12, atol=1e-12
            )
        assert candidate._records == base._records
        truth = candidate._simulation.state.concentration_ug_m3[mask]
        np.testing.assert_array_equal(
            truth, base._simulation.state.concentration_ug_m3[mask]
        )
        now = candidate._simulation.clock.utc_at(info["elapsed_s"])
        fixed = tuple(
            r
            for r in candidate._records[len(history) :]
            if r.sensor_type is SensorType.FIXED
        )
        if tick == 1:
            assert fixed and any(r.sensor_type is SensorType.MOBILE for r in history)
        before = GaussianProcessFieldReconstructor(gp).reconstruct(
            ReconstructionRequest(
                grid=candidate.domain.grid,
                reconstruction_timestamp_utc=now,
                measurements=(*history, *fixed),
            )
        )
        after = candidate._posterior.mean_ug_m3[mask]
        expected = np.sum(
            weights * ((before.mean_ug_m3[mask] - truth) ** 2 - (after - truth) ** 2)
        ) / (mask.sum() * gp.prior_stddev_ug_m3**2)
        expected -= info["distance_cost"] + candidate.invalid_action_penalty * int(
            info["invalid_actions"].sum()
        )
        assert reward == pytest.approx(expected, abs=1e-10)
        assert (
            candidate._updater.current_snapshot.sequence_number
            == base._updater.current_snapshot.sequence_number
        )


def test_worse_reconstruction_has_negative_gain_even_when_variance_falls(make_env):
    env = error_environment(make_env(mobile_noise_stddev_ug_m3=1))
    env.reset(seed=42)
    [batch] = env._simulation.advance(env.positions)
    records = tuple(
        replace(r, value_ug_m3=100) if r.sensor_type is SensorType.MOBILE else r
        for r in batch.records
    )
    gain = env._assimilate(
        MeasurementBatch(batch.simulation_time_s, batch.timestamp_utc, records)
    )
    assert gain < 0
    assert env._tick_uncertainty_gain > 0
    env.reset(seed=42)
    assert env._tick_uncertainty_gain == 0


def test_fixed_only_batches_have_zero_reward_and_wrong_mobile_time_is_rejected(
    make_env,
):
    env = error_environment(make_env())
    env.reset(seed=42)
    batches = env._simulation.advance(env.positions)
    batch = batches[-1]
    early_records = tuple(
        replace(
            r, timestamp_utc=env._simulation.clock.utc_at(150), simulation_time_s=150
        )
        for r in batch.records
    )
    with pytest.raises(ValueError, match="mismo instante"):
        env._assimilate(
            MeasurementBatch(150, env._simulation.clock.utc_at(150), early_records)
        )
    # Solo fijas: no se premia ni envejecimiento ni actualización de estaciones.
    for batch in env._simulation.advance(env.positions):
        fixed = tuple(r for r in batch.records if r.sensor_type is SensorType.FIXED)
        if fixed:
            assert (
                env._assimilate(
                    MeasurementBatch(
                        batch.simulation_time_s, batch.timestamp_utc, fixed
                    )
                )
                == 0
            )


def test_historical_replay_is_rejected_before_building_the_simulation(
    make_env, template
):
    env = error_environment(
        make_env(episode_factory=lambda seed: Episode(template, historical_fixed=()))
    )
    with pytest.raises(ValueError, match="replay histórico"):
        env.reset(seed=42)
    assert env._simulation is None


def test_checkpoint_protocol_and_inference_do_not_require_truth_reward(
    make_env, tmp_path
):
    config = TrainingConfig(
        device="cpu", hidden_size=8, capacity_steps=12, batch_size=2, learning_starts=1
    )
    env = error_environment(make_env())
    with pytest.raises(ValueError, match="recompensa"):
        Trainer(env, config)
    trainer = Trainer(env, config, protocol=error_reward_protocol({}))
    trainer.train_episode()
    trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    restored = Trainer(
        error_environment(make_env()), config, protocol=error_reward_protocol({})
    )
    restored.load_checkpoint(tmp_path / "checkpoint.pt")
    assert restored.replay.episodes[0].rewards == trainer.replay.episodes[0].rewards
    assert restored.updates == trainer.updates > 0
    with pytest.raises(ValueError, match="protocolo"):
        Trainer(make_env(), config).load_checkpoint(tmp_path / "checkpoint.pt")
    experience = trainer.replay.episodes[0]
    first_tick = len(experience.decisions[0])
    if first_tick > 1:
        assert trainer.replay.transition(0)["reward"] == 0
        assert trainer.replay.transition(0)["discount"] == 1
    assert trainer.replay.transition(first_tick - 1)["reward"] == pytest.approx(
        experience.rewards[0]
    )
    assert trainer.replay.transition(first_tick - 1)["discount"] == config.gamma
    trainer.export_policy(tmp_path / "policy.pt")
    normal = make_env()
    synthetic = error_environment(make_env())
    policies = [load_policy(tmp_path / "policy.pt", device="cpu")[0] for _ in range(2)]
    observations = [e.reset(seed=99)[0] for e in (normal, synthetic)]
    for _ in range(3):
        actions = [
            p.act(e, o)
            for p, e, o in zip(policies, (normal, synthetic), observations, strict=True)
        ]
        np.testing.assert_array_equal(*actions)
        observations = [
            e.step(a)[0] for e, a in zip((normal, synthetic), actions, strict=True)
        ]
        for key in observations[0]:
            np.testing.assert_array_equal(observations[0][key], observations[1][key])
