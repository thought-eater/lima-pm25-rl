import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import torch

from lima_pm25_rl.pilot import controlled_environment, controlled_oracle
from lima_pm25_rl.policies import PlanningDecision
from lima_pm25_rl.replay import EpisodeExperience, SequenceReplay
from lima_pm25_rl.train_cli import prepare_experiment, run_training
from lima_pm25_rl.training import Trainer, TrainingConfig, load_policy


def config(**kwargs):
    return replace(
        TrainingConfig(
            device="cpu",
            hidden_size=8,
            history_length=2,
            capacity_steps=6,
            batch_size=2,
            learning_starts=1,
            target_update_interval=2,
            epsilon_start=0.6,
            epsilon_end=0.2,
            epsilon_decay_steps=20,
        ),
        **kwargs,
    )


def fake_episode(offset=0):
    observations = []
    for time in range(3):
        maps = np.zeros((7, 2, 2), dtype=np.float32)
        maps[0:2] = 1
        maps[2] = time + offset
        for vehicle, cell in enumerate(((0, 0), (0, 1), (1, 0))):
            maps[4 + vehicle, *cell] = 1
        observations.append(
            {
                "maps": maps,
                "vehicles": np.zeros((3, 2), dtype=np.float32),
                "remaining": np.array([1 - time / 2], dtype=np.float32),
                "action_mask": np.ones((3, 8), dtype=np.int8),
            }
        )
    choices = tuple(
        PlanningDecision(
            i,
            2,
            np.array([[int(v >= i)] * 8 for v in range(3)], dtype=np.int8),
            tuple(((0, 0), (1, 1))[:i]),
            tuple(range(i, 3)),
            ((0.0, 0.0),) * 3,
        )
        for i in range(3)
    )
    return EpisodeExperience(
        observations, [choices, choices], [0.1, 0.2], [False, True]
    )


def test_planning_rewards_and_discounts_do_not_triple_time_or_reward():
    replay = SequenceReplay(4, 4, 0.9, seed=1)
    replay.add(fake_episode())
    transitions = [replay.transition(i) for i in range(6)]
    assert [item["reward"] for item in transitions] == [0, 0, 0.1, 0, 0, 0.2]
    assert [item["discount"] for item in transitions] == [1, 1, 0.9, 1, 1, 0.9]
    assert [item["done"] for item in transitions] == [False] * 5 + [True]
    assert sum(item["reward"] for item in transitions) == pytest.approx(0.3)


def test_sequences_are_causal_and_preserve_the_next_reservation_context():
    replay = SequenceReplay(4, 4, 0.9, seed=1)
    replay.add(fake_episode())
    first = replay.transition(0)
    assert np.all(first["observations"][0][:, 2] == 0)
    assert np.all(first["next_observations"][0][:, :, 2] == 0)
    assert first["next_observations"][0][1, -1, 6, 0, 0] == 1
    assert np.all(first["next_observations"][0][:, :-1, 6] == 0)
    end_tick = replay.transition(2)
    np.testing.assert_array_equal(
        end_tick["next_observations"][0][0, :, 2, 0, 0], [0, 0, 0, 1]
    )
    assert end_tick["next_mask"].any()
    assert not replay.transition(5)["next_mask"].any()


def test_replay_evicts_whole_episodes_and_never_joins_histories():
    replay = SequenceReplay(3, 4, 0.9, seed=1)
    replay.add(fake_episode())
    replay.add(fake_episode(100))
    assert replay.steps == 2
    assert len(replay) == 6 and len(replay.episodes) == 1
    assert np.all(replay.transition(0)["observations"][0][:, 2] == 100)
    assert replay.nbytes > 0


def test_replay_checkpoint_restores_sampling_and_uses_weights_only(tmp_path):
    replay = SequenceReplay(4, 2, 0.9, seed=1)
    replay.add(fake_episode())
    replay.add(fake_episode(100))
    path = tmp_path / "replay.pt"
    torch.save(replay.state_dict(), path)
    restored = SequenceReplay(4, 2, 0.9, seed=99)
    restored.load_state_dict(torch.load(path, weights_only=True))
    first = replay.sample(4, torch.device("cpu"))
    second = restored.sample(4, torch.device("cpu"))
    for name in ("actions", "rewards", "discounts", "next_masks", "done"):
        assert torch.equal(first[name], second[name])
    assert torch.equal(first["observations"][0], second["observations"][0])


def test_three_decisions_are_recorded_before_a_single_environment_transition(env):
    trainer = Trainer(env, config())
    row = trainer.train_episode()
    assert row["episode_steps"] == 3 and row["physical_steps"] == 3
    assert row["planning_decisions"] == 9
    assert len(trainer.replay) == 9
    assert row["optimizer_updates"] == 3 and row["target_updates"] == 1
    assert row["invalid_actions"] == 0
    assert all(parameter.grad is None for parameter in trainer.target.parameters())


def test_forced_holds_are_not_learned_as_actions_and_keep_the_joint_reward():
    trainer = Trainer(controlled_environment(), config())
    row = trainer.train_episode()
    assert row["planning_decisions"] == 1
    assert len(trainer.replay) == 1
    transition = trainer.replay.transition(0)
    assert transition["reward"] == pytest.approx(row["reward"])
    assert transition["done"]
    assert row["optimizer_updates"] == 0
    oracle = controlled_oracle()
    assert oracle[6] > oracle[2]


def test_epsilon_decays_by_physical_ticks_not_vehicle_count(env):
    trainer = Trainer(
        env, config(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=10)
    )
    trainer.train_episode()
    assert trainer.epsilon == pytest.approx(0.7)


def test_warmup_does_not_update_network_weights(env):
    trainer = Trainer(env, config(learning_starts=100))
    before = {key: value.clone() for key, value in trainer.online.state_dict().items()}
    row = trainer.train_episode()
    assert row["loss"] is None and trainer.updates == 0
    assert all(
        torch.equal(value, trainer.online.state_dict()[key])
        for key, value in before.items()
    )


def test_resume_reproduces_continuous_cpu_training_at_episode_boundaries(
    make_env, tmp_path
):
    parameters = config()
    continuous = Trainer(make_env(), parameters)
    for _ in range(4):
        continuous.train_episode()
    split = Trainer(make_env(), parameters)
    for _ in range(2):
        split.train_episode()
    split.save_checkpoint(tmp_path / "checkpoint.pt")
    resumed = Trainer(make_env(), parameters)
    resumed.load_checkpoint(tmp_path / "checkpoint.pt")
    for _ in range(2):
        resumed.train_episode()
    assert resumed.physical_steps == continuous.physical_steps == 12
    assert resumed.updates == continuous.updates
    assert resumed.target_updates == continuous.target_updates
    assert (
        resumed.controller.rng.bit_generator.state
        == continuous.controller.rng.bit_generator.state
    )
    assert (
        resumed.replay.rng.bit_generator.state
        == continuous.replay.rng.bit_generator.state
    )
    for key, value in continuous.online.state_dict().items():
        assert torch.equal(value, resumed.online.state_dict()[key])
    for key, value in continuous.target.state_dict().items():
        assert torch.equal(value, resumed.target.state_dict()[key])


def test_checkpoint_rejects_different_training_protocol_and_exports_a_loadable_policy(
    make_env, tmp_path
):
    trainer = Trainer(make_env(), config(), protocol={"test": 1})
    trainer.train_episode()
    trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    other = Trainer(make_env(), config(), protocol={"test": 2})
    with pytest.raises(ValueError, match="protocolo"):
        other.load_checkpoint(tmp_path / "checkpoint.pt")
    trainer.export_policy(tmp_path / "policy.pt")
    policy, protocol = load_policy(tmp_path / "policy.pt", device="cpu")
    assert protocol == {"test": 1}
    observation, _ = other.env.reset(seed=42)
    action = policy.act(other.env, observation)
    _, _, _, _, info = other.env.step(action)
    assert info["invalid_actions"].sum() == 0
    with pytest.raises(ValueError, match="checkpoint de entrenamiento"):
        trainer.load_checkpoint(tmp_path / "policy.pt")


def test_validation_does_not_train_add_replay_data_or_change_exploration(make_env):
    trainer = Trainer(make_env(), config())
    trainer.train_episode()
    weights = {key: value.clone() for key, value in trainer.online.state_dict().items()}
    rng = trainer.controller.rng.bit_generator.state
    metrics = trainer.validate(make_env(), [100, 101])
    assert metrics["invalid_actions"] == 0
    assert trainer.updates == 3 and trainer.replay.steps == 3
    assert trainer.controller.rng.bit_generator.state == rng
    assert all(
        torch.equal(value, trainer.online.state_dict()[key])
        for key, value in weights.items()
    )


def test_runner_writes_metrics_checkpoints_and_never_overwrites_an_existing_run(
    env, make_env, tmp_path
):
    trainer = Trainer(env, config())
    output = tmp_path / "run"
    run_training(
        trainer,
        episodes=1,
        output=output,
        validation_env=make_env(),
        validation_seeds=[100],
        validation_every=1,
    )
    assert (output / "metrics.csv").read_text().count("\n") == 2
    assert (output / "config.json").is_file()
    assert (output / "checkpoint-last.pt").is_file()
    assert (output / "policy-last.pt").is_file()
    assert (output / "policy-best.pt").is_file()
    initial = torch.load(output / "policy-initial.pt", weights_only=True)
    assert initial["completed_episodes"] == 0
    assert initial["physical_steps"] == 0
    assert (output / "validation-initial.json").is_file()
    saved = (output / "metrics.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_training(trainer, episodes=2, output=output)
    assert (output / "metrics.csv").read_bytes() == saved


def test_validation_uses_common_horizon_after_early_termination(make_env, monkeypatch):
    trainer = Trainer(make_env(), config())
    validation = make_env()
    original = validation.action_masks

    def masks(reserved=()):
        return (
            np.zeros((3, 8), dtype=np.int8)
            if validation._steps >= 1
            else original(reserved)
        )

    monkeypatch.setattr(validation, "action_masks", masks)
    result = trainer.validate(validation, [100])
    assert validation._steps == 1
    assert validation._simulation.state.simulation_time_s == 900
    assert (
        result["final_integrated_variance"] == validation._info()["integrated_variance"]
    )
    assert trainer.updates == 0 and trainer.replay.steps == 0


@pytest.mark.parametrize(
    "parameters",
    [
        {"history_length": 0},
        {"gamma": 1.5},
        {"epsilon_start": 0.1, "epsilon_end": 0.9},
        {"learning_rate": -1},
        {"capacity_steps": 1, "batch_size": 2},
        {"seed": -1},
    ],
)
def test_invalid_training_configuration_is_rejected(parameters):
    with pytest.raises(ValueError):
        TrainingConfig(**parameters)


def test_controlled_configuration_is_explicitly_not_a_lima_experiment(tmp_path):
    env, validation, protocol = prepare_experiment(
        {"kind": "controlled", "steps_per_episode": 1}, tmp_path
    )
    assert protocol["kind"] == "controlled" and validation is None
    assert env.domain.grid.shape == (5, 5)


@pytest.fixture
def experiment_inputs(
    monkeypatch,
    tmp_path,
    domain,
    template,
    estimator_config,
    historical_weather,
    generation_config,
):
    import lima_pm25.simulation as core_simulation

    import lima_pm25_rl.train_cli as entry

    for name in ("nominal.json", "stress.json"):
        (tmp_path / name).write_text(json.dumps({"fixture": name}))
    monkeypatch.setattr(
        entry,
        "load_study_resources",
        lambda *args, **kwargs: (domain, template, estimator_config),
    )
    monkeypatch.setattr(entry, "_sha256", lambda path: "unit-fixture")
    monkeypatch.setattr(
        core_simulation,
        "load_historical_weather_csv",
        lambda *args, **kwargs: historical_weather,
    )
    monkeypatch.setattr(
        core_simulation,
        "load_episode_generation_config",
        lambda path: generation_config,
    )
    spec = {
        "kind": "generated",
        "core_root": "fixture",
        "steps_per_episode": 1,
        "gp_fit_end_utc": generation_config.training_end_utc.isoformat(),
        "reserved_final_start_utc": generation_config.evaluation_end_utc.isoformat(),
        "validation_seeds": [100, 101],
        "episode_configs": [
            {"path": "nominal.json", "weight": 0.8, "member": "control"},
            {"path": "stress.json", "weight": 0.2, "member": "event"},
        ],
    }
    return spec, tmp_path


def test_generated_training_uses_separate_validation_and_reserves_final_replay(
    experiment_inputs,
):
    spec, directory = experiment_inputs
    env, validation, protocol = prepare_experiment(spec, directory)
    trainer = Trainer(env, config(), protocol=protocol)
    trainer.train_episode()
    assert (
        json.loads(env._simulation.request.episode_parameters_json)["partition"]
        == "training"
    )
    trainer.validate(validation, spec["validation_seeds"])
    assert (
        json.loads(validation._simulation.request.episode_parameters_json)["partition"]
        == "evaluation"
    )
    assert protocol["policy_fit_end_utc"] == protocol["reserved_final_start_utc"]
    assert [item["member"] for item in protocol["scenarios"]] == ["control", "event"]


def test_field_variation_is_in_protocol_and_short_runs_preserve_initial_field(
    experiment_inputs,
):
    spec, directory = experiment_inputs
    spec["field_variation"] = {
        "ambient_range_ug_m3": [13.6, 25.6],
        "road_source_scale": 20,
        "initial_plume_count": 2,
        "initial_plume_amplitude_range_ug_m3": [5, 15],
        "initial_plume_sigma_range_m": [100, 200],
        "provenance": "Prueba sintética de integración",
    }
    env, _, protocol = prepare_experiment(spec, directory)
    short = env._episode_factory(42).request
    spec["steps_per_episode"] = 2
    longer_env, _, longer_protocol = prepare_experiment(spec, directory)
    longer = longer_env._episode_factory(42).request
    np.testing.assert_array_equal(
        short.initial_state.concentration_ug_m3,
        longer.initial_state.concentration_ug_m3,
    )
    assert short.config.duration_s == 300 and longer.config.duration_s == 600
    assert protocol["field_variation"] == longer_protocol["field_variation"]
    assert protocol["field_variation"]["road_source_scale"] == 20


def test_experiment_rejects_final_replay_overlap(experiment_inputs):
    spec, directory = experiment_inputs
    spec["reserved_final_start_utc"] = spec["gp_fit_end_utc"]
    with pytest.raises(ValueError, match="replay final"):
        prepare_experiment(spec, directory)


def test_mixture_cannot_fit_weather_on_another_members_validation_period(
    experiment_inputs, monkeypatch, generation_config
):
    import lima_pm25.simulation as core_simulation

    spec, directory = experiment_inputs
    changed = replace(
        generation_config,
        evaluation_start_utc=generation_config.evaluation_start_utc
        + timedelta(hours=1),
    )
    monkeypatch.setattr(
        core_simulation,
        "load_episode_generation_config",
        lambda path: changed if path.name == "stress.json" else generation_config,
    )
    with pytest.raises(ValueError, match="compartir las particiones"):
        prepare_experiment(spec, directory)


def test_check_composes_all_scenario_weather_partition_combinations(
    experiment_inputs, monkeypatch, template
):
    from lima_pm25_rl.episodes import Episode, GeneratedEpisodes

    seen = []

    def compose(factory, seed):
        seen.append((factory.partition, factory.weather_mode))
        return Episode(template)

    monkeypatch.setattr(GeneratedEpisodes, "__call__", compose)
    spec, directory = experiment_inputs
    prepare_experiment(spec, directory, check=True)
    assert len(seen) == 12
    assert all(
        seen.count((partition, mode)) == 2
        for partition in ("training", "evaluation")
        for mode in ("historical", "historical_perturbed", "synthetic")
    )


@pytest.mark.parametrize("problem", ["cutoff", "weather", "final_end"])
def test_experiment_checks_the_declared_gp_partition(
    experiment_inputs, problem, historical_weather
):
    spec, directory = experiment_inputs
    spec["gp_partition_manifest"] = "partition.json"
    spec["reserved_final_end_utc"] = "2025-01-06T00:00:00+00:00"
    manifest = {
        "calibration_end_utc": spec["gp_fit_end_utc"],
        "weather_resource_sha256": historical_weather.resource_sha256,
        "replay_start_utc": spec["gp_fit_end_utc"],
        "replay_end_utc": spec["reserved_final_end_utc"],
    }
    if problem == "cutoff":
        manifest["calibration_end_utc"] = "2025-01-01T00:00:00+00:00"
    elif problem == "weather":
        manifest["weather_resource_sha256"] = "different-weather"
    else:
        manifest["replay_end_utc"] = spec["reserved_final_start_utc"]
    root = directory / spec["core_root"]
    root.mkdir()
    (root / "partition.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifiesto|reserva histórica"):
        prepare_experiment(spec, directory)


def test_experiment_records_the_final_upper_bound(experiment_inputs):
    spec, directory = experiment_inputs
    spec["reserved_final_end_utc"] = "2025-01-06T00:00:00+00:00"
    _, _, protocol = prepare_experiment(spec, directory)
    assert protocol["reserved_final_end_utc"] == spec["reserved_final_end_utc"]
    spec["reserved_final_end_utc"] = spec["reserved_final_start_utc"]
    with pytest.raises(ValueError, match="duración positiva"):
        prepare_experiment(spec, directory)


def test_loaded_policy_can_be_used_by_the_existing_cli(
    make_env, domain, template, estimator_config, tmp_path, monkeypatch, capsys
):
    import sys

    import lima_pm25_rl.cli as entry

    protocol = {
        "grid_shape": list(domain.grid.shape),
        "gp_profile_sha256": estimator_config.profile_sha256,
    }
    trainer = Trainer(make_env(), config(), protocol=protocol)
    trainer.train_episode()
    path = tmp_path / "policy.pt"
    trainer.export_policy(path)
    monkeypatch.setattr(
        entry,
        "load_study_resources",
        lambda *args, **kwargs: (domain, template, estimator_config),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lima-pm25-rl",
            "--policy",
            "neural",
            "--checkpoint",
            str(path),
            "--steps",
            "1",
        ],
    )
    entry.main()
    report = json.loads(capsys.readouterr().out)
    assert report["steps"] == 1 and report["invalid_actions"] == 0


@pytest.mark.parametrize(
    "extra_arguments",
    [
        ["--policy-fit-end", "2026-02-01T00:00:00+00:00"],
        ["--gp-fit-end", "2025-01-01T00:00:00+00:00"],
        [],
    ],
)
def test_replay_cli_respects_the_fitting_and_reserved_period_in_the_policy(
    make_env, tmp_path, monkeypatch, extra_arguments
):
    import sys

    import lima_pm25_rl.cli as entry

    protocol = {
        "kind": "generated",
        "gp_fit_end_utc": "2026-01-01T00:00:00+00:00",
        "policy_fit_end_utc": "2026-04-01T00:00:00+00:00",
        "reserved_final_start_utc": "2026-05-01T00:00:00+00:00",
    }
    trainer = Trainer(make_env(), config(), protocol=protocol)
    path = tmp_path / "policy.pt"
    trainer.export_policy(path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lima-pm25-rl",
            "--policy",
            "neural",
            "--checkpoint",
            str(path),
            "--replay-start",
            "2026-04-15T00:00:00+00:00",
            *extra_arguments,
        ],
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


def test_replay_cli_rejects_a_window_extending_beyond_the_reserved_end(
    make_env, tmp_path, monkeypatch, capsys
):
    import sys

    import lima_pm25_rl.cli as entry

    trainer = Trainer(
        make_env(),
        config(),
        protocol={
            "gp_fit_end_utc": "2026-01-18T09:00:00+00:00",
            "policy_fit_end_utc": "2026-04-01T05:00:00+00:00",
            "reserved_final_start_utc": "2026-04-01T05:00:00+00:00",
            "reserved_final_end_utc": "2026-07-01T05:00:00+00:00",
        },
    )
    path = tmp_path / "policy.pt"
    trainer.export_policy(path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lima-pm25-rl",
            "--policy",
            "neural",
            "--checkpoint",
            str(path),
            "--replay-start",
            "2026-07-01T04:55:00+00:00",
            "--steps",
            "2",
        ],
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert "extremo final reservado" in capsys.readouterr().err
