import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from lima_pm25_rl.gp_diagnostics import diagnose_case
from lima_pm25_rl.gp_spatial_experiment import (
    load_frozen_case,
    partition_bounds,
    plot_pair,
    score_frozen_case,
    select_scale,
    spatial_variant,
    summarize_validation,
)
from lima_pm25_rl.network import BiLGD3QN, NeuralController
from lima_pm25_rl.training import TrainingConfig


def test_only_one_isotropic_spatial_scale_is_changed(estimator_config):
    original = estimator_config.reconstruction_config
    variant = spatial_variant(original, 600)
    expected = asdict(original)
    expected.update(x_length_scale_m=600, y_length_scale_m=600)
    assert asdict(variant) == expected
    assert original.x_length_scale_m == original.y_length_scale_m == 300


def test_selection_uses_pooled_case_mse_and_can_retain_original():
    rows = [
        {"case_seed": seed, "spatial_scale_m": scale, "temporal_mse": mse}
        for scale, values in ((150, (1, 9)), (300, (4, 4)), (600, (6, 6)))
        for seed, mse in zip((1, 2), values, strict=True)
    ]
    selected = select_scale(rows, [150, 300, 600], [1, 2], 300)
    assert selected["selected_scale_m"] == 300
    # Promediar RMSE erróneamente empataría 150 y 300. Se promedia MSE.
    assert selected["calibration_scores"][0]["mean_case_temporal_mse"] == 5
    for row in rows:
        row["temporal_mse"] = 2
    assert select_scale(rows, [150, 300, 600], [1, 2], 300)["selected_scale_m"] == 300
    with pytest.raises(ValueError, match="todos los casos"):
        select_scale(rows[:-1], [150, 300, 600], [1, 2], 300)


def test_offline_gp_reproduces_original_and_does_not_use_future_readings(
    make_env, estimator_config, tmp_path, monkeypatch
):
    from lima_pm25_rl import gp_spatial_experiment as experiment

    env = make_env(concentration_priority=1, mobile_noise_stddev_ug_m3=1)
    torch.set_num_threads(1)
    torch.manual_seed(2)
    policy = NeuralController(BiLGD3QN(hidden_size=8).eval())
    folder = tmp_path / "input"
    diagnose_case(env, policy, seed=42, gamma=0.99, output=folder)
    weights = {
        name: value.clone() for name, value in policy.network.state_dict().items()
    }
    records = tuple(env._records)
    before = env._posterior
    bounds = {
        "start_utc": datetime(2025, 1, 1, tzinfo=UTC),
        "end_utc": datetime(2025, 1, 2, tzinfo=UTC),
    }
    frozen = load_frozen_case(folder, env.domain.grid, **bounds)
    model = experiment.GaussianProcessFieldReconstructor
    calls = []

    class Spy(model):
        def reconstruct(self, request):
            assert all(
                r.timestamp_utc <= request.reconstruction_timestamp_utc
                for r in request.measurements
            )
            calls.append(request)
            return super().reconstruct(request)

    monkeypatch.setattr(experiment, "GaussianProcessFieldReconstructor", Spy)
    base = score_frozen_case(
        env.domain.grid,
        estimator_config,
        frozen,
        length_m=300,
        output=tmp_path / "original",
        control=True,
    )
    variant = score_frozen_case(
        env.domain.grid,
        estimator_config,
        frozen,
        length_m=600,
        output=tmp_path / "variant",
    )
    assert len(calls) == 8
    assert len(calls[0].measurements) == 1
    assert base["temporal_mse"] == pytest.approx(frozen[0]["metrics"]["temporal_mse"])
    assert variant["temporal_mse"] != pytest.approx(base["temporal_mse"])
    assert tuple(env._records) == records and env._posterior is before
    assert all(
        torch.equal(v, policy.network.state_dict()[k]) for k, v in weights.items()
    )
    plot_pair(
        env.domain.grid,
        frozen,
        tmp_path / "original/posterior-private.npz",
        tmp_path / "variant/posterior-private.npz",
        ticks=[1],
        output=tmp_path,
    )
    assert (tmp_path / "tick-01.png").is_file()
    with pytest.raises(ValueError, match="partición temporal"):
        load_frozen_case(
            folder,
            env.domain.grid,
            start_utc=bounds["end_utc"],
            end_utc=bounds["end_utc"] + timedelta(days=1),
        )
    wrong = json.loads((folder / "report.json").read_text())
    wrong["signatures"]["field_sha256"] = "incorrect"
    (folder / "report.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="huella"):
        load_frozen_case(folder, env.domain.grid, **bounds)


def test_summary_does_not_select_from_validation_and_uses_mse():
    rows = [
        {
            "case_seed": seed,
            "spatial_scale_m": scale,
            "temporal_mse": mse,
            "final_mse": mse,
            "coverage_68": 0.8,
            "coverage_95": 0.98,
            "mean_variance": 10,
            "spearman_variance_squared_error": None,
            "top_uncertainty_error_enrichment": 1,
        }
        for scale, values in ((300, (1, 9)), (600, (9, 25)))
        for seed, mse in zip((1, 2), values, strict=True)
    ]
    result = summarize_validation(rows, 300, 600, [1, 2])
    assert result["variants"][0]["final_rmse"] == pytest.approx(np.sqrt(5))
    assert result["paired"][0]["selected_minus_original_final_mse"] > 0
    assert result["variants"][1]["spatial_scale_m"] == 600
    only_control = summarize_validation(rows[:2], 300, 300, [1, 2])
    assert len(only_control["variants"]) == 1


def test_partition_rejects_calibration_using_validation_dates():
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
    assert (
        partition_bounds(protocol, "calibration")["end_utc"]
        == partition_bounds(protocol, "validation")["start_utc"]
    )
    wrong = deepcopy(protocol)
    wrong["scenarios"][0]["configuration"]["training_end_utc"] = (
        "2025-03-01T00:00:00+00:00"
    )
    with pytest.raises(ValueError, match="reservados"):
        partition_bounds(wrong, "calibration")


def test_cli_freezes_training_selection_before_loading_validation(
    make_env, template, tmp_path, monkeypatch, capsys
):
    from lima_pm25_rl import Episode
    from lima_pm25_rl import gp_spatial_experiment as experiment

    torch.set_num_threads(1)
    parameters = TrainingConfig(device="cpu", hidden_size=8, history_length=2)
    period = {
        "training_start_utc": "2025-01-01T00:00:00+00:00",
        "training_end_utc": "2025-02-01T00:00:00+00:00",
        "evaluation_start_utc": "2025-02-01T00:00:00+00:00",
        "evaluation_end_utc": "2025-03-01T00:00:00+00:00",
    }
    protocol = {
        "scenarios": [{"configuration": period}],
        "gp_fit_end_utc": period["training_end_utc"],
        "reserved_final_start_utc": period["evaluation_end_utc"],
    }
    start = datetime(2025, 2, 1, tzinfo=UTC)

    def validation_env():
        return make_env(
            episode_factory=lambda seed: Episode(
                replace(
                    template,
                    config=replace(template.config, start_utc=start, seed=seed),
                    weather=replace(template.weather, start_utc=start),
                )
            )
        )

    policy = NeuralController(BiLGD3QN(hidden_size=8).eval(), history_length=2)
    checkpoint = tmp_path / "policy.pt"
    # Estado artificial para verificar el flujo, no entrenamiento de 200 episodios.
    torch.save(
        {
            "version": 2,
            "kind": "policy",
            "training_config": asdict(parameters),
            "protocol": protocol,
            "completed_episodes": 200,
            "online": policy.network.state_dict(),
        },
        checkpoint,
    )
    source = tmp_path / "source"
    source.mkdir()
    diagnose_case(
        validation_env(), policy, seed=100, gamma=0.99, output=source / "case-100"
    )
    (source / "case-100/reproduction.json").write_text('{"match": true}')
    (source / "summary.json").write_text("{}")
    (source / "config.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "training": asdict(parameters),
                "checkpoint": str(checkpoint),
                "policy_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "case_seeds": [100],
            }
        )
    )
    spec_path = tmp_path / "experiment.json"
    spec_path.write_text(
        json.dumps(
            {
                "kind": "generated",
                "episodes": 200,
                "steps_per_episode": 3,
                "training": asdict(parameters),
                "validation_seeds": [100],
            }
        )
    )
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "experiment_config": spec_path.name,
                "calibration_cases": 1,
                "spatial_scales_m": [150, 300, 600],
                "selection_metric": "mean_case_temporal_mse",
                "map_ticks": [1],
            }
        )
    )
    monkeypatch.setattr(
        experiment,
        "prepare_experiment",
        lambda *a, **k: (make_env(), validation_env(), protocol),
    )
    arguments = [
        "compare-gp",
        "--config",
        str(settings_path),
        "--diagnostic",
        str(source),
    ]
    original_load = experiment.load_policy
    monkeypatch.setattr(
        experiment,
        "load_policy",
        lambda *a, **k: pytest.fail("check no debe cargar la red"),
    )
    monkeypatch.setattr(sys, "argv", [*arguments, "--check"])
    experiment.main()
    assert json.loads(capsys.readouterr().out)["simulation_executed"] is False
    monkeypatch.setattr(experiment, "load_policy", original_load)
    output = tmp_path / "result"
    original_loader = experiment.load_frozen_case
    validation_reads = []

    def guarded(path, *args, **kwargs):
        if path == source / "case-100":
            selected = json.loads((output / "selection.json").read_text())
            assert selected["selection_partition"] == "training"
            validation_reads.append(path)
        return original_loader(path, *args, **kwargs)

    monkeypatch.setattr(experiment, "load_frozen_case", guarded)
    monkeypatch.setattr(sys, "argv", [*arguments, "--output", str(output)])
    experiment.main()
    assert len(validation_reads) == 1
    calibration = json.loads((output / "calibration.json").read_text())
    assert len(calibration) == 3 and all(r["case_seed"] != 100 for r in calibration)
    assert (output / "validation/case-100/tick-01.png").is_file()
