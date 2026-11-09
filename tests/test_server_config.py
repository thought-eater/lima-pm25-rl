"""Contrato del experimento inicial; no ejecuta entrenamientos sobre Lima."""

import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from lima_pm25.domain import PM25SourceRole
from lima_pm25.simulation import (
    EpisodeWeatherMode,
    RoadLineSource,
    compose_episode,
    load_episode_generation_config,
)
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/training-server.json"
GP_END = datetime(2026, 1, 18, 9, tzinfo=UTC)
FINAL_START = datetime(2026, 4, 1, 5, tzinfo=UTC)
FINAL_END = datetime(2026, 7, 1, 5, tzinfo=UTC)


def server_inputs():
    spec = json.loads(CONFIG.read_text())
    configs = [
        load_episode_generation_config(CONFIG.parent / item["path"])
        for item in spec["episode_configs"]
    ]
    return spec, configs


def test_server_config_uses_the_requested_budget_and_separate_final_period():
    from lima_pm25_rl.training import TrainingConfig

    spec, configs = server_inputs()
    training = TrainingConfig(**spec["training"])
    assert spec["kind"] == "generated"
    assert spec["episodes"] == 1000
    assert spec["steps_per_episode"] == 36
    assert training.hidden_size == 64 and training.history_length == 4
    assert training.capacity_steps >= spec["steps_per_episode"]
    assert training.learning_starts <= spec["episodes"] * spec["steps_per_episode"]
    assert datetime.fromisoformat(spec["gp_fit_end_utc"]) == GP_END
    assert datetime.fromisoformat(spec["reserved_final_start_utc"]) == FINAL_START
    assert datetime.fromisoformat(spec["reserved_final_end_utc"]) == FINAL_END
    for config in configs:
        assert config.training_start_utc < config.training_end_utc == GP_END
        assert GP_END == config.evaluation_start_utc
        assert config.evaluation_end_utc == FINAL_START < FINAL_END
        assert all(weight > 0 for weight in config.weather_weights)
        # No fabricar evidencia nominal ni usar resúmenes de todo el producto.
        assert config.stress and not config.constraint_evidence_ids
        assert not config.area_sources
    assert [item["member"] for item in spec["episode_configs"]] == ["control", "event"]
    assert [item["weight"] for item in spec["episode_configs"]] == [0.8, 0.2]


def test_validation_seeds_cover_every_family_and_weather_twice_without_outcomes():
    spec, configs = server_inputs()
    seeds = spec["validation_seeds"]
    assert len(seeds) == len(set(seeds)) == 12
    coverage = Counter()
    for seed in seeds:
        # evaluate_episode pasa la semilla a env.reset; el entorno deriva una
        # semilla de escenario, a diferencia del override usado al entrenar.
        scenario_seed = int(np.random.default_rng(seed).integers(2**32))
        family = int(np.random.default_rng([scenario_seed, 7]).choice(2, p=[0.8, 0.2]))
        mode_stream = np.random.SeedSequence([1, scenario_seed]).spawn(4)[0]
        weights = np.asarray(configs[family].weather_weights)
        mode = int(
            np.random.default_rng(mode_stream).choice(3, p=weights / weights.sum())
        )
        coverage[family, mode] += 1
    assert coverage == Counter(
        {(family, mode): 2 for family in range(2) for mode in range(3)}
    )


def test_validation_coverage_uses_the_actual_reset_seed_mapping(make_env, template):
    from lima_pm25_rl.episodes import Episode
    from lima_pm25_rl.evaluation import evaluate_episode
    from lima_pm25_rl.policies import BaselinePolicy

    seen = []

    def episode(seed):
        seen.append(seed)
        return Episode(replace(template, config=replace(template.config, seed=seed)))

    env = make_env(episode_factory=episode)
    evaluate_episode(env, BaselinePolicy("random", seed=42), seed=100002)
    assert seen == [int(np.random.default_rng(100002).integers(2**32))]


@pytest.mark.parametrize("family", [0, 1])
@pytest.mark.parametrize("partition", ["training", "evaluation"])
@pytest.mark.parametrize("mode", list(EpisodeWeatherMode))
def test_server_sources_compose_in_every_weather_and_partition(
    family, partition, mode, template, historical_weather, generation_config
):
    spec, configs = server_inputs()
    config = replace(
        configs[family],
        training_start_utc=generation_config.training_start_utc,
        training_end_utc=generation_config.training_end_utc,
        evaluation_start_utc=generation_config.evaluation_start_utc,
        evaluation_end_utc=generation_config.evaluation_end_utc,
    )
    sources = tuple(
        RoadLineSource(
            source_id=item.source_id,
            line=LineString(((100, 100), (700, 100))),
            emission_rate_ug_m_s=100,
            provenance="Geometría y tasa artificiales para comprobar la configuración",
            starts_at_s=3600 if item.source_id == "alfonso-ugarte-event" else 0,
            ends_at_s=7200 if item.source_id == "alfonso-ugarte-event" else 10800,
            role=PM25SourceRole.COUNTERFACTUAL_EVENT
            if item.source_id == "alfonso-ugarte-event"
            else PM25SourceRole.ROUTINE_ENVIRONMENTAL,
        )
        for item in config.source_variations
    )
    template = replace(
        template, sources=sources, config=replace(template.config, duration_s=10800)
    )
    for seed in (0, 17, 42):
        request = compose_episode(
            template,
            historical_weather,
            config,
            seed=seed,
            partition=partition,
            weather_mode=mode,
        )
        metadata = json.loads(request.episode_parameters_json)
        assert metadata["partition"] == partition and metadata["weather_mode"] == mode
        assert len(request.sources) == (3 if family == 0 else 4)
        start = (
            config.training_start_utc
            if partition == "training"
            else config.evaluation_start_utc
        )
        end = (
            config.training_end_utc
            if partition == "training"
            else config.evaluation_end_utc
        )
        assert start <= request.config.start_utc
        assert request.config.start_utc + timedelta(seconds=10800) < end
        for source in request.sources:
            assert 0 <= source.starts_at_s < source.ends_at_s <= 10800
    assert spec["steps_per_episode"] * 300 == template.config.duration_s


def test_transfer_lists_are_explicit_and_include_runtime_resources():
    def entries(name):
        return [
            line
            for line in (ROOT / "config" / name).read_text().splitlines()
            if line and not line.startswith("#")
        ]

    core = entries("server-core-files.txt")
    rl = entries("server-rl-files.txt")
    assert json.loads(CONFIG.read_text())["gp_partition_manifest"] in core
    assert "resultados/r1_2_entorno/domain/road_graph.graphml" in core
    assert "resultados/r1_2_entorno/domain/grid.npz" in core
    assert "data/var_meteorologicas_lima_2025_2026_imputed.csv" in core
    assert {"src/", "config/", "docs/", "uv.lock", "tests/"} <= set(rl)
    for entry in (*core, *rl):
        path = Path(entry)
        assert not path.is_absolute() and ".." not in path.parts
        assert not {".git", ".venv", "cache", "__pycache__"}.intersection(path.parts)
    assert all((ROOT / path).exists() for path in rl)


def test_check_cli_never_constructs_a_trainer_or_resets_an_environment(
    tmp_path, monkeypatch, capsys
):
    import sys

    import lima_pm25_rl.train_cli as entry

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "kind": "controlled",
                "episodes": 1,
                "steps_per_episode": 1,
                "training": {"device": "cpu"},
            }
        )
    )
    monkeypatch.setattr(
        entry, "Trainer", lambda *a, **kw: pytest.fail("No debe construir la red")
    )
    monkeypatch.setattr(
        entry.LimaPM25Env, "reset", lambda *a, **kw: pytest.fail("No debe simular")
    )
    monkeypatch.setattr(
        sys, "argv", ["lima-pm25-train", "--config", str(path), "--check"]
    )
    entry.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "configuration_checked"
    assert result["training_executed"] is False
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "extra",
    [
        ["--output", "unused-output"],
        ["--resume"],
        ["--resume", "unused.pt"],
        ["--episodes", "1000"],
    ],
)
def test_check_rejects_training_arguments(monkeypatch, extra):
    import sys

    import lima_pm25_rl.train_cli as entry

    monkeypatch.setattr(
        sys, "argv", ["lima-pm25-train", "--config", str(CONFIG), "--check", *extra]
    )
    monkeypatch.setattr(
        entry,
        "prepare_experiment",
        lambda *a, **kw: pytest.fail("Debe rechazar antes de cargar"),
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


@pytest.mark.parametrize("field", ["episodes", "checkpoint_every", "validation_every"])
def test_check_rejects_invalid_execution_budgets(tmp_path, monkeypatch, field):
    import sys

    import lima_pm25_rl.train_cli as entry

    spec = {"kind": "controlled", "episodes": 1, "steps_per_episode": 1, field: 0}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(
        sys, "argv", ["lima-pm25-train", "--config", str(path), "--check"]
    )
    with pytest.raises(ValueError):
        entry.main()
