import csv
import json

import numpy as np
import pytest

from lima_pm25_rl.training_report import (
    block_summary,
    comparison_section,
    generate_report,
    load_training,
    trailing_mean,
)


def segment(path, start, end, *, gamma=0.99):
    path.mkdir()
    config = {
        "training": {"gamma": gamma, "seed": 42, "device": "cpu"},
        "protocol": {"fixture": "same_scenarios"},
        "validation_seeds": [100],
        "validation_every": 2,
        "start_episode": start,
        "target_episodes": end,
        "resumed_from": "original/checkpoint-last.pt" if start else None,
        "start_physical_steps": start * 3,
        "start_optimizer_updates": start * 2,
        "start_target_updates": start,
    }
    (path / "config.json").write_text(json.dumps(config))
    rows = []
    for ep in range(start + 1, end + 1):
        validation = ep % 2 == 0
        rows.append(
            {
                "episode": ep,
                "scenario_seed": int(
                    np.random.SeedSequence([42, 0, ep - 1]).generate_state(1)[0]
                ),
                "episode_steps": 3,
                "physical_steps": ep * 3,
                "optimizer_updates": ep * 2,
                "target_updates": ep,
                "invalid_actions": 0,
                "epsilon_end": 0.05,
                "reward": ep / 10,
                "loss": 1 / ep if ep >= 2 else None,
                "information_gain": ep / 10 + 0.01,
                "distance_cost": 0.01,
                "distance_m": ep * 100,
                "final_integrated_variance": 100 - ep,
                "mean_integrated_variance": 110 - ep,
                "validation_reward": ep / 5 if validation else None,
                "validation_variance": 90 - ep if validation else None,
                "validation_mean_variance": 100 - ep if validation else None,
            }
        )
    with (path / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if start == 0:
        (path / "validation-initial.json").write_text(
            json.dumps(
                {
                    "reward": 0.1,
                    "final_integrated_variance": 90,
                    "mean_integrated_variance": 100,
                }
            )
        )
    return path


def test_smoothing_is_retrospective_and_does_not_fill_missing_losses():
    result = trailing_mean([np.nan, 2, 4, np.nan, 8], 2)
    np.testing.assert_allclose(result, [np.nan, 2, 3, np.nan, 8], equal_nan=True)
    changed_future = trailing_mean([np.nan, 2, 4, np.nan, 100], 2)
    np.testing.assert_array_equal(result[:4], changed_future[:4])
    with pytest.raises(ValueError):
        trailing_mean([1, 2], 0)


def test_joined_runs_keep_episode_numbers_counters_and_validation_gaps(tmp_path):
    first = segment(tmp_path / "first", 0, 2)
    second = segment(tmp_path / "second", 2, 4)
    run = load_training([first, second], "Gamma .99")
    assert [r["episode"] for r in run["rows"]] == [1, 2, 3, 4]
    assert run["boundaries"] == [2]
    assert run["rows"][-1]["physical_steps"] == 12
    assert [r["episode"] for r in run["validation"]] == [0, 2, 4]
    assert run["rows"][0]["loss"] is None
    blocks = block_summary(run, 2)
    assert [b["validations"] for b in blocks] == [1, 1]
    assert blocks[1]["optimizer_updates"] == 4
    assert blocks[1]["validation_mean_final_integrated_variance"] == 86


@pytest.mark.parametrize(
    "start,gamma,error",
    [(3, 0.99, "huecos"), (1, 0.99, "solapamientos"), (2, 0.5, "distinto protocolo")],
)
def test_cannot_splice_missing_overlapping_or_different_gamma_runs(
    tmp_path, start, gamma, error
):
    first = segment(tmp_path / "first", 0, 2)
    second = segment(tmp_path / "second", start, 4, gamma=gamma)
    with pytest.raises(ValueError, match=error):
        load_training([first, second], "Invalid")


def test_counter_reset_is_rejected(tmp_path):
    first = segment(tmp_path / "first", 0, 2)
    second = segment(tmp_path / "second", 2, 4)
    config = json.loads((second / "config.json").read_text())
    config["start_optimizer_updates"] = 0
    (second / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contador"):
        load_training([first, second], "Invalid")


def test_interrupted_segment_can_continue_without_inventing_missing_episodes(tmp_path):
    first = segment(tmp_path / "first", 0, 2)
    config = json.loads((first / "config.json").read_text())
    config["target_episodes"] = 4
    (first / "config.json").write_text(json.dumps(config))
    partial = load_training([first], "Partial")
    assert len(partial["rows"]) == 2 and partial["target_episodes"] == 4
    second = segment(tmp_path / "resumed", 2, 4)
    full = load_training([first, second], "Full")
    assert [r["episode"] for r in full["rows"]] == [1, 2, 3, 4]


def test_explicit_checkpoint_cut_excludes_tail_and_its_validation(tmp_path):
    first = segment(tmp_path / "first", 0, 4)
    second = segment(tmp_path / "resumed", 2, 6)
    original = (first / "metrics.csv").read_bytes()
    run = load_training([first, second], "Continued", end_episodes=[2, None])
    assert [r["episode"] for r in run["rows"]] == [1, 2, 3, 4, 5, 6]
    assert [r["episode"] for r in run["validation"]] == [0, 2, 4, 6]
    assert run["sources"][0]["excluded_tail_episodes"] == 2
    assert run["sources"][0]["recorded_end_episode"] == 4
    assert (first / "metrics.csv").read_bytes() == original
    config = json.loads((second / "config.json").read_text())
    config["start_physical_steps"] += 1
    (second / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contador"):
        load_training([first, second], "Invalid", end_episodes=[2, None])


@pytest.mark.parametrize("ends", [[0], [5], [True], [-1], []])
def test_invalid_explicit_segment_ends_are_rejected(tmp_path, ends):
    first = segment(tmp_path / "first", 0, 4)
    with pytest.raises(ValueError):
        load_training([first], "Invalid", end_episodes=ends)


def test_report_supports_gamma_comparison_without_changing_scenarios(
    tmp_path, monkeypatch
):
    import lima_pm25_rl.training_report as report

    first = segment(tmp_path / "control", 0, 4)
    candidate = segment(tmp_path / "candidate", 0, 4, gamma=0.5)
    settings = {
        "training_runs": [
            {"label": "Control", "sources": [first.name]},
            {"label": "Gamma .5", "sources": [candidate.name]},
        ],
        "smoothing_episodes": 2,
        "block_episodes": 2,
        "comparisons": [],
        "pending_note": "Todavía falta la evaluación.",
    }
    output = tmp_path / "report"
    monkeypatch.setattr(report, "product", lambda name: tmp_path / name)
    page = generate_report(settings, output)
    assert "Gamma .5" in page.read_text()
    assert "no prueba convergencia" in page.read_text()
    for name in ("learning", "operation", "schedule", "validation"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (output / f"{name}.svg").is_file()
    with (output / "training-1.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["training_executed"] and not manifest["simulation_executed"]
    with pytest.raises(FileExistsError):
        generate_report(settings, output)


def test_report_rejects_smoke_comparison_before_presenting_results(
    tmp_path, monkeypatch
):
    import lima_pm25_rl.training_report as report

    source = tmp_path / "source"
    folder = source / "evaluation-1"
    folder.mkdir(parents=True)
    (source / "protocol.json").write_text(json.dumps({"version": 2}))
    (folder / "source.json").write_text(json.dumps({"smoke_only": True}))
    (folder / "summary.json").write_text(json.dumps({"smoke_only": True}))
    output = tmp_path / "report"
    output.mkdir()
    monkeypatch.setattr(
        report, "product", lambda key: source if key == "protocol" else folder
    )
    with pytest.raises(ValueError, match="smoke"):
        comparison_section(
            {"protocol_product": "protocol", "products": {"evaluation-1": "Smoke"}},
            output,
            1,
        )
