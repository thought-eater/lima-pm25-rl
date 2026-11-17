"""Comprobaciones de la evidencia publicada; no repiten los experimentos."""

import csv
import json

import pytest

from lima_pm25_rl.files import PROJECT_ROOT, product


@pytest.mark.parametrize(
    "result",
    [
        "r3_1_formulacion",
        "r3_2_entrenamiento",
        "r3_3_evaluacion",
        "r3_4_visualizacion",
    ],
)
def test_each_result_has_a_verification_description(result):
    assert (PROJECT_ROOT / "resultados" / result / "README.md").is_file()


@pytest.mark.parametrize(
    "key,first,last,ticks",
    [
        ("entrenamiento-original", 1, 1000, 35984),
        ("continuacion", 1001, 2000, 71984),
        ("gamma05", 1, 2000, 71979),
    ],
)
def test_r32_training_records_budget_and_valid_actions(key, first, last, ticks):
    folder = product(key)
    with (folder / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(r["episode"]) for r in rows] == list(range(first, last + 1))
    assert int(rows[-1]["physical_steps"]) == ticks
    assert all(int(r["invalid_actions"]) == 0 for r in rows)
    assert (folder / "checkpoint-last.pt").is_file()


@pytest.mark.parametrize("comparison", ["duracion", "descuento"])
def test_r33_comparison_preserves_all_references_and_cases(comparison):
    folder = product(comparison)
    for label in ("evaluation-1", "evaluation-2"):
        summary = json.loads((folder / label / "summary.json").read_text())
        assert not summary["smoke_only"] and summary["reproduction_passed"]
        assert {r["strategy"] for r in summary["strategies"]} == {
            "fixed",
            "neural",
            "random",
            "greedy",
            "sweep",
        }
        assert all(r["cases"] == 12 for r in summary["strategies"])
        assert all(r["target_percent"] == 20 for r in summary["neural_vs_references"])


def test_control_policies_are_preserved_without_window_runs():
    folder = product("controles-bilg")
    assert not list(folder.glob("window-*"))
    for seed in (42, 43, 44):
        assert (folder / f"bilg-{seed}/policy-last.pt").is_file()


def test_r34_report_contains_local_figures():
    import re

    folder = product("informe")
    page = folder / "index.html"
    assert page.is_file()
    for link in re.findall(r"(?:src|href)='([^']+)'", page.read_text()):
        assert (folder / link).is_file()
