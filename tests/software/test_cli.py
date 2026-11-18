import json
import re
import sys
from types import SimpleNamespace

import pytest

from lima_pm25_rl import cli, study
from lima_pm25_rl.files import output_directory


def test_no_subcommand_prints_usage_without_starting_work(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lima-pm25-rl"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_new_directory_name_contains_only_date_and_seconds():
    path = output_directory("r3_2_entrenamiento/gamma099")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", path.name)
    assert path.parent.name == "gamma099"


@pytest.mark.parametrize(
    "command",
    [
        ["entrenar", "adaptativo", "--comprobar"],
        ["experimento", "gp-adaptativo", "--comprobar"],
        ["entrenar", "adaptativo-continuacion"],
    ],
)
def test_removed_cli_modes_fail_before_any_work(monkeypatch, command):
    monkeypatch.setattr(
        study, "Trainer", lambda *a, **k: pytest.fail("No debe entrenar")
    )
    monkeypatch.setattr(sys, "argv", ["lima-pm25-rl", *command])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_visualize_uses_the_folium_viewer(monkeypatch):
    from lima_pm25_rl import map_viewer

    calls = []
    monkeypatch.setattr(map_viewer, "run", lambda: calls.append("viewer"))
    monkeypatch.setattr(sys, "argv", ["lima-pm25-rl", "visualizar"])
    cli.main()
    assert calls == ["viewer"]


def test_experiment_inputs_are_explicit_without_cache_search():
    args = cli.experiment_arguments("gp-temporal")
    assert "resultados" in args.output.parts
    assert args.spatial_experiment.is_dir() and args.diagnostic.is_dir()
    assert "resultados" in args.spatial_experiment.parts
    assert args.device == "cuda"


@pytest.mark.parametrize(
    "command",
    [
        ["entrenar", "adaptativo"],
        ["evaluar", "adaptativo-20000"],
        ["experimento", "gp-adaptativo"],
    ],
)
def test_device_flag_is_not_part_of_public_cli(monkeypatch, command):
    monkeypatch.setattr(sys, "argv", ["lima-pm25-rl", *command, "--device", "cpu"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("check", [False, True])
def test_cuda_configuration_cannot_silently_fall_back_to_cpu(
    tmp_path, monkeypatch, check
):
    (tmp_path / "adaptativo.json").write_text(
        json.dumps(
            {
                "kind": "controlled",
                "episodes": 1,
                "steps_per_episode": 1,
                "training": {"device": "cuda"},
            }
        )
    )
    monkeypatch.setattr(cli, "CONFIG", tmp_path)
    monkeypatch.setattr(study.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        study,
        "prepare_experiment",
        lambda *a, **kw: pytest.fail("Debe detenerse antes de cargar escenarios"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lima-pm25-rl",
            "entrenar",
            "adaptativo",
            *(["--comprobar"] if check else []),
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == [tmp_path / "adaptativo.json"]


def test_evaluate_command_runs_global_and_local_evaluation_in_order(
    tmp_path, monkeypatch
):
    from lima_pm25_rl import evaluate, hotspot_evaluation
    from lima_pm25_rl.experiments import gp_temporal_experiment

    (tmp_path / "entrenamiento.json").write_text("{}")
    (tmp_path / "evaluacion.json").write_text(
        json.dumps(
            {
                "protocol_product": "duracion",
                "truth_product": "diagnostico-gp",
                "map_ticks": [1, 12, 24, 36],
            }
        )
    )
    (tmp_path / "duracion").mkdir()
    (tmp_path / "duracion/protocol.json").write_text('{"version": 2}')
    calls = []
    env = SimpleNamespace(
        domain=SimpleNamespace(grid="grid"),
        estimator_config="gp",
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(cli, "CONFIG", tmp_path)
    monkeypatch.setattr(cli, "product", lambda name: tmp_path / name)
    monkeypatch.setattr(cli, "output_directory", lambda name: tmp_path / "output")
    monkeypatch.setattr(
        evaluate, "run", lambda args: calls.append(("global", args.checkpoint.name))
    )
    monkeypatch.setattr(study, "prepare_experiment", lambda *a: (env, env, {}))
    monkeypatch.setattr(gp_temporal_experiment, "check_control", lambda *a: None)
    monkeypatch.setattr(hotspot_evaluation, "frozen_cases", lambda *a: "private-fields")

    def local(source, frozen, grid, gp, protocol, output, **kwargs):
        assert source == tmp_path / "output/misiones"
        assert frozen == "private-fields" and output == tmp_path / "output/focos"
        assert kwargs["map_ticks"] == [1, 12, 24, 36]
        calls.append("local")

    monkeypatch.setattr(hotspot_evaluation, "evaluate_saved", local)
    monkeypatch.setattr(sys, "argv", ["lima-pm25-rl", "evaluar", "gamma099-2000"])
    cli.main()
    assert calls == [("global", "policy-episode-2000.pt"), "local", "close", "close"]
