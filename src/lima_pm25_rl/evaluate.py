"""Un comando para comparar la política aprendida y sus referencias en validación."""

import argparse
import json
from dataclasses import asdict

import torch

from .comparison import run_comparison
from .study import (
    _sha256,
    prepare_experiment,
)
from .training import TrainingConfig, load_policy, resolve_device


def _compatible(state, config: TrainingConfig, protocol) -> bool:
    if (
        not isinstance(state, dict)
        or state.get("version") != 2
        or state.get("kind") != "policy"
    ):
        return False
    if not isinstance(state.get("training_config"), dict):
        return False
    hardware = {"device", "cpu_threads"}
    try:
        saved_config = asdict(TrainingConfig(**state["training_config"]))
    except TypeError, ValueError:
        return False
    actual = {key: value for key, value in saved_config.items() if key not in hardware}
    expected = {
        key: value for key, value in asdict(config).items() if key not in hardware
    }
    return actual == expected and state.get("protocol") == json.loads(
        json.dumps(protocol)
    )


def run(args) -> None:
    parser = argparse.ArgumentParser(
        description="Comparación diagnóstica en validación; no entrena ni abre el replay final."
    )
    spec = json.loads(args.config.read_text())
    if spec["kind"] != "generated":
        parser.error(
            "La comparación requiere los escenarios generados con validación reservada"
        )
    config = TrainingConfig(**spec.get("training", {}))
    device = resolve_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    training_env, env, protocol = prepare_experiment(spec, args.config.resolve().parent)
    training_env.close()
    assert env is not None
    try:
        path = args.checkpoint.resolve()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not _compatible(state, config, protocol):
            parser.error(
                "La política no corresponde a los hiperparámetros y protocolo configurados"
            )
        expected_episode = getattr(args, "expected_episode", None)
        if (
            expected_episode is not None
            and state["completed_episodes"] != expected_episode
        ):
            raise ValueError(
                "Los episodios de la política no coinciden con la selección"
            )
        del state
        policy, _ = load_policy(path, device=str(device))
        print(f"Dispositivo: {device}", flush=True)
        output = args.output
        print(f"Política: {path}", flush=True)
        print(f"Salida: {output}", flush=True)
        summary = run_comparison(
            env,
            policy,
            case_seeds=spec["validation_seeds"],
            policy_seeds=spec.get("evaluation_policy_seeds", [0, 1, 2, 3, 4]),
            output=output,
            metadata={
                "protocol": protocol,
                "training": asdict(config),
                "checkpoint_path": str(path),
                "checkpoint_sha256": _sha256(path),
                "torch_version": str(torch.__version__),
                "device": str(next(policy.network.parameters()).device),
                "partition": "validation",
                "baseline_rules": {
                    "random": "uniform_over_valid_actions_with_sequential_destination_reservations",
                    "greedy": "max_current_GP_cell_uncertainty; random_tie_breaking",
                    "sweep": "geometric_east_west_row_sweep; directed_roads; no_exhaustive_coverage_guarantee",
                },
            },
        )
        for row in summary["strategies"]:
            print(
                f"{row['strategy']}: RMSE={row['synthetic_rmse_ug_m3']:.4f}, varianza={row['final_integrated_variance']:.6g}, terminaciones anticipadas={row['early_terminations']}"
            )
    finally:
        env.close()
