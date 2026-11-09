"""Un comando para comparar la política aprendida y sus referencias en validación."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from .comparison import run_comparison
from .train_cli import (
    CACHE_DIRECTORY,
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    _sha256,
    prepare_experiment,
)
from .training import TrainingConfig, load_policy


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
    actual = {
        key: value
        for key, value in state["training_config"].items()
        if key not in hardware
    }
    expected = {
        key: value for key, value in asdict(config).items() if key not in hardware
    }
    return actual == expected and state.get("protocol") == json.loads(
        json.dumps(protocol)
    )


def find_policy(config: TrainingConfig, protocol) -> Path:
    """Incluye una política importada a la raíz y las ejecuciones de cache/."""
    candidates = [
        PROJECT_ROOT / "policy-best.pt",
        *CACHE_DIRECTORY.rglob("policy-best.pt"),
    ]
    candidates = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for path in candidates:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if _compatible(state, config, protocol):
            return path
    raise FileNotFoundError(
        "No hay una policy-best.pt compatible en la raíz del proyecto ni en cache/."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparación diagnóstica en validación; no entrena ni abre el replay final."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Configuración del experimento; usa training-server.json por defecto",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Política específica; por defecto busca la policy-best.pt compatible más reciente",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Salida excepcional; por defecto crea cache/evaluation/<fecha UTC>/",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if spec["kind"] != "generated":
        parser.error(
            "La comparación requiere los escenarios generados con validación reservada"
        )
    config = TrainingConfig(**spec.get("training", {}))
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    training_env, env, protocol = prepare_experiment(spec, args.config.resolve().parent)
    training_env.close()
    assert env is not None
    try:
        path = (
            args.checkpoint.resolve()
            if args.checkpoint
            else find_policy(config, protocol)
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not _compatible(state, config, protocol):
            parser.error(
                "La política no corresponde a los hiperparámetros y protocolo configurados"
            )
        del state
        policy, _ = load_policy(path, device=args.device)
        output = args.output or (
            CACHE_DIRECTORY
            / "evaluation"
            / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        )
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
