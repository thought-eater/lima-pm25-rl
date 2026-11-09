"""Inspección del campo oculto de entrenamiento: CPU, sin aprender una política."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .diagnostics import run_diagnostics
from .train_cli import CACHE_DIRECTORY, DEFAULT_CONFIG, _sha256, prepare_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico de contraste y dinámica, sin entrenamiento neuronal"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()
    if args.cases < 2:
        parser.error("--cases debe ser al menos 2")
    spec = json.loads(args.config.read_text())
    if spec["kind"] != "generated":
        parser.error("Se requiere una configuración de episodios generados")
    output = args.output or (
        CACHE_DIRECTORY / "diagnostics" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    )
    if output.exists():
        parser.error("La salida debe ser un directorio nuevo")
    # Mismas semillas de los primeros episodios del entrenador. No se selecciona
    # ningún caso por contraste, recompensa ni desempeño de una política.
    seed = spec.get("training", {}).get("seed", 42)
    seeds = [
        int(np.random.SeedSequence([seed, 0, index]).generate_state(1)[0])
        for index in range(args.cases)
    ]
    env, validation, protocol = prepare_experiment(spec, args.config.resolve().parent)
    try:
        print(
            f"Diagnóstico CPU: {len(seeds)} campos, {spec['steps_per_episode']} ticks por campo. Salida: {output}",
            flush=True,
        )
        report = run_diagnostics(
            env,
            seeds=seeds,
            fit_cases=len(seeds) // 2,
            output=output,
            metadata={
                "experiment_config_sha256": _sha256(args.config),
                "protocol": protocol,
                "selection": "first_training_episode_seeds; configured_scenario_and_weather_mixture",
                "gp_prior_mean_ug_m3": env.estimator_config.reconstruction_config.prior_mean_ug_m3,
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        env.close()
        if validation is not None:
            validation.close()


if __name__ == "__main__":
    main()
