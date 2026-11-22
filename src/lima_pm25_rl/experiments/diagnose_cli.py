"""Inspección del campo oculto de entrenamiento: CPU, sin aprender una política."""

import argparse
import json

import numpy as np

from ..study import _sha256, prepare_experiment
from .diagnostics import run_diagnostics


def run(args):
    parser = argparse.ArgumentParser(
        description="Diagnóstico de contraste y dinámica, sin entrenamiento neuronal"
    )
    if args.cases < 2:
        parser.error("--cases debe ser al menos 2")
    spec = json.loads(args.config.read_text())
    if spec["kind"] != "generated":
        parser.error("Se requiere una configuración de episodios generados")
    output = args.output
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
