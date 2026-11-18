"""Interfaz única de los procedimientos reproducibles de O3."""

import argparse
import json
from importlib import import_module
from pathlib import Path

from .files import PROJECT_ROOT, _write_json, output_directory, product

CONFIG = PROJECT_ROOT / "config"
POLICIES = {
    "adaptativo-20000": ("adaptativo", "policy-last.pt", "adaptativo.json"),
    "adaptativo-30000": (
        "adaptativo-continuacion",
        "policy-last.pt",
        "adaptativo-continuacion.json",
    ),
    "adaptativo-mejor": (
        "adaptativo-continuacion",
        "best-rmse.json",
        "adaptativo-continuacion.json",
    ),
    "gamma099-1000": ("continuacion", "policy-episode-1000.pt", "entrenamiento.json"),
    "gamma099-2000": ("continuacion", "policy-episode-2000.pt", "entrenamiento.json"),
    "gamma05-2000": ("gamma05", "policy-last.pt", "gamma05.json"),
    "inicial": ("entrenamiento-original", "policy-initial.pt", "entrenamiento.json"),
    "best-original": ("entrenamiento-original", "policy-best.pt", "entrenamiento.json"),
}
EXPERIMENTS = {
    "escenarios": ("diagnose_cli", "entrenamiento.json"),
    "movilidad": ("mobility_diagnostics", "entrenamiento.json"),
    "llegadas": ("arrival_experiment", "entrenamiento.json"),
    "valores-q": ("value_diagnostics", "entrenamiento.json"),
    "diagnostico-gp": ("gp_diagnostics", "controles-bilg.json"),
    "gp-espacial": ("gp_spatial_experiment", "experimentos/gp-espacial.json"),
    "gp-temporal": ("gp_temporal_experiment", "experimentos/gp-temporal.json"),
    "gp-adaptativo": ("gp_adaptive_experiment", "adaptativo.json"),
    "gp-realimentado": ("gp_feedback_experiment", "controles-bilg.json"),
    "historia": ("history_experiment", "experimentos/historia.json"),
    "recompensa": ("reward_experiment", "experimentos/recompensa.json"),
    "focos-controlados": ("hotspot_sampling", "experimentos/focos-controlados.json"),
    "gp-focos-espacial": ("hotspot_gp_experiment", None),
    "gp-focos-temporal": ("hotspot_gp_time_experiment", None),
    "gp-mezcla": ("hotspot_gp_mixture", None),
    "gp-multiescala": ("gp_multiscale_experiment", None),
}


def experiment_arguments(name, *, check=False):
    """Entradas concretas de los experimentos publicados; no descubrimiento automático."""
    _, config = EXPERIMENTS[name]
    config = CONFIG / config if config else None
    settings = json.loads(config.read_text()) if config else {}
    if "experiment_config" in settings:
        settings = json.loads(
            (config.parent / settings["experiment_config"]).read_text()
        )
    sources = {
        "gp-focos-espacial": "focos-controlados",
        "gp-focos-temporal": "gp-focos-espacial",
        "gp-mezcla": "gp-focos-temporal",
    }
    return argparse.Namespace(
        config=config,
        output=None
        if check
        else output_directory(f"r3_3_evaluacion/experimentos/{name.replace('-', '_')}"),
        check=check,
        device=settings.get("training", {}).get("device", "cuda"),
        cases=12,
        case=100002,
        vehicle=0,
        all_cases=True,
        ticks=[1, 12, 24],
        noise_replicas=3,
        checkpoint=(product("controles-bilg") / "bilg-42/policy-last.pt")
        if name in {"diagnostico-gp", "gp-espacial", "gp-realimentado"}
        else product("entrenamiento-original") / "policy-best.pt",
        training_run=product("entrenamiento-original"),
        evaluation_run=product("evaluacion-best"),
        controls=product("controles-bilg"),
        diagnostic=product("diagnostico-gp"),
        spatial_experiment=product("gp-espacial"),
        temporal_experiment=product("gp-temporal"),
        source=product(sources[name]) if name in sources else None,
        training_source=product("gp-espacial"),
        hotspot_source=product("focos-controlados"),
    )


def evaluate_policy(name):
    """Ejecutar la política fijada y entregar también sus métricas localizadas."""
    from .evaluate import run
    from .experiments.gp_temporal_experiment import check_control
    from .hotspot_evaluation import evaluate_saved, frozen_cases
    from .study import prepare_experiment

    key, filename, config_name = POLICIES[name]
    config = CONFIG / config_name
    checkpoint = product(key) / filename
    suffix = name.rsplit("-", 1)[-1]
    expected_episode = int(suffix) if suffix.isdigit() else None
    if filename == "best-rmse.json":
        from .training_artifacts import policy_name

        best = json.loads(checkpoint.read_text())
        expected_episode = best["episode"]
        if best["policy_file"] != policy_name(expected_episode):
            raise ValueError("El registro del mejor no identifica su política")
        checkpoint = product(key) / best["policy_file"]
    output = output_directory(f"r3_3_evaluacion/politicas/{name}")
    run(
        argparse.Namespace(
            config=config,
            checkpoint=checkpoint,
            expected_episode=expected_episode,
            output=output / "misiones",
        )
    )
    spec = json.loads(config.read_text())
    settings = json.loads((CONFIG / "evaluacion.json").read_text())
    train, validation, protocol = prepare_experiment(spec, CONFIG)
    try:
        check_control(train.domain.grid, train.estimator_config)
        frozen_protocol = json.loads(
            (product(settings["protocol_product"]) / "protocol.json").read_text()
        )
        frozen = frozen_cases(
            product(settings["truth_product"]), train.domain.grid, protocol
        )
        evaluate_saved(
            output / "misiones",
            frozen,
            train.domain.grid,
            train.estimator_config,
            frozen_protocol,
            output / "focos",
            map_ticks=settings["map_ticks"],
        )
    finally:
        train.close()
        if validation is not None:
            validation.close()
    print(f"Resultados: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Procedimientos y evidencia de la tesis PM2.5 — O3"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "entrenar", help="Entrenamiento desde cero o reanudación explícita"
    )
    train.add_argument(
        "perfil",
        choices=("principal", "gamma05", "adaptativo", "adaptativo-continuacion"),
    )
    train.add_argument("--reanudar", type=Path)
    train.add_argument("--comprobar", action="store_true")
    evaluate = commands.add_parser(
        "evaluar", help="Evaluación global, focos y recorrido de una política"
    )
    evaluate.add_argument("politica", choices=tuple(POLICIES))
    compare = commands.add_parser(
        "comparar", help="Tablas y figuras de evaluaciones conservadas"
    )
    compare.add_argument("contraste", choices=("duracion", "descuento"))
    experiment = commands.add_parser(
        "experimento", help="Reproducir un procedimiento complementario"
    )
    experiment.add_argument("nombre", choices=tuple(EXPERIMENTS))
    experiment.add_argument("--comprobar", action="store_true")
    commands.add_parser(
        "informe", help="Generar el informe visual con los productos publicados"
    )
    commands.add_parser("visualizar", help="Localizar el informe visual conservado")
    args = parser.parse_args()
    try:
        if args.command == "entrenar":
            from .study import run

            config = (
                CONFIG
                / {
                    "principal": "entrenamiento.json",
                    "gamma05": "gamma05.json",
                    "adaptativo": "adaptativo.json",
                    "adaptativo-continuacion": "adaptativo-continuacion.json",
                }[args.perfil]
            )
            run(
                argparse.Namespace(
                    config=config,
                    check=args.comprobar,
                    output=None
                    if args.comprobar
                    else output_directory(
                        f"r3_2_entrenamiento/{'gamma099' if args.perfil == 'principal' else 'adaptativo' if args.perfil.startswith('adaptativo') else args.perfil}"
                    ),
                    resume=args.reanudar,
                    episodes=None,
                    seed=None,
                )
            )
        elif args.command == "evaluar":
            evaluate_policy(args.politica)
        elif args.command == "experimento":
            if args.comprobar and args.nombre in {
                "escenarios",
                "movilidad",
                "llegadas",
                "valores-q",
            }:
                # Estos diagnósticos no tienen fase de composición: validar sus entradas.
                options = experiment_arguments(args.nombre, check=True)
                for path in (options.config, options.checkpoint):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                print("Entradas disponibles; no se ejecutaron misiones.")
                return
            module, _ = EXPERIMENTS[args.nombre]
            import_module(f".experiments.{module}", __package__).run(
                experiment_arguments(args.nombre, check=args.comprobar)
            )
        elif args.command == "comparar":
            from .training_report import comparison_section

            output = output_directory(f"r3_3_evaluacion/{args.contraste}")
            output.mkdir(parents=True, exist_ok=False)
            comparisons = json.loads((CONFIG / "informe.json").read_text())[
                "comparisons"
            ]
            spec = next(
                item
                for item in comparisons
                if item["protocol_product"] == args.contraste
            )
            section, sources = comparison_section(spec, output, 1)
            _write_json(
                output / "config.json",
                {"procedure": "comparar", "settings": spec, "sources": sources},
            )
            (output / "index.html").write_text(
                "<!doctype html><meta charset='utf-8'><style>body{font:16px sans-serif}img{max-width:100%}td,th{padding:.5em}</style>"
                + section
            )
            print(output / "index.html")
        elif args.command == "informe":
            from .training_report import run

            run(
                argparse.Namespace(
                    config=CONFIG / "informe.json",
                    output=output_directory("r3_4_visualizacion/informes"),
                )
            )
        else:
            path = product("informe") / "index.html"
            if not path.is_file():
                raise FileNotFoundError(path)
            print(path)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
