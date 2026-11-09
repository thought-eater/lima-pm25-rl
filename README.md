# Lima PM2.5 RL

Entorno de Gymnasium para un agente centralizado que planifica el muestreo de
**tres vehículos** sobre las calles de Lima. Reutiliza el dominio, la física,
los sensores y el proceso gaussiano (GP) de `lima-pm25-core`.

Cada ciclo elige tres destinos en las ocho direcciones de la grilla, evitando
destinos ocupados o reservados. El decodificador obtiene recorridos viales
dirigidos y se asume que todos terminan al siguiente tick de **cinco minutos**.
Solo se mide al final, sin simular tráfico ni lecturas durante el trayecto.

## Preparación

Se necesitan Python 3.14 o posterior, [uv](https://docs.astral.sh/uv/) y el
repositorio `lima-pm25-core` versión `0.1.0` en `../lima-pm25-core`.

```bash
uv sync --locked
```

PyTorch es una dependencia obligatoria y se instala con la misma sincronización;
no hace falta seleccionar dependencias opcionales para usar la red o entrenar.
Tras cambiar las dependencias de `pyproject.toml`, regenerar `uv.lock` con
`uv lock` antes de usar `--locked`. Sin `--upgrade`, uv conserva las versiones
bloqueadas que siguen siendo compatibles.

En Linux, PyTorch de PyPI incluye soporte NVIDIA/CUDA; el servidor debe tener
un controlador compatible. No se utiliza ni modifica el PyTorch global. El
entorno, la red y sus dependencias se resuelven en la `.venv` de este proyecto.

El núcleo es una dependencia **editable**: se importa como `lima_pm25` y sus
cambios de código se reflejan sin copiarlo. Este proyecto tiene su propio
`uv.lock`. Sus recursos geográficos y científicos se cargan explícitamente;
instalar la dependencia no incorpora `config/`, `data/` ni `resultados/`.

## Demostración corta, sin entrenamiento

Desde la raíz de este proyecto:

```bash
uv run --locked lima-pm25-rl --policy greedy --steps 3
uv run --locked lima-pm25-rl --policy random --steps 3 --noise 1
uv run --locked lima-pm25-rl --policy sweep --steps 3 --image cache/entorno.png
uv run --locked lima-pm25-rl --policy neural --steps 3
```

`--core-root` permite indicar otra ubicación de los recursos conservados.
La carga no descarga mapas ni escribe en el repositorio vecino. La imagen
es opcional y su destino no se sobrescribe.

La demostración usa el escenario sintético ilustrativo con evento de `core`;
`--scenario control` permite su alternativa sin el evento. No son observaciones
de Lima ni un currículo definitivo. La opción `neural` utiliza una **red sin
entrenar**, únicamente para comprobar la integración.

## Componentes

- `LimaPM25Env`: `reset()`, `step()`, observaciones, máscaras, recompensa y render.
- `RoadRouter`: A* sobre vías dirigidas, incluidas celdas sin intersecciones OSM.
- `GeneratedEpisodes` / `HistoricalReplay`: adaptación de escenarios de `core`.
- `BaselinePolicy`: aleatoria, greedy por incertidumbre y barrido geométrico.
- `BiLGD3QN` / `NeuralController`: CNN, Bi-LSTM, Bi-GRU y cabeza dueling;
  selección central secuencial con memoria histórica y máscaras.
- `double_dqn_loss`: objetivo Double DQN y pérdida Smooth L1 para comprobar una
  actualización de la red. No inicia entrenamiento automáticamente.

La [especificación del entorno](docs/environment.md) explica acciones,
observaciones, recompensa, finalización, escenarios y límites del replay.

## Entrenamiento

El entrenador mínimo utiliza PyTorch directamente: memoria temporal, exploración
enmascarada, Adam, pérdida Double DQN y actualización periódica de la red objetivo.
No añade frameworks de aprendizaje ni inicia entrenamientos al importar módulos.

Piloto artificial pequeño para comprobar que realmente puede aprender:

```bash
uv run --locked lima-pm25-train \
  --config config/training-pilot.json --output cache/pilot-01
```

Comprobación corta sobre el dominio de Lima: dos episodios de tres ticks,
no un entrenamiento experimental suficiente para obtener una buena política:

```bash
uv run --locked lima-pm25-train \
  --config config/training-lima-smoke.json --output cache/lima-check
```

Cada ejecución exige un directorio nuevo y conserva `config.json`, `metrics.csv`,
`checkpoint-last.pt` y `policy-last.pt`. Con validación reservada también guarda
`policy-best.pt`, seleccionada por incertidumbre final, no por replay histórico.

Para continuar, usar el mismo protocolo y un nuevo directorio. `--episodes`
indica el **total deseado**, no episodios adicionales:

```bash
uv run --locked lima-pm25-train \
  --config config/training-pilot.json --resume cache/pilot-01/checkpoint-last.pt \
  --episodes 200 --output cache/pilot-continued
```

Para utilizar una política guardada sobre la misma grilla y configuración del GP:

```bash
uv run --locked lima-pm25-rl --policy neural \
  --checkpoint cache/lima-check/policy-last.pt --steps 3
```

El piloto controlado no representa Lima ni prueba generalización. El currículo
y las particiones definitivas deben fijarse antes del entrenamiento largo.
La [guía de entrenamiento](docs/training.md) detalla esa configuración, la
reanudación, las subdecisiones y las verificaciones realizadas. No se ha
ejecutado el entrenamiento experimental largo ni la evaluación final de la tesis.

### Preparación del servidor

La [guía del servidor](docs/server.md) incluye el protocolo temporal inicial,
listas explícitas de transferencia y comandos de instalación, prueba y
reanudación. `config/training-server.json` mezcla control sintético y estrés
con meteorología generada; **por defecto solo ejecuta cuatro episodios**.
El control no se presenta como un nominal de Lima validado.

Para comprobar recursos y componer todas las familias sin simular ni entrenar:

```bash
uv run --locked lima-pm25-train \
  --config config/training-server.json --check
```

La ejecución larga requiere fijar y autorizar su presupuesto por separado.

## Verificación

```bash
uv run --locked pytest
uv build
```

Las pruebas usan dominios pequeños en memoria, comprueban el contrato de
Gymnasium, la integración con `core`, las rutas y reservas, el tratamiento
temporal del GP, las particiones y la red. La comprobación CUDA se ejecuta
solo cuando hay una GPU compatible; no es un entrenamiento largo.
