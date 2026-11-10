# Lima PM2.5 RL

Entorno de Gymnasium para un agente centralizado que planifica el muestreo de
**tres vehículos** sobre las calles de Lima. Reutiliza el dominio, la física,
los sensores y el proceso gaussiano (GP) de `lima-pm25-core`.

Cada ciclo elige tres destinos en las ocho direcciones de la grilla, evitando
destinos ocupados o reservados. La red elige el mayor Q válido entre todos los
vehículos pendientes y recalcula después de cada reserva. El decodificador obtiene recorridos viales
dirigidos y se asume que todos terminan al siguiente tick de **cinco minutos**.
Solo se mide al final, sin simular tráfico ni lecturas durante el trayecto.

## Preparación

Se necesitan Python 3.14 o posterior, [uv](https://docs.astral.sh/uv/) y el
repositorio `lima-pm25-core` versión `0.1.0` en `../lima-pm25-core`.

```bash
uv sync
```

PyTorch es una dependencia obligatoria y se instala con la misma sincronización;
no hace falta seleccionar dependencias opcionales para usar la red o entrenar.
`uv.lock` conserva las versiones resueltas del proyecto.

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

Para entrenar:

```bash
uv run lima-pm25-train
```

Lee automáticamente `config/training-server.json` y crea una carpeta nueva en
`cache/training/<fecha UTC>/`, cuya ruta imprime al empezar. El presupuesto se
define con `episodes` en ese JSON; está configurado en 1000 episodios.

La configuración actual incluye contraste espacial, fondo variable, ruido móvil
independiente de 1 µg/m³ y recompensa informativa con prioridad acotada por
concentración y coste vial. Son parámetros experimentales, no valores instrumentales
o de emisiones medidos. La [guía de la revisión](docs/training-v2.md) explica los
cambios y el diagnóstico ejecutado. Las políticas/checkpoints anteriores al formato
2 requieren una ejecución nueva: cambió el contexto de decisión de la red.

Para continuar, después de aumentar el total en el JSON si ya se alcanzó:

```bash
uv run lima-pm25-train --resume
```

Busca en `cache/` el checkpoint más reciente compatible con la configuración y
el protocolo actuales. La continuación también crea una carpeta nueva; nunca
trunca ni sobrescribe ejecuciones anteriores. El total es el objetivo acumulado,
no una cantidad de episodios adicionales.

Solo para elegir un punto específico:

```bash
uv run lima-pm25-train --resume cache/server-check-01/checkpoint-last.pt
```

Cada carpeta conserva `config.json`, `metrics.csv`, `checkpoint-last.pt` y
`policy-last.pt`. Con validación también guarda `policy-best.pt`, seleccionada
por incertidumbre final, no por replay histórico.

`--config` y `--output` siguen disponibles para excepciones. Por ejemplo, los
pilotos de software usan otras configuraciones:

```bash
uv run lima-pm25-train --config config/training-pilot.json
uv run lima-pm25-train --config config/training-lima-smoke.json
```

El piloto controlado no representa Lima ni prueba generalización. Las
configuraciones iniciales no constituyen un currículo nominal definitivo.
La [guía de entrenamiento](docs/training.md) detalla esa configuración, la
reanudación, las subdecisiones y las verificaciones realizadas. Los diagnósticos
de validación no sustituyen la evaluación final independiente de la tesis.

### Preparación del servidor

La [guía del servidor](docs/server.md) incluye el protocolo temporal inicial,
listas explícitas de transferencia y comandos de instalación, prueba y
reanudación. `config/training-server.json` mezcla control sintético y estrés
con meteorología generada y **1000 episodios por defecto**.
El control no se presenta como un nominal de Lima validado.

Para comprobar recursos y componer todas las familias sin simular ni entrenar:

```bash
uv run lima-pm25-train --check
```

La ejecución larga requiere fijar y autorizar su presupuesto por separado.

### Diagnóstico del campo antes de entrenar

```bash
uv run --locked lima-pm25-diagnose
```

Simula en CPU los primeros doce escenarios de entrenamiento y guarda contraste
espacial por tick, variación temporal por celda, lecturas fijas y campos completos
en una carpeta nueva de `cache/diagnostics/`. No necesita GPU ni entrena una red.
Incluye un predictor constante ajustado en seis casos y evaluado en los otros seis.
La [guía del diagnóstico](docs/diagnostics.md) explica las métricas y sus límites.

Para auditar el bloqueo del caso 100002 con los archivos de resultados recibidos:

```bash
uv run --locked lima-pm25-audit-mobility
```

Reproduce la política `best` y registra Q, reservas, recorridos y conectividad,
con salida nueva en `cache/mobility-diagnostics/`. La [auditoría de movilidad](docs/mobility-diagnostic.md)
describe el sumidero encontrado y la alternativa de llegada dentro de la misma celda.

La [comparación de criterios de llegada](docs/arrival-experiment.md) se ejecuta
con `uv run --locked lima-pm25-compare-arrivals`: compara el caso 100002 sin
reentrenar. `--all-cases` amplía a los doce casos y las cuatro políticas móviles.

El [diagnóstico de Q y retornos](docs/value-diagnostic.md) se ejecuta con
`uv run --locked lima-pm25-audit-values --ticks 1` para el piloto, o
`--all-cases` para los doce casos en ticks prefijados. Compara alternativas
legales con continuación congelada y ruido futuro emparejado, guardando
resultados nuevos en `cache/value-diagnostics/`.

## Comparación de estrategias

### Piloto de arquitectura en el servidor

La [ablación BiLG frente a D3QN sin recurrencia](docs/architecture-pilot.md)
conserva la misma ventana histórica y usa tres semillas emparejadas:

```bash
uv run --locked lima-pm25-compare-architectures --check
uv run --locked lima-pm25-compare-architectures
```

El segundo comando ejecuta seis entrenamientos de 200 episodios y sus evaluaciones
en una carpeta nueva de `cache/architecture-experiments/`. Se prepara para el
servidor; no lanza el diagnóstico exhaustivo de alternativas. La comparación
principal utiliza los pesos finales con el mismo presupuesto de episodios.

### Diagnóstico del GP

El [diagnóstico acotado del GP](docs/gp-diagnostic.md) reproduce doce misiones
con BiLG-42 del episodio 200 del piloto, comparando incertidumbre con error y
aislando el aporte de las lecturas móviles en cada tick. Para el servidor:

```bash
uv run --locked lima-pm25-audit-gp --check
uv run --locked lima-pm25-audit-gp
```

Guarda resultados nuevos en `cache/gp-diagnostics/` y verifica que las misiones
reproduzcan las trazas del piloto de arquitectura.

### Política frente a referencias

```bash
uv run lima-pm25-evaluate
```

Compara la política `best` con solo fijas, aleatoria, greedy y barrido sobre los
mismos doce casos de validación. La política y la carpeta de salida se eligen
automáticamente; se imprimen sus rutas. Con las cinco semillas configuradas para
aleatoria y greedy ejecuta 144 misiones de inferencia, **sin entrenar ni abrir
el replay final**.

Guarda métricas por caso y agregadas, diferencias pareadas, lecturas, recorridos
y cinco mapas del mismo caso preseleccionado en `cache/evaluation/`.
La [guía de evaluación](docs/evaluation.md) explica las reglas, el horizonte
común y cómo interpretar los resultados sin confundir validación con prueba final.

## Verificación

```bash
uv run --locked pytest
uv build
```

Las pruebas usan dominios pequeños en memoria, comprueban el contrato de
Gymnasium, la integración con `core`, las rutas y reservas, el tratamiento
temporal del GP, las particiones y la red. La comprobación CUDA se ejecuta
solo cuando hay una GPU compatible; no es un entrenamiento largo.
