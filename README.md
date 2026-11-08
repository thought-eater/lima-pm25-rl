# Lima PM2.5 RL

Proyecto de aprendizaje por refuerzo para la planificación de sensores móviles
de PM2.5, basado en el simulador `lima-pm25-core`.

## Dependencia local

Se requieren Python 3.14 o posterior y [uv](https://docs.astral.sh/uv/).
El repositorio `lima-pm25-core`, con versión de paquete `0.1.0`, debe estar
en `../lima-pm25-core`, junto a este proyecto.

Desde la raíz de `lima-pm25-rl`, preparar su entorno virtual y sus dependencias:

```bash
uv sync --locked
```

El núcleo se instala en modo editable: sus cambios de código se reflejan en
las siguientes ejecuciones sin copiarlo ni reinstalarlo. Se importa mediante
`lima_pm25`, aunque la distribución se llama `lima-pm25-core`.

Este proyecto utiliza su propia `.venv` y su propio `uv.lock`; no reutiliza
los del núcleo. La instalación no incorpora automáticamente los recursos de
`config/`, `data/` o `resultados/` del repositorio vecino.

## Entorno inicial de Gymnasium

`LimaPM25Env`, definido en `src/lima_pm25_rl/env.py`, hereda de `gymnasium.Env`
y compone los tipos existentes del núcleo. Recibe tres argumentos explícitos:

- `domain`: un `SpatialDomain` ya preparado, con grilla, máscara y grafo vial.
- `physics`: un `PM25PhysicsConfig`, sin parámetros físicos implícitos.
- `initial_baseline`: un `AmbientBaseline`, con concentración inicial y procedencia.

Con estos objetos ya preparados, se construye directamente:

```python
from lima_pm25_rl import LimaPM25Env

env = LimaPM25Env(
    domain=domain,
    physics=physics,
    initial_baseline=initial_baseline,
)
env.close()
```

La construcción conserva el dominio sin modificarlo y crea un `PM25Transport`
y un `PM25State` propios. El campo inicial es uniforme sobre las celdas activas,
con cero en las inactivas y tiempo simulado igual a cero. No se avanza la física
ni se descargan mapas o escriben archivos al construir el entorno.

El método interno `_initialize_simulation(seed=...)` prepara el reinicio físico
para el futuro `reset()`: utiliza la inicialización aleatoria de Gymnasium y
crea un campo nuevo. Una semilla explícita reproduce la secuencia aleatoria;
`None` conserva la secuencia existente. La concentración inicial es determinista
para una misma configuración. El campo verdadero permanece en `_state`, sin
devolverse como observación del agente.

**Esta clase todavía no es un entorno entrenable.** `reset()` y `step()` lanzan
`NotImplementedError` sin modificar el estado. Quedan pendientes las observaciones,
acciones, avance temporal, recompensas, condiciones de término, sensores y
estimador. No se declaran espacios provisionales ni modos de visualización,
y no se registra el entorno para `gymnasium.make()` ni se valida con `check_env`.
La grilla del campo no se interpreta como una red de celdas transitables.

## Pruebas

```bash
uv run --locked pytest
uv build
```

Las pruebas usan un dominio sintético pequeño construido en memoria. Comprueban
la composición con el núcleo, la inicialización y el aislamiento de campos y
generadores aleatorios, las semillas y las operaciones aún pendientes. No ejecutan
episodios, descargan cartografía ni modifican los recursos del núcleo.
