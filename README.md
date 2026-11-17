# Lima PM2.5 — planificación de sensores móviles

Este repositorio presenta el código y la evidencia del **objetivo específico O3**
de la tesis: planificar ubicaciones dinámicas de tres sensores móviles para reducir
la incertidumbre de mapas urbanos de PM2.5. Reutiliza el simulador y estimador de
[`lima-pm25-core`](../lima-pm25-core/).

## Resultados y medios de verificación

| Resultado | Producto examinable |
| --- | --- |
| R3.1 — Formulación del MDP y recompensa | [Especificación y correspondencia con el código](resultados/r3_1_formulacion/README.md) |
| R3.2 — Agente implementado y entrenado | [Modelos, configuraciones y bitácoras](resultados/r3_2_entrenamiento/README.md) |
| R3.3 — Evaluación frente a referencias | [Métricas, experimentos y limitaciones](resultados/r3_3_evaluacion/README.md) |
| R3.4 — Visualización | [Informes, mapas y curvas](resultados/r3_4_visualizacion/README.md) |

El [índice de evidencia e indicadores](resultados/README.md) relaciona cada IOV con
los productos conservados. La explicación principal y la discusión se desarrollan
en el documento de tesis; este repositorio facilita su inspección y reproducción.

## Instalación y procedimientos

Se requieren Python 3.14, `uv` y ambos repositorios vecinos:

```text
directorio/
├── lima-pm25-core/
└── lima-pm25-rl/
```

```bash
uv sync --locked
uv run --locked lima-pm25-rl --help
uv run --locked lima-pm25-rl visualizar
```

Hay una sola entrada pública: **`lima-pm25-rl`**. Sus subcomandos son `entrenar`,
`evaluar`, `comparar`, `experimento`, `informe` y `visualizar`. Los procedimientos,
entradas y costes se explican en [REPRODUCIR.md](REPRODUCIR.md).
Los perfiles de entrenamiento declaran CUDA en el JSON y se detienen si no está
disponible. El progreso muestra el mejor RMSE de validación conocido y su episodio.

## Organización

```text
src/lima_pm25_rl/              entorno, red, entrenamiento y evaluación
src/lima_pm25_rl/experiments/  procedimientos complementarios de la discusión
config/                       parámetros e inventario explícito de productos
resultados/                   evidencia experimental de R3.1–R3.4
tests/software/               contratos y regresiones de software
tests/tesis/                  comprobaciones de la evidencia publicada
```

Las entradas conservadas están identificadas en `config/productos.json`, mediante
rutas relativas. Una reproducción escribe en una carpeta nueva con fecha y hora
UTC hasta segundos (`AAAA-MM-DD_HH-MM-SS`); no reemplaza la evidencia publicada.
La selección de productos se hace explícitamente en la configuración.

`docs/` contiene notas de trabajo privadas y se excluye de Git. `cache/` se
reserva a salidas temporales; `dist/`, a los paquetes construidos.

## Alcance

- Un agente centralizado BiLG-D3QN asigna destinos a tres vehículos. Las tres
  elecciones pertenecen a un único tick físico de 300 s.
- La estimación utiliza un GP espacio-temporal. La verdad sintética es privada
  del evaluador y no entra en las observaciones del agente.
- Los resultados sintéticos no demuestran precisión atmosférica real, beneficio
  sanitario ni desempeño de vehículos desplegados en Lima.
- Se presentan mejoras y deterioros. La meta general del 20 % no se declara
  alcanzada por mejorar únicamente la distancia o una métrica localizada.
- El replay histórico final sigue pendiente. El candidato con ajuste GP por
  verosimilitud marginal y recompensa de cambio del modelo está implementado;
  su entrenamiento de 20 000 episodios sigue pendiente. Véase el
  [protocolo adaptativo](resultados/r3_1_formulacion/adaptativo.md).

## Verificación del software

```bash
uv run --locked pytest
uv build
```

Las pruebas de cálculo no equivalen a demostrar los indicadores experimentales.
Véase [tests/README.md](tests/README.md).
