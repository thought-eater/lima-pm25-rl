# R3.2 — Agente implementado y entrenado

## Productos conservados

- [Gamma 0,99](gamma099/): entrenamiento original de 1000 episodios y continuación hasta
  2000. La continuación conserva online, target, Adam, replay y estados aleatorios.
- [Gamma 0,5](gamma05/): entrenamiento desde cero hasta 2000 episodios, con inicialización
  y secuencia de escenarios emparejadas.
- [Controles BiLG](controles_bilg/): controles de 200 episodios con semillas 42/43/44 utilizados
  por los experimentos de historia, recompensa y diagnóstico GP.

Cada ejecución conserva configuración, `metrics.csv`, políticas y checkpoint
completo. Los nombres `initial`, `best` y `last` tienen funciones distintas:
la comparación de duración/descuento utiliza finales al mismo presupuesto.

## Verificación e interpretación

| Ejecución | Episodios | Ticks físicos acumulados | Actualizaciones Adam | Acciones inválidas |
| --- | ---: | ---: | ---: | ---: |
| Gamma 0,99 original | 1000 | 35984 | 35876 | 0 |
| Gamma 0,99 continuado | 2000 | 71984 | 71876 | 0 |
| Gamma 0,5 | 2000 | 71979 | 71871 | 0 |

Los ticks pueden diferir por terminaciones anticipadas; no se fabrican
transiciones para igualarlos. Las curvas originales y suavizadas, epsilon,
actualizaciones y validación se presentan en el informe de R3.4.

La pérdida cae marcadamente al inicio y después se mantiene con oscilaciones.
La validación también oscila; una meseta o una pendiente de pérdida no demuestra
convergencia ni superioridad operacional. El IOV debe discutirse con las curvas
completas y con la evaluación de R3.3.

Reproducción: `lima-pm25-rl entrenar principal` y
`lima-pm25-rl entrenar gamma05`, según [REPRODUCIR.md](../../REPRODUCIR.md).

## Candidato pendiente de entrenamiento completo

`config/adaptativo.json` prepara BiLG-D3QN desde cero con gamma 0,99, GP
ajustable y recompensa de cambio del modelo para **20 000 episodios totales**.
El diagnóstico corto de software/coste no sustituye esa corrida. Todavía no
hay un producto `adaptativo` registrado ni una política final de ese presupuesto.
Protocolo y comando: [REPRODUCIR.md](../../REPRODUCIR.md).
