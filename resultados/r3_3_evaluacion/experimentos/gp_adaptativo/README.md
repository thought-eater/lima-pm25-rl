# Diagnóstico del GP adaptativo

Ejecución completa de la comprobación:
[`2026-11-17_04-50-30/summary.json`](2026-11-17_04-50-30/summary.json).
Entradas y configuración:
[`config.json`](2026-11-17_04-50-30/config.json).

Se reconstruyeron doce casos **de entrenamiento**, con las mismas lecturas y
rutas del diagnóstico espacial conservado. El control de GP fijo reproduce
numéricamente los posteriores originales. La verdad retenida solo puntúa los
posteriores; no se entrega al ajuste ni a la recompensa.

| Métrica | GP fijo | GP adaptativo |
| --- | ---: | ---: |
| RMSE global temporal (µg/m³) | 4,3994 | 4,2731 |
| RMSE global final (µg/m³) | 4,7907 | 4,6540 |
| RMSE temporal en focos (µg/m³) | 13,5366 | 13,5849 |
| Cobertura empírica del intervalo del 95 % | 99,32 % | 93,04 % |

Cada caso tiene el mismo peso; el RMSE se obtiene después de promediar MSE.
La cobertura promedia ticks posteriores a t=0 y después casos. Los focos usan
el umbral ya fijado de 24,555472595740433 µg/m³.

El RMSE global temporal mejora aproximadamente 2,87 %, pero el localizado
empeora 0,36 %. La cobertura nominal del 95 % no se cumple uniformemente:
varía aproximadamente entre 87,9 % y 97,4 % según caso. No se demuestra la
meta general del 20 % ni ventaja de una política entrenada con este GP.

## Ajustes y recompensa

- 804 intentos de ajuste en las dos ramas; todos convergieron en estos casos.
- Límites alcanzados en amplitud/espacio/tiempo: 466/403/104 ajustes, respectivamente.
- En 91 ticks el aporte firmado de reducción de varianza fue negativo.
- Sumas diagnósticas: cambio de media 2,3465 y cambio de desviación 2,3381.
  El peso 0,5 de incertidumbre aporta aproximadamente la mitad del término
  de media sobre estas rutas; no se ajustaron pesos buscando mejorar resultados.

Se mantienen los límites 200–4000 m, 300–21600 s y amplitud 0,25–2 veces σ₀
como regularización experimental. La frecuencia de soluciones de borde impide
interpretarlas como escalas físicas identificadas. La mejora global pequeña y
los deterioros locales no justifican reemplazar automáticamente el GP principal.
Los archivos `case-*.json` conservan parámetros, verosimilitud, límites,
tiempos y resultados favorables y desfavorables.

## Comprobación integrada y coste

Se ejecutaron **cuatro episodios cortos de entrenamiento**, 144 ticks físicos
y 36 actualizaciones Adam en total, con cero acciones inválidas. Los primeros
tres llenaron el replay hasta el calentamiento configurado; el cuarto incluyó
36 actualizaciones de la red.

En esta CPU, con BLAS y PyTorch a un hilo, el cuarto episodio tardó **18,56 s**:
1,44 s en entorno/planificación y 17,12 s en optimización de la red. Los ajustes
GP y sus predicciones durante ticks sumaron 0,71 s, incluidos ambos posteriores
y la referencia solo fijas. Véase
[`integrated-episode-4.json`](2026-11-17_04-50-30/integrated-episode-4.json).
Una extrapolación lineal de ese único episodio daría **103 h de entrenamiento
en esta CPU**, sin validación ni checkpoints. No es una estimación medida del
servidor con GPU ni incluye la variación de coste entre misiones.

La ejecución preliminar
[`2026-11-17_04-47-20/`](2026-11-17_04-47-20/) conserva la misma reconstrucción
y un solo episodio de calentamiento. Su extrapolación de 18,46 h **no incluye
Adam** y no debe usarse como coste del entrenamiento completo. Se conserva
para explicar la corrección de la medición.

La corrida posterior de 20 000 episodios está conservada en
[entrenamiento adaptativo](../../../r3_2_entrenamiento/adaptativo/README.md).
Este diagnóstico sigue siendo una comprobación corta. Procedimiento reproducible:
`lima-pm25-rl experimento gp-adaptativo`, según
[REPRODUCIR.md](../../../../REPRODUCIR.md).
