# Entrenamiento adaptativo y conservación de modelos

La corrida [2026-11-17_06-01-32](2026-11-17_06-01-32/) completó **20 000 episodios**
desde cero, 719 968 ticks y 719 860 actualizaciones Adam, sin acciones inválidas.
Los pesos del checkpoint final coinciden con `policy-last.pt` y el corte 20 000.

El RMSE global final de validación fue **5,141073 µg/m³**. El menor valor registrado
fue **4,832575 µg/m³ en el episodio 19 600**, pero sus pesos no se conservaron:
el guardado anterior retenía solo los cortes prefijados y sobrescribía el último
checkpoint completo. Ese récord es una métrica histórica, no una política disponible.
La [evaluación posterior de 28 400 y 30 000](../../r3_3_evaluacion/adaptativo/README.md)
ya está completa, incluidos errores globales/localizados y referencias.

La corrida `2026-11-18_05-28-17/` completó la continuación hasta **30 000 episodios**.
Su mejor es 28 400 (RMSE 4,831652), con pesos y checkpoint conservados; el final
obtuvo 5,429510. `config/adaptativo-continuacion.json` permanece como especificación
histórica, no como perfil público de ejecución.

La corrida recibida utilizó esta retención masiva:

- `policy-episode-NNNN.pt` en cada episodio, incluidos el inicio del tramo y el final.
- `checkpoint-episode-NNNN.pt` cada 50 episodios, en cada validación y al terminar.
- `validation-episode-NNNN.json`, con casos, métricas y rutas de ambos archivos.
- `best-rmse.json`, que identifica el mejor par recuperable. No es solo el valor:
  los dos archivos `.pt` señalados existen y se mantienen aunque haya otro mejor.
- `checkpoint-last.pt`, copia del último checkpoint, y `policy-last.pt` al finalizar.

Para corridas nuevas, el código conserva mejor por RMSE, final y un último
checkpoint rotativo cada 50 episodios, con una ocupación estable aproximada de
0,82 GB. Los resultados originales se mantienen hasta autorizar expresamente
la limpieza; se protegen los modelos usados para evaluación y discusión.
Con `entrenar adaptativo --reanudar RUTA`, una carpeta selecciona el último estado
completo y un archivo selecciona ese checkpoint exacto. Para partir del mejor de
esta corrida, el archivo es `2026-11-18_05-28-17/checkpoint-episode-28400.pt`.
Véase [REPRODUCIR.md](../../../REPRODUCIR.md).

La limpieza del 18 de noviembre se ejecutó con autorización explícita: se retiraron
9950 políticas de episodios sin validación y 150 checkpoints intermedios.
Se conservaron los 104 archivos `.pt` que incluyen todos los estados de validación,
mejor, final y últimos, además de CSV, JSON, gráficos y archivos de transferencia.
