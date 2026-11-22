# R3.4 — Visualización de entrenamiento y misiones

**[Abrir el visor integrado: misiones sintéticas y replay SENAMHI](mapas/2026-11-22_11-58-49/index.html)**.

El informe HTML reúne curvas de recompensa, pérdida, validación, recorrido,
epsilon y actualizaciones; tablas de evaluación, puntos pareados por caso y
mapas de concentración, incertidumbre, estaciones, muestras y trayectorias.

**[Abrir el informe a presupuesto común](informes/2026-11-20_22-57-54/index.html)**:
incluye el contraste principal de finales y el secundario de mejores, diferencias
fijo menos adaptativo en los doce casos, intervalos descriptivos y evolución
del RMSE por instante.

```bash
uv run --locked lima-pm25-rl visualizar
uv run --locked lima-pm25-rl informe
```

La primera llamada indica dónde abrir el visor publicado; la segunda reproduce el
informe en `cache/`. PNG y SVG del informe permiten incorporar las figuras al documento.

El visor permite reproducir y pausar una misión, cambiar de instante, sincronizar
zoom/encuadre y consultar una celda en todos los mapas. Presenta recorrido,
estimación (o incertidumbre), verdad sintética y error real menos estimado.
Rojo significa subestimación; azul, sobreestimación. Las calles OSM conservadas
sirven de fondo y las bibliotecas habituales de Folium se cargan por CDN.
Véase [config/visualizacion.json](../../config/visualizacion.json).

## Lectura y verificación visual

- Se conserva la serie original de episodios y una media retrospectiva de 50.
- Las reanudaciones están marcadas; no reinician la numeración ni los contadores.
- La validación muestra solo instantes registrados.
- Los bloques descriptivos son de 200 episodios.
- La pérdida usa escala logarítmica con zona lineal cerca de cero; ausencias
  de warmup no se transforman en ceros.
- La comparación incluye todos los casos; los mapas usan el caso prefijado
  100002 y los ticks 1/12/24/36, con escalas comunes.
- En las figuras estáticas, la trayectoria completa se muestra al final. El visor
  Folium recorre los 37 instantes registrados, incluido t=0, y muestra solo la
  trayectoria disponible en cada instante.

Las verificaciones de presentación comprueban los enlaces locales y la presencia
de figuras/tablas. Las imágenes conservadas y las curvas generadas se inspeccionan
para comprobar ejes, unidades, leyendas y correspondencia con los resultados.
El visualizador complementa la explicación principal de la tesis.
