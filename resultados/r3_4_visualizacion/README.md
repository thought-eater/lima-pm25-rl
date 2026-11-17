# R3.4 — Visualización de entrenamiento y misiones

El informe HTML reúne curvas de recompensa, pérdida, validación, recorrido,
epsilon y actualizaciones; tablas de evaluación, puntos pareados por caso y
mapas de concentración, incertidumbre, estaciones, muestras y trayectorias.

**[Abrir el informe completo conservado](informes/2026-11-17_03-41-33/index.html)**:
incluye entrenamiento 1–2000, ambos descuentos, comparación de duración,
referencias y mapas.

```bash
uv run --locked lima-pm25-rl visualizar
uv run --locked lima-pm25-rl informe
```

La primera llamada indica el informe conservado; la segunda genera uno nuevo
con los productos seleccionados. Abrir `index.html` en un navegador, conservando
su carpeta completa. PNG y SVG permiten incorporar las figuras al documento.

## Lectura y verificación visual

- Se conserva la serie original de episodios y una media retrospectiva de 50.
- Las reanudaciones están marcadas; no reinician la numeración ni los contadores.
- La validación muestra solo instantes registrados.
- Los bloques descriptivos son de 200 episodios.
- La pérdida usa escala logarítmica con zona lineal cerca de cero; ausencias
  de warmup no se transforman en ceros.
- La comparación incluye todos los casos; los mapas usan el caso prefijado
  100002 y los ticks 1/12/24/36, con escalas comunes.
- La trayectoria completa se muestra al final de la misión. Los cuatro mapas
  temporales no se presentan como una animación de todos los ticks.

Las verificaciones de presentación comprueban los enlaces locales y la presencia
de figuras/tablas. Las imágenes conservadas y las curvas generadas se inspeccionan
para comprobar ejes, unidades, leyendas y correspondencia con los resultados.
El visualizador complementa la explicación principal de la tesis.
