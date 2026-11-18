# R3.4 — Visualización de entrenamiento y misiones

**[Abrir el visor Folium del mejor y el final](mapas/2026-11-18_19-04-15/index.html)** ·
[Mejor 28 400](mapas/2026-11-18_19-04-15/captura-mejor-tick-36.png) ·
[Final 30 000](mapas/2026-11-18_19-04-15/captura-final-tick-36.png).

El informe HTML reúne curvas de recompensa, pérdida, validación, recorrido,
epsilon y actualizaciones; tablas de evaluación, puntos pareados por caso y
mapas de concentración, incertidumbre, estaciones, muestras y trayectorias.

**[Abrir el informe completo conservado](informes/2026-11-18_19-04-15/index.html)**:
incluye entrenamiento hasta 30 000, ambos descuentos, duración, mejor frente a
final y contraste entre métodos, con sus propias referencias y mapas.

```bash
uv run --locked lima-pm25-rl visualizar
uv run --locked lima-pm25-rl informe
```

La primera llamada genera el visor Folium de cuatro mapas; la segunda genera un
informe de curvas y comparaciones con los productos seleccionados. Abrir
`index.html` en un navegador, conservando
su carpeta completa. PNG y SVG permiten incorporar las figuras al documento.

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
