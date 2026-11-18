# Adaptativo: mejor 28 400 y final 30 000

Ambos RMSE se reprodujeron desde los pesos antes de evaluar. Se completaron
156 misiones por política: doce casos, cinco réplicas aleatorias y greedy, barrido,
neural y una referencia solo fijas independiente. Las referencias de ambas
evaluaciones coinciden. La política 28 400 fue seleccionada en estos casos de
validación; no se presenta como una prueba independiente.

| Indicador | Mejor 28 400 | Final 30 000 |
| --- | ---: | ---: |
| RMSE global temporal (µg/m³) | 4,5518 | 4,5856 |
| RMSE global final (µg/m³) | 4,8317 | 5,4295 |
| RMSE en focos temporal (µg/m³) | 12,3235 | 12,5317 |
| RMSE en focos final (µg/m³) | 10,5342 | 10,5092 |
| Déficit local temporal (µg/m³) | 8,4411 | 8,6078 |
| Sensibilidad temporal | 24,56 % | 23,80 % |
| Falsa área alta media (km²) | 6,9725 | 8,7293 |
| Recorrido de flota (km) | 72,0980 | 52,2433 |
| Eventos completados omitidos | 77,81 % | 80,03 % |

El mejor reduce RMSE global final frente a solo fijas (12,52 %), aleatoria
(10,67 %), greedy (11,48 %) y barrido (10,29 %). Frente a greedy, el beneficio
temporal global es solo 0,53 % y el recorrido baja 2,88 %. Su incertidumbre final
es mayor que la de todas las referencias adaptativas: no equivale a error real.
La menor varianza del final tampoco implica mejor reconstrucción.

No hay superioridad uniforme ni cumplimiento general del 20 %. El mejor tiene
más recorrido que el final; el final tiene RMSE localizado final ligeramente
menor. Se conservan todos los casos y deterioros.

## Evidencia

- [Reproducción de RMSE y auditorías numéricas](../adaptativo_verificacion/2026-11-18_18-50-50/).
- [Misiones, mediciones y rutas del mejor](../politicas/adaptativo-mejor/2026-11-18_18-51-03/misiones/).
- [Misiones, mediciones y rutas del final](../politicas/adaptativo-30000/2026-11-18_18-51-03/misiones/).
- [Métricas localizadas y posteriores del mejor](../focos/adaptativo-mejor/2026-11-18_18-59-57/).
- [Métricas localizadas y posteriores del final](../focos/adaptativo-30000/2026-11-18_19-01-28/).
- [Informe completo con curvas 1–30 000 y comparaciones](../../r3_4_visualizacion/informes/2026-11-18_19-04-15/index.html).
- [Visor Folium de ambos modelos](../../r3_4_visualizacion/mapas/2026-11-18_19-04-15/index.html).

Se verificó el mismo campo sintético y la misma información fija. El cotejo de
coordenadas admite hasta 10⁻⁶ m de redondeo de proyección; valores, ruido, reloj
y procedencia coinciden. No se alteraron las lecturas originales. Las primeras
salidas `focos/` dentro de las carpetas de misiones quedaron interrumpidas por
el cotejo binario de una diferencia de 1,16×10⁻¹⁰ m; las salidas completas son
las enlazadas arriba. Se reutilizaron las misiones ya guardadas.

## Modelos que deben conservarse

En la corrida `2026-11-18_05-28-17`, preservar los pares de política/checkpoint
de 28 400 y 30 000, `policy-last.pt`, `checkpoint-last.pt` y `best-rmse.json`,
además de configuraciones, CSV y registros de validación. Conservar también la
corrida de 20 000 que originó la continuación y los productos previos utilizados
en los contrastes. La limpieza de otros checkpoints requiere autorización expresa.
