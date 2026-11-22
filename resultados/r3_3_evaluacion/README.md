# R3.3 — Evaluación comparativa y experimentos

## Comparación principal

Las políticas se evaluaron en doce casos sintéticos retenidos, con escenarios,
lecturas fijas, posiciones iniciales y horizontes comunes. Aleatoria y greedy
incluyen cinco réplicas de decisiones por caso. Se promedian réplicas dentro
de caso y después casos; RMSE es raíz del MSE agregado.

| Política con GP fijo | RMSE global promedio | RMSE global final | Sensibilidad temporal | Falsa área alta media (km²) | Recorrido (km) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gamma 0,99, 1000 episodios | 4,951 | 5,400 | 13,56 % | 4,267 | 83,78 |
| Gamma 0,99, 2000 episodios | 4,885 | 5,096 | 11,16 % | 2,727 | 66,14 |
| Gamma 0,5, 2000 episodios | 4,846 | 5,126 | 11,49 % | 3,283 | 65,20 |
| Gamma 0,99, 30 000 episodios (mejor y final) | 4,737 | 4,548 | 15,44 % | 3,170 | 72,52 |
| Aleatoria | 4,918 | 5,312 | 12,60 % | 2,963 | 49,30 |
| Greedy | 4,827 | 5,040 | 12,35 % | 2,388 | 84,05 |
| Barrido | 4,889 | 5,234 | 13,16 % | 2,624 | 45,26 |
| Solo fijas (sin flota) | 5,176 | 5,536 | 5,87 % | 1,739 | 0,00 |

Errores en µg/m³. Las cifras completas, casos y diferencias están en las carpetas
`duracion/` y `descuento/`. `politicas/` conserva evaluaciones anteriores y las
lecturas/rutas originales disponibles. El inventario exacto es
[`config/productos.json`](../../config/productos.json).

Para el fijo a 30 000, una sola evaluación respalda el mejor y el final:
[misiones y rutas](politicas/fijo-30000/2026-11-20_11-01-46/misiones/),
[métricas por caso](politicas/fijo-30000/2026-11-20_11-01-46/focos/cases.csv)
y [contrastes agregados](politicas/fijo-30000/2026-11-20_11-01-46/focos/references.csv).
Frente a las tres estrategias móviles, las reducciones medias fueron de 2,89 %
en RMSE global promedio, 12,40 % en RMSE global final y 5,87 % en varianza
integrada final. Frente a solo fijas fueron de 8,49 %, 17,84 % y 19,11 %.
La sensibilidad aumentó, pero también el área media clasificada como alta por
error. Se omitió el 82,88 % de los eventos por celda completados (promedio
entre casos). Son resultados sobre los casos de selección, no prueba independiente.

La [comparación a 30 000 episodios](../r3_4_visualizacion/informes/2026-11-20_22-57-54/index.html)
reúne las tres políticas, los intervalos descriptivos y los resultados de focos y movilidad.
En el contraste de finales, el fijo tuvo menor RMSE global final en 8/12 casos,
pero menor RMSE global promedio en solo 3/12 y recorrió 20,27 km más por misión
en promedio. En el contraste de mejores, el agregado final favoreció al fijo
aunque este solo obtuvo menor error final en 4/12 casos.

El umbral de foco es el percentil 90 de doce campos de entrenamiento, sin t=0:
24,555472595740433 µg/m³. Es experimental, no sanitario. Se mantienen déficit,
sensibilidad, falsas alarmas y censura de eventos separados.

## Discusión apoyada por los experimentos

La [auditoría de validación](auditoria_validacion/2026-11-21_00-38-05/)
coteja el campo latente con los intervalos del 95 % y el CRPS del GP, y mide
la velocidad requerida por vehículo y tick sobre las rutas guardadas.
Los extremos de recorrido impiden presentar todos los desplazamientos como
físicamente viables en cinco minutos, aun respetando el sentido de las vías.
Los productos incluyen las referencias de cada estimador y no usan replay histórico.

La [sensibilidad con tope de 30 km/h](experimentos/velocidad/2026-11-21_00-53-40/)
reevaluó las rutas antes de cada decisión. En validación, la reducción media
del RMSE global final frente a las tres referencias móviles pasó de 12,40 %
a 0,13 % en el fijo a 30 000 y de 10,81 % a 8,62 % en el adaptativo 28 400.
El adaptativo final no tuvo ventaja media en ese indicador con ninguna regla.
El tope se aplicó solo en este contraste de movilidad.

Los experimentos complementarios están en [experimentos/](experimentos/):

- **Escenarios, movilidad, llegadas y valores Q:** contraste del campo,
  conectividad vial, restricciones de llegada y relación entre Q y retorno.
- **GP espacial y temporal:** sensibilidad de reconstrucciones sobre entradas
  congeladas; distinguirla del cambio de política en `gp_realimentado/`.
- **Historia y recompensa:** ablaciones con controles BiLG emparejados. No se
  comparan directamente retornos nativos de recompensas diferentes.
- **Focos controlados y GP en focos:** mecanismos artificiales persistentes y
  pulsantes; no constituyen calibración atmosférica de Lima.
- **Mezcla y multiescala:** candidatos estadísticos evaluados con entradas
  comunes. No se adoptan automáticamente por mejorar un subconjunto.
- **[GP adaptativo](experimentos/gp_adaptativo/README.md):** doce reconstrucciones
  de entrenamiento y cuatro episodios cortos para comprobar ajuste, recompensa
  y coste. Mejora global pequeña, deterioro local leve y menor cobertura; el
  entrenamiento llegó a 30 000 episodios. La
  [evaluación completa del mejor y el final](adaptativo/README.md) conserva
  156 misiones por política, resultados localizados, diferencias y mapas.

Los resultados desfavorables se conservan para discutir por qué una modificación
no se incorporó a la configuración principal.

## Estado del IOV y límites

La comparación incluye las tres referencias requeridas, incertidumbre,
porcentajes, cobertura, distancia y errores. **No demuestra una mejora general
del 20 %.** El menor recorrido frente a greedy coexiste con diferencias pequeñas
o desfavorables en otros indicadores.

Los descuentos producen compromisos entre omisiones, falsas alarmas y coste.
Una sola semilla de entrenamiento limita la generalización. Las evaluaciones
son de validación sintética, no replay histórico final ni evidencia sanitaria.
El replay final de O3 se ejecuta por separado en el servidor mediante
`uv run --locked lima-pm25-rl evaluar replay-historico`. El protocolo fija
doce ventanas antes de abrir sus resultados. Se evalúa contra observaciones
de estación al cierre de la hora; no dispone de una verdad espacial histórica.
El [replay completo](replay_historico/2026-11-22_09-10-33/) y la
[prueba independiente](prueba_independiente/2026-11-21_15-25-58/) sustentan
los contrastes finales; sus `completed.json` registran el cierre de las corridas.
La primera misión histórica se consulta en el [visor integrado](../r3_4_visualizacion/mapas/2026-11-22_11-58-49/index.html).
