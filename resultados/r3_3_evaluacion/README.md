# R3.3 — Evaluación comparativa y experimentos

## Comparación principal

Las políticas se evaluaron en doce casos sintéticos retenidos, con escenarios,
lecturas fijas, posiciones iniciales y horizontes comunes. Aleatoria y greedy
incluyen cinco réplicas de decisiones por caso. Se promedian réplicas dentro
de caso y después casos; RMSE es raíz del MSE agregado.

| Política final | RMSE temporal | RMSE final | Sensibilidad temporal | Falsa área alta media (km²) | Recorrido (km) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gamma 0,99, 1000 episodios | 4,951 | 5,400 | 13,56 % | 4,267 | 83,78 |
| Gamma 0,99, 2000 episodios | 4,885 | 5,096 | 11,16 % | 2,727 | 66,14 |
| Gamma 0,5, 2000 episodios | 4,846 | 5,126 | 11,49 % | 3,283 | 65,20 |
| Aleatoria | 4,918 | 5,312 | 12,60 % | 2,963 | 49,30 |
| Greedy | 4,827 | 5,040 | 12,35 % | 2,388 | 84,05 |
| Barrido | 4,889 | 5,234 | 13,16 % | 2,624 | 45,26 |

Errores en µg/m³. Las cifras completas, casos y diferencias están en las carpetas
`duracion/` y `descuento/`. `politicas/` conserva evaluaciones anteriores y las
lecturas/rutas originales disponibles. El inventario exacto es
[`config/productos.json`](../../config/productos.json).

El umbral de foco es el percentil 90 de doce campos de entrenamiento, sin t=0:
24,555472595740433 µg/m³. Es experimental, no sanitario. Se mantienen déficit,
sensibilidad, falsas alarmas y censura de eventos separados.

## Discusión apoyada por los experimentos

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
El replay final se encuentra pendiente y no se publica una carpeta vacía como
si fuera un resultado realizado.
