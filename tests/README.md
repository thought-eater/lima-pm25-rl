# Pruebas y evidencia de tesis

`software/` contiene pruebas de cálculo, contratos, causalidad, acciones,
entrenamiento, persistencia, métricas y procedimientos experimentales. Usan
datos pequeños o sintéticos; no ejecutan los entrenamientos publicados.

`tesis/` comprueba la presencia y coherencia de la configuración y los productos
de R3.1–R3.4. Examinar la evidencia conservada no equivale a repetir los
experimentos ni a demostrar un indicador porcentual favorable.

```bash
uv run --locked pytest tests/software
uv run --locked pytest tests/tesis
uv build
```

Las pruebas de la variante Window y las interfaces retiradas no forman parte
de la entrega. Las verificaciones relevantes se mantienen sobre BiLG y la CLI
única. Los totales de pytest son una medida del software, no el denominador
del IOV experimental ni sustituto de la validación del experto.
