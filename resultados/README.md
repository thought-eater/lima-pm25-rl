# Evidencia del objetivo específico O3

Este índice relaciona cada resultado con los medios de verificación y los
indicadores declarados en la tesis. Los datos y procedimientos permiten examinar
la evidencia técnica; la validación académica del experto no se sustituye por
un resultado de pytest.

| Resultado | Medio de verificación | IOV y evidencia |
| --- | --- | --- |
| [R3.1 — Formulación](r3_1_formulacion/README.md) | Especificación del entorno, MDP y recompensa | Estado, acciones, restricciones, presupuesto, término y recompensa explícitos; código y pruebas de contratos. Validación del experto: corresponde al documento académico. |
| [R3.2 — Agente](r3_2_entrenamiento/README.md) | Código, modelos, configuración y bitácora | Curvas de recompensa, pérdida e incertidumbre; semillas, contadores y movimientos inválidos registrados. |
| [R3.3 — Evaluación](r3_3_evaluacion/README.md) | Tablas, gráficos y procedimientos | Comparación con aleatoria, barrido y greedy; incertidumbre, reducción porcentual, cobertura, distancia y errores. El 20 % general no se declara alcanzado. |
| [R3.4 — Visualización](r3_4_visualizacion/README.md) | Código, informes, mapas y capturas | Concentración, incertidumbre, estaciones, trayectorias, muestras y métricas. |

Los procedimientos se describen en [REPRODUCIR.md](../REPRODUCIR.md).
`config/productos.json` identifica las carpetas exactas utilizadas. El archivo
[origenes.csv](origenes.csv) documenta la organización de los productos originales;
renombrar o trasladar una ejecución no significa volver a ejecutarla.

Las carpetas generadas utilizan fecha/hora UTC hasta segundos. Los artefactos
conservados pueden contener rutas de procedencia originales y firmas históricas;
no son requisitos para instalar una copia en otra máquina. La configuración
actual utiliza rutas relativas.

Las matrices sintéticas completas y los posteriores numéricos forman parte
del evaluador. Su presencia en esta evidencia no autoriza usarlas como entradas
del agente. Los resultados no describen un despliegue físico.
