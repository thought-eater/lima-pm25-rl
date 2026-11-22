# R3.1 — Formulación del problema

## Resultado

Un controlador centralizado selecciona movimientos para tres sensores móviles
iguales sobre una red vial dirigida. Cada ciclo tiene tres asignaciones
secuenciales de destinos y un único avance físico de 300 s.

| Elemento | Definición e implementación |
| --- | --- |
| Estado observable | Mapas GP de concentración e incertidumbre, soporte espacial, posiciones, reservas, vehículos pendientes e historial causal. `env.py`, `policies.py`. |
| Acción | Par vehículo–dirección codificado como `8 * vehículo + dirección`; ocho celdas vecinas y máscaras de conectividad/reserva. `routing.py`, `policies.py`. |
| Transición | Una misión conjunta por tick; mediciones móviles al final, actualizaciones simultáneas del GP. `simulation.py`, `env.py`. |
| Recompensa | Ganancia de varianza GP ponderada por concentración previa, menos distancia vial. Comparación antes/después en el mismo instante, sin atribuir nuevas fijas a los móviles. `env.py`. |
| Presupuesto y término | Tiempo restante del episodio; 36 ticks nominales. Término anticipado si la flota no tiene acciones válidas. |
| Aprendizaje | Double DQN dueling con CNN, Bi-LSTM y Bi-GRU. Descuento 1 entre asignaciones y gamma una vez por tick. `network.py`, `replay.py`, `training.py`. |

## Verificación

Las pruebas de software examinan máscaras, reservas, transiciones conjuntas,
recompensas, causalidad del historial y aislamiento de la verdad sintética.
Las comprobaciones de tesis enlazan la configuración publicada y sus productos.

El GP resume un historial limitado y el agente utiliza observaciones parciales;
la formulación es una representación operacional para planificación, no una
afirmación de observabilidad completa del proceso atmosférico.
La aprobación de la formulación por el experto se documenta en la tesis.

El [candidato adaptativo](adaptativo.md) incorpora ajuste GP y recompensa de
cambio del modelo. Su protocolo y diagnóstico se presentan por separado de
la formulación utilizada en los entrenamientos conservados de 2000 episodios.
