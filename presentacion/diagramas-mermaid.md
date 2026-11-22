# Diagramas editables de O3

Fuentes Mermaid para las figuras de R3.2. Los gráficos numéricos de entrenamiento
y validación se generan por separado con `lima-pm25-rl informe`.

## Figura 25. Selección centralizada de movimientos de la flota

```mermaid
flowchart TD
    subgraph Net["Una sola red BiLG-D3QN, compartida por los tres vehículos"]
        M["4 observaciones de la misma misión<br/>7 mapas de 59 × 78 por observación"] --> C["CNN aplicada a cada observación<br/>Canales: 7 → 32 → 32 → 64<br/>Promedio 3 × 3 y proyección: 576 → 128"]
        C --> X["Secuencia de 4 vectores de 155 componentes"]
        A["27 variables auxiliares por observación"] --> X
        X --> L["Bi-LSTM → Bi-GRU<br/>64 unidades por sentido en cada capa"]
        L --> H["Resumen temporal<br/>128 componentes"]
        H --> V["Rama de valor<br/>128 → 64 → 1"]
        H --> D["Rama de ventaja<br/>128 → 64 → 8"]
        V --> Q["8 valores Q para la perspectiva consultada"]
        D --> Q
    end
    Q --> S["Comparar los pares vehículo–dirección admisibles"]
    S --> R["Seleccionar el mayor Q y reservar el destino"]
    R -->|"Algún vehículo pendiente tiene un movimiento válido"| P["Actualizar perspectiva, reservas y máscara<br/>Sin avanzar el reloj ni obtener lecturas"]
    P --> M
    P --> A
    R -->|"No quedan pares válidos pendientes"| T["Ejecutar los movimientos asignados<br/>Los vehículos sin opción permanecen en su sitio<br/>Un intervalo físico de 300 s"]
```

La historia se amplía una vez por intervalo físico, no al reservar un destino.
La red puede consultar el mismo historial desde otra perspectiva; las nuevas
reservas y la máscara se aplican a la decisión actual.
La selección del mayor Q representa la explotación; durante el entrenamiento
también se seleccionan pares válidos al azar con probabilidad ε.

## Figura 26. Flujo de interacción con el entorno y actualización del agente BiLG-D3QN

```mermaid
flowchart TD
    I["Iniciar misión: escenario, posiciones y observación inicial"] --> A["Asignar destinos válidos con la red en aprendizaje<br/>Explorar entre pares admisibles con probabilidad ε"]
    A --> B["Avanzar un intervalo de 300 s<br/>Medir, actualizar el GP y obtener recompensa conjunta"]
    B --> C["Registrar observaciones, asignaciones y recompensa"]
    C -->|"Queda misión"| A
    C -->|"Fin de la misión"| D["Guardar el episodio completo en memoria"]
    D --> K{"¿Hay al menos 128 pasos físicos acumulados<br/>y un lote de 16 decisiones disponibles?"}
    K -->|"No"| I
    K -->|"Sí"| E["Muestrear 16 decisiones y reconstruir sus historias causales"]
    E --> F["Construir el objetivo Double DQN<br/>La red en aprendizaje selecciona; la red objetivo evalúa"]
    E --> Q["Estimar Q de la acción registrada<br/>Red en aprendizaje"]
    F --> G["Comparar predicción y objetivo<br/>Pérdida Smooth L1"]
    Q --> G
    G --> H["Actualizar la red en aprendizaje mediante Adam"]
    H -. "Cada 200 actualizaciones del optimizador" .-> J["Copiar parámetros a la red objetivo"]
    H -->|"Quedan actualizaciones del episodio"| E
    H -->|"Finaliza el bloque de optimización"| I
```

Se optimiza después de incorporar el episodio, si se cumplen el mínimo de
experiencia y el tamaño de lote. En las asignaciones intermedias se usa
recompensa cero y descuento unitario. La última asignación aprendible del
intervalo recibe la recompensa conjunta; no se añade un valor futuro al término
de la misión.
