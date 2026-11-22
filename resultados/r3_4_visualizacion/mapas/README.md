# Visor Folium de misiones conservadas

**[Abrir el visor final](2026-11-22_11-58-49/index.html)**.

La publicación actual contiene los dos casos prefijados 100002 y 100003 del
fijo (mejor y final 30 000) y del adaptativo: mejor en validación (28 400) y final (30 000). Para
cada modelo/caso permite elegir BiLG-D3QN, greedy y aleatoria (réplica 0),
barrido o solo fijas. Las referencias comunes de ambos adaptativos aparecen una vez.

El mismo selector incluye el replay SENAMHI del 8 de abril de 2026. En este caso,
los dos últimos mapas presentan las observaciones SENAMHI y las diferencias
horarias en sus ubicaciones, no un campo real interpolado. Los destinos que
acaba de elegir la flota se distinguen del recorrido ejecutado. Se puede
reproducir el HTML completo en `cache/` con `visualizar --replay CARPETA`.

Las nuevas salidas separan misión, estrategia y modelo. La ventana ilustrativa
del replay guarda los Q antes de cada reserva para los dos agentes; **Valores Q**
permite inspeccionarlos junto a los destinos. `visualizar --en-vivo` añade
escenarios nuevos a este mismo visor, con los Q de la inferencia actual.

Los cuatro mapas muestran recorrido/sensores, media GP (o desviación estándar),
verdad sintética y error real menos estimado. Rojo representa subestimación y
azul sobreestimación. Tiempo, encuadre y zoom están sincronizados. Al pulsar una
celda se consultan sus valores y se destaca en los cuatro paneles.

Se recorren los 37 instantes registrados de cada misión, de 0 a 180 min. Las
rutas se recortan al extremo correspondiente a cada tick y solo se muestran
mediciones ya recibidas. No se generan posiciones o lecturas posteriores a una
terminación anticipada. Las métricas superiores corresponden al tick elegido.

Se comprobó la coincidencia de campo, configuración, pesos, grilla y reloj, así
como RMSE y distancia por tick contra los archivos originales. Las imágenes
están reproyectadas a Web Mercator sin suavizado adicional; sus escalas son fijas
en toda la publicación. La verdad completa permanece en el evaluador/visor.

La interfaz de la publicación anterior se verificó en Firefox: reproducción/pausa, cambio de tiempo y misión,
consulta por celda, zoom sincronizado, selección de incertidumbre y muestras.
El fondo utiliza la red vial OSM conservada; no solicita teselas en línea.
El HTML sí requiere Internet para cargar las bibliotecas habituales de Folium.
El HTML contiene los valores y la proyección de los mapas; no necesita imágenes contiguas.

`uv run --locked lima-pm25-rl visualizar` imprime la ruta del HTML final.
Entradas explícitas en [`config/visualizacion.json`](../../../config/visualizacion.json).
