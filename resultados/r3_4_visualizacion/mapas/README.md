# Visor Folium de misiones conservadas

**[Abrir el visor publicado](2026-11-18_19-04-15/index.html)** ·
[Captura del mejor](2026-11-18_19-04-15/captura-mejor-tick-36.png) ·
[Captura del final](2026-11-18_19-04-15/captura-final-tick-36.png).

La publicación actual contiene los dos casos prefijados 100002 y 100003 del
adaptativo: mejor seleccionado en validación (28 400) y final (30 000). Para
cada modelo/caso permite elegir BiLG-D3QN, greedy y aleatoria (réplica 0),
barrido o solo fijas: veinte misiones con escalas comunes.
La [publicación inicial gamma 0,5](2026-11-18_06-08-20/index.html) sigue conservada.

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

La interfaz se verificó en Firefox: reproducción/pausa, cambio de tiempo y misión,
consulta por celda, zoom sincronizado, selección de incertidumbre y muestras.
El fondo utiliza la red vial OSM conservada; no solicita teselas en línea.
El HTML sí requiere Internet para cargar las bibliotecas habituales de Folium.
Conservar `frames/` junto a `index.html`.

Generación: `lima-pm25-rl visualizar`. Entradas explícitas en
[`config/visualizacion.json`](../../../config/visualizacion.json). Una reproducción
crea otra carpeta UTC; el inventario publicado no cambia automáticamente.
