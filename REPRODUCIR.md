# Reproducción de los resultados de O3

## Entradas y entorno

Instalar con `uv sync --locked`, manteniendo `lima-pm25-core` junto a este
repositorio. Core aporta su dominio retenido, grafo vial, meteorología, estaciones
y perfil GP. No se descargan mapas ni se recalibran esos recursos al evaluar.

`config/productos.json` identifica las carpetas utilizadas. Los parámetros del
entrenamiento están en `config/entrenamiento.json` y `config/gamma05.json`; los
procedimientos complementarios, en `config/experimentos/`. `config/informe.json`
selecciona las curvas y comparaciones que se presentan.

La distribución de resultados se entrega ya extraída. Los tar originales son
medios de transferencia, no entradas de los procedimientos. `resultados/origenes.csv`
documenta el traslado. Para ejecuciones cuyo nombre original no incluía fecha,
la carpeta lleva la fecha de organización, no un inicio de entrenamiento inventado.

## Inspeccionar la evidencia conservada

```bash
uv run --locked lima-pm25-rl visualizar
uv run --locked lima-pm25-rl informe
uv run --locked lima-pm25-rl comparar duracion
uv run --locked lima-pm25-rl comparar descuento
```

`visualizar` genera un visor Folium nuevo e imprime su ruta. `informe` genera una presentación
nueva de las curvas de ambos descuentos, tablas, casos pareados y mapas existentes.
`comparar` presenta las evaluaciones declaradas del contraste. Estas operaciones
no entrenan ni generan misiones.

La duración compara finales 1000/2000 de gamma 0,99. El descuento compara finales
de 2000 episodios con gamma 0,99/0,5. Las políticas `best` no reemplazan esos cortes.

### Visor de cuatro mapas

Abrir el `index.html` indicado por `lima-pm25-rl visualizar`, conservando a su lado
la carpeta `frames/`. La salida está en
`resultados/r3_4_visualizacion/mapas/AAAA-MM-DD_HH-MM-SS/`.

Los cuatro mapas comparten reloj, encuadre y zoom:

1. Calles OSM conservadas, posiciones actuales, estaciones fijas y recorrido pasado.
2. Media GP; el selector permite mostrar su desviación estándar.
3. Verdad sintética del mismo caso e instante.
4. Error **real − estimado**: rojo = subestimación; azul = sobreestimación.

Hay reproducción/pausa, avance por tick, barra temporal, velocidad y consulta por
celda. Las métricas mostradas corresponden al instante, no al promedio del episodio.
Las posiciones cambian en los ticks registrados de 300 s; no se inventan velocidades
intermedias. Una terminación anticipada mantiene el último móvil y no añade muestras.

`config/visualizacion.json` declara productos, políticas, casos y réplicas. La
configuración publicada incluye los casos prefijados 100002/100003 del mejor
adaptativo (28 400) y del final (30 000), con BiLG, greedy/aleatoria réplica 0,
barrido y solo fijas. Son
ejemplos individuales; las conclusiones agregadas pertenecen al informe completo.
Para un candidato nuevo se requieren primero sus mediciones, rutas y posteriores
de evaluación. Añadirlos a la configuración no vuelve a entrenar ni ajustar el GP.

El generador comprueba episodio/pesos, protocolo, grilla, reloj, identidad del campo,
lecturas fijas, RMSE por tick y distancia contra los productos originales.
Reproyecta las imágenes UTM a Web Mercator con vecino más próximo, sin suavizar
el campo. Las métricas se calculan desde los arrays originales; las consultas del
HTML conservan cinco decimales. Las escalas son fijas y compartidas entre misiones
seleccionadas, con una escala común para verdad/media y otra simétrica para error.

Folium, Branca y PyProj se declaran como dependencias Python. No hay instalación
separada de Leaflet ni gestor JavaScript. El HTML carga las bibliotecas habituales
de Folium por CDN, por lo que requiere conexión al abrirse. Las calles OSM y los
campos son locales: no solicita teselas ni descarga nueva cartografía.

## Entrenar

Para reproducir un entrenamiento completo en servidor:

```bash
uv run --locked lima-pm25-rl entrenar principal
uv run --locked lima-pm25-rl entrenar gamma05
```

Cada perfil fija 2000 episodios, semilla 42 y la programación de exploración
original. `principal` usa gamma 0,99; `gamma05`, gamma 0,5. El segundo comienza
desde cero, no desde los pesos del primero. Se guardan configuración, métricas,
políticas y un checkpoint completo entre episodios.

El dispositivo se declara en `training.device` del JSON. Los perfiles de servidor
usan `cuda`: si no está disponible, el comando se detiene antes de preparar los
escenarios, sin pasar a CPU. El entrenamiento imprime el dispositivo efectivo al
inicio. La CLI no admite `--device`. Las comprobaciones locales en CPU requieren
declararlo explícitamente en su configuración.

`--reanudar RUTA` continúa un estado del mismo protocolo en otra salida. Si
`RUTA` es un archivo, se usa exactamente ese checkpoint completo; una política
de solo pesos no sirve. Si es una carpeta, busca el mayor episodio entre `checkpoint-last.pt`,
`checkpoint-best.pt` y `checkpoint-final.pt` de esa carpeta, comprobando su
contenido. No descubre carpetas por antigüedad ni usa una ruta fija al episodio
20 000. No cambia el total del JSON ni reinicia exploración. Los
entrenamientos son operaciones costosas; graficar resultados existentes no
requiere ejecutarlos.

CUDA, configuración y compatibilidad se comprueban automáticamente al ejecutar.
La CLI no tiene modo separado de comprobación.

### Candidato adaptativo

```bash
uv run --locked lima-pm25-rl experimento gp-adaptativo
```

El experimento reconstruye doce casos **de entrenamiento** conservados y ejecuta
cuatro episodios nuevos para medir también Adam después del calentamiento del
replay. Guarda ajustes, límites alcanzados, cobertura, errores globales/locales y
componentes de recompensa. No produce la política de 20 000 episodios.

Comando de entrenamiento en el servidor:

```bash
uv run --locked lima-pm25-rl entrenar adaptativo
```

`config/adaptativo.json` fija actualmente **30 000 episodios totales**, gamma 0,99,
exploración decreciente durante 240 000 ticks, validación cada 200 episodios,
checkpoint rotativo cada 50. Las corridas recibidas conservaron los cortes
2000/5000/10000/15000/20000 y, durante la continuación, una política por episodio;
esa retención masiva ya no se utiliza en entrenamientos nuevos.
Se conservan el final y el mejor por RMSE, identificados por separado; no se
selecciona `best` por varianza de un GP cuya amplitud cambia.
BLAS se limita a un hilo durante entrenamiento y comparación;
`cpu_threads` controla los hilos de PyTorch. El ajuste GP se realiza en CPU,
incluso si la red utiliza CUDA.

El progreso se imprime cada 50 episodios y en cada validación. Muestra recompensa,
pérdida, actualizaciones y el **mejor RMSE global final de validación,
con su episodio**. Cada 200 episodios muestra además el RMSE recién medido sobre
los mismos doce casos. El RMSE se obtiene del MSE agregado, no promediando RMSE
por caso. El episodio 0 identifica la política inicial.

Entre validaciones se indica «sin nueva validación». Hay un solo mejor por RMSE,
identificado en `best-rmse.json` y respaldado por `checkpoint-best.pt` y
`policy-best.pt`. El checkpoint completo del mejor es el estado autoritativo:
su publicación es atómica y permite recuperar también la política si se interrumpe
su exportación. No hay parada anticipada; un récord aislado no prueba convergencia.

La salida se crea en `resultados/r3_2_entrenamiento/adaptativo/AAAA-MM-DD_HH-MM-SS/`.
Para reanudar, añadir `--reanudar RUTA`; el protocolo debe
coincidir. Al terminar, registrar explícitamente esa carpeta como `adaptativo`
en `config/productos.json`; entonces `lima-pm25-rl evaluar adaptativo-20000`
ejecuta política y referencias con el mismo GP adaptativo, incluida una referencia
solo fijas independiente. `gp-fits.jsonl` conserva las tres ramas de ajuste.
La reconstrucción localizada reproduce cronológicamente esos ajustes desde las
lecturas guardadas y usa el umbral de focos ya fijado.

La [formulación](resultados/r3_1_formulacion/adaptativo.md) especifica la adaptación
de Barrionuevo y sus límites. Cambian conjuntamente estimador, recompensa,
exploración y presupuesto: no es una ablación aislada del GP.

### Guardado y reanudación

La corrida original y su continuación están conservadas en
`2026-11-17_06-01-32/` y `2026-11-18_05-28-17/` bajo
`resultados/r3_2_entrenamiento/adaptativo/`. El archivo
`config/adaptativo-continuacion.json` se conserva como configuración histórica;
no corresponde a un perfil ejecutable actual.

```bash
uv run --locked lima-pm25-rl entrenar adaptativo \
  --reanudar resultados/r3_2_entrenamiento/adaptativo/2026-11-18_05-28-17
```

Con el objetivo actual de 30 000, el comando anterior indica que esa corrida
ya lo alcanzó y termina sin entrenar. Para una ampliación autorizada, cambiar
`episodes` en el JSON: un último estado de 27 350 y un total de 30 000 dejan
2650 episodios. No se reinician redes, Adam, replay, RNG ni epsilon.

Una nueva ejecución conserva:

- `checkpoint-best.pt` y `policy-best.pt`: mejor por RMSE, actualizados al mejorar.
- `checkpoint-final.pt` y `policy-last.pt`: estado completo y política finales.
- `checkpoint-last.pt`: un único estado rotativo, cada 50 episodios.
- CSV por episodio y registros pequeños por validación. Se conserva también la
  política inicial ligera para los contrastes inicial/final de los experimentos.

La ocupación estable de modelos es aproximadamente **0,82 GB** con el tamaño
actual del replay, más espacio temporal para escrituras atómicas. No se acumulan
los checkpoints anteriores. Las salidas antiguas no se modifican al reanudar.
El mejor anterior se transfiere desde su checkpoint completo a la carpeta nueva.
El récord perdido de 19 600 se discute solo en la documentación histórica.

`lima-pm25-rl evaluar adaptativo-mejor` carga los pesos señalados por `best-rmse.json`;
`lima-pm25-rl evaluar adaptativo-30000` utiliza el final. Se comprueba el episodio
interno del archivo, no solo su nombre. El mejor seleccionado por validación se
presenta como tal, sin sustituir silenciosamente una comparación de finales.

### Informes tras una interrupción

El CSV se escribe cada episodio; el checkpoint adaptativo, cada 50. Si el proceso
se interrumpe en el episodio 127 y se reanuda desde el checkpoint 100, el segundo
tramo repetirá los episodios 101–127. No borrar ni editar el CSV original.
Registrar ambas carpetas en `config/productos.json` y declarar en
`config/informe.json`, dentro de `training_runs`:

```json
{
  "label": "Adaptativo",
  "sources": ["adaptativo-tramo-1", "adaptativo-tramo-2"],
  "end_episodes": [100, null]
}
```

`end_episodes` tiene un elemento por fuente: un entero incluye hasta ese episodio;
`null` incluye todo el tramo. Si se omite, se incluyen todos los registros. El
informe comprueba continuidad de episodios, protocolo y contadores del checkpoint,
excluye las colas indicadas de curvas y bloques, y registra cuántos episodios
excluyó en el HTML y el manifiesto. Los CSV fuente permanecen intactos.

### Devolver el entrenamiento para análisis

Desde la raíz del repositorio en el servidor, empaquetar las carpetas completas
del entrenamiento adaptativo, incluidos todos los tramos si hubo reanudaciones:

```bash
tar --zstd -cf adaptativo-30000.tar.zst \
  resultados/r3_2_entrenamiento/adaptativo \
  config/adaptativo.json config/adaptativo-continuacion.json
```

Para la continuación, el archivo de transferencia se llama **`adaptativo-30000.tar.zst`**.
Incluye configuraciones, métricas, validaciones, todas las políticas y checkpoints;
no basta con enviar `policy-last.pt`. El tar es solo transporte, no una entrada
de la CLI.

## Evaluar una política

```bash
uv run --locked lima-pm25-rl evaluar gamma099-1000
uv run --locked lima-pm25-rl evaluar gamma099-2000
uv run --locked lima-pm25-rl evaluar gamma05-2000
```

Cada llamada ejecuta los doce escenarios fijados y las referencias, y entrega
en una salida nueva `misiones/` (lecturas, rutas y métricas globales) y `focos/`
(reconstrucción localizada, incertidumbre y recorrido). No hay una segunda
llamada pública con rutas de protocolo, entradas y salida que deba ensamblarse
manualmente. El umbral experimental se conserva en el protocolo publicado.

Las entradas `inicial` y `best-original` permiten examinar las políticas anteriores.
La evaluación completa simula misiones y puede tardar varios minutos. Los campos
verdaderos siguen siendo privados del evaluador.

Las nuevas salidas no sustituyen automáticamente los productos del documento.
Para presentar una reproducción, actualizar explícitamente las rutas relativas de
`config/productos.json` y `config/informe.json`.

## Experimentos complementarios

```bash
uv run --locked lima-pm25-rl experimento gp-espacial
```

| Nombre | Procedimiento | Trabajo principal |
| --- | --- | --- |
| `escenarios` | Contraste espacial y dinámica del campo | Simulación en CPU |
| `movilidad` | Conectividad y decisiones del caso prefijado | Misión con pesos fijos |
| `llegadas` | Llegada más corta frente a parada con continuidad | Misiones comparativas |
| `valores-q` | Valores Q y retornos de alternativas | Misiones ramificadas |
| `diagnostico-gp` | Aporte móvil, incertidumbre y error | Doce misiones |
| `gp-espacial` | Escalas 500/1000/2000/4000 m | Reconstrucciones y casos de entrenamiento |
| `gp-temporal` | Escalas 900/1800/3600/7200 s | Reconstrucciones congeladas |
| `gp-adaptativo` | Ajuste causal de amplitud y escalas; cambio del modelo | Doce casos de entrenamiento congelados y cuatro episodios nuevos |
| `gp-realimentado` | Cambio temporal con decisiones de la política | Misiones con pesos fijos |
| `historia` | Historia explícita frente a presente repetido | 600 episodios nuevos |
| `recompensa` | Incertidumbre frente a error sintético | 600 episodios nuevos |
| `focos-controlados` | Planes sobre focos persistentes y pulsantes | Simulación artificial |
| `gp-focos-espacial` | Escala espacial en focos | Reconstrucciones congeladas |
| `gp-focos-temporal` | Escala temporal en focos | Reconstrucciones congeladas |
| `gp-mezcla` | GP lento/rápido y mezclas fija/adaptativa | Reconstrucciones congeladas |
| `gp-multiescala` | Kernel conjunto espacial 2000+500 m | Reconstrucciones congeladas |

Las fuentes se resuelven desde `config/productos.json`; no se eligen por fecha
de modificación ni se exige que coincidan hashes de archivos de código.
Las comprobaciones numéricas automáticas de cada experimento distinguen
reconstrucción, incertidumbre y efecto
de cambiar rutas; no convierten un caso artificial en validación de Lima.

Los controles BiLG de 200 episodios y semillas 42/43/44 se conservan porque
sustentan las pruebas de historia, recompensa y GP. La comparación BiLG/Window
y su variante no forman parte de esta entrega.

## Pruebas y límites de reproducción

```bash
uv run --locked pytest tests/software
uv run --locked pytest tests/tesis
uv build
```

Los cálculos se verifican con tolerancias numéricas y contratos de dominio.
Se conserva la separación temporal de entrenamiento, validación y replay final.
Las diferencias entre hardware pueden alterar detalles numéricos de un entrenamiento;
las semillas no constituyen una promesa de identidad bit a bit.
