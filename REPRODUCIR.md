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

`visualizar` indica el HTML consolidado que ya está publicado. `informe` y
`comparar duracion/descuento` reproducen presentaciones en `cache/`, sin
duplicar las publicaciones de `resultados/`. Estas operaciones
no entrenan ni generan misiones.

El informe incluye `fijo-adaptativo/`: ventanas y estratos por semilla, dos
contrastes (fijo menos adaptativo), intervalos percentiles del 95 % por
remuestreo estratificado de casos y RMSE en los 37 instantes comunes.
Se enumeran las 4096 remuestras ordenadas (dos extracciones con reposición
por cada pareja dentro de los seis estratos); no hay semilla Monte Carlo.
Son descripciones de las políticas congeladas en validación, no prueba externa.

`.venv/bin/lima-pm25-rl comparar auditoria-validacion` genera en R3.3 una
auditoría de los mismos productos sin simular. `missions.csv`, `cases.csv` y
`summary.csv` contienen cobertura predictiva del 95 % (porcentaje), anchura
y CRPS gaussianos (µg/m³) para el campo latente sintético, en promedio y al
cierre. `intervals.csv` registra diferencias candidato menos referencia con
intervalos descriptivos por estrato (cobertura en puntos porcentuales).
`segments.csv` registra cada vehículo y tick móvil; `speeds*.csv` resume
las velocidades medias requeridas sobre los tramos en movimiento, por misión,
caso y estrategia. La fracción inmóvil excluye los ticks tras el término. Un
percentil de velocidad se calcula sobre tramos, no promediando percentiles;
no representa una velocidad observada ni una restricción aplicada al MDP.
`fixed` en la columna `strategy` significa solo estaciones fijas; en `study`
identifica el método con GP fijo. `adaptive_refs` son las referencias comunes
a ambos checkpoints adaptativos, verificadas y contadas una sola vez.

**Sensibilidad aislada de movilidad:** `.venv/bin/lima-pm25-rl experimento velocidad`.
Usa CUDA y ejecuta 300 misiones móviles con pesos congelados en los doce casos
de validación. `config/experimentos/velocidad.json` declara el techo de
30 km/h (2500 m viales por vehículo y tick) y la referencia normativa que
inspira ese supuesto uniforme. Filtra antes de seleccionar y reservar, tanto
para BiLG como para aleatoria, barrido y greedy. Los controles sin límite y
solo fijas se reutilizan de las evaluaciones originales; las referencias
adaptativas comunes se ejecutan una vez. La salida `experimentos/velocidad/`
de R3.3 conserva misiones, MSE y distancia por vehículo/tick, cambios de cada
estrategia y cambios de ventaja porcentual por caso y agregados (`all`).
La prueba independiente, el replay y los perfiles normales no usan el tope.

La duración compara finales 1000/2000 de gamma 0,99. El descuento compara finales
de 2000 episodios con gamma 0,99/0,5. Las políticas `best` no reemplazan esos cortes.

### Visor de cuatro mapas

**En vivo:** desde la raíz, ejecutar `uv run --locked lima-pm25-rl visualizar --en-vivo`
y abrir en el navegador la URL `http://127.0.0.1:PUERTO/` impresa. Usa los
perfiles CUDA y los tres archivos de política declarados en `config/prueba-independiente.json`.
Para ejecutarlo en el servidor, se puede acceder desde el equipo local mediante
un túnel SSH al puerto impreso. Ctrl+C cierra la sesión. No escribe resultados de
evaluación: es una demostración sintética, con las reglas de movilidad originales.

**Nuevo escenario** sortea una semilla. Dentro de una misión se puede cambiar de
estrategia o modelo en el mismo minuto: se avanza la nueva rama desde su propio
inicio, con el mismo campo, las mismas estaciones y posiciones iniciales; las
lecturas móviles corresponden a su propia ruta. Se guardan en memoria los pasos
consultados del escenario activo. Al regresar a otro escenario de la sesión se
reconstruye con su semilla. El navegador espera cada paso de inferencia; la
velocidad de reproducción no cambia el intervalo físico de 300 s. Los límites
de color de `live_limits` son comunes a todas las ramas y no cambian las cifras.

**Valores Q**, junto a la acción del mapa de recorrido, abre las asignaciones de un
mismo tick. Cada fila es un vehículo; las ocho columnas son sus direcciones. El
máximo válido elegido está destacado y las opciones bloqueadas aparecen con «—».
Al elegir otra asignación cambia el destino destacado sin mover el reloj. No se
atribuyen Q a las referencias ni a misiones antiguas que solo conservan rutas.
La consulta de celda aparece sobre el mapa.

Abrir `resultados/r3_4_visualizacion/mapas/2026-11-22_11-58-49/index.html`.
Los datos visuales y su proyección se incluyen comprimidos en el HTML (sin
`frames/`); el navegador requiere conexión para las bibliotecas Folium por CDN.
El selector combina las misiones sintéticas y la histórica en orden local de Lima.
Cada panel puede ampliarse con ⛶ y cerrarse con **Cerrar** o **Esc**, sin cambiar
el minuto ni el encuadre. En vivo se abre el mismo HTML con «Nuevo escenario».

Los cuatro mapas comparten reloj, encuadre y zoom:

1. Calles OSM conservadas, posiciones actuales, estaciones fijas y recorrido pasado.
2. Media GP; el selector permite mostrar su desviación estándar.
3. Campo simulado o, en replay, última observación horaria en cada estación.
4. Error **referencia − estimado**: espacial en sintético, solo puntual y
   desde la primera hora cerrada en replay.

Hay reproducción/pausa, avance por tick, barra temporal, velocidad y consulta por
celda. Las métricas mostradas corresponden al instante, no al promedio del episodio.
Las posiciones cambian en los ticks registrados de 300 s; no se inventan velocidades
intermedias. Una terminación anticipada mantiene el último móvil y no añade muestras.

`config/visualizacion.json` declara productos, políticas, casos y réplicas. La
configuración publicada incluye los casos prefijados 100002/100003 del fijo
(mejor y final 30 000) y del adaptativo (mejor 28 400 y final 30 000), con BiLG, greedy/aleatoria réplica 0,
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

`principal` fija actualmente 30 000 episodios, `gamma05` fija 2000. Ambos usan
semilla 42 y la programación de exploración original. `principal` usa gamma 0,99;
`gamma05`, gamma 0,5. El segundo comienza
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

### Continuar el GP fijo de 2000 a 30 000

Desde la raíz de RL, con los dos repositorios actualizados en el servidor:

```bash
uv run --locked lima-pm25-rl entrenar principal \
  --reanudar resultados/r3_2_entrenamiento/gamma099/2026-11-14_23-34-29/checkpoint-last.pt
```

Se esperan 2000 episodios completados, 28 000 restantes y epsilon 0,05. El GP,
recompensa, Adam, replay, redes, semillas y configuración de exploración siguen
siendo los del fijo. Se reevalúa el punto de partida y se guarda como primer
mejor por RMSE del tramo. La comparación de mejores usa solo 2000, 2200, …,
30 000 (141 mediciones, mismos doce casos); la comparación principal usa los dos
finales. Solo cambia la frecuencia de medir la validación, no el método de
aprendizaje. En una interrupción reanudar desde la **carpeta nueva** indicada
en `Salida:`. Anotar cada carpeta si hubo varios tramos.

Después del entrenamiento, sustituir `FECHA_UTC` por el nombre real de la nueva
carpeta y empaquetar desde la raíz de RL (añadir explícitamente cualquier otro
tramo nuevo que haya sido necesario para llegar al final):

```bash
tar --zstd -cf gp-fijo-30000.tar.zst \
  resultados/r3_2_entrenamiento/gamma099/FECHA_UTC config/entrenamiento.json
```

Para recibirlo, extraer en una **carpeta nueva** sin sobrescribir el repositorio:

```bash
mkdir recibido-gp-fijo-30000 && tar --zstd -xf gp-fijo-30000.tar.zst -C recibido-gp-fijo-30000
```

Conservar las carpetas originales de 1–2000. El tramo recibido está registrado
como `fijo-30000` en `config/productos.json`. En esta corrida, el mejor y el
final son el episodio 30 000 y sus pesos coinciden tensor a tensor. Ejecutar
una sola vez `uv run --locked lima-pm25-rl evaluar fijo-30000`: la misma salida
sirve para los contrastes de finales y de mejores sobre los doce casos de validación.

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

### Prueba independiente en el servidor

Desde la raíz de `lima-pm25-rl`, con la instalación editable y el entorno CUDA
ya utilizados para entrenar:

```bash
uv run --locked lima-pm25-rl evaluar independiente
```

Actualizar el repositorio del servidor con Git en la rama de trabajo.
Si los pesos ya están en sus carpetas de origen no hay que transferirlos. Comprobar
estas tres rutas relativas bajo `resultados/r3_2_entrenamiento/`:

- `gamma099/2026-11-19_03-10-56/policy-last.pt`
- `adaptativo/2026-11-18_05-28-17/policy-last.pt`
- `adaptativo/2026-11-18_05-28-17/policy-episode-28400.pt`

Se reutilizan `config/{entrenamiento,adaptativo}.json`,
`config/episodes/server-{control,event-stress}.json` y el vecino
`../lima-pm25-core` ya utilizado para ambos entrenamientos: configuración del
dominio y del estimador, grafo y grilla, calibración final, `partition.json`
y CSV meteorológico. No se requieren checkpoints de Adam/replay, CSV de
entrenamiento, evaluaciones anteriores, visor ni CSV histórico de PM2.5.

**Diseño congelado.** `config/prueba-independiente.json` fija 24 casos en
enero–marzo de 2026 antes de examinar sus resultados: seis estratos de cuatro
ventanas, tres horas cada una, guarda mínima de 24 h entre ventanas de misión
y respecto de los casos conocidos. Las fechas y estratos se verificaron contra
el generador original. Son ventanas nuevas del periodo sintético posterior
al ajuste, no otro dataset meteorológico; una guarda no garantiza independencia
atmosférica. Los estratos pesan igual por diseño, no por frecuencia en Lima.
Se mantiene el umbral de focos de entrenamiento. Las políticas congeladas usan
las reglas viales originales y las referencias su propio GP; aleatoria y greedy
tienen cinco réplicas por ventana. En total 600 misiones móviles y 48
referencias solo fijas, compartiendo las adaptativas entre sus dos políticas.

**Indicadores.** La reducción media de varianza integrada final del fijo
frente a las tres rutas móviles es el indicador principal. Los porcentajes
se calculan por referencia tras agregar, después se promedian; solo fijas
se informa aparte. El test de signos fijo–greedy evalúa frecuencia de ventaja,
no una magnitud del 20 %. El bootstrap pareado estratificado de 10 000
remuestras estima intervalos por ventana. La comparación de métodos es
final/final y, secundariamente, mejor/mejor. El contraste 4 km menos 2 km usa
**solo las lecturas de la misma ruta neural fija**, con el control de 2 km
reproducido en cada tick. No es una ruta con GP de 4 km ni sirve para elegir
otro checkpoint.

**Qué devolver.** La ruta UTC exacta aparece en `Salida:`. Conservar y traer
la carpeta completa `resultados/r3_3_evaluacion/prueba_independiente/<UTC>/`
cuando aparezca `completed.json`. Incluye protocolo, `summary.csv`, contrastes
con IC, `mobile-references.csv`, pares por escala y evidencia por escenario
(verdad privada única, errores por tick, rutas y lecturas). Sin
`completed.json` la corrida es parcial. Las comprobaciones de preparación
local usaron un caso antiguo, no son resultados de los 24 nuevos.

### Replay histórico final de O3

Actualizar el código y los JSON en el servidor con Git. Desde la raíz de RL:

```bash
git pull --ff-only
uv run --locked lima-pm25-rl evaluar replay-historico
```

Esta orden reproduce la evaluación y conserva sus entradas visuales. El visor
publicado ya incluye la misión histórica seleccionada.
Requiere el entorno CUDA y las dos políticas **finales** de 30 000 ya usadas
en `evaluar independiente`, el core vecino con sus recursos habituales y los
dos CSV de R1.1 en `../lima-pm25-core/data/`: `pm25_lima_2025_2026_imputed.csv`
y `var_meteorologicas_lima_2025_2026_imputed.csv`. Se comprueba su identidad con
el manifiesto de particiones. El CSV de PM2.5 sí es necesario para este replay.

**Selección.** `config/replay-historico.json` fija 12 ventanas semanales de
tres horas en abril–junio; se excluye el contexto de 24 h y las ventanas
anteriores de O2. La serie de PM2.5 debe ser observada en las tres estaciones
en los cuatro sellos y la meteorología debe cubrir la misión. El bloque sin
soporte no se reemplaza. Se condiciona la conclusión a esa disponibilidad.
El ejemplo visual es la primera ventana, no la que obtiene mejor resultado.

**Qué se simula.** Para ambas variantes y todas las estrategias, misma
meteorología histórica, posiciones iniciales, campo móvil, ruido por lectura,
información fija y horizonte. Como en O2 se usan fuentes viales rutinarias a
la escala congelada y un fondo y frontera uniformes, inicializados solo con
la mediana de las fijas **anteriores al inicio**, no con valores de la misión.
El GP se reinicia y recibe las lecturas disponibles desde t=0. Se asimilan
las observaciones elegibles hasta la hora de cierre inclusive. No hay fijas
imputadas ni Ceres. Un bloque histórico no proporciona una superficie de verdad.

**Qué se puntúa.** En la coordenada real de cada estación se retienen doce
predicciones GP causales (una al inicio de cada intervalo de cinco minutos).
Su media queda fijada antes de asimilar la lectura que cierra la hora.
Se comparan MAE, RMSE y sesgo horarios, varianza integrada, cobertura móvil
y distancia. Horas, estaciones y réplicas no son unidades independientes:
la incertidumbre de las diferencias se estima por ventana pareada. La única
prueba formal es unilateral para MAE del fijo frente a greedy (empates
excluidos); también se reporta el IC del RMSE. No se usa error espacial sintético.

**Qué traer de vuelta.** Conservar la carpeta UTC impresa bajo
`resultados/r3_3_evaluacion/replay_historico/`. Contiene `protocol.json`,
misiones, decisiones, rutas, lecturas, comparaciones estación–hora,
`cases.csv`, `summary.csv`, `contrasts.csv`, `mobile-references.csv` (media de
las reducciones frente a las tres móviles, tras agregar) y, al terminar,
`completed.json`. La inferencia descriptiva usa doce ventanas, no 108 ensayos
independientes; la prueba de signos contrasta frecuencia de ventaja, no magnitud.
Para la ventana histórica hay diez combinaciones (GP fijo y adaptativo, agente, aleatoria y greedy réplica 0,
barrido y solo fijas); las figuras de observación y error muestran **puntos
de estación**, no superficies. El error del mapa puntual es observado menos
estimado; el sesgo en CSV es predicho menos observado.

Para reconstruir el visor completo con un replay explícito, usar `uv run --locked
lima-pm25-rl visualizar --replay CARPETA_DEL_REPLAY`. Las reproducciones van a
`cache/visualizacion/<UTC>/`. Sin argumentos se muestra la publicación final.

### Evaluar políticas en desarrollo

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
