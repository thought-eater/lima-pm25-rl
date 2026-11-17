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

`visualizar` imprime la ruta del HTML conservado. `informe` genera una presentación
nueva de las curvas de ambos descuentos, tablas, casos pareados y mapas existentes.
`comparar` presenta las evaluaciones declaradas del contraste. Estas operaciones
no entrenan ni generan misiones.

La duración compara finales 1000/2000 de gamma 0,99. El descuento compara finales
de 2000 episodios con gamma 0,99/0,5. Las políticas `best` no reemplazan esos cortes.

## Entrenar

```bash
uv run --locked lima-pm25-rl entrenar principal --comprobar
uv run --locked lima-pm25-rl entrenar gamma05 --comprobar
```

La comprobación verifica el dispositivo configurado y compone escenarios sin
avanzar la simulación ni construir el entrenador. Para reproducir un entrenamiento
completo en servidor:

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

`--reanudar RUTA/checkpoint-last.pt` continúa un estado del mismo protocolo en
otra salida. No cambia el total del JSON ni reinicia exploración. Los
entrenamientos son operaciones costosas; graficar resultados existentes no
requiere ejecutarlos.

### Candidato adaptativo de 20 000 episodios

```bash
uv run --locked lima-pm25-rl entrenar adaptativo --comprobar
uv run --locked lima-pm25-rl experimento gp-adaptativo --comprobar
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

`config/adaptativo.json` fija **20 000 episodios totales desde cero**, gamma 0,99,
exploración decreciente durante 240 000 ticks, validación cada 200 episodios,
checkpoint completo cada 50 y políticas en 2000/5000/10000/15000/20000.
Se usa el final prefijado; no se selecciona `best` por varianza de un GP cuya
amplitud cambia. BLAS se limita a un hilo durante entrenamiento y comparación;
`cpu_threads` controla los hilos de PyTorch. El ajuste GP se realiza en CPU,
incluso si la red utiliza CUDA.

El progreso se imprime cada 50 episodios y en cada validación. Muestra recompensa,
pérdida, actualizaciones y el **mejor RMSE global final de validación conocido,
con su episodio**. Cada 200 episodios muestra además el RMSE recién medido sobre
los mismos doce casos. El RMSE se obtiene del MSE agregado, no promediando RMSE
por caso. El episodio 0 identifica la política inicial.

Entre validaciones se indica «sin nueva validación». El mejor valor y su episodio
se guardan en el checkpoint y en `metrics.csv`; se conservan al reanudar. Un
checkpoint antiguo sin ese seguimiento comienza con «todavía no registrado».
Este seguimiento no elige otro modelo final, no guarda un `best` adaptativo ni
activa parada anticipada. Un récord aislado no prueba convergencia.

La salida se crea en `resultados/r3_2_entrenamiento/adaptativo/AAAA-MM-DD_HH-MM-SS/`.
Para reanudar, añadir `--reanudar RUTA/checkpoint-last.pt`; el protocolo debe
coincidir. Al terminar, registrar explícitamente esa carpeta como `adaptativo`
en `config/productos.json`; entonces `lima-pm25-rl evaluar adaptativo-20000`
ejecuta política y referencias con el mismo GP adaptativo, incluida una referencia
solo fijas independiente. `gp-fits.jsonl` conserva las tres ramas de ajuste.
La reconstrucción localizada reproduce cronológicamente esos ajustes desde las
lecturas guardadas y usa el umbral de focos ya fijado.

La [formulación](resultados/r3_1_formulacion/adaptativo.md) especifica la adaptación
de Barrionuevo y sus límites. Cambian conjuntamente estimador, recompensa,
exploración y presupuesto: no es una ablación aislada del GP.

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
tar --zstd -cf adaptativo-20000.tar.zst \
  resultados/r3_2_entrenamiento/adaptativo \
  config/adaptativo.json
```

El archivo de transferencia se llama **`adaptativo-20000.tar.zst`**. Incluye
configuraciones, métricas, validación inicial, políticas y checkpoint completo;
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
uv run --locked lima-pm25-rl experimento gp-espacial --comprobar
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
`--comprobar` valida la preparación sin iniciar la batería. Las comprobaciones
numéricas de cada experimento distinguen reconstrucción, incertidumbre y efecto
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
