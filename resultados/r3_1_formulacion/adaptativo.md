# Candidato: GP adaptativo y recompensa de cambio del modelo

## Qué se adapta de Barrionuevo

Barrionuevo et al., DOI [10.1109/ACCESS.2024.3402980](https://doi.org/10.1109/ACCESS.2024.3402980),
§IV-D.2, p. 71843, ecuación 12, emplean cambios **absolutos de desviación
estándar por celda**, antes de sumar. R1 combina cambios de media e incertidumbre.
La motivación es que reajustar hiperparámetros permite aumentos de incertidumbre.
[Yanes et al., §IV-D](https://arxiv.org/html/2401.04631v1) también describen esta
heurística. Es cambio del modelo, no garantía de reducción del error ni una
ganancia de información exacta.

El mejor resultado de Barrionuevo corresponde a la variante con mapa de
conocimiento. Este candidato adapta la rama basada solo en GP: no reproduce
aquella configuración ganadora. Su campo espacial estático, sensores heterogéneos,
RBF y presupuesto de 100 000 misiones difieren de este problema.

## Estimador

Se conserva Matérn ν=1,5 espacio-temporal. La media previa, el ruido instrumental
por lectura y el jitter del perfil original permanecen fijados. Se ajustan
amplitud de covarianza, una escala espacial común a x/y y escala temporal,
maximizando la log-verosimilitud marginal de las mediciones disponibles.

La implementación está en `lima_pm25.estimation.adaptive` de core. Utiliza
L-BFGS-B acotado de SciPy, con gradientes de scikit-learn y arranque desde los
parámetros anteriores, en vez de añadir GPyTorch. Solo acepta mejoras finitas
superiores a la tolerancia numérica; conserva el mejor punto evaluado si se
alcanza el límite de iteraciones. No supone que una parada sea convergencia.

| Parámetro | Inicial | Límites experimentales |
| --- | ---: | ---: |
| Escala espacial isotrópica | 2000 m | 200–4000 m |
| Escala temporal | 3600 s | 300–21600 s |
| Desviación de la amplitud | σ₀ del perfil conservado | 0,25 σ₀–2 σ₀ |

Historial: 6 h. Máximo: 20 iteraciones por ajuste. Se requieren al menos doce
lecturas elegibles, 900 s entre la primera y la última y tres posiciones
distintas. Cada misión reinicia parámetros e historial. La consulta predictiva
no optimiza ni altera el estado. Los límites son regularización experimental,
no escalas atmosféricas medidas.

## Recompensa centralizada

En cada instante de medición se construyen dos ramas con el mismo inicio y
presupuesto de optimización:

- **Referencia:** historial anterior, incluidos móviles anteriores, más fijas nuevas.
- **Factual:** ese mismo conjunto más móviles nuevos.

La referencia y el posterior factual se predicen en el mismo instante. Solo
el factual se conserva. Sea U la unión de discos de radio 2000 m centrados en
las nuevas mediciones móviles elegibles, N el número de celdas activas y σ₀ la
desviación previa original:

```text
Δμ = Σ[c∈U] |μ_factual(c) − μ_referencia(c)| / (N σ₀)
Δσ = Σ[c∈U] |σ_factual(c) − σ_referencia(c)| / (N σ₀)
r = Δμ + 0,5 Δσ − 0,0005 × distancia_vial_km
```

Las zonas solapadas cuentan una sola vez. No se atribuye envejecimiento ni una
nueva lectura fija al aporte móvil; sin móviles elegibles no se premia cambio
del modelo. La ganancia firmada de varianza se conserva como diagnóstico
separado (`uncertainty_gain`/`information_gain`), no como esta recompensa.
Se mantiene la penalización operacional por acciones inválidas; el controlador
enmascarado debe producir cero acciones inválidas.

La incertidumbre observable es `σ/(σ+σ₀)`, monótona incluso cuando σ supera σ₀.
La normalización de la media sigue anclada al perfil original. Ni el ajuste ni
la recompensa acceden al campo sintético completo.

## Evaluación e interpretación

La referencia **solo fijas** mantiene un estimador independiente: no reutiliza
parámetros inferidos con móviles. Aleatoria, greedy, barrido y red utilizan el
mismo método adaptativo al evaluar este candidato. Las referencias con GP fijo
de los experimentos previos se conservan como otro método.

Se registran errores, cobertura de intervalos, parámetros, convergencia y
límites alcanzados. `at_bounds` está ordenado como amplitud/espacio/tiempo.
Los tiempos de ajuste incluyen predicción; los contadores por tick incluyen
ambas ramas y el estimador solo fijas, pero excluyen el reset inicial.
Una amplitud menor puede bajar la varianza sin mejorar el error. Los informes
separan curvas con recompensas/estimadores distintos y conservan sus referencias.

El [diagnóstico de entrenamiento](../r3_3_evaluacion/experimentos/gp_adaptativo/)
no es validación final. El presupuesto de 20 000 episodios, la exploración y la
recompensa cambian conjuntamente con el GP. Los cortes prefijados y el modelo
final son los productos de interés, no un `best` escogido por varianza.
La corrida original completó 20 000 episodios y su continuación llegó a 30 000.
Se evaluaron el mejor por RMSE (28 400) y el final. El guardado de nuevas corridas
conserva mejor, final y un último checkpoint rotativo cada 50 episodios, sin
retención masiva por episodio ni parada anticipada.
El mejor seleccionado por validación y el final se presentan por separado.
Se conserva el umbral de focos de 24,555472595740433 µg/m³, los IOV originales
y la meta general del 20 %. El replay histórico final sigue pendiente.

Referencias técnicas:
[GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html),
[L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
