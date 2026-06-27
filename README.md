# Scouting LATAM — Inteligencia de Mercado Deportivo

**AD3010 · Business Analytics · Grupo 3**
Detectar jugadores **subvalorados** de Latinoamérica para un fondo de inversión deportivo, comparando un **valor justo modelado** contra el **precio de mercado real**.

`FIFA 24 (EA Sports)` · `Transfermarkt` · `World Bank`

---

## Resumen ejecutivo (TL;DR)

Un fondo quiere comprar talento joven latinoamericano, pero **el precio rara vez refleja el valor real**. Construimos un modelo de valor justo y medimos la brecha contra el precio de mercado. Hallazgos:

| Pregunta | Método | Resultado |
|---|---|---|
| ¿Existe ineficiencia? | Correlación + regresión simple | El talento explica solo el **24%** del precio |
| ¿Cuánto vale de verdad? | Regresión lineal múltiple (log) | **R² = 0,573** en test; **43% sin explicar por diseño** |
| ¿Quiénes están baratos? | Umbral estadístico de **1σ** | **90 candidatos** LATAM ≤23 subvalorados |
| ¿Es sistemático? | Reg. logística + árbol + Random Forest | Bolsillo donde la tasa de ganga sube **13% → 30%** |
| ¿Se aprecian? | **Backtest** temporal (12 meses) | Subvalorados **+25%**, sobrevalorados **−16,7%** |
| ¿Cómo invertir? | K-Means (k=3) | 3 perfiles: **36 / 10 / 44** jugadores |

**Conclusión:** la ineficiencia existe, es sistemática y la señal la anticipa. Recomendamos armar una cartera de los 90 candidatos repartida en tres perfiles de riesgo.

---

## 1. La pregunta de investigación y la hipótesis

**Pregunta:** ¿hay oportunidades de inversión en jugadores de fútbol latinoamericanos que permitan **maximizar el ROI** de un fondo?

**Hipótesis:** el mercado **no siempre paga lo que un jugador realmente vale**. Esa brecha entre *precio* y *valor justo* es medible, sistemática y aprovechable.

La hipótesis se vuelve medible con dos números:

- `tm_value_eur` — **precio de mercado**: lo que el mercado paga hoy (dato real, Transfermarkt).
- `fair_value_eur` — **valor justo**: lo que *debería* costar según talento, margen, edad y posición (lo predice el modelo).

$$\text{mispricing} = \log_{10}(\text{precio}) - \log_{10}(\text{valor justo})$$

Negativo → **barato → comprar**.

---

## 2. Datos y fuentes

| Fuente | Archivo | Filas crudas | Para qué |
|---|---|---|---|
| FIFA / EA Sports | `male_players.csv` | 180 021 | Calidad (overall, potential, 6 habilidades) |
| Transfermarkt (jugadores) | `players.csv` | 47 702 | Precio real, caps, posición |
| Transfermarkt (histórico) | `player_valuations.csv` | 616 377 | Serie de valuaciones (backtest, PA5) |
| World Bank | `API_NY.GDP.PCAP.CD…csv` | 266 países | PIB per cápita (contexto, PA4) |

### Descarga de los datasets

- **FIFA / EA Sports FC 24 — Complete Player Dataset** (Stefano Leone) → archivo `male_players.csv` (incluye todas las versiones FIFA 15–24 vía la columna `fifa_version`).
  https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset
- **Transfermarkt — Football Data** (David Cariboo, *player-scores*) → archivos `players.csv` y `player_valuations.csv` (precios reales, `international_caps` e histórico mensual de valuaciones).
  https://www.kaggle.com/datasets/davidcariboo/player-scores
  *(Espejo del código/datos en GitHub: https://github.com/dcaribou/transfermarkt-datasets)*
- **World Bank — PIB per cápita (US$ corrientes), indicador `NY.GDP.PCAP.CD`** → archivo `API_NY.GDP.PCAP.CD_*.csv` (descargar "CSV" desde la página del indicador).
  https://data.worldbank.org/indicator/NY.GDP.PCAP.CD

```python
df_fifa_raw = pd.read_csv(FIFA_FILE, low_memory=False)   # 180 021
df_tm_raw   = pd.read_csv(TM_PLAYERS, low_memory=False)   #  47 702
df_gdp_raw  = pd.read_csv(GDP_FILE, skiprows=4)           #  266 (4 líneas de cabecera del Banco Mundial)
```

---

## 3. Bloque 0 — Construcción de la base (la decisión "Ruta B")

**Objetivo:** una tabla donde *1 fila = 1 jugador*, combinando calidad (FIFA), precio (Transfermarkt) y contexto (PIB).

**La decisión clave — Ruta B:** entrenamos los modelos con **todo el mundo** (para que aprendan cómo el mercado fija precios en general), pero **cazamos solo en LATAM joven** (nuestro encargo). Por eso marcamos cada jugador con `is_latam` (¿dónde cazo?) e `is_matched` (¿tiene precio real, entra al modelo?), sin filtrar por país al construir la base.

**Preparación de FIFA 24:**

```python
fifa = df_fifa_raw[df_fifa_raw['fifa_version'].astype(float) == 24.0].copy()
fifa = fifa[fifa['player_positions'] != 'GK'].copy()      # sin arqueros (otra lógica de valor)

# Imputar nulos de las 6 habilidades con la MEDIANA de su posición
SKILLS = ['pace','shooting','passing','dribbling','defending','physic']
for c in SKILLS:
    fifa[c] = fifa.groupby('player_positions')[c].transform(lambda x: x.fillna(x.median()))

fifa = fifa.dropna(subset=['value_eur']); fifa = fifa[fifa['value_eur'] > 0]

# Variables derivadas
fifa['skill_index'] = fifa[SKILLS].mean(axis=1)
fifa['growth_room'] = fifa['potential'] - fifa['overall']   # techo: cuánto puede mejorar
```

**El reto técnico — el cruce FIFA ↔ Transfermarkt:** no comparten ID. Los unimos por una **clave de nombre** (inicial + apellido, sin tildes) y desempatamos por **año de nacimiento** (≤1 año) y **nacionalidad**, para evitar homónimos.

```python
def make_key(name):                       # 'Lionel Andrés Messi' -> 'l messi'
    t = _strip(name).split()
    return t[0] if len(t) == 1 else t[0][0] + ' ' + t[-1]

cand['byear_diff'] = (cand['fifa_byear'] - cand['tm_byear']).abs()
df = cand.sort_values(['byear_diff','nat_match'], ascending=[True, False]).drop_duplicates('player_id', keep='first')
df.loc[df['byear_diff'] > 1, ['tm_value_eur', ...]] = np.nan   # rechaza match si nacieron con >1 año de diferencia
```

> **Limitación honesta:** el cruce por nombre pierde apodos y grafías raras. Lo asumimos como sesgo de cobertura.

---

## 4. El universo de caza — el embudo a 303

**Objetivo:** acotar dónde invertir sin perder señal.

**Método:** filtrado secuencial. El corte de edad **no es a dedo**: se eligió con una **tabla de cortes** que compara cantidad vs. margen de crecimiento.

```python
# Tabla de cortes (margen de crecimiento medio entre paréntesis)
#   ≤20 → 116 (12.7)   ≤21 → 172 (11.9)   ≤22 → 235 (10.9)
#   ≤23 → 303 ( 9.9)   ≤24 → 370 ( 9.1)   ≤25 → 435 ( 8.3)
EDAD_MAX = 23
candidatos = base[base['is_latam'] & base['is_matched'] & (base['tm_value_eur'] > 0) & (base['age'] <= 23)]
```

```
16 217  jugadores de campo (mundo)
 →  2 982  nacionalidad LATAM
 →    833  con precio real (Transfermarkt)
 →    303  jóvenes ≤23 con precio   (151 mediocampistas · 93 defensas · 59 delanteros)
```

📊 *Figura:* `figuras/embudo.png`
**Interpretación:** de ~16 000 jugadores nos concentramos en **303** jóvenes LATAM con precio. A los ≤23 el techo todavía es amplio (margen medio 9,9) y hay masa suficiente para diversificar. *Filtrar = dónde cazar; el modelo = a cuál disparar.*

---

## 5. PA1 — ¿Existe la ineficiencia? El talento explica solo el 24%

**Objetivo:** probar que el talento **no basta** para explicar el precio (si lo explicara todo, no habría negocio).

**Método:** correlación de Pearson + regresión lineal simple (calidad → precio).

📊 *Figura:* `figuras/pa1_talento_precio.png`

| Métrica | Valor | Lectura |
|---|---|---|
| r (talento ↔ precio) | **0,49** | relación positiva pero lejos de perfecta |
| R² | **0,24** | el talento explica el **24%** del precio |
| Sin explicar | **76%** | la oportunidad |

**Conclusión:** tres de cada cuatro euros que paga el mercado responden a algo distinto del talento puro. Ahí empieza el negocio.

---

## 6. PA2 — ¿Qué mueve el precio? El potencial manda; la edad pesa como curva

**Objetivo:** entender las reglas del mercado más allá del talento.

**Método:** matriz de correlaciones de cada atributo con el precio.

| Atributo | Correlación con el precio |
|---|---|
| Potencial | **+0,67** |
| Talento (overall) | +0,49 |
| Habilidad (skill_index) | +0,40 |
| Fama (caps) | +0,24 |
| Margen (growth_room) | +0,14 |
| Edad | **−0,25** |

📊 *Figura:* `figuras/pa2_correlaciones.png` y `figuras/pa2_curva_edad.png`
**Interpretación:** el **potencial** pesa más que el talento actual. La **edad** correlaciona negativo: el valor hace pico ~21 años y se deprecia rápido. *Comprar joven, antes del pico, es donde está el margen.*

### El confound del margen — a igual talento, el mercado sí paga el margen

El margen parecía irrelevante (**+0,14**)… pero era una trampa estadística:

```python
# edad y margen están casi perfectamente ligadas
corr_edad_margen = base['age'].corr(base['growth_room'])   # ≈ -0.88
```

📊 *Figura:* `figuras/pa2_confound.png`
**Interpretación:** edad y margen correlacionan **−0,88** — los de más margen son justo los más jóvenes y de menor overall, así que el efecto del margen se confundía con el de la edad. Al comparar **a igual talento** (estratificando por overall), el margen **sí** paga. Conclusión metodológica: necesitamos un modelo que controle **todas las variables a la vez** → PA3.

---

## 7. PA3 — El motor de valor justo (explica el 57% y excluye fama/liga a propósito)

**Objetivo:** estimar cuánto vale *de verdad* cada jugador, controlando todo simultáneamente.

**Método:** regresión lineal múltiple sobre `log10(precio)`.

- **Predictores:** `overall`, `growth_room`, `age`, `age²`, posición (dummies).
- **Exclusión intencional:** fama, liga y país — para que aparezcan como *desviación* (oportunidad), no como *precio correcto*.
- **`age²`:** la relación edad-valor es una curva (sube, pica, cae). Un término cuadrático la captura; una recta no.
- **Limpieza:** se quitan outliers a >3σ del primer ajuste (matches imposibles).
- **Validación:** train/test 75/25 → el R² se reporta sobre datos no vistos.

```python
t['log_value'] = np.log10(t['tm_value_eur'])   # precio en log (es exponencial)
t['age_sq']    = t['age']**2                   # término curvo

Xnum = ['overall', 'growth_room', 'age', 'age_sq']
X = pd.concat([t[Xnum], pd.get_dummies(t['pos_group'], prefix='pos', drop_first=True)], axis=1)

resid0 = y - LinearRegression().fit(X, y).predict(X)
clean  = resid0.abs() <= 3*resid0.std()        # Matches implausibles removidos: 106 de 8569

Xtr, Xte, ytr, yte = train_test_split(X[clean], y[clean], test_size=0.25, random_state=42)
r2 = r2_score(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))   # 0.573
```

**Resultados:**

| Métrica | Valor |
|---|---|
| R² test (solo talento) | 0,24 |
| **R² test (modelo completo)** | **0,573** |
| Sin explicar (por diseño) | ~43% |
| Outliers removidos (3σ) | 106 de 8 569 |

**Lectura de los coeficientes** (efecto sobre el precio *a igualdad del resto*, como `10^β − 1`):

| Factor | Efecto |
|---|---|
| +1 punto de overall | **+22%** de precio |
| +1 punto de margen | +4% |
| Ser delantero | +8% (vs. defensa) |
| Ser mediocampista | +2% (vs. defensa) |
| Edad | efecto en curva (pico y caída) |

**Conclusión:** el modelo pasa de explicar **1 de cada 4 euros a más de la mitad**, y como es *test*, no memoriza. El ~43% restante es **intencional**: ahí viven los subvalorados. Un R² de 0,95 sería *mala* noticia — significaría que el mercado ya pone el precio correcto y no quedaría brecha que explotar.

### 7.1 La brecha y su dispersión — una campana centrada en cero

```python
base['mispricing'] = np.log10(base['tm_value_eur']) - np.log10(base['fair_value_eur'])
mu, sd = mm.mean(), mm.std()          # μ ≈ -0.001 ,  σ ≈ 0.452
disc1  = (1 - 10**(-sd)) * 100        # 1σ ≈ 65%
```

📊 *Figura:* `figuras/pa3_histograma_mispricing.png`
**Interpretación:** la media ≈ 0 confirma que el modelo **no está sesgado**. La desviación típica (1σ ≈ 0,452) equivale a un ~**65% de descuento/sobreprecio**. La mayoría cotiza cerca de su valor justo (el centro); el negocio vive en las **dos colas**. *¿Por qué log?* Un desvío relativo pesa igual en un jugador de €1M y en uno de €100M.

### 7.2 La señal 1σ → 90 candidatos

```python
SIGMA = 1.0; thr = SIGMA * sd
base['signal'] = np.select(
    [base['mispricing'] <= -thr, base['mispricing'] >= thr],
    ['SUBVALORADO', 'SOBREVALORADO'], default='NEUTRAL')
```

📊 *Figura:* `figuras/pa3_brecha_top8.png`

| Segmento (mundo, con precio) | Conteo |
|---|---|
| SUBVALORADO (≤ −1σ → comprar) | 1 113 |
| NEUTRAL | 6 291 |
| SOBREVALORADO (≥ +1σ → evitar) | 1 165 |

**Resultado:** aplicado a LATAM ≤23 → **90 candidatos de compra**. Es una vara **estadística**, no un criterio a dedo: 1σ es la dispersión normal del propio mercado.

---

## 8. PA4 — ¿Sistemático o azar? La ganga tiene un patrón, no es suerte

**Objetivo:** comprobar que la subvaloración responde a un patrón de contexto y no al azar.

**Método (evitando circularidad):** clasificamos "ganga / no-ganga" usando **solo rasgos contextuales nuevos** — `log_gdp`, `caps` (fama), `league_level` — *distintos* de los que definieron la etiqueta. Tres modelos (S6), cada uno con un propósito:

- **Regresión logística** → ¿el contexto *predice* la ganga? (con `class_weight='balanced'` porque las gangas son minoría)
- **Árbol de decisión (depth 3)** → la regla legible (el "bolsillo" de gangas)
- **Random Forest** → qué rasgo manda (importancia de variables)

```python
d['log_gdp']      = np.log10(d['gdp_per_capita_2023'])
d['caps']         = d['international_caps'].fillna(0)
d['league_level'] = d.groupby('league_name')['overall'].transform('median')

feats = ['log_gdp', 'caps', 'league_level']
clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(Xtr_s, ytr)
```

📊 *Figura:* `figuras/pa4_importancia.png` y `figuras/pa4_arbol.png`

| Importancia (Random Forest) | Peso |
|---|---|
| País (PIB) | **42%** |
| Fama (caps) | 36% |
| Nivel de liga | 22% |

**Resultado:** el árbol aísla un **bolsillo** donde la tasa de ganga sube de **13% → 30%** (más del doble). La fama correlaciona **−0,60** con ser ganga (menos fama → más ganga). Hay un patrón claro — **menos fama, menor PIB, liga más débil** —, no es azar.

> **Limitación honesta:** la predicción *individual* de gangas es débil (por eso recomendamos perfiles de scouting, no apuestas ciegas). El criterio humano sigue siendo necesario para descartar gangas "falsas" (lesiones, fin de contrato) que el modelo no ve.

---

## 9. PA5 — El backtest: los baratos suben, los caros caen

**Objetivo:** la prueba decisiva — ¿las gangas se aprecian de verdad?

**Método:** backtest temporal **sin circularidad**. Fijamos la señal en **sept-2023** (alineado con FIFA 24) y medimos la apreciación a **sept-2024** (12 meses). El futuro **nunca** entra en el cálculo de la señal. Usamos el histórico mensual de Transfermarkt (616 377 valuaciones).

```python
T0 = pd.Timestamp('2023-09-01')      # señal anclada aquí
T1 = pd.Timestamp('2024-09-01')      # apreciación medida 12 meses después
def value_asof(vals, t):
    return vals[vals['date'] <= t].sort_values('date').groupby('player_id').tail(1)...
```

📊 *Figura:* `figuras/pa5_convergencia.png`

| Segmento | Apreciación 12m | n | Edad mediana |
|---|---|---|---|
| **SUBVALORADO** | **+25,0%** | 1 297 | 21 |
| NEUTRAL | 0,0% | 5 944 | 25 |
| **SOBREVALORADO** | **−16,7%** | 955 | 28 |

**Conclusión:** si fuera una marea general de mercado, todos subirían. En cambio, los neutrales quedaron planos y los sobrevalorados **cayeron**. El mercado corrige hacia el valor justo y **nuestra señal lo anticipa**. Esta es la evidencia más fuerte del proyecto: es una predicción a futuro puesta a prueba, no un ajuste dentro de muestra.

---

## 10. PA6 — Tres perfiles de cartera que equilibran riesgo y upside

**Objetivo:** con 90 candidatos no se invierte de forma uniforme; armar perfiles de riesgo accionables.

**Método:** clustering **K-Means** (aprendizaje no supervisado, S10), con variables estandarizadas (`StandardScaler`, porque K-Means mide distancias). El número de grupos se eligió con **método del codo + coeficiente de silueta**.

```python
feats = ['overall', 'growth_room', 'age', 'caps', 'vs_fair_pct']
X = StandardScaler().fit_transform(clu[feats])
clu['cluster'] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
```

📊 *Figura:* `figuras/pa6_codo_silueta.png` y `figuras/pa6_perfiles.png`

**Por qué k=3:** el codo no marca un quiebre brusco y la silueta deja 3 y 4 casi empatados; con ~90 candidatos, **tres** grupos dan perfiles interpretables y con masa suficiente, mientras que cuatro fragmenta en grupos demasiado chicos para defender. *El "mejor k" de negocio no siempre es el de la silueta.*

| Perfil | n | Overall | Margen | Edad | Caps | Descuento | Precio mediano | Lectura |
|---|---|---|---|---|---|---|---|---|
| **Apuesta joven** | 36 | 63,8 | 12,0 | 19,6 | 0,3 | −83,7% | €0,20M | Mucho techo, muy jóvenes. Alto riesgo / alto retorno |
| **Joya consagrada** | 10 | 67,7 | 12,9 | 19,6 | 12,4 | −74,2% | €0,73M | Techo alto y ya con minutos en selección. El menor riesgo |
| **Valor maduro** | 44 | 69,3 | 7,3 | 22,4 | 0,7 | −85,3% | €0,30M | Menor techo, algo mayores. Rendimiento inmediato |

**Interpretación del radar:** la apuesta joven manda en techo y juventud; la joya consagrada, en trayectoria (caps); el valor maduro, en calidad actual y descuento. Los tres perfiles suman exactamente los **90** candidatos.

---

## 11. Conclusiones — cuatro hallazgos con su evidencia

1. **La oportunidad existe y es modelable** — el modelo de valor justo alcanza **R² = 0,573** en test.
2. **La señal funciona** — **+25%** a 12 meses, verificado en backtest sin circularidad.
3. **Dónde cazar** — el perfil de scouting es talento joven, poco mediático, de ligas/países de bajo ingreso.
4. **Cómo invertir** — cartera de 3 perfiles que combina riesgo y upside.

---

## 12. Recomendación (CTA)

- **Recomendación:** armar la cartera de los **90 candidatos** LATAM ≤23 en los tres perfiles (36 / 10 / 44).
- **Respaldo:** descuento mediano profundo (la señal exige ≥1σ ≈ 65% bajo el valor justo; el candidato típico cotiza ~84% por debajo), **+25%** de retorno verificado y **R² = 0,573**.
- **Costo de no actuar:** el margen es perecedero — los baratos convergen a su valor justo en ~12 meses; cada ventana de fichajes que pasa, el descuento se evapora.
- **Próximo paso:** aprobar un piloto sobre el perfil **Apuesta joven** (n=36, mayor techo) y validar los 10 de mayor brecha con scouting de campo.

---

## 13. Limitaciones honestas

- El modelo de valor justo deja **~43% sin explicar** — por diseño; ahí vive la oportunidad.
- El cruce FIFA ↔ Transfermarkt **pierde algunos apodos** y solo ~28% de los LATAM tenían precio real (sesgo de cobertura).
- FIFA es un **proxy de talento**, no verdad absoluta; la validación real es el backtest contra precios de mercado.
- La **predicción individual** de gangas es débil → recomendamos perfiles, no apuestas ciegas; el criterio humano descarta gangas "falsas" (lesiones, contratos).

---

## 14. Reproducibilidad

```
Business_Analytics_PC1/
├── INFORME.md                     ← este archivo
├── Notebook_Scouting_LATAM.ipynb  ← pipeline completo (Bloque 0 → PA6)
├── figuras/                       ← exports de los gráficos del notebook
└── data/                          ← CSV (no versionados; ver rutas abajo)
```

**Datos requeridos** (en `data/` o Google Drive): `male_players.csv` (FIFA), `players.csv` y `player_valuations.csv` (Transfermarkt), `API_NY.GDP.PCAP.CD_*.csv` (World Bank). Links de descarga en la [sección 2](#2-datos-y-fuentes).

**Stack:** Python · pandas · numpy · scikit-learn (`LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `KMeans`, `StandardScaler`, `train_test_split`) · matplotlib. Ejecutar las celdas en orden; el Bloque 0 construye `base` y cada PA depende de la anterior.

---

## 15. Glosario

| Término | Definición |
|---|---|
| **mispricing** | Distancia entre precio y valor justo en log. Negativo = barato. |
| **1σ (sigma)** | La vara: más barato de lo que la dispersión normal del mercado explicaría. |
| **Ruta B** | Entrenar con el mundo entero; cazar solo en LATAM joven. |
| **Exclusión intencional** | Dejar fuera fama, liga y país para que sean oportunidad, no "precio correcto". |
| **Circularidad** | No clasificar con las mismas variables que ya definieron la etiqueta. |
| **Backtest** | Medir a futuro con la señal fijada en el pasado: el futuro no entra en el cálculo. |
| **growth_room (margen)** | `potential − overall`: cuánto puede mejorar el jugador. |
| **caps** | Partidos internacionales: proxy de fama / trayectoria. |
