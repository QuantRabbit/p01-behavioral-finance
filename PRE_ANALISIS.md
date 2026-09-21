# Pre-análisis: Proyecto P01

**Simulador de finanzas conductuales con *ground truth* conocido**

- Alumnos: Juan Pablo Sánchez, Matteo Nelson, Edgardo González
- Profesor: Prof. Luis Felipe Gómez Estrada
- **Fecha de redacción: 18 de septiembre de 2026**

> Este documento se escribe y se registra en git **antes** de ejecutar un solo
> estimador. Su función es fijar por escrito qué parámetros se van a inyectar y
> qué se espera obtener, de modo que cualquier desacuerdo posterior entre lo
> inyectado y lo estimado sea un hallazgo y no una racionalización *ex post*.
> El commit que contiene únicamente este archivo fecha el registro previo.

---

## 1. Qué se está probando

Con datos reales de transacciones no existe *ground truth*: un coeficiente
positivo puede ser un sesgo psicológico, un artefacto institucional o un error
de código, y los tres son observacionalmente equivalentes. Aquí los parámetros
de comportamiento se inyectan, por lo tanto se conocen. La pregunta del
proyecto no es "¿existe el efecto disposición?" sino **"¿cuándo mienten los
estimadores estándar de la literatura?"**.

Dos parámetros por agente constituyen la verdad conocida:

- $\delta_i \in [0,1]$: intensidad del efecto disposición.
- $\kappa_i \in [0,1]$: sobreprecisión o propensión a rotar la cartera.

Ninguno de los dos es una probabilidad de venta ni una tasa de rotación.
Modulan una **tasa de riesgo diaria** (*hazard rate*), un nivel por debajo de las
cantidades que se van a estimar:

$$h_0(\kappa_i) = h_{\text{base}} + c_\kappa \kappa_i \quad \text{con} \quad h_{\text{base}} = 0.015, \quad c_\kappa = 0.060$$

$$h_i(t) = \begin{cases} 
\min(0.99, \, h_0(1 + \delta_i)) & \text{si } P_t > P_{\text{ref}} \quad (\text{posición en ganancia}) \\ 
\max(10^{-4}, \, h_0(1 - \delta_i)) & \text{si } P_t < P_{\text{ref}} \quad (\text{posición en pérdida}) \\ 
h_0 & \text{si } P_t = P_{\text{ref}} \quad (\text{posición neutral}) 
\end{cases}$$

`PGR`, `PLR`, el turnover, el número de operaciones y los horizontes de tenencia
son **outputs emergentes** de la interacción entre esta tasa, la trayectoria de
precios y el tiempo. Nunca son inputs. En ningún punto del código existe una
sentencia de la forma `if ganancia: vender con probabilidad p`.

`h_base = 0.015` corresponde a un horizonte medio de tenencia de
`1/0.015 ≈ 67` días hábiles (≈ 3 meses) cuando `δ = κ = 0`, del orden de los
horizontes reportados por Odean (1998) para cuentas minoristas.

---

## 2. Parámetros exactos que se van a inyectar

### 2.1 Mercado (idéntico en todos los escenarios salvo la semilla)

| Parámetro | Valor |
|---|---|
| Activos `M` | 60 |
| Días hábiles `T` | 504 (2 años) |
| Factor 1 (mercado) | `μ = 8%` anual, `σ = 16%` anual |
| Factor 2 (sectorial) | `μ = 0`, `σ = 12%` anual |
| Factor 3 (estilo, ortogonal) | `μ = 0`, `σ = 8%` anual |
| Betas | `β₁ ~ U[0.65,1.35]`, `β₂ ~ U[−0.6,0.6]`, `β₃ ~ U[−0.4,0.4]` |
| Vol. idiosincrática | `~ U[12%, 28%]` anual |
| Precio inicial | `~ U[25, 150]` |
| Autocorrelación de retornos | **cero por construcción** (i.i.d. en el tiempo) |

### 2.2 Costos de transacción

| Parámetro | Valor |
|---|---|
| Comisión | **$1.00 USD por orden** |
| Spread bid-ask total | **10 puntos base** (medio spread = 5 bps = 0.0005) |
| Ejecución de compras | ask = `P·(1+0.0005)` |
| Ejecución de ventas | bid = `P·(1−0.0005)` |

### 2.3 Población

| Parámetro | Valor |
|---|---|
| Agentes `N` | 1000 |
| Riqueza inicial `W_i` | log-uniforme en `[$10,000, $500,000]` |
| Capacidad `n_i` | entera uniforme en `[5, 30]` posiciones |
| Distribución de `δ_i` y `κ_i` | Beta reparametrizada, `E[·] = objetivo`, `ν = 20` |

Para un objetivo `m > 0`: `δ_i ~ Beta(a,b)` con `a = m·ν`, `b = (1−m)·ν`. Con
`ν = 20` y `m = 0.8` eso da `a = 16`, `b = 4`, `E[δ] = 0.80`, `sd(δ) ≈ 0.087`.
Para `m = 0` no existe Beta admisible: en ese caso **`δ_i = 0` exacto para todos
los agentes**, porque la ausencia del sesgo es precisamente lo que el escenario
quiere probar. Lo mismo para `κ`. Esto implica que en esos escenarios la
correlación `Corr(δ, κ)` **no está definida** (varianza cero) y así se reportará:
`n/d`, nunca `0.0000`.

`δ` y `κ` se sortean de **dos flujos independientes del mismo generador de
población**, de modo que su correlación muestral sea ruido de orden `1/√N`.

### 2.4 Escenarios

| # | Clave | `δ` objetivo | dist. `δ` | `κ` objetivo | dist. `κ` | Confound |
|---|---|---|---|---|---|---|
| 1 | `nulo` | 0.0 | punto | 0.0 | punto | - |
| 2 | `disposicion_baja` | 0.3 | Beta(6,14) | 0.0 | punto | - |
| 3 | `disposicion_alta` | 0.8 | Beta(16,4) | 0.0 | punto | - |
| 4 | `rotacion_baja` | 0.0 | punto | 0.3 | Beta(6,14) | - |
| 5 | `rotacion_alta` | 0.0 | punto | 0.8 | Beta(16,4) | - |
| 6 | `ambos_activos` | 0.8 | Beta(16,4) | 0.8 | Beta(16,4) | - |
| 7 | `confound_rebalanceo` | 0.0 | punto | 0.0 | punto | rebalanceo |
| 8 | `confound_reversion` | 0.0 | punto | 0.0 | punto | reversión |
| 9 | `heterogeneo` | - | `U[0,1]` | - | `U[0,1]` | - |

Además: barrido de monotonicidad `δ ∈ {0, 0.2, 0.4, 0.6, 0.8}` con `κ = 0` y
`κ ∈ {0, 0.2, 0.4, 0.6, 0.8}` con `δ = 0`; y 20 réplicas del escenario nulo con
semillas distintas.

**Confounds (ambos con `δ = 0` y `κ = 0`, es decir sin psicología alguna):**

- *Rebalanceo*: peso objetivo `1/n_i`, banda de no-operar de ±15% relativo.
  Fuera de la banda por arriba recorta la posición hasta el peso objetivo
  (venta **parcial**); por abajo compra para completar. Condiciona en el
  **peso relativo**, nunca en el precio de compra. Es un agente perfectamente
  racional.
- *Reversión*: cree (falsamente, porque los retornos son i.i.d.) que lo que
  subió va a bajar. `h = h_base · (1 + 4.0 · clip(r_pasado_10d / 0.10, −1, +1))`.
  Condiciona en el **retorno reciente**, nunca en el precio de compra.

---

## 3. Expectativas registradas

### 3.1 Estimador de disposición de Odean (1998)

`PGR = ΣG_r/(ΣG_r+ΣG_p)`, `PLR = ΣL_r/(ΣL_r+ΣL_p)`, contados sólo en días con
al menos una venta.

Como `PGR ≈ h_gain` y `PLR ≈ h_loss` (más una inflación por selección al
condicionar en días con venta), la predicción algebraica es:

```
PGR − PLR ≈ 2 · δ · h_0        PGR / PLR ≈ (1+δ)/(1−δ)
```

| Escenario | `PGR−PLR` esperado | signo | `PGR/PLR` esperado |
|---|---|---|---|
| 1 nulo | 0.000 ± ruido | no significativo | ≈ 1.0 |
| 2 δ=0.3 | ≈ +0.009 | positivo, significativo | ≈ 1.9 |
| 3 δ=0.8 | ≈ +0.024 | positivo, significativo | ≈ 9 |
| 4 κ=0.3 | 0.000 ± ruido | **callado** | ≈ 1.0 |
| 5 κ=0.8 | 0.000 ± ruido | **callado** | ≈ 1.0 |
| 6 ambos | ≈ +0.10 (h₀ es ~4× mayor) | positivo y mucho más grande | ≈ 9 |
| 7 rebalanceo | **positivo** (falso positivo) | positivo, significativo | > 1 |
| 8 reversión | **positivo** (falso positivo) | positivo, significativo | > 1 |
| 9 heterogéneo | ≈ +0.015 | positivo | ≈ 3 |

**Predicción central y arriesgada:** `PGR − PLR` **no** es una estimación de `δ`.
Escala con `h_0`, es decir con `κ`. El escenario 6 tendrá un `PGR−PLR` cuatro
veces mayor que el escenario 3 pese a tener **exactamente el mismo `δ`**. Si
eso se confirma, queda demostrado que la magnitud publicada del efecto
disposición en la literatura no es comparable entre poblaciones con distinta
propensión a operar.

### 3.2 Estimador estructural por razón de hazards

```
ĥ_gain = Σ sells_from_gain / Σ position_days_in_gain
ĥ_loss = Σ sells_from_loss / Σ position_days_in_loss
HR = ĥ_gain/ĥ_loss          δ̂ = (HR−1)/(HR+1)
```

Expectativa: **este sí recupera el parámetro inyectado**, con error absoluto
`|δ̂ − E[δ]| < 0.05` en los escenarios 1–6.

Dos fuentes de sesgo previstas, que se van a medir:

1. **Agregación.** `ĥ` es una razón de sumas, no una media de razones. Los
   agentes con `δ` alto acumulan muchos más días-posición en pérdida (lock-in) y
   muchos menos en ganancia, de modo que la razón agregada pondera de forma
   desigual a los agentes. Predicción: en el escenario 9 (`δ ~ U[0,1]`,
   `E[δ] = 0.5`) el `δ̂` agregado saldrá **por encima de 0.5**. Se reporta
   también la media transversal de `δ̂_i` para separar los dos efectos.
2. **Piso de la hazard.** `h_loss` tiene un piso de `1e-4`. Con `δ` cercano a 1
   el piso se activa y comprime `HR` hacia abajo. Afecta sobre todo al escenario 9.

### 3.3 Estimador de sobreconfianza de Barber & Odean (2000)

```
r_i = α + β·Turnover_i + γ₁·ln(W_i) + γ₂·n_i + γ₃·σ_{p,i} + γ₄·cash_i + γ₅·HHI_i + ε_i
```

| Escenario | `β_gross` | `β_net` |
|---|---|---|
| 1 nulo | ≈ 0 | **negativo** (ver advertencia abajo) |
| 2, 3 (δ>0, κ=0) | **positivo** por causalidad reversa | ambiguo |
| 4, 5 (κ>0, δ=0) | ≈ 0 | negativo, más negativo en el 5 |
| 6 | positivo | negativo |
| 7, 8 | ≈ 0 | negativo |
| 9 | ≈ 0 | **negativo y significativo** (resultado Barber-Odean de libro) |

**Advertencia registrada de antemano**: la expectativa ingenua de que en el
escenario nulo "los dos estimadores estén callados" es probablemente falsa para
`β_net`. Aunque `κ = 0` para todos, el turnover realizado sigue variando entre
cuentas por puro azar de los sorteos de venta, y ese turnover cuesta dinero.
La regresión de Barber-Odean encontrará entonces una pendiente negativa
significativa **sin que exista un solo agente sobreconfiado**. Si eso ocurre, el
hallazgo es que la regresión identifica el costo mecánico de operar, no la
sobreconfianza: cualquier fuente de rotación (ruido, rebalanceo, impuestos,
liquidez) produce el mismo coeficiente. Se registra aquí para que no pueda
presentarse después como si se hubiera anticipado trivialmente.

Magnitud esperada de `β_net`: el costo por dólar operado es
`5 bps (medio spread) + comisión/valor de orden ≈ 6–8 bps`. Con turnover anual
`τ`, el costo anual es `≈ 2·τ·0.0007` del valor de la cartera, o sea
`≈ 0.0014·τ` por año y `≈ 0.0028·τ` en los dos años simulados. Con `r_i` medido
como retorno total a dos años, **`β_net` debería caer en el rango
`[−0.004, −0.001]`**. Un valor fuera de ese rango indica un error de contabilidad
de costos o una fuente adicional de pérdida no prevista.

### 3.4 IV / 2SLS

`κ_i` es instrumento válido por construcción: se sortea al azar de un stream
independiente, no toca los precios, y afecta el turnover únicamente por la vía
de la frecuencia de operación.

- `F` de primera etapa: esperado **>> 10** en los escenarios con varianza en `κ`
  (4, 5, 6, 9). **No identificado** en los escenarios con `κ` constante (1, 2, 3,
  7, 8): ahí el instrumento tiene varianza cero y se reportará `n/d`, no un
  número.
- Predicción: `β_IV^gross ≈ 0` mientras que `β_OLS^gross > 0` en los escenarios
  con `δ > 0`. La brecha mide la causalidad reversa.

### 3.5 Diagnósticos de independencia

| Cantidad | Expectativa |
|---|---|
| `Corr(δ, κ)` | ≈ 0 (`|·| < 0.05`), o `n/d` si alguno es degenerado |
| `Corr(δ, Turnover)` | **negativa y apreciable** (`< −0.2`) por lock-in |
| `Corr(κ, Turnover)` | **fuertemente positiva** (`> +0.6`) |

El punto conceptual registrado de antemano: `δ` y `κ` son independientes **como
parámetros inyectados**, pero `δ` contamina el turnover **realizado**, que es un
output. Retener indefinidamente las perdedoras reduce el número de operaciones.

### 3.6 Celdas discriminantes (matriz de firmas)

| Celda | Definición | Disposición | Rebalanceo | Reversión |
|---|---|---|---|---|
| A: "rebote bajo el agua" | en pérdida **y** `r_10d > +2%` | tasa **baja** | ≈ base | tasa **alta** |
| B: "ganadora infraponderada" | en ganancia **y** peso < objetivo·(1−15%) | tasa **alta** | tasa **baja** | ≈ base |
| C: "perdedora sobreponderada" | en pérdida **y** peso > objetivo·(1+15%) | tasa **baja** | tasa **alta** | ≈ base |

Predicción: los escenarios 3, 7 y 8 serán **indistinguibles** por `PGR−PLR`
(los tres dan positivo y significativo) pero **separables** por el patrón de
signos sobre A, B, C.

### 3.7 Contabilidad

Convención base: `average_cost` × `partial_as_full` × referencia `mid`.
Se va a correr la rejilla completa `{fifo, average_cost, per_lot} ×
{partial_as_full, partial_as_fraction} × {mid, ask_paid}` sobre el escenario 3.

Predicción: `PGR−PLR` cambia **poco** entre bases de costo (`< 15%` relativo)
pero cambia **de forma apreciable** con la referencia `ask_paid`, porque comparar
el precio medio de hoy contra el ask pagado desplaza sistemáticamente la
clasificación hacia "pérdida" en medio spread. Bajo `ask_paid` la referencia no
es sólo una convención de medición: también cambia el punto de referencia del
agente, y por lo tanto su conducta.

---

## 4. Predicciones que podrían fallar, y qué significaría

1. **`δ̂` no recupera `δ` en algún escenario.** Si el error supera 0.05 en los
   escenarios 1–6, el estimador estructural está mal especificado o la
   clasificación gana/pierde del estimador no coincide con la que usa el agente.
   Sería un fallo de diseño y hay que encontrarlo, no maquillarlo.
2. **`β_gross` sale positiva.** Hay cinco causas posibles: (i) fuga de
   información, (ii) *cash drag*, (iii) contaminación por spread, (iv) efectos
   de composición, (v) causalidad reversa. Las pruebas de no-fuga, el control de
   `cash_weight`, los dos libros contables paralelos y la comparación OLS/IV
   están diseñados para discriminar entre ellas. Si es causalidad reversa, la
   brecha `β_OLS − β_IV` debe ser grande y `β_IV` debe colapsar a cero.
3. **El escenario nulo rechaza más del 5% de las veces.** Sería un problema de
   tamaño del test, probablemente por errores estándar mal agrupados. Se
   reportaría como hallazgo de primer orden, no se escondería.
4. **`Corr(δ, turnover)` sale cero.** Si la población es degenerada en `δ`, la
   correlación no está definida y se reporta `n/d`. Si no es degenerada y aun
   así sale cero, el mecanismo de lock-in no está funcionando y hay que
   revisar la hazard.
5. **El barrido de monotonicidad no es monótono.** Se reportaría el punto de
   ruptura y se investigaría si es ruido de muestreo o saturación del piso/techo
   de la hazard.
6. **La conservación de valor falla.** Tolerancia 1e-6 sobre
   `efectivo + posiciones + costos pagados = riqueza inicial + P&L de mercado`.
   Cualquier violación es un bug de contabilidad y bloquea el resto.

---

## 5. Compromisos metodológicos

- Los precios se generan **antes** de instanciar el primer agente, con un
  generador propio. Una clase `PriceView` lanza excepción ante cualquier lectura
  de un precio con índice `> current_day`.
- Errores estándar por **bootstrap agrupado por cuenta** (B = 1000). Se reporta
  además, sólo para el escenario 3, el bootstrap por transacción (incorrecto a
  propósito) para cuantificar el factor de subestimación con datos propios.
- Dos libros contables paralelos (bruto al medio sin comisión, neto al
  bid/ask con comisión) con **idénticas cantidades en unidades y fechas**, de
  modo que `r_gross − r_net` sea exactamente el costo de transacción acumulado.
- Todo resultado se interpreta **contra el parámetro inyectado**. Un desacuerdo
  entre lo inyectado y lo estimado es el producto del trabajo.

---

*Fin del pre-análisis. Todo lo que siga en la historia de git es resultado.*
