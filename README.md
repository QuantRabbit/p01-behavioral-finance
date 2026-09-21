# P01 - Simulador de finanzas conductuales con *ground truth*

Mercado sintético basado en agentes cuyo propósito es **auditar estimadores
econométricos de sesgos conductuales usando parámetros conocidos**. Con datos
reales de transacciones no hay forma de distinguir un sesgo psicológico de una
regla institucional o de un error de código. Aquí los parámetros se inyectan, de
modo que la pregunta deja de ser *¿existe el efecto disposición?* y pasa a ser
**¿cuándo mienten los estimadores?**.

- Alumnos: Juan Pablo Sánchez, Matteo Nelson, Edgardo González
- Profesor: Prof. Luis Felipe Gómez Estrada
- Reporte completo: [`REPORTE.md`](REPORTE.md)
- Registro previo: [`PRE_ANALISIS.md`](PRE_ANALISIS.md) (commit `2d5dfc9`, anterior a cualquier resultado)
- Cuaderno ejecutado: [`notebooks/P01_Analisis.ipynb`](notebooks/P01_Analisis.ipynb)
- Ejecutar en la nube: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantRabbit/p01-behavioral-finance/blob/main/notebooks/P01_Analisis.ipynb)

---

## Ejecución interactiva en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantRabbit/p01-behavioral-finance/blob/main/notebooks/P01_Analisis.ipynb)

Para abrir y recorrer el cuaderno completo de análisis en el navegador sin requerir instalación local:
1. Clic en el botón superior **Open in Colab**.
2. En el menú de Colab, presionar **Entorno de ejecución → Ejecutar todas** (*Runtime → Run all*).
3. El cuaderno se autoconfigura, cargando los resultados de `outputs/` y visualizando las tablas y las diez figuras analíticas.

---

## Instalación y ejecución

```bash
pip install -r requirements.txt
python run_simulation.py --agents 1000 --days 504 --assets 60 --bootstrap 1000 --seed 42 --all
```

Ese comando reproduce **todo** `outputs/` desde cero en unos 6.5 minutos. Atajos:

```bash
make test       # bateria completa de pruebas (129)
make run        # el comando de arriba
make figures    # regenera solo las figuras desde outputs/
make notebook   # ejecuta el cuaderno y lo guarda con sus salidas
make all        # test + run + notebook
```

Banderas útiles de `run_simulation.py`:

| bandera | efecto |
|---|---|
| `--quick` | corrida de humo (250 agentes, 120 días, 30 activos) |
| `--commission` / `--spread-bps` | análisis de sensibilidad a los costos |
| `--cost-basis {average_cost,fifo,per_lot}` | base de costo por defecto |
| `--partial-counting`, `--reference` | convenciones de conteo y de precio de referencia |
| `--null-reps`, `--disp-reps` | número de réplicas con semillas distintas |
| `--scenarios-only`, `--no-figures`, `--figures-only` | control de etapas |

---

## Estructura

```
p01-behavioral-finance/
├── PRE_ANALISIS.md            registro previo (commiteado antes de los resultados)
├── REPORTE.md                 reporte final
├── run_simulation.py          CLI principal
├── src/
│   ├── config.py              dataclasses de configuracion y semillas por stream
│   ├── price_process.py       mercado multifactorial + PriceView anti-lookahead
│   ├── agents.py              poblacion, hazard rates y confounds
│   ├── portfolio.py           lotes, bases de costo, ventas parciales, conteos de Odean
│   ├── simulator.py           motor diario vectorizado con dos libros contables
│   ├── estimators.py          Odean, razon de hazards, Barber-Odean HC1, IV 2SLS
│   ├── diagnostics.py         independencia, causalidad reversa, celdas discriminantes
│   ├── scenarios.py           escenarios, barridos, replicas, rejilla contable
│   └── plots.py               las diez figuras
├── tests/                     129 pruebas
├── notebooks/                 cuaderno ejecutado + script que lo genera
└── outputs/                   tablas, JSON, CSV y figuras
```

---

## Las decisiones de diseño que importan

**Cero fuga de información.** La matriz de precios se genera con su propio
generador **antes** de instanciar el primer agente. Los agentes sólo la ven a
través de `PriceView`, cuyo reloj fija el motor y que levanta `LookAheadError`
ante cualquier lectura de un día futuro o ante cualquier intento de hacer
retroceder el reloj. Hay pruebas de que la excepción se levanta, de que los
retornos no tienen autocorrelación en los rezagos 1 a 10 y de que los activos
comprados no baten al mercado entre trayectorias independientes.

**El parámetro está un nivel por debajo de lo estimado.** No existe ningún
`if ganancia: vender con probabilidad p`. `δ` y `κ` modulan una tasa de riesgo
diaria; `PGR`, `PLR`, el turnover y los horizontes de tenencia **emergen**.

**Distribuciones, no escalares.** `δ_i` y `κ_i` se sortean de Betas
reparametrizadas por media y concentración desde streams independientes. Cuando
el objetivo es 0 el valor es exactamente 0 para todos y las correlaciones se
reportan como `n/d`, nunca como `0.0000`.

**Dos libros contables en paralelo.** Neto (bid/ask con comisión) y bruto
(precio medio, sin comisión), con idénticas decisiones, fechas y cantidades en
unidades, de modo que `r_gross − r_net` sea exactamente el costo acumulado.
La conservación de valor se verifica con error relativo de ~1e-15.

**Reproducibilidad.** Semilla maestra 42; por escenario se derivan cinco streams
nombrados (`prices`, `population`, `decisions`, `bootstrap`, `placebo`) con
`SeedSequence`. Dos corridas con la misma semilla dan resultados idénticos bit a
bit, y hay una prueba unitaria que lo verifica.

---

## Salidas

| archivo | contenido |
|---|---|
| `tabla_principal.{md,csv}` | 9 escenarios × estimadores, errores estándar y correlaciones |
| `tabla_diagnosticos.{md,csv}` | causalidad reversa, placebo, cash drag, spread, celdas |
| `resultados_completos.json` | todo lo anterior más metadatos de la corrida |
| `barrido_monotonicidad.{md,csv}` | 5 puntos en `δ` y 5 en `κ` |
| `replicas_nulo.{md,csv}` | 20 réplicas del nulo con semillas distintas |
| `replicas_disposicion_alta.{md,csv}` | 10 réplicas del escenario 3 (variación entre trayectorias) |
| `sensibilidad_contable{,_spread_ancho}.{md,csv}` | rejilla 3 bases × 2 conteos × 2 referencias |
| `matriz_firmas.{md,csv}`, `distancias_firmas`, `clasificacion_firmas` | celdas discriminantes |
| `cuentas_esc*.csv` | métricas por cuenta de cinco escenarios |
| `figuras/fig01..fig10.png` | las diez figuras del reporte |

---

## Resultados en una línea

El estimador estructural por razón de hazards recupera `δ` con error ≤ 0.014;
`PGR − PLR` cambia 53% entre dos poblaciones con el **mismo** `δ`, varía 75% con
las convenciones contables, tiene errores estándar inflados unas 35 veces y da
falsos positivos ante mecanismos sin psicología. Las tres celdas discriminantes
sí separan disposición, rebalanceo y creencia en reversión. Detalle completo en
[`REPORTE.md`](REPORTE.md).
