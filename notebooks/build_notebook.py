"""
Construye ``P01_Analisis.ipynb`` a partir de las celdas definidas aqui.

El notebook se entrega **ejecutado**. Este script solo genera el archivo; la
ejecucion la hace ``make notebook`` (nbconvert --execute --inplace).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

MD = "markdown"
CODE = "code"

CELLS = [
(MD, r"""# P01 - Simulador de finanzas conductuales con *ground truth*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuantRabbit/p01-behavioral-finance/blob/main/notebooks/P01_Analisis.ipynb)

**Alumnos:** Juan Pablo Sánchez, Matteo Nelson, Edgardo González  
**Profesor:** Prof. Luis Felipe Gómez Estrada  

Este cuaderno recorre los resultados de `outputs/`, generados por

```
python run_simulation.py --agents 1000 --days 504 --assets 60 --bootstrap 1000 --seed 42 --all
```

La pregunta del proyecto no es *¿existe el efecto disposición?* sino
**¿cuándo mienten los estimadores estándar?**. Como los parámetros de
comportamiento se inyectan, se conocen; todo desacuerdo entre lo inyectado y lo
estimado es un resultado, no un defecto."""),

(CODE, r"""import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

# Compatibilidad con Google Colab: clona el repositorio si se ejecuta en la nube
ROOT = Path.cwd()
if not (ROOT / "outputs").exists() and not (ROOT.parent / "outputs").exists():
    if "google.colab" in sys.modules or not (ROOT / "outputs").exists():
        print("Entorno Google Colab detectado. Clonando repositorio para acceder a outputs...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/QuantRabbit/p01-behavioral-finance.git"], check=False)
        if (ROOT / "p01-behavioral-finance" / "outputs").exists():
            os.chdir("p01-behavioral-finance")
            ROOT = Path.cwd()

if not (ROOT / "outputs").exists() and (ROOT.parent / "outputs").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
FIG = OUT / "figuras"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

resultados = json.loads((OUT / "resultados_completos.json").read_text(encoding="utf-8"))
tabla = pd.read_csv(OUT / "tabla_principal.csv")
diag = pd.read_csv(OUT / "tabla_diagnosticos.csv")
barrido = pd.read_csv(OUT / "barrido_monotonicidad.csv")
nulos = pd.read_csv(OUT / "replicas_nulo.csv")
disp_reps = pd.read_csv(OUT / "replicas_disposicion_alta.csv")
rejilla = pd.read_csv(OUT / "sensibilidad_contable.csv")
rejilla_ancha = pd.read_csv(OUT / "sensibilidad_contable_spread_ancho.csv")
firmas = pd.read_csv(OUT / "matriz_firmas.csv").set_index("escenario")
clasif = pd.read_csv(OUT / "clasificacion_firmas.csv")

meta = resultados["meta"]
print(f"Semilla maestra: {meta['semilla_maestra']}   Python {meta['python']}   numpy {meta['numpy']}")
print(f"Tiempo total de computo: {meta['tiempo_total_s']:.1f} s")
print(f"Escenarios en la tabla principal: {len(tabla)}")"""),

(MD, r"""## 1. El mercado

Tres factores comunes (mercado, sector, estilo) más ruido idiosincrático, todos
i.i.d. en el tiempo. **No hay momentum ni reversión real**: cualquier patrón
temporal que un agente crea ver es una creencia falsa. Eso es lo que convierte
al escenario 8 en un *confound* genuino y no en una estrategia que funciona."""),

(CODE, r"""display(Image(str(FIG / "fig01_trayectorias_precios.png")))"""),

(CODE, r"""from src.config import MarketConfig, derive_rngs
from src.price_process import generate_market, pooled_autocorrelation

mk = generate_market(MarketConfig(), derive_rngs("esc1_nulo", 42)["prices"])
acf = [pooled_autocorrelation(mk.log_returns, k) for k in range(1, 11)]
print("Autocorrelacion agrupada, rezagos 1..10:")
print(np.round(acf, 5))
print(f"\nmaximo |acf| = {max(abs(a) for a in acf):.5f}")
corr = np.corrcoef(mk.log_returns.T)
print(f"correlacion transversal media entre activos = {corr[~np.eye(60, dtype=bool)].mean():.3f}")"""),

(MD, r"""## 2. Los parámetros inyectados son distribuciones, no escalares

Si todos los agentes compartieran el mismo `delta`, la correlación con `kappa`
saldría exactamente cero **por varianza cero**, no por independencia, y la
regresión de Barber-Odean se quedaría sin variación transversal. En cada
escenario con objetivo distinto de cero, `delta` y `kappa` se sortean de una
Beta reparametrizada por media y concentración."""),

(CODE, r"""display(Image(str(FIG / "fig02_distribucion_delta_kappa.png")))"""),

(CODE, r"""print(tabla[["#", "escenario", "delta_iny_media", "delta_iny_sd",
             "kappa_iny_media", "kappa_iny_sd"]].round(4).to_string(index=False))"""),

(MD, r"""## 3. Tabla principal

Nueve filas: los ocho escenarios canónicos más la población heterogénea
completa."""),

(CODE, r"""cols = ["#", "escenario", "PGR", "PLR", "PGR_menos_PLR", "SE_boot", "z", "PGR_sobre_PLR"]
print(tabla[cols].round(5).to_string(index=False))"""),

(CODE, r"""cols = ["#", "escenario", "delta_iny_media", "delta_hat", "SE_delta_hat",
        "error_recuperacion", "delta_hat_por_agente"]
print(tabla[cols].round(5).to_string(index=False))
print(f"\nError absoluto maximo de recuperacion en escenarios sin confound: "
      f"{tabla.loc[~tabla['#'].isin([7, 8]), 'error_recuperacion'].abs().max():.5f}")"""),

(MD, r"""### El hallazgo central

`PGR - PLR` **no** es una estimación de `delta`. El estimador estructural por
razón de hazards sí lo es. En la figura los escenarios sin *confound* caen sobre
la diagonal de 45 grados; los dos *confounds*, con `delta = 0` inyectado, quedan
muy por encima del cero."""),

(CODE, r"""display(Image(str(FIG / "fig04_delta_recuperado_vs_inyectado.png")))"""),

(CODE, r"""# Mismo delta, distinto kappa: PGR - PLR cambia, delta_hat no.
comp = tabla[tabla["#"].isin([3, 6])][
    ["#", "escenario", "delta_iny_media", "kappa_iny_media",
     "PGR_menos_PLR", "PGR_sobre_PLR", "delta_hat", "turnover_medio"]
]
print(comp.round(4).to_string(index=False))
f3, f6 = comp.iloc[0], comp.iloc[1]
print(f"\nMismo delta inyectado ({f3.delta_iny_media:.3f} vs {f6.delta_iny_media:.3f}) "
      f"pero PGR-PLR cambia un {100 * (f6.PGR_menos_PLR / f3.PGR_menos_PLR - 1):+.0f}%, "
      f"mientras que delta_hat cambia un {100 * (f6.delta_hat / f3.delta_hat - 1):+.2f}%.")"""),

(MD, r"""## 4. Monotonicidad

`PGR - PLR` es monótona creciente en `delta` pero vive en otra escala; `delta_hat`
sigue la diagonal. En el barrido de `kappa` el estimador de disposición se queda
callado, que es la prueba de silencio cruzado."""),

(CODE, r"""display(Image(str(FIG / "fig05_pgr_menos_plr_vs_delta.png")))"""),

(CODE, r"""print(barrido[["parametro", "valor_inyectado", "media_realizada", "PGR_menos_PLR",
               "PGR_sobre_PLR", "delta_hat", "beta_net", "p_beta_net",
               "turnover_medio"]].round(5).to_string(index=False))"""),

(CODE, r"""d = barrido[barrido.parametro == "delta"]
teorico = (1 + d.media_realizada) / (1 - d.media_realizada)
cmp_ratio = pd.DataFrame({
    "delta": d.media_realizada.round(3),
    "PGR/PLR observado": d.PGR_sobre_PLR.round(3),
    "HR teorico (1+d)/(1-d)": teorico.round(3),
    "subestimacion %": (100 * (d.PGR_sobre_PLR / teorico - 1)).round(1),
})
print(cmp_ratio.to_string(index=False))
print("\nLa RAZON PGR/PLR va en la direccion correcta pero subestima la razon de")
print("hazards, y la subestimacion crece con delta: condicionar en dias con venta")
print("no es lo mismo que medir la tasa de riesgo.")"""),

(MD, r"""## 5. Validación bajo el nulo, y por qué hay que desconfiar de los estadísticos z

Veinte réplicas del escenario nulo con semillas distintas: cada réplica es una
economía independiente (trayectoria de precios nueva y población nueva)."""),

(CODE, r"""display(Image(str(FIG / "fig09_bootstrap_nulo.png")))"""),

(CODE, r"""print(f"Replicas: {len(nulos)}")
print(f"Rechazos al 5% de PGR-PLR : {int(nulos.rechaza_5pct.sum())}/{len(nulos)} "
      f"({nulos.rechaza_5pct.mean():.0%})")
print(f"Rechazos al 5% de delta_hat: {int(nulos.rechaza_delta_5pct.sum())}/{len(nulos)}")
print(f"Rechazos al 5% de beta_net : {int(nulos.rechaza_beta_net_5pct.sum())}/{len(nulos)}")
print(f"Cero dentro del IC al 95%  : {int(nulos.cero_dentro_ci.sum())}/{len(nulos)}")"""),

(CODE, r"""res = resultados["resumen_replicas"]
filas = []
for k, v in res.items():
    filas.append({
        "cantidad": k,
        "sd entre trayectorias": v["sd_entre_trayectorias"],
        "SE bootstrap medio": v["se_bootstrap_medio"],
        "factor": v["factor_sd_sobre_se"],
        "replicas": v["n_replicas"],
    })
print(pd.DataFrame(filas).round(6).to_string(index=False))
print('''
Lectura: bajo el nulo el bootstrap por cuenta esta bien calibrado (factor ~1).
Pero en el escenario de disposicion alta la dispersion de PGR-PLR ENTRE
trayectorias es 16.6 veces su error estandar bootstrap. El bootstrap agrupado
por cuenta esta condicionado a UNA sola realizacion del mercado y no puede ver
esa fuente de variabilidad. El z = 91 que reporta la tabla principal es un
z condicional; el z incondicional relevante es del orden de 5.5.
delta_hat, en cambio, tiene factor 1.12: sus z si son honestos.''')"""),

(CODE, r"""boot = resultados["bootstrap_comparacion"]
print("Escenario 3, PGR - PLR:")
print(f"  SE bootstrap agrupado por cuenta : {boot['se_agrupado_por_cuenta']:.6f}  (z = {boot['z_agrupado']:.1f})")
print(f"  SE bootstrap por transaccion     : {boot['se_por_transaccion']:.6f}  (z = {boot['z_por_transaccion']:.1f})")
print(f"  Factor de subestimacion por ignorar el agrupamiento: {boot['factor_subestimacion']:.2f}x")
print(f"\n  Y si ademas se ignora la trayectoria de precios, el factor acumulado es "
      f"{boot['factor_subestimacion'] * res['disp_PGR_menos_PLR']['factor_sd_sobre_se']:.0f}x.")"""),

(MD, r"""## 6. El estadístico de horizontes de tenencia está arrastrado por la deriva

Odean (1998) documenta que los inversionistas retienen las perdedoras más tiempo
que las ganadoras. Bajo el nulo (`delta = kappa = 0`) esa brecha **no** debería
existir. Sobre veinte trayectorias independientes, su signo lo decide el retorno
realizado del mercado."""),

(CODE, r"""der = resultados["deriva_y_brecha_de_tenencia"]
print(nulos[["replica", "retorno_mercado_ew", "tenencia_ganadoras",
             "tenencia_perdedoras", "brecha_tenencia"]].round(3).to_string(index=False))
print(f"\nCorr(retorno del mercado, brecha de tenencia) = {der['corr_retorno_mercado_brecha_tenencia']:+.3f} "
      f"(t = {der['t']:.2f}, n = {der['n_replicas']})")
neg = int((nulos.brecha_tenencia < 0).sum())
print(f"Replicas con brecha negativa, es decir con la lectura clasica de "
      f"'retiene perdedoras': {neg}/{len(nulos)}  -- y en todas ellas delta = 0.")"""),

(MD, r"""## 7. Barber-Odean, causalidad reversa e IV"""),

(CODE, r"""cols = ["#", "escenario", "beta_gross", "p_beta_gross", "beta_net", "p_beta_net",
        "beta_iv_net", "F_primera_etapa"]
print(tabla[cols].round(5).to_string(index=False))
print("\nNaN en las columnas de IV = kappa es degenerado en ese escenario, por lo que")
print("el modelo NO esta identificado. No se reporta un numero inventado.")"""),

(CODE, r"""display(Image(str(FIG / "fig06_turnover_vs_retorno.png")))"""),

(CODE, r"""display(Image(str(FIG / "fig07_placebo_permutacion.png")))"""),

(CODE, r"""cols = ["#", "escenario", "beta_hacia_atras", "t_hacia_atras",
        "beta_ols_net", "beta_iv_net", "brecha_net"]
print(diag[cols].round(5).to_string(index=False))
print('''
La regresion 'hacia atras' regresa el turnover contra el retorno bruto. Si el
turnover causara el retorno y no al reves, no deberia encontrar nada. En los
escenarios con delta > 0 sale fuertemente positiva: la cuenta que tuvo suerte
acumula ganadoras, las vende (porque la regla condiciona en el precio de compra)
y registra mas turnover.''')"""),

(MD, r"""### Diagnóstico de la pendiente bruta positiva

El enunciado enumera cinco causas posibles: fuga de información, *cash drag*,
contaminación por spread, efectos de composición y causalidad reversa. Se
discriminan una por una con las cuentas exportadas."""),

(CODE, r"""cuentas = {k: pd.read_csv(OUT / f"cuentas_{k}.csv")
           for k in ["esc1_nulo", "esc3_disposicion_alta", "esc7_confound_rebalanceo",
                     "esc8_confound_reversion", "esc9_heterogeneo"]}

filas = []
for k, a in cuentas.items():
    filas.append({
        "escenario": k.replace("esc", "").replace("_", " "),
        "corr(turnover, r_gross)": np.corrcoef(a.turnover, a.gross_return)[0, 1],
        "corr(turnover, vol)": np.corrcoef(a.turnover, a.portfolio_vol)[0, 1],
        "corr(turnover, hhi)": np.corrcoef(a.turnover, a.hhi)[0, 1],
        "corr(turnover, capacidad)": np.corrcoef(a.turnover, a.capacity)[0, 1],
        "corr(turnover, efectivo)": np.corrcoef(a.turnover, a.mean_cash_weight)[0, 1],
    })
print(pd.DataFrame(filas).round(3).to_string(index=False))"""),

(CODE, r"""from src.estimators import barber_odean_regression

a7 = cuentas["esc7_confound_rebalanceo"]
r7 = barber_odean_regression(a7, "gross_return")
print("Escenario 7 (rebalanceo), regresion bruta completa:")
for n, b, t in zip(r7["names"], r7["beta"], r7["t"]):
    print(f"   {n:<18} beta = {b:+10.5f}   t = {t:+7.2f}")
print(f"\nR2 = {r7['r2']:.3f}")

r7b = barber_odean_regression(a7, "gross_return", controls=["log_wealth", "capacity"])
print(f"\nMisma regresion sin los controles de volatilidad, efectivo y HHI: "
      f"beta_turnover = {r7b['beta_turnover']:+.5f} (t = {r7b['t_turnover']:.2f})")
print(f"Correlacion simple turnover-retorno bruto: "
      f"{np.corrcoef(a7.turnover, a7.gross_return)[0, 1]:+.4f}")
print('''
Diagnostico: en el escenario 7 la relacion CRUDA entre turnover y retorno bruto
es practicamente nula. La pendiente positiva aparece SOLO al condicionar en la
volatilidad de la cartera, que en esta trayectoria explica casi todo el retorno
(t = 16) y esta negativamente correlacionada con el turnover. Es un efecto de
composicion inducido por los controles, no causalidad reversa ni fuga.''')"""),

(CODE, r"""a3 = cuentas["esc3_disposicion_alta"]
print("Escenario 3 (disposicion alta):")
print(f"  correlacion simple turnover-retorno bruto = {np.corrcoef(a3.turnover, a3.gross_return)[0, 1]:+.4f}")
print(f"  regresion hacia atras: beta = {diag.loc[diag['#'] == 3, 'beta_hacia_atras'].iloc[0]:.3f}, "
      f"t = {diag.loc[diag['#'] == 3, 't_hacia_atras'].iloc[0]:.2f}")
print(f"  IV: {'identificado' if np.isfinite(tabla.loc[tabla['#'] == 3, 'beta_iv_net'].iloc[0]) else 'NO identificado (kappa degenerado)'}")
print('''
Aqui la relacion cruda si existe y es fuerte: es causalidad reversa genuina.
Y no se puede corregir, porque en una poblacion donde todos comparten el mismo
kappa no existe variacion exogena que sirva de instrumento. El econometrista se
queda sin herramienta justo cuando mas la necesita.''')"""),

(CODE, r"""filas = []
for _, r in diag.iterrows():
    filas.append({
        "#": int(r["#"]), "escenario": r["escenario"],
        "pendiente spread observada": r["spread_pendiente_obs"],
        "pendiente spread teorica": r["spread_pendiente_teorica"],
        "error relativo": r["spread_error_rel"],
        "peso de comisiones": r["peso_comisiones"],
        "desplazamiento por cash drag": r["cash_drag_desplazamiento"],
    })
print(pd.DataFrame(filas).round(5).to_string(index=False))
print("\nLa contaminacion por spread queda descartada por construccion: los dos libros")
print("contables paralelos hacen que r_gross - r_net sea exactamente el costo pagado,")
print("y la pendiente observada coincide con la teorica dentro de un 12%.")"""),

(MD, r"""## 8. Independencia y *lock-in*"""),

(CODE, r"""cols = ["#", "escenario", "corr_delta_kappa", "corr_delta_turnover",
        "corr_kappa_turnover", "turnover_medio"]
t = tabla[cols].copy()
print(t.round(4).fillna("n/d").to_string(index=False))
print('''
n/d significa que la variable es degenerada en ese escenario y la correlacion NO
esta definida. Reportar 0.0000 ahi seria afirmar independencia cuando lo que hay
es varianza cero.

delta y kappa son independientes como parametros INYECTADOS. Pero delta
contamina el turnover REALIZADO, que es un output: el agente con delta alto
retiene indefinidamente las perdedoras y opera menos.''')"""),

(MD, r"""## 9. Celdas discriminantes: la sección que sí separa los mecanismos

Los escenarios 3, 7 y 8 son indistinguibles con `PGR - PLR`: los tres dan
positivo y significativo. Pero los tres mecanismos condicionan en variables
distintas, y eso deja firmas separables."""),

(CODE, r"""cols = ["PGR_menos_PLR", "SE_boot", "z", "delta_hat"]
print("Los tres escenarios que PGR - PLR no puede separar:")
print(tabla[tabla["#"].isin([3, 7, 8])][["#", "escenario"] + cols].round(5).to_string(index=False))"""),

(CODE, r"""display(Image(str(FIG / "fig08_matriz_firmas.png")))"""),

(CODE, r"""print(clasif[["escenario", "patron", "mecanismo_por_direccion", "coseno",
              "coseno_segundo", "intensidad_norma"]].round(3).to_string(index=False))
print('''
La clasificacion usa la DIRECCION del vector de tres log-razones (coseno), no su
magnitud: un agente con delta = 0.3 deja la misma firma direccional que uno con
delta = 0.8, solo que mas debil. Cuando la norma es despreciable se dice
'sin mecanismo detectable' en vez de forzar una etiqueta sobre ruido.''')"""),

(MD, r"""## 10. Las decisiones contables mueven la medida estándar

Rejilla completa: tres bases de costo x dos convenciones de conteo de ventas
parciales x dos precios de referencia."""),

(CODE, r"""display(Image(str(FIG / "fig10_sensibilidad_contable.png")))"""),

(CODE, r"""print(rejilla[["base_costo", "conteo_parcial", "referencia", "PGR", "PLR",
               "PGR_menos_PLR", "PGR_sobre_PLR", "delta_hat"]].round(5).to_string(index=False))
r = rejilla["PGR_menos_PLR"]
d = rejilla["delta_hat"]
print(f"\nPGR-PLR: min {r.min():.5f}  max {r.max():.5f}  rango relativo {100 * (r.max() / r.min() - 1):.0f}%")
print(f"delta_hat: min {d.min():.5f}  max {d.max():.5f}  rango relativo {100 * (d.max() / d.min() - 1):.2f}%")
print(f"delta inyectado = {rejilla['delta_inyectado'].iloc[0]:.5f}")"""),

(CODE, r"""ef = resultados["efecto_referencia_por_spread"]
print("Efecto de usar 'ask pagado' en vez de 'precio medio' como referencia:")
print(f"  con spread de {ef['spread_bps_base']:.0f} bps : {ef['delta_PGR_menos_PLR_base']:+.5f}")
print(f"  con spread de {ef['spread_bps_ancho']:.0f} bps: {ef['delta_PGR_menos_PLR_ancho']:+.5f}")
print(f"  factor: {ef['delta_PGR_menos_PLR_ancho'] / ef['delta_PGR_menos_PLR_base']:.1f}x")
print('''
El pre-analisis predijo que la referencia 'ask pagado' moveria PGR-PLR de forma
apreciable. Con 10 bps de spread la prediccion FALLA: el sesgo de medio spread
es de 5 bps, despreciable frente a un movimiento diario tipico de ~1.6%. Al
repetir la rejilla con un spread de mercado ilquido (200 bps) el efecto aparece
y escala casi linealmente con el spread. La prediccion era correcta en el
mecanismo y equivocada en la magnitud para activos liquidos.''')"""),

(MD, r"""## 11. Integridad del diseño"""),

(CODE, r"""esc = resultados["escenarios"]
filas = []
for k, v in esc.items():
    i = v["integrity"]
    filas.append({
        "escenario": v["name"],
        "error de conservacion": i["conservation_error_rel"],
        "bruto-neto = costos": i["gross_minus_net_equals_costs"],
        "descomposicion de costos": i["cost_decomposition_error"],
        "dias-unidad neutros": i["neutral_unit_days"],
        "dias-unidad totales": i["total_unit_days"],
        "fusiones de lote": i["merge_events"],
    })
f = pd.DataFrame(filas)
print(f.to_string(index=False))
print(f"\nMaximo error de conservacion en los nueve escenarios: {f['error de conservacion'].max():.2e}")
print(f"Posiciones exactamente al costo: {f['dias-unidad neutros'].sum():.0f} de "
      f"{f['dias-unidad totales'].sum():.0f} dias-unidad (con precios continuos la")
print("igualdad exacta no ocurre nunca; la regla existe y esta probada con un caso")
print("construido a mano en tests/test_portfolio.py).")"""),

(CODE, r"""print("Lotes abiertos por agente y fusiones (los lotes multiples existen de verdad):")
for k, v in esc.items():
    print(f"  {v['name']:<24} lotes/agente = {v['num_lots_opened_mean']:7.1f}   "
          f"fusiones = {v['integrity']['merge_events']:.0f}")"""),

(MD, r"""## 12. Resumen

1. **El estimador clásico no recupera el parámetro; el estructural sí.**
   `delta_hat` reproduce `delta` con error absoluto máximo de 0.006 en los siete
   escenarios sin *confound*, mientras que `PGR - PLR` cambia un 53% entre dos
   escenarios con **el mismo** `delta` y distinta propensión a operar.
2. **Los confounds disparan falsos positivos en los dos estimadores.** Con
   `delta = 0` inyectado, el rebalanceo produce `PGR - PLR = +0.012` y
   `delta_hat = +0.16`; la creencia en reversión produce `+0.041` y `+0.38`.
   Ninguno de los dos agentes tiene psicología alguna.
3. **Las celdas discriminantes sí separan los tres mecanismos**, con firmas
   direccionales que no se confunden entre sí (coseno ≥ 0.97 con el mecanismo
   correcto y ≤ 0.02 con el siguiente)."""),
]


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == MD else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    destino = Path(__file__).resolve().parent / "P01_Analisis.ipynb"
    nbf.write(nb, destino)
    return destino


if __name__ == "__main__":
    print(build())
