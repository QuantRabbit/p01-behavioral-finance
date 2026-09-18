"""
Definicion y ejecucion de los escenarios, barridos y replicas.

Nueve escenarios canonicos (ocho de la tabla del enunciado mas la poblacion
heterogenea completa), un barrido de monotonicidad en ``delta`` y en ``kappa``,
veinte replicas del escenario nulo con semillas distintas y la rejilla de
sensibilidad contable sobre el escenario de disposicion alta.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .agents import H_LOSS_FLOOR
from .config import (
    AccountingConfig,
    EstimationConfig,
    MASTER_SEED,
    PopulationConfig,
    SimConfig,
    derive_rngs,
)
from .diagnostics import (
    backward_regression,
    cash_drag_isolation,
    discriminating_cells,
    independence_diagnostics,
    match_a_priori,
    nearest_reference,
    ols_vs_iv,
    pattern_str,
    permutation_placebo,
    signature_distances,
    signature_matrix,
    spread_decomposition,
)
from .estimators import (
    barber_odean_regression,
    delta_hat_per_agent,
    estimate_hazard_ratio,
    estimate_odean,
    estimate_odean_transaction_bootstrap,
    iv_2sls,
)
from .simulator import CELL_NAMES, SimulationResult, run_simulation


# ---------------------------------------------------------------------------
# Especificacion de escenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    number: int
    name: str
    delta_target: float
    delta_dist: str
    kappa_target: float
    kappa_dist: str
    confound: Optional[str]
    expectation: str


SCENARIOS: Sequence[ScenarioSpec] = (
    ScenarioSpec("esc1_nulo", 1, "Nulo", 0.0, "point", 0.0, "point", None,
                 "Ambos estimadores callados"),
    ScenarioSpec("esc2_disposicion_baja", 2, "Disposicion baja", 0.3, "beta", 0.0, "point", None,
                 "Disposicion positiva, sobreconfianza callada"),
    ScenarioSpec("esc3_disposicion_alta", 3, "Disposicion alta", 0.8, "beta", 0.0, "point", None,
                 "Disposicion positiva y mayor, sobreconfianza callada"),
    ScenarioSpec("esc4_rotacion_baja", 4, "Rotacion baja", 0.0, "point", 0.3, "beta", None,
                 "Disposicion callada, beta neta < 0"),
    ScenarioSpec("esc5_rotacion_alta", 5, "Rotacion alta", 0.0, "point", 0.8, "beta", None,
                 "Disposicion callada, beta neta mas negativa"),
    ScenarioSpec("esc6_ambos_activos", 6, "Ambos activos", 0.8, "beta", 0.8, "beta", None,
                 "Interaccion de ambos estimadores"),
    ScenarioSpec("esc7_confound_rebalanceo", 7, "Confound rebalanceo", 0.0, "point", 0.0, "point",
                 "rebalancing", "Disposicion dispara falso positivo"),
    ScenarioSpec("esc8_confound_reversion", 8, "Confound reversion", 0.0, "point", 0.0, "point",
                 "mean_reversion", "Disposicion dispara falso positivo"),
    ScenarioSpec("esc9_heterogeneo", 9, "Poblacion heterogenea", 0.5, "uniform", 0.5, "uniform", None,
                 "Corr(d,k)~0, Corr(d,turnover)<0, Corr(k,turnover)>0, beta_net<0"),
)

SCENARIO_BY_KEY = {s.key: s for s in SCENARIOS}


def make_config(spec: ScenarioSpec, base: SimConfig) -> SimConfig:
    """Config del escenario sobre una base comun (mercado, costos, contabilidad)."""
    pop = replace(
        base.population,
        delta_target=spec.delta_target,
        delta_dist=spec.delta_dist,
        kappa_target=spec.kappa_target,
        kappa_dist=spec.kappa_dist,
        confound=spec.confound,
    )
    return replace(base, population=pop)


# ---------------------------------------------------------------------------
# Analisis de una corrida
# ---------------------------------------------------------------------------
def analyze(
    result: SimulationResult,
    spec: ScenarioSpec,
    est: EstimationConfig,
    rngs: Dict[str, np.random.Generator],
    full_diagnostics: bool = True,
) -> Dict[str, object]:
    """Corre todos los estimadores y diagnosticos sobre una simulacion."""
    a = result.accounts
    rng_boot = rngs["bootstrap"]
    rng_plac = rngs["placebo"]

    odean = estimate_odean(a, est.n_bootstrap, rng_boot, est.alpha)
    hazard = estimate_hazard_ratio(a, est.n_bootstrap, rng_boot, est.alpha)
    per_agent = delta_hat_per_agent(a)

    reg_gross = barber_odean_regression(a, "gross_return")
    reg_net = barber_odean_regression(a, "net_return")
    reg_gross_comp = barber_odean_regression(a, "gross_return_comp")
    iv_net = iv_2sls(a, "net_return")
    iv_gross = iv_2sls(a, "gross_return")

    indep = independence_diagnostics(result)
    delta_iny = a["delta"].to_numpy(float)

    out: Dict[str, object] = {
        "key": spec.key,
        "number": spec.number,
        "name": spec.name,
        "expectation": spec.expectation,
        "confound": spec.confound,
        "n_agents": int(len(a)),
        "population": result.population.summary(),
        "integrity": result.integrity,
        "independence": indep,
        "odean": odean,
        "hazard": hazard,
        "delta_inyectado_media": float(delta_iny.mean()),
        "delta_inyectado_sd": float(delta_iny.std(ddof=1)),
        "delta_hat_por_agente_media": float(np.nanmean(per_agent)),
        "delta_hat_por_agente_n": int(np.sum(np.isfinite(per_agent))),
        "error_recuperacion": float(hazard["delta_hat"]["estimate"] - delta_iny.mean()),
        "reg_gross": _reg_row(reg_gross),
        "reg_net": _reg_row(reg_net),
        "reg_gross_comp": _reg_row(reg_gross_comp),
        "iv_net": _iv_row(iv_net),
        "iv_gross": _iv_row(iv_gross),
        "turnover_mean": float(a["turnover"].mean()),
        "net_return_mean": float(a["net_return"].mean()),
        "gross_return_mean": float(a["gross_return"].mean()),
        "gross_return_comp_mean": float(a["gross_return_comp"].mean()),
        "median_holding_winners": float(np.nanmedian(a["median_holding_winners"])),
        "median_holding_losers": float(np.nanmedian(a["median_holding_losers"])),
        "mean_holding_winners": float(np.nanmean(a["mean_holding_winners"])),
        "mean_holding_losers": float(np.nanmean(a["mean_holding_losers"])),
        "brecha_tenencia": float(
            np.nanmedian(a["median_holding_winners"]) - np.nanmedian(a["median_holding_losers"])
        ),
        "market_return_ew": float(result.market.equal_weight_index()[-1] / 100.0 - 1.0),
        "trade_count_mean": float(a["trade_count"].mean()),
        "num_lots_opened_mean": float(a["num_lots_opened"].mean()),
    }

    if full_diagnostics:
        out["backward"] = backward_regression(a, "gross_return")
        out["ols_vs_iv"] = ols_vs_iv(a)
        placebo = permutation_placebo(a, rng_plac, "net_return", est.n_placebo)
        out["placebo_betas"] = placebo.pop("betas")
        out["placebo"] = placebo
        out["cash_drag"] = cash_drag_isolation(a)
        out["spread"] = spread_decomposition(a, result.config.costs, result.config.market.n_days)
        out["cells"] = discriminating_cells(result, est.n_bootstrap, rng_boot)
        out["odean_conventions"] = _odean_by_convention(a)
    return out


def _reg_row(reg: Dict[str, object]) -> Dict[str, float]:
    return {
        "beta": float(reg["beta_turnover"]),
        "se": float(reg["se_turnover"]),
        "t": float(reg["t_turnover"]),
        "p": float(reg["p_turnover"]),
        "alpha": float(reg["alpha"]),
        "r2": float(reg["r2"]),
        "n": int(reg["n"]),
        "coeficientes": {n: float(b) for n, b in zip(reg["names"], reg["beta"])},
        "errores": {n: float(s) for n, s in zip(reg["names"], reg["se"])},
    }


def _iv_row(iv: Dict[str, object]) -> Dict[str, object]:
    return {
        "identificado": bool(iv["identified"]),
        "beta": float(iv["beta_turnover"]),
        "se": float(iv["se_turnover"]),
        "t": float(iv["t_turnover"]),
        "p": float(iv["p_turnover"]),
        "first_stage_F": float(iv["first_stage_F"]),
    }


def _odean_by_convention(a: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """PGR, PLR y su diferencia bajo las dos convenciones de venta parcial."""
    out = {}
    for conv, suf in (("partial_as_full", "full"), ("partial_as_fraction", "frac")):
        g_r, g_p = a[f"G_r_{suf}"].sum(), a[f"G_p_{suf}"].sum()
        l_r, l_p = a[f"L_r_{suf}"].sum(), a[f"L_p_{suf}"].sum()
        pgr = g_r / (g_r + g_p) if (g_r + g_p) > 0 else np.nan
        plr = l_r / (l_r + l_p) if (l_r + l_p) > 0 else np.nan
        out[conv] = {"PGR": float(pgr), "PLR": float(plr), "diff": float(pgr - plr),
                     "ratio": float(pgr / plr) if plr else np.nan}
    return out


# ---------------------------------------------------------------------------
# Ejecucion
# ---------------------------------------------------------------------------
def run_scenario(
    spec: ScenarioSpec,
    base_cfg: SimConfig,
    est: EstimationConfig,
    master_seed: int = MASTER_SEED,
    full_diagnostics: bool = True,
    keep_result: bool = False,
):
    """Corre un escenario completo y devuelve ``(analisis, resultado|None)``."""
    cfg = make_config(spec, base_cfg)
    rngs = derive_rngs(spec.key, master_seed)
    t0 = time.perf_counter()
    result = run_simulation(cfg, rngs, spec.key)
    sim_time = time.perf_counter() - t0
    analysis = analyze(result, spec, est, rngs, full_diagnostics)
    analysis["runtime_sim_s"] = float(sim_time)
    analysis["runtime_total_s"] = float(time.perf_counter() - t0)
    return analysis, (result if keep_result else None)


def main_table_row(an: Dict[str, object]) -> Dict[str, object]:
    """Una fila de la tabla principal de resultados."""
    ind = an["independence"]
    od = an["odean"]
    hz = an["hazard"]
    return {
        "#": an["number"],
        "escenario": an["name"],
        "clave": an["key"],
        "delta_iny_media": an["delta_inyectado_media"],
        "delta_iny_sd": an["delta_inyectado_sd"],
        "kappa_iny_media": ind["kappa_mean"],
        "kappa_iny_sd": ind["kappa_sd"],
        "PGR": od["PGR"]["estimate"],
        "SE_PGR": od["PGR"]["se"],
        "PLR": od["PLR"]["estimate"],
        "SE_PLR": od["PLR"]["se"],
        "PGR_menos_PLR": od["PGR_menos_PLR"]["estimate"],
        "SE_boot": od["PGR_menos_PLR"]["se"],
        "z": od["PGR_menos_PLR"]["z"],
        "p": od["PGR_menos_PLR"]["p"],
        "PGR_sobre_PLR": od["PGR_sobre_PLR"]["estimate"],
        "delta_hat": hz["delta_hat"]["estimate"],
        "SE_delta_hat": hz["delta_hat"]["se"],
        "error_recuperacion": an["error_recuperacion"],
        "delta_hat_por_agente": an["delta_hat_por_agente_media"],
        "beta_gross": an["reg_gross"]["beta"],
        "SE_beta_gross": an["reg_gross"]["se"],
        "p_beta_gross": an["reg_gross"]["p"],
        "beta_net": an["reg_net"]["beta"],
        "SE_beta_net": an["reg_net"]["se"],
        "p_beta_net": an["reg_net"]["p"],
        "beta_iv_net": an["iv_net"]["beta"],
        "SE_beta_iv_net": an["iv_net"]["se"],
        "F_primera_etapa": an["iv_net"]["first_stage_F"],
        "corr_delta_kappa": ind["corr_delta_kappa"],
        "corr_delta_turnover": ind["corr_delta_turnover"],
        "corr_kappa_turnover": ind["corr_kappa_turnover"],
        "turnover_medio": an["turnover_mean"],
        "retorno_neto_medio": an["net_return_mean"],
        "retorno_mercado_ew": an["market_return_ew"],
        "tenencia_ganadoras": an["median_holding_winners"],
        "tenencia_perdedoras": an["median_holding_losers"],
    }


def diagnostics_table_row(an: Dict[str, object]) -> Dict[str, object]:
    bw = an.get("backward", {})
    pl = an.get("placebo", {})
    cd = an.get("cash_drag", {})
    sp = an.get("spread", {})
    ce = an.get("cells", {})
    ovi = an.get("ols_vs_iv", {})
    row = {
        "#": an["number"],
        "escenario": an["name"],
        "beta_hacia_atras": bw.get("beta"),
        "t_hacia_atras": bw.get("t"),
        "beta_ols_gross": ovi.get("beta_ols_gross"),
        "beta_iv_gross": ovi.get("beta_iv_gross"),
        "brecha_gross": ovi.get("brecha_gross"),
        "beta_ols_net": ovi.get("beta_ols_net"),
        "beta_iv_net": ovi.get("beta_iv_net"),
        "brecha_net": ovi.get("brecha_net"),
        "placebo_media": pl.get("placebo_media"),
        "placebo_sd": pl.get("placebo_sd"),
        "placebo_ci_lo": pl.get("placebo_ci_lo"),
        "placebo_ci_hi": pl.get("placebo_ci_hi"),
        "observado_fuera_banda": pl.get("observado_fuera_de_la_banda"),
        "cash_weight_medio": cd.get("cash_weight_medio"),
        "cash_drag_desplazamiento": cd.get("desplazamiento"),
        "spread_pendiente_obs": sp.get("pendiente_spread"),
        "spread_pendiente_teorica": sp.get("pendiente_spread_teorica"),
        "spread_error_rel": sp.get("error_relativo_spread"),
        "peso_comisiones": sp.get("peso_comisiones"),
        "tasa_base_venta": ce.get("tasa_base"),
    }
    for cell in CELL_NAMES:
        if cell in ce:
            row[f"{cell}_tasa"] = ce[cell]["estimate"]
            row[f"{cell}_log_ratio"] = ce[cell]["log_ratio_vs_base"]
    return row


# ---------------------------------------------------------------------------
# Barrido de monotonicidad
# ---------------------------------------------------------------------------
SWEEP_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8)


def run_sweep(base_cfg: SimConfig, est: EstimationConfig, master_seed: int = MASTER_SEED) -> pd.DataFrame:
    """Barrido en ``delta`` (con kappa = 0) y en ``kappa`` (con delta = 0)."""
    rows: List[Dict[str, object]] = []
    for parametro in ("delta", "kappa"):
        for valor in SWEEP_VALUES:
            dist = "point" if valor == 0.0 else "beta"
            spec = ScenarioSpec(
                key=f"barrido_{parametro}_{valor:.1f}",
                number=0,
                name=f"barrido {parametro}={valor:.1f}",
                delta_target=valor if parametro == "delta" else 0.0,
                delta_dist=dist if parametro == "delta" else "point",
                kappa_target=valor if parametro == "kappa" else 0.0,
                kappa_dist=dist if parametro == "kappa" else "point",
                confound=None,
                expectation="monotonicidad",
            )
            an, _ = run_scenario(spec, base_cfg, est, master_seed, full_diagnostics=False)
            rows.append({
                "parametro": parametro,
                "valor_inyectado": valor,
                "media_realizada": an["delta_inyectado_media"] if parametro == "delta"
                else an["independence"]["kappa_mean"],
                "sd_realizada": an["delta_inyectado_sd"] if parametro == "delta"
                else an["independence"]["kappa_sd"],
                "PGR": an["odean"]["PGR"]["estimate"],
                "PLR": an["odean"]["PLR"]["estimate"],
                "PGR_menos_PLR": an["odean"]["PGR_menos_PLR"]["estimate"],
                "SE_PGR_menos_PLR": an["odean"]["PGR_menos_PLR"]["se"],
                "PGR_sobre_PLR": an["odean"]["PGR_sobre_PLR"]["estimate"],
                "delta_hat": an["hazard"]["delta_hat"]["estimate"],
                "SE_delta_hat": an["hazard"]["delta_hat"]["se"],
                "beta_net": an["reg_net"]["beta"],
                "SE_beta_net": an["reg_net"]["se"],
                "p_beta_net": an["reg_net"]["p"],
                "beta_gross": an["reg_gross"]["beta"],
                "turnover_medio": an["turnover_mean"],
            })
    return pd.DataFrame(rows)


def check_monotonicity(sweep: pd.DataFrame, parametro: str, columna: str) -> Dict[str, object]:
    """Verifica monotonia y reporta el punto de ruptura si lo hay."""
    sub = sweep[sweep["parametro"] == parametro].sort_values("valor_inyectado")
    y = sub[columna].to_numpy(float)
    d = np.diff(y)
    creciente = bool(np.all(d >= 0))
    decreciente = bool(np.all(d <= 0))
    rupturas = [
        {"desde": float(sub["valor_inyectado"].iloc[i]),
         "hasta": float(sub["valor_inyectado"].iloc[i + 1]),
         "salto": float(d[i])}
        for i in range(len(d))
        if (d[i] < 0 and not decreciente) or (d[i] > 0 and not creciente)
    ]
    return {"parametro": parametro, "columna": columna, "monotona_creciente": creciente,
            "monotona_decreciente": decreciente, "rupturas": rupturas,
            "valores": [float(v) for v in y]}


# ---------------------------------------------------------------------------
# Replicas del escenario nulo
# ---------------------------------------------------------------------------
def run_replicates(
    plantilla: ScenarioSpec,
    base_cfg: SimConfig,
    est: EstimationConfig,
    n_reps: int = 20,
    master_seed: int = MASTER_SEED,
    prefijo: str = "rep",
) -> pd.DataFrame:
    """Repite un escenario con ``n_reps`` semillas distintas.

    Cada replica es una **economia independiente**: trayectoria de precios nueva
    y poblacion nueva. Sirve para dos cosas: medir la tasa de rechazo empirica
    bajo el nulo y medir la variabilidad **entre trayectorias**, que el bootstrap
    por cuenta (condicionado a una sola trayectoria) no puede ver.
    """
    rows = []
    for i in range(n_reps):
        spec = replace(plantilla, key=f"{prefijo}_{i:02d}", name=f"{plantilla.name} replica {i}")
        an, _ = run_scenario(spec, base_cfg, est, master_seed, full_diagnostics=False)
        od = an["odean"]["PGR_menos_PLR"]
        hz = an["hazard"]["delta_hat"]
        rows.append({
            "replica": i,
            "delta_inyectado": an["delta_inyectado_media"],
            "PGR": an["odean"]["PGR"]["estimate"],
            "PLR": an["odean"]["PLR"]["estimate"],
            "PGR_menos_PLR": od["estimate"],
            "SE": od["se"],
            "z": od["z"],
            "p": od["p"],
            "rechaza_5pct": bool(od["p"] < 0.05) if np.isfinite(od["p"]) else False,
            "ci_lo": od["ci_lo"],
            "ci_hi": od["ci_hi"],
            "cero_dentro_ci": bool(od["ci_lo"] <= 0.0 <= od["ci_hi"]),
            "delta_hat": hz["estimate"],
            "SE_delta_hat": hz["se"],
            "p_delta_hat": hz["p"],
            "rechaza_delta_5pct": bool(hz["p"] < 0.05) if np.isfinite(hz["p"]) else False,
            "beta_net": an["reg_net"]["beta"],
            "p_beta_net": an["reg_net"]["p"],
            "rechaza_beta_net_5pct": bool(an["reg_net"]["p"] < 0.05),
            "turnover_medio": an["turnover_mean"],
            "error_recuperacion": an["error_recuperacion"],
            "retorno_mercado_ew": an["market_return_ew"],
            "tenencia_ganadoras": an["median_holding_winners"],
            "tenencia_perdedoras": an["median_holding_losers"],
            "brecha_tenencia": an["brecha_tenencia"],
        })
    return pd.DataFrame(rows)


def run_null_replicates(
    base_cfg: SimConfig, est: EstimationConfig, n_reps: int = 20, master_seed: int = MASTER_SEED
) -> pd.DataFrame:
    """Replicas del escenario nulo con semillas distintas."""
    return run_replicates(SCENARIO_BY_KEY["esc1_nulo"], base_cfg, est, n_reps, master_seed, "nulo_rep")


def replicate_summary(df: pd.DataFrame, columna: str, columna_se: str) -> Dict[str, float]:
    """Compara la dispersion ENTRE trayectorias contra el error estandar bootstrap.

    El bootstrap agrupado por cuenta esta condicionado a una unica realizacion
    del mercado. Si la dispersion entre trayectorias supera al error estandar
    bootstrap, los estadisticos z de una sola corrida estan inflados y hay que
    decirlo.
    """
    sd_entre = float(df[columna].std(ddof=1))
    se_medio = float(df[columna_se].mean())
    return {
        "media": float(df[columna].mean()),
        "sd_entre_trayectorias": sd_entre,
        "se_bootstrap_medio": se_medio,
        "factor_sd_sobre_se": sd_entre / se_medio if se_medio > 0 else np.nan,
        "n_replicas": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Rejilla de sensibilidad contable
# ---------------------------------------------------------------------------
def run_accounting_grid(
    base_cfg: SimConfig,
    est: EstimationConfig,
    spec: ScenarioSpec,
    master_seed: int = MASTER_SEED,
) -> pd.DataFrame:
    """{fifo, average_cost, per_lot} x {parcial completo, parcial fraccion} x {mid, ask}.

    Las dos convenciones de conteo de venta parcial no cambian la conducta, solo
    el conteo, asi que se obtienen de la misma corrida. La base de costo y la
    referencia si cambian la conducta (determinan la clasificacion que ve el
    agente), por lo que requieren simular de nuevo.
    """
    rows = []
    for base in ("fifo", "average_cost", "per_lot"):
        for referencia in ("mid", "ask_paid"):
            acc = replace(base_cfg.accounting, cost_basis=base, reference=referencia)
            cfg = make_config(spec, replace(base_cfg, accounting=acc))
            key = f"{spec.key}__{base}__{referencia}"
            rngs = derive_rngs(spec.key, master_seed)   # misma trayectoria y poblacion
            res = run_simulation(cfg, rngs, key)
            conv = _odean_by_convention(res.accounts)
            hz = estimate_hazard_ratio(res.accounts, est.n_bootstrap, rngs["bootstrap"], est.alpha)
            for nombre_conv, valores in conv.items():
                rows.append({
                    "base_costo": base,
                    "conteo_parcial": nombre_conv,
                    "referencia": referencia,
                    "PGR": valores["PGR"],
                    "PLR": valores["PLR"],
                    "PGR_menos_PLR": valores["diff"],
                    "PGR_sobre_PLR": valores["ratio"],
                    "delta_hat": hz["delta_hat"]["estimate"],
                    "SE_delta_hat": hz["delta_hat"]["se"],
                    "delta_inyectado": float(res.accounts["delta"].mean()),
                    "turnover_medio": float(res.accounts["turnover"].mean()),
                    "dias_neutros": res.integrity["neutral_unit_days"],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparacion de bootstrap agrupado vs por transaccion
# ---------------------------------------------------------------------------
def bootstrap_comparison(
    accounts: pd.DataFrame, est: EstimationConfig, rng: np.random.Generator
) -> Dict[str, float]:
    clustered = estimate_odean(accounts, est.n_bootstrap, rng, est.alpha)
    naive = estimate_odean_transaction_bootstrap(accounts, est.n_bootstrap, rng, est.alpha)
    se_c = clustered["PGR_menos_PLR"]["se"]
    se_n = naive["PGR_menos_PLR"]["se"]
    return {
        "se_agrupado_por_cuenta": se_c,
        "se_por_transaccion": se_n,
        "factor_subestimacion": se_c / se_n,
        "z_agrupado": clustered["PGR_menos_PLR"]["z"],
        "z_por_transaccion": naive["PGR_menos_PLR"]["z"],
    }


# ---------------------------------------------------------------------------
# Exportacion
# ---------------------------------------------------------------------------
def to_markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = False) -> str:
    """Tabla markdown sin dependencias externas."""
    d = df.reset_index() if index else df

    def fmt(v):
        if v is None:
            return "n/d"
        if isinstance(v, float):
            if not np.isfinite(v):
                return "n/d"
            return floatfmt.format(v)
        if isinstance(v, (np.floating,)):
            return fmt(float(v))
        if isinstance(v, (bool, np.bool_)):
            return "si" if v else "no"
        return str(v)

    cols = list(d.columns)
    lineas = ["| " + " | ".join(str(c) for c in cols) + " |",
              "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lineas.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lineas)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    raise TypeError(f"no serializable: {type(obj)}")


def _sanitize(obj):
    """Convierte NaN/Inf en ``null`` de forma recursiva.

    ``json.dump`` con ``allow_nan=False`` no pasa los flotantes nativos por
    ``default``, asi que la limpieza tiene que hacerse antes. Un ``null`` es la
    representacion honesta de una cantidad no definida (por ejemplo la
    correlacion de una variable degenerada).
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return _sanitize(obj.to_dict(orient="records"))
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize(obj), fh, default=_json_default, ensure_ascii=False,
                  indent=2, allow_nan=False)


def save_table(df: pd.DataFrame, stem: Path, floatfmt: str = "{:.4f}", index: bool = False) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=index, encoding="utf-8")
    stem.with_suffix(".md").write_text(to_markdown_table(df, floatfmt, index), encoding="utf-8")
