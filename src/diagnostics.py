"""
Diagnosticos: independencia, causalidad reversa, descomposicion del spread y
celdas discriminantes.

La seccion de celdas discriminantes es la contribucion central del proyecto.
``PGR - PLR`` no puede separar disposicion, rebalanceo y creencia en reversion
porque las tres reglas venden ganadoras. Pero las tres **condicionan en
variables distintas**:

    Disposicion  ->  precio actual vs PRECIO DE COMPRA
    Rebalanceo   ->  peso actual   vs PESO OBJETIVO
    Reversion    ->  RETORNO RECIENTE del activo

Construyendo particiones de los dias-posicion donde los mecanismos hacen
predicciones **opuestas**, la tasa de venta condicional deja una firma de signos
distinta para cada mecanismo.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import CostConfig
from .estimators import (
    CONTROLS,
    baseline_sell_rate,
    barber_odean_regression,
    conditional_sell_rate,
    correlation_or_nd,
    design_matrix,
    iv_2sls,
    ols_hc1,
)
from .simulator import CELL_NAMES, SimulationResult


# ---------------------------------------------------------------------------
# (D1) Independencia
# ---------------------------------------------------------------------------
def independence_diagnostics(result: SimulationResult) -> Dict[str, object]:
    """Los cuatro puntos de independencia del enunciado.

    ``Corr(delta, kappa)`` debe ser ruido por construccion. ``Corr(delta,
    turnover)`` **no** sera cero y debe salir negativa: el agente con ``delta``
    alto retiene indefinidamente las perdedoras (lock-in), lo que reduce su
    numero de operaciones. Es decir, ``delta`` y ``kappa`` son independientes
    como parametros **inyectados**, pero ``delta`` contamina el turnover
    **realizado**, que es un output.
    """
    a = result.accounts
    delta = a["delta"].to_numpy(float)
    kappa = a["kappa"].to_numpy(float)
    turn = a["turnover"].to_numpy(float)
    return {
        "delta_mean": float(delta.mean()),
        "delta_sd": float(delta.std(ddof=1)),
        "kappa_mean": float(kappa.mean()),
        "kappa_sd": float(kappa.std(ddof=1)),
        "delta_degenerado": bool(np.std(delta) <= 0),
        "kappa_degenerado": bool(np.std(kappa) <= 0),
        "corr_delta_kappa": correlation_or_nd(delta, kappa),
        "corr_delta_turnover": correlation_or_nd(delta, turn),
        "corr_kappa_turnover": correlation_or_nd(kappa, turn),
        "turnover_mean": float(turn.mean()),
        "turnover_sd": float(turn.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# (D2) Sondas de causalidad reversa
# ---------------------------------------------------------------------------
def backward_regression(accounts: pd.DataFrame, dependent: str = "gross_return") -> Dict[str, float]:
    """Regresion "hacia atras": Turnover ~ retorno + controles.

    Si el turnover fuera una causa del retorno y no al reves, esta regresion no
    deberia encontrar nada. En los escenarios con ``delta > 0`` sale fuertemente
    positiva: la cuenta que tuvo suerte acumula ganadoras, las vende (porque la
    regla condiciona en el precio de compra) y registra mas turnover.
    """
    X = design_matrix(accounts)
    cols = [c for c in X.columns if c != "turnover"]
    Xb = X[cols].copy()
    Xb.insert(1, dependent, accounts[dependent].to_numpy(float))
    fit = ols_hc1(accounts["turnover"].to_numpy(float), Xb.to_numpy(float))
    i = list(Xb.columns).index(dependent)
    return {
        "beta": float(fit["beta"][i]),
        "se": float(fit["se"][i]),
        "t": float(fit["t"][i]),
        "p": float(fit["p"][i]),
        "r2": float(fit["r2"]),
    }


def permutation_placebo(
    accounts: pd.DataFrame,
    rng: np.random.Generator,
    dependent: str = "net_return",
    n_placebo: int = 200,
) -> Dict[str, object]:
    """Placebo de permutacion: se permuta el turnover entre cuentas.

    Al romper el emparejamiento cuenta-turnover, cualquier relacion genuina
    desaparece y ``beta`` debe colapsar a cero. Si no colapsa, lo que la
    regresion recoge no es el turnover sino alguna caracteristica de la cuenta
    correlacionada con el.
    """
    observed = barber_odean_regression(accounts, dependent)["beta_turnover"]
    df = accounts.copy()
    betas = np.empty(n_placebo)
    turn = accounts["turnover"].to_numpy(float)
    for b in range(n_placebo):
        df["turnover"] = rng.permutation(turn)
        betas[b] = barber_odean_regression(df, dependent)["beta_turnover"]
    lo, hi = np.quantile(betas, [0.025, 0.975])
    return {
        "beta_observado": float(observed),
        "placebo_media": float(betas.mean()),
        "placebo_sd": float(betas.std(ddof=1)),
        "placebo_ci_lo": float(lo),
        "placebo_ci_hi": float(hi),
        "observado_fuera_de_la_banda": bool(observed < lo or observed > hi),
        "n_placebo": int(n_placebo),
        "betas": betas,
    }


def ols_vs_iv(accounts: pd.DataFrame) -> Dict[str, object]:
    """Compara OLS contra IV en bruto y en neto. La brecha mide la causalidad
    reversa, medida y no argumentada."""
    out: Dict[str, object] = {}
    for dep in ("gross_return", "net_return"):
        ols = barber_odean_regression(accounts, dep)
        iv = iv_2sls(accounts, dep)
        key = "gross" if dep == "gross_return" else "net"
        out[f"beta_ols_{key}"] = ols["beta_turnover"]
        out[f"se_ols_{key}"] = ols["se_turnover"]
        out[f"t_ols_{key}"] = ols["t_turnover"]
        out[f"p_ols_{key}"] = ols["p_turnover"]
        out[f"r2_{key}"] = ols["r2"]
        out[f"identificado_{key}"] = iv["identified"]
        out[f"beta_iv_{key}"] = iv["beta_turnover"]
        out[f"se_iv_{key}"] = iv["se_turnover"]
        out[f"t_iv_{key}"] = iv["t_turnover"]
        out[f"first_stage_F"] = iv["first_stage_F"]
        out[f"brecha_{key}"] = (
            ols["beta_turnover"] - iv["beta_turnover"] if iv["identified"] else np.nan
        )
    return out


def cash_drag_isolation(accounts: pd.DataFrame) -> Dict[str, float]:
    """Aisla y cuantifica el canal de *cash drag*.

    Se corre la regresion bruta con y sin el control ``mean_cash_weight``. El
    desplazamiento de ``beta_gross`` es la parte de la pendiente que viajaba por
    el canal del efectivo ocioso.
    """
    con = barber_odean_regression(accounts, "gross_return", controls=CONTROLS)
    sin = barber_odean_regression(
        accounts, "gross_return", controls=[c for c in CONTROLS if c != "mean_cash_weight"]
    )
    return {
        "beta_gross_con_control": con["beta_turnover"],
        "beta_gross_sin_control": sin["beta_turnover"],
        "desplazamiento": sin["beta_turnover"] - con["beta_turnover"],
        "corr_turnover_cash": float(
            np.corrcoef(accounts["turnover"], accounts["mean_cash_weight"])[0, 1]
        ),
        "cash_weight_medio": float(accounts["mean_cash_weight"].mean()),
    }


# ---------------------------------------------------------------------------
# (D3) Descomposicion del spread dentro del retorno bruto
# ---------------------------------------------------------------------------
def spread_decomposition(
    accounts: pd.DataFrame, costs: CostConfig, n_days: int
) -> Dict[str, float]:
    """Verifica que ``r_gross - r_net`` sea exactamente el costo de transaccion.

    Por definicion del turnover anualizado ``tau``, el volumen negociado en toda
    la simulacion es ``2 * tau * V_medio * (T/252)``. Entonces

        spread_cost / W0 = 2 * half_spread * (T/252) * tau * (V_medio / W0)

    de modo que la pendiente de la regresion de ``spread_cost/W0`` contra
    ``tau`` tiene un valor teorico cerrado. Si no coincide hay un error de
    contabilidad.
    """
    w0 = accounts["initial_wealth"].to_numpy(float)
    tau = accounts["turnover"].to_numpy(float)
    y_total = (accounts["gross_return"] - accounts["net_return"]).to_numpy(float)
    y_spread = accounts["spread_cost"].to_numpy(float) / w0
    y_comm = accounts["commission_cost"].to_numpy(float) / w0
    ratio_v = accounts["mean_portfolio_value"].to_numpy(float) / w0

    X = np.column_stack([np.ones(len(tau)), tau])
    fit_total = ols_hc1(y_total, X)
    fit_spread = ols_hc1(y_spread, X)

    teorico = 2.0 * costs.half_spread * (n_days / 252.0) * float(np.mean(ratio_v))
    return {
        "identidad_max_error": float(
            np.max(np.abs(y_total - (y_spread + y_comm)))
        ),
        "pendiente_total": float(fit_total["beta"][1]),
        "r2_total": float(fit_total["r2"]),
        "pendiente_spread": float(fit_spread["beta"][1]),
        "pendiente_spread_teorica": teorico,
        "error_relativo_spread": float(abs(fit_spread["beta"][1] - teorico) / teorico),
        "costo_medio_sobre_w0": float(np.mean(y_total)),
        "peso_comisiones": float(np.sum(y_comm) / np.sum(y_total)),
    }


# ---------------------------------------------------------------------------
# (D4) Celdas discriminantes
# ---------------------------------------------------------------------------
CELL_LABELS = {
    CELL_NAMES[0]: "A: en perdida con rebote reciente (>+2% a 10d)",
    CELL_NAMES[1]: "B: en ganancia e infraponderada",
    CELL_NAMES[2]: "C: en perdida y sobreponderada",
}

#: Prediccion cualitativa de cada mecanismo sobre cada celda. Se registra aqui
#: para poder contrastarla con lo observado sin reescribir la expectativa.
CELL_PREDICTIONS = {
    "disposicion": {CELL_NAMES[0]: "baja", CELL_NAMES[1]: "alta", CELL_NAMES[2]: "baja"},
    "rebalanceo": {CELL_NAMES[0]: "base", CELL_NAMES[1]: "baja", CELL_NAMES[2]: "alta"},
    "reversion": {CELL_NAMES[0]: "alta", CELL_NAMES[1]: "base", CELL_NAMES[2]: "base"},
}


def discriminating_cells(
    result: SimulationResult, n_boot: int, rng: np.random.Generator
) -> Dict[str, object]:
    """Tasa de venta condicional en cada celda, con intervalo bootstrap por cuenta.

    Se reporta ademas la **razon a la tasa base** (log-razon), que es la firma
    comparable entre escenarios con tasas base muy distintas.
    """
    a = result.accounts
    base = baseline_sell_rate(a)
    out: Dict[str, object] = {"tasa_base": base}
    for cell in CELL_NAMES:
        est = conditional_sell_rate(a, cell, n_boot, rng)
        est["label"] = CELL_LABELS[cell]
        # Log-razon con correccion de continuidad: una celda donde el mecanismo
        # NUNCA vende da tasa exactamente 0 (el caso del rebalanceo sobre
        # ganadoras infraponderadas) y log(0) no es reportable. La correccion
        # (s + 0.5) / (n + 1) mantiene la firma finita y muy negativa, que es
        # justo la informacion que interesa.
        opp = est.get("opportunities", 0.0)
        if opp > 0 and np.isfinite(base) and base > 0:
            s = float(a[f"cell_{cell}_sell"].sum())
            tasa_corr = (s + 0.5) / (opp + 1.0)
            est["log_ratio_vs_base"] = float(np.log(tasa_corr / base))
        else:
            est["log_ratio_vs_base"] = np.nan
        out[cell] = est
    return out


def signature_matrix(cells_by_scenario: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Matriz de firmas: escenarios en filas, celdas en columnas, log-razon en
    las entradas. Cada mecanismo deja un patron de signos distinto."""
    rows = []
    for key, cells in cells_by_scenario.items():
        row = {"escenario": key, "tasa_base": cells["tasa_base"]}
        for cell in CELL_NAMES:
            row[cell] = cells[cell]["log_ratio_vs_base"]
            row[f"{cell}_tasa"] = cells[cell]["estimate"]
            row[f"{cell}_ci_lo"] = cells[cell]["ci_lo"]
            row[f"{cell}_ci_hi"] = cells[cell]["ci_hi"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("escenario")


def signature_pattern(row: pd.Series, tol: float = 0.15) -> Dict[str, str]:
    """Traduce una fila de la matriz de firmas a un patron de signos.

    ``tol`` es la magnitud minima (en log-razon) para llamar a una celda alta o
    baja en vez de indistinguible de la tasa base.
    """
    def sgn(x):
        if not np.isfinite(x):
            return "n/d"
        if x > tol:
            return "alta"
        if x < -tol:
            return "baja"
        return "base"

    return {cell: sgn(row[cell]) for cell in CELL_NAMES}


def pattern_str(row: pd.Series, tol: float = 0.15) -> str:
    firma = signature_pattern(row, tol)
    return ", ".join(f"{c.split('_')[0]}={firma[c]}" for c in CELL_NAMES)


def match_a_priori(row: pd.Series, tol: float = 0.15) -> Optional[str]:
    """Devuelve el mecanismo cuya prediccion *registrada de antemano* coincide.

    Devuelve ``None`` si el patron observado no corresponde a ninguna de las
    predicciones del pre-analisis. Un ``None`` no invalida la separabilidad: la
    firma observada puede ser igualmente distintiva y simplemente distinta de la
    que se habia anticipado. Esa discrepancia se reporta como hallazgo.
    """
    firma = signature_pattern(row, tol)
    for mecanismo, pred in CELL_PREDICTIONS.items():
        if all(firma[c] == pred[c] for c in CELL_NAMES):
            return mecanismo
    return None


def nearest_reference(row: pd.Series, reference: pd.DataFrame) -> Dict[str, object]:
    """Clasifica una firma por distancia euclidiana a las firmas de referencia.

    Este es el procedimiento de identificacion propuesto: en vez de leer un solo
    numero (``PGR - PLR``), se compara el vector de tres log-razones contra las
    firmas de los mecanismos puros.
    """
    v = np.array([row[c] for c in CELL_NAMES], dtype=float)
    dists = {}
    for name, ref in reference.iterrows():
        w = np.array([ref[c] for c in CELL_NAMES], dtype=float)
        dists[name] = float(np.sqrt(np.nansum((v - w) ** 2)))
    orden = sorted(dists, key=dists.get)
    return {
        "mas_cercano": orden[0],
        "distancia": dists[orden[0]],
        "segundo": orden[1] if len(orden) > 1 else None,
        "distancia_segundo": dists[orden[1]] if len(orden) > 1 else np.nan,
        "todas": dists,
    }


def signature_distances(matrix: pd.DataFrame) -> pd.DataFrame:
    """Matriz de distancias entre las firmas de los escenarios."""
    names = list(matrix.index)
    vals = matrix[list(CELL_NAMES)].to_numpy(float)
    out = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            out[i, j] = float(np.sqrt(np.nansum((vals[i] - vals[j]) ** 2)))
    return pd.DataFrame(out, index=names, columns=names)
