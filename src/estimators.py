"""
Estimadores econometricos que el proyecto audita.

(A) Odean (1998): PGR, PLR, su diferencia y su razon.
(B) Estimador estructural por razon de hazards: el unico que recupera ``delta``.
(C) Barber & Odean (2000): regresion transversal retorno ~ turnover.
(D) IV / 2SLS con ``kappa`` como instrumento, para medir la causalidad reversa.

Todos los errores estandar del lado de Odean se obtienen por **bootstrap
agrupado por cuenta**: se remuestrean cuentas completas con reemplazo, nunca
transacciones sueltas. El remuestreo se implementa con pesos multinomiales, que
es exactamente equivalente a remuestrear indices con reemplazo pero permite
calcular las B replicas con un unico producto matricial.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

CONTROLS = ("log_wealth", "capacity", "portfolio_vol", "mean_cash_weight", "hhi")


# ---------------------------------------------------------------------------
# Bootstrap agrupado por cuenta
# ---------------------------------------------------------------------------
def cluster_bootstrap_sums(matrix: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(B, k) de sumas por columna tras remuestrear CUENTAS con reemplazo.

    Remuestrear ``n`` cuentas con reemplazo equivale a sortear un vector de
    conteos multinomial ``c ~ Mult(n, 1/n)`` y calcular ``c @ matrix``.
    """
    n = matrix.shape[0]
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    return weights @ matrix


def _summary_from_draws(point: float, draws: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    draws = draws[np.isfinite(draws)]
    if draws.size < 10:
        return {"estimate": point, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "z": np.nan, "p": np.nan, "n_boot": int(draws.size)}
    se = float(draws.std(ddof=1))
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    z = point / se if se > 0 else np.nan
    p = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {"estimate": float(point), "se": se, "ci_lo": float(lo), "ci_hi": float(hi),
            "z": float(z) if np.isfinite(z) else np.nan, "p": p, "n_boot": int(draws.size)}


def _safe_div(a, b):
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, np.asarray(a, dtype=float) / np.where(b > 0, b, 1.0), np.nan)


# ---------------------------------------------------------------------------
# (A) Estimador de disposicion de Odean (1998)
# ---------------------------------------------------------------------------
ODEAN_COLS = ("G_r", "G_p", "L_r", "L_p")


def odean_point(sums: np.ndarray) -> Dict[str, np.ndarray]:
    """PGR, PLR, diferencia y razon a partir de las sumas agregadas."""
    g_r, g_p, l_r, l_p = (sums[..., i] for i in range(4))
    pgr = _safe_div(g_r, g_r + g_p)
    plr = _safe_div(l_r, l_r + l_p)
    return {"PGR": pgr, "PLR": plr, "PGR_menos_PLR": pgr - plr, "PGR_sobre_PLR": _safe_div(pgr, plr)}


def estimate_odean(
    accounts: pd.DataFrame, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """Estimador de Odean con errores estandar por bootstrap agrupado por cuenta."""
    mat = accounts[list(ODEAN_COLS)].to_numpy(dtype=float)
    point = odean_point(mat.sum(axis=0))
    draws = odean_point(cluster_bootstrap_sums(mat, n_boot, rng))
    return {k: _summary_from_draws(float(point[k]), draws[k], alpha) for k in point}


def estimate_odean_transaction_bootstrap(
    accounts: pd.DataFrame, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """Bootstrap POR TRANSACCION: incorrecto a proposito, para cuantificar el sesgo.

    Ignorar el agrupamiento equivale a tratar cada dia-posicion como una
    observacion independiente. Como las observaciones de una misma cuenta
    comparten ``delta_i``, ``kappa_i`` y la misma cartera, la independencia es
    falsa y el error estandar sale demasiado pequenio.

    Bajo independencia el remuestreo no parametrico de ``n`` indicadores 0/1 con
    proporcion ``p`` produce exactamente ``Binomial(n, p)`` exitos, asi que el
    bootstrap se implementa con esa equivalencia (identico resultado, coste
    despreciable frente a remuestrear millones de filas).
    """
    g_r, g_p, l_r, l_p = (float(accounts[c].sum()) for c in ODEAN_COLS)
    n_gain, n_loss = g_r + g_p, l_r + l_p
    pgr, plr = g_r / n_gain, l_r / n_loss
    gr_b = rng.binomial(int(round(n_gain)), pgr, size=n_boot) / n_gain
    lr_b = rng.binomial(int(round(n_loss)), plr, size=n_boot) / n_loss
    draws = {"PGR": gr_b, "PLR": lr_b, "PGR_menos_PLR": gr_b - lr_b,
             "PGR_sobre_PLR": _safe_div(gr_b, lr_b)}
    point = {"PGR": pgr, "PLR": plr, "PGR_menos_PLR": pgr - plr, "PGR_sobre_PLR": pgr / plr}
    return {k: _summary_from_draws(point[k], draws[k], alpha) for k in point}


# ---------------------------------------------------------------------------
# (B) Estimador estructural por razon de hazards  <- el que recupera delta
# ---------------------------------------------------------------------------
HAZARD_COLS = ("sells_from_gain", "position_days_in_gain", "sells_from_loss", "position_days_in_loss")


def hazard_point(sums: np.ndarray) -> Dict[str, np.ndarray]:
    """Tasas de riesgo empiricas, su razon y el ``delta`` implicado.

    ``PGR - PLR`` no es un estimador consistente de ``delta``: es una
    transformacion monotona pero sesgada que depende del horizonte, de la
    volatilidad y de la deriva. El parametro se recupera invirtiendo
    ``HR = (1 + delta) / (1 - delta)``.
    """
    s_g, d_g, s_l, d_l = (sums[..., i] for i in range(4))
    h_gain = _safe_div(s_g, d_g)
    h_loss = _safe_div(s_l, d_l)
    hr = _safe_div(h_gain, h_loss)
    delta_hat = (hr - 1.0) / (hr + 1.0)
    return {"h_gain": h_gain, "h_loss": h_loss, "HR": hr, "delta_hat": delta_hat}


def estimate_hazard_ratio(
    accounts: pd.DataFrame, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    mat = accounts[list(HAZARD_COLS)].to_numpy(dtype=float)
    point = hazard_point(mat.sum(axis=0))
    draws = hazard_point(cluster_bootstrap_sums(mat, n_boot, rng))
    return {k: _summary_from_draws(float(point[k]), draws[k], alpha) for k in point}


def delta_hat_per_agent(accounts: pd.DataFrame, min_obs: int = 30) -> np.ndarray:
    """``delta`` implicado cuenta por cuenta (para medir el sesgo de agregacion).

    El estimador agregado es una razon de sumas, no una media de razones: las
    cuentas con ``delta`` alto acumulan muchos mas dias-posicion en perdida
    (lock-in) y muchos menos en ganancia, de modo que ponderan distinto en el
    numerador y en el denominador.
    """
    d_g = accounts["position_days_in_gain"].to_numpy(float)
    d_l = accounts["position_days_in_loss"].to_numpy(float)
    h_g = _safe_div(accounts["sells_from_gain"].to_numpy(float), d_g)
    h_l = _safe_div(accounts["sells_from_loss"].to_numpy(float), d_l)
    hr = _safe_div(h_g, h_l)
    out = (hr - 1.0) / (hr + 1.0)
    out[(d_g < min_obs) | (d_l < min_obs) | ~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------------
# (C) Regresion de Barber & Odean (2000)
# ---------------------------------------------------------------------------
def design_matrix(accounts: pd.DataFrame, controls: Sequence[str] = CONTROLS) -> pd.DataFrame:
    """Matriz de disenio: constante, turnover y controles, sin columnas degeneradas."""
    df = pd.DataFrame(index=accounts.index)
    df["const"] = 1.0
    df["turnover"] = accounts["turnover"].to_numpy(float)
    disponibles = {
        "log_wealth": np.log(accounts["initial_wealth"].to_numpy(float)),
        "capacity": accounts["capacity"].to_numpy(float),
        "portfolio_vol": accounts["portfolio_vol"].to_numpy(float),
        "mean_cash_weight": accounts["mean_cash_weight"].to_numpy(float),
        "hhi": accounts["hhi"].to_numpy(float),
    }
    for name in controls:
        col = disponibles[name]
        if np.nanstd(col) > 0:
            df[name] = col
    return df


def ols_hc1(y: np.ndarray, X: np.ndarray) -> Dict[str, np.ndarray]:
    """OLS con errores estandar robustos a heterocedasticidad HC1, implementado
    a mano para verificacion cruzada contra ``statsmodels``.

        Var(b) = n/(n-k) * (X'X)^-1 X' diag(e^2) X (X'X)^-1
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = (X * (resid ** 2)[:, None]).T @ X
    cov = xtx_inv @ meat @ xtx_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pval = 2.0 * stats.norm.sf(np.abs(tstat))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": tstat, "p": pval, "cov": cov,
            "r2": 1.0 - ss_res / ss_tot, "n": n, "k": k, "resid": resid}


def barber_odean_regression(
    accounts: pd.DataFrame,
    dependent: str = "net_return",
    controls: Sequence[str] = CONTROLS,
) -> Dict[str, object]:
    """Regresion transversal, una observacion por cuenta, con HC1 de ``statsmodels``."""
    X = design_matrix(accounts, controls)
    y = accounts[dependent].to_numpy(float)
    fit = sm.OLS(y, X.to_numpy(float)).fit(cov_type="HC1")
    names = list(X.columns)
    i = names.index("turnover")
    return {
        "dependent": dependent,
        "names": names,
        "beta": np.asarray(fit.params),
        "se": np.asarray(fit.bse),
        "t": np.asarray(fit.tvalues),
        "p": np.asarray(fit.pvalues),
        "r2": float(fit.rsquared),
        "n": int(fit.nobs),
        "beta_turnover": float(fit.params[i]),
        "se_turnover": float(fit.bse[i]),
        "t_turnover": float(fit.tvalues[i]),
        "p_turnover": float(fit.pvalues[i]),
        "alpha": float(fit.params[names.index("const")]),
    }


# ---------------------------------------------------------------------------
# (D) IV / 2SLS: kappa como instrumento del turnover
# ---------------------------------------------------------------------------
def iv_2sls(
    accounts: pd.DataFrame,
    dependent: str = "net_return",
    controls: Sequence[str] = CONTROLS,
    instrument: str = "kappa",
) -> Dict[str, object]:
    """2SLS exactamente identificado con errores estandar robustos.

    ``kappa_i`` es instrumento valido por construccion: se sortea de un stream
    aleatorio independiente, no toca los precios y afecta al turnover solo por
    la frecuencia de operacion (relevancia), sin ninguna otra via hacia el
    retorno (exclusion).

    Si el instrumento es degenerado (``kappa`` constante, que es el caso de los
    escenarios 1, 2, 3, 7 y 8) el modelo NO esta identificado y se devuelve
    ``identified = False`` con todo en NaN. No se reporta un numero inventado.
    """
    z_raw = accounts[instrument].to_numpy(float)
    X = design_matrix(accounts, controls)
    names = list(X.columns)
    if np.std(z_raw) <= 0.0:
        return {"identified": False, "reason": f"{instrument} es degenerado (varianza cero)",
                "beta_turnover": np.nan, "se_turnover": np.nan, "t_turnover": np.nan,
                "p_turnover": np.nan, "first_stage_F": np.nan, "n": len(accounts),
                "dependent": dependent}

    Xm = X.to_numpy(float)
    Z = Xm.copy()
    i = names.index("turnover")
    Z[:, i] = z_raw
    y = accounts[dependent].to_numpy(float)

    # --- primera etapa: turnover ~ controles + instrumento ---------------
    first = ols_hc1(Xm[:, i], Z)
    f_stat = float(first["t"][i] ** 2)

    # --- 2SLS exactamente identificado ----------------------------------
    n, k = Xm.shape
    zx_inv = np.linalg.pinv(Z.T @ Xm)
    beta = zx_inv @ (Z.T @ y)
    resid = y - Xm @ beta
    meat = (Z * (resid ** 2)[:, None]).T @ Z
    cov = zx_inv @ meat @ zx_inv.T * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    return {
        "identified": True,
        "dependent": dependent,
        "names": names,
        "beta": beta,
        "se": se,
        "t": tstat,
        "beta_turnover": float(beta[i]),
        "se_turnover": float(se[i]),
        "t_turnover": float(tstat[i]),
        "p_turnover": float(2.0 * stats.norm.sf(abs(tstat[i]))),
        "first_stage_F": f_stat,
        "first_stage_pi": float(first["beta"][i]),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Tasas de venta condicionales (celdas discriminantes)
# ---------------------------------------------------------------------------
def conditional_sell_rate(
    accounts: pd.DataFrame, cell: str, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> Dict[str, float]:
    """Tasa de venta condicional en una celda, con intervalo bootstrap por cuenta."""
    cols = [f"cell_{cell}_sell", f"cell_{cell}_opp"]
    mat = accounts[cols].to_numpy(float)
    total_opp = float(mat[:, 1].sum())
    if total_opp <= 0:
        return {"estimate": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "z": np.nan, "p": np.nan, "n_boot": 0, "opportunities": 0.0}
    point = float(mat[:, 0].sum() / total_opp)
    sums = cluster_bootstrap_sums(mat, n_boot, rng)
    draws = _safe_div(sums[:, 0], sums[:, 1])
    out = _summary_from_draws(point, draws, alpha)
    out["opportunities"] = total_opp
    return out


def baseline_sell_rate(accounts: pd.DataFrame) -> float:
    """Tasa de venta incondicional sobre todos los dias-unidad clasificados."""
    ventas = accounts[["sells_from_gain", "sells_from_loss", "sells_from_neutral"]].to_numpy().sum()
    dias = accounts[["position_days_in_gain", "position_days_in_loss", "position_days_neutral"]].to_numpy().sum()
    return float(ventas / dias) if dias > 0 else np.nan


def correlation_or_nd(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Correlacion de Pearson, o ``None`` si alguna variable es degenerada.

    Devolver ``None`` (que se reporta como ``n/d``) en vez de 0.0000 evita el
    anti-patron de afirmar independencia cuando lo que hay es varianza cero.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.std(x) <= 0 or np.std(y) <= 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])
