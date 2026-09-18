"""Pruebas de los estimadores: Odean, razon de hazards, HC1 e IV."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.config import MarketConfig, PopulationConfig, SimConfig, derive_rngs
from src.estimators import (
    barber_odean_regression,
    conditional_sell_rate,
    correlation_or_nd,
    delta_hat_per_agent,
    estimate_hazard_ratio,
    estimate_odean,
    estimate_odean_transaction_bootstrap,
    hazard_point,
    iv_2sls,
    odean_point,
    ols_hc1,
)
from src.simulator import run_simulation


# ---------------------------------------------------------------------------
# (A) Odean con datos construidos a mano
# ---------------------------------------------------------------------------
def test_pgr_plr_con_datos_calculados_a_mano():
    """Dos cuentas: G_r=3,G_p=7 y G_r=1,G_p=9 -> PGR = 4/20 = 0.20.
    L_r=1,L_p=9 y L_r=1,L_p=19 -> PLR = 2/30 = 0.0666..."""
    df = pd.DataFrame({"G_r": [3.0, 1.0], "G_p": [7.0, 9.0],
                       "L_r": [1.0, 1.0], "L_p": [9.0, 19.0]})
    out = odean_point(df[["G_r", "G_p", "L_r", "L_p"]].to_numpy().sum(axis=0))
    assert out["PGR"] == pytest.approx(0.20)
    assert out["PLR"] == pytest.approx(2.0 / 30.0)
    assert out["PGR_menos_PLR"] == pytest.approx(0.20 - 2.0 / 30.0)
    assert out["PGR_sobre_PLR"] == pytest.approx(3.0)


def test_odean_sin_observaciones_devuelve_nan():
    df = pd.DataFrame({"G_r": [0.0], "G_p": [0.0], "L_r": [0.0], "L_p": [0.0]})
    out = odean_point(df.to_numpy().sum(axis=0))
    assert np.isnan(out["PGR"]) and np.isnan(out["PLR"])


# ---------------------------------------------------------------------------
# (B) El estimador estructural recupera delta
# ---------------------------------------------------------------------------
def _cuentas_sinteticas(delta: float, h0: float, n_agents: int, dias: int, seed: int):
    """Cuentas sinteticas con una hazard CONOCIDA impuesta directamente.

    No hay simulacion de mercado: se sortean directamente las ventas como
    Bernoulli con las tasas verdaderas. Si el estimador no recupera ``delta``
    aqui, el estimador esta mal, no el simulador.
    """
    rng = np.random.default_rng(seed)
    h_gain = h0 * (1.0 + delta)
    h_loss = h0 * (1.0 - delta)
    d_g = rng.integers(dias // 2, dias, size=n_agents).astype(float)
    d_l = rng.integers(dias // 2, dias, size=n_agents).astype(float)
    s_g = rng.binomial(d_g.astype(int), h_gain).astype(float)
    s_l = rng.binomial(d_l.astype(int), h_loss).astype(float)
    return pd.DataFrame({
        "sells_from_gain": s_g, "position_days_in_gain": d_g,
        "sells_from_loss": s_l, "position_days_in_loss": d_l,
        "G_r": s_g, "G_p": d_g - s_g, "L_r": s_l, "L_p": d_l - s_l,
    })


@pytest.mark.parametrize("delta", [0.0, 0.2, 0.5, 0.8])
def test_razon_de_hazards_recupera_delta_con_error_menor_a_002(delta):
    df = _cuentas_sinteticas(delta, h0=0.02, n_agents=500, dias=2000, seed=11)
    cols = ["sells_from_gain", "position_days_in_gain", "sells_from_loss", "position_days_in_loss"]
    out = hazard_point(df[cols].to_numpy().sum(axis=0))
    assert abs(out["delta_hat"] - delta) < 0.02, f"delta_hat={out['delta_hat']:.4f}"


def test_el_bootstrap_del_estimador_estructural_cubre_el_valor_verdadero():
    df = _cuentas_sinteticas(0.6, 0.02, 800, 3000, seed=3)
    out = estimate_hazard_ratio(df, n_boot=400, rng=np.random.default_rng(0))
    assert out["delta_hat"]["ci_lo"] < 0.6 < out["delta_hat"]["ci_hi"]
    assert out["delta_hat"]["se"] > 0
    assert out["delta_hat"]["ci_lo"] < out["delta_hat"]["estimate"] < out["delta_hat"]["ci_hi"]


def test_delta_hat_por_agente_promedia_cerca_del_verdadero():
    df = _cuentas_sinteticas(0.5, 0.03, 800, 3000, seed=5)
    per = delta_hat_per_agent(df)
    assert abs(np.nanmean(per) - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Bootstrap agrupado vs por transaccion
# ---------------------------------------------------------------------------
def test_el_bootstrap_por_cuenta_da_errores_estandar_mayores():
    """Ignorar el agrupamiento subestima el error estandar."""
    rng = np.random.default_rng(7)
    n = 400
    # Heterogeneidad real entre cuentas: cada una tiene su propia tasa.
    h_gain = rng.beta(2, 60, size=n)
    h_loss = rng.beta(1, 90, size=n)
    d_g = np.full(n, 400.0)
    d_l = np.full(n, 400.0)
    df = pd.DataFrame({
        "G_r": rng.binomial(400, h_gain).astype(float), "L_r": rng.binomial(400, h_loss).astype(float),
        "position_days_in_gain": d_g, "position_days_in_loss": d_l,
    })
    df["G_p"] = d_g - df["G_r"]
    df["L_p"] = d_l - df["L_r"]

    clustered = estimate_odean(df, 500, np.random.default_rng(1))
    naive = estimate_odean_transaction_bootstrap(df, 500, np.random.default_rng(1))
    se_c = clustered["PGR_menos_PLR"]["se"]
    se_n = naive["PGR_menos_PLR"]["se"]
    assert se_c > se_n, f"clustered {se_c:.5f} deberia superar a por transaccion {se_n:.5f}"
    assert se_c / se_n > 2.0, "el factor de subestimacion deberia ser grande"


# ---------------------------------------------------------------------------
# (C) HC1 manual contra statsmodels
# ---------------------------------------------------------------------------
def test_hc1_manual_coincide_con_statsmodels_a_1e_8():
    rng = np.random.default_rng(42)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n), rng.uniform(size=n)])
    y = X @ np.array([0.5, -0.3, 0.2, 1.0]) + rng.normal(size=n) * (0.5 + np.abs(X[:, 1]))
    manual = ols_hc1(y, X)
    fit = sm.OLS(y, X).fit(cov_type="HC1")
    assert np.allclose(manual["beta"], fit.params, atol=1e-10)
    assert np.allclose(manual["se"], fit.bse, atol=1e-8), (manual["se"], fit.bse)
    assert np.allclose(manual["cov"], fit.cov_params(), atol=1e-10)
    assert abs(manual["r2"] - fit.rsquared) < 1e-10


def test_hc1_difiere_de_los_errores_clasicos_bajo_heterocedasticidad():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = 1.0 + 0.0 * x + rng.normal(size=n) * (0.2 + 2.0 * np.abs(x))
    hc1 = ols_hc1(y, X)["se"][1]
    clasico = sm.OLS(y, X).fit().bse[1]
    assert abs(hc1 - clasico) / clasico > 0.10


# ---------------------------------------------------------------------------
# (D) IV
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim_heterogeneo():
    cfg = SimConfig(
        market=MarketConfig(n_assets=40, n_days=252),
        population=PopulationConfig(n_agents=500, delta_dist="uniform", kappa_dist="uniform"),
    )
    return run_simulation(cfg, derive_rngs("test_iv"), "heterogeneo")


def test_el_f_de_primera_etapa_supera_10_en_el_escenario_heterogeneo(sim_heterogeneo):
    out = iv_2sls(sim_heterogeneo.accounts, dependent="net_return")
    assert out["identified"]
    assert out["first_stage_F"] > 10.0, f"F = {out['first_stage_F']:.1f}"
    assert out["first_stage_pi"] > 0.0, "kappa debe aumentar el turnover"


def test_iv_no_identificado_cuando_kappa_es_degenerado():
    cfg = SimConfig(
        market=MarketConfig(n_assets=30, n_days=100),
        population=PopulationConfig(n_agents=150, delta_target=0.5, delta_dist="beta"),
    )
    res = run_simulation(cfg, derive_rngs("iv_degen"), "degen")
    out = iv_2sls(res.accounts)
    assert out["identified"] is False
    assert np.isnan(out["beta_turnover"])


def test_regresion_barber_odean_devuelve_estructura_completa(sim_heterogeneo):
    out = barber_odean_regression(sim_heterogeneo.accounts, "net_return")
    assert "turnover" in out["names"]
    assert np.isfinite(out["beta_turnover"]) and out["se_turnover"] > 0
    assert 0.0 <= out["r2"] <= 1.0
    assert out["n"] == len(sim_heterogeneo.accounts)


def test_la_pendiente_bruta_supera_a_la_neta_por_el_costo_de_operar(sim_heterogeneo):
    """Identidad contable: r_gross - r_net = costos / W0, creciente en turnover."""
    g = barber_odean_regression(sim_heterogeneo.accounts, "gross_return")
    n = barber_odean_regression(sim_heterogeneo.accounts, "net_return")
    dif = g["beta_turnover"] - n["beta_turnover"]
    assert dif > 0.0
    assert 0.0005 < dif < 0.01, f"costo implicito por unidad de turnover fuera de rango: {dif:.5f}"


def test_el_iv_corrige_la_causalidad_reversa_hacia_abajo(sim_heterogeneo):
    """Con delta > 0 la suerte genera ganancias que se venden: el turnover
    observado es en parte CONSECUENCIA del retorno, no su causa. El IV, que
    solo usa la variacion exogena de kappa, debe quedar por debajo del OLS."""
    ols = barber_odean_regression(sim_heterogeneo.accounts, "net_return")
    iv = iv_2sls(sim_heterogeneo.accounts, "net_return")
    assert iv["identified"]
    assert iv["beta_turnover"] < ols["beta_turnover"]


def test_la_pendiente_neta_es_negativa_cuando_solo_hay_churn():
    """Sin efecto disposicion no hay causalidad reversa y el costo domina."""
    cfg = SimConfig(
        market=MarketConfig(n_assets=40, n_days=252),
        population=PopulationConfig(n_agents=500, kappa_target=0.8, kappa_dist="beta"),
    )
    res = run_simulation(cfg, derive_rngs("t_kappa_alto"), "kappa_alto")
    out = barber_odean_regression(res.accounts, "net_return")
    assert out["beta_turnover"] < 0.0


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def test_correlacion_devuelve_nd_si_hay_varianza_cero():
    assert correlation_or_nd(np.zeros(10), np.arange(10)) is None
    assert correlation_or_nd(np.arange(10), np.arange(10)) == pytest.approx(1.0)


def test_tasa_condicional_en_celda_vacia_es_nan():
    df = pd.DataFrame({"cell_X_sell": [0.0], "cell_X_opp": [0.0]})
    out = conditional_sell_rate(df, "X", 50, np.random.default_rng(0))
    assert np.isnan(out["estimate"]) and out["opportunities"] == 0.0
