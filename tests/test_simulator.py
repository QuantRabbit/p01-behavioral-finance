"""Pruebas del motor de simulacion: contabilidad, no-fuga y determinismo."""

import numpy as np
import pytest

from src.config import (
    AccountingConfig,
    MarketConfig,
    PopulationConfig,
    SimConfig,
    derive_rngs,
)
from src.simulator import run_simulation


def _cfg(**pop_kw):
    return SimConfig(
        market=MarketConfig(n_assets=30, n_days=120),
        population=PopulationConfig(n_agents=250, **pop_kw),
    )


@pytest.fixture(scope="module")
def sim_disposicion():
    cfg = _cfg(delta_target=0.8, delta_dist="beta")
    return run_simulation(cfg, derive_rngs("test_sim_disp"), "test_disp")


@pytest.fixture(scope="module")
def sim_grande():
    """Corrida mas grande, necesaria para las pruebas estadisticas de no-fuga."""
    cfg = SimConfig(
        market=MarketConfig(n_assets=60, n_days=300),
        population=PopulationConfig(n_agents=600, kappa_target=0.6, kappa_dist="beta"),
    )
    return run_simulation(cfg, derive_rngs("test_sim_grande"), "test_grande")


# ---------------------------------------------------------------------------
# Conservacion de valor
# ---------------------------------------------------------------------------
def test_conservacion_de_valor(sim_disposicion):
    """efectivo + posiciones + costos pagados = riqueza inicial + P&L de mercado."""
    assert sim_disposicion.integrity["conservation_error_rel"] < 1e-6


def test_libro_bruto_menos_neto_es_exactamente_el_costo(sim_disposicion):
    assert sim_disposicion.integrity["gross_minus_net_equals_costs"] < 1e-9


def test_libros_con_identicas_cantidades_y_fechas(sim_disposicion):
    """El libro bruto replica unidad por unidad y fecha por fecha al neto."""
    log = sim_disposicion.buy_log
    assert log, "el log de compras debe registrarse"
    assert np.array_equal(log["shares_net"], log["shares_gross"])
    assert log["day"].shape == log["agent"].shape == log["asset"].shape


def test_valor_de_cuenta_nunca_negativo(sim_disposicion):
    assert np.all(sim_disposicion.daily_value >= 0.0)
    assert np.all(sim_disposicion.accounts["final_net"] > 0.0)


# ---------------------------------------------------------------------------
# Contabilidad de Odean
# ---------------------------------------------------------------------------
def test_agente_sin_dias_de_venta_no_aporta_conteos(sim_disposicion):
    a = sim_disposicion.accounts
    sin_ventas = a["sale_days"] == 0
    if sin_ventas.any():
        sub = a.loc[sin_ventas, ["G_r", "G_p", "L_r", "L_p"]]
        assert (sub.to_numpy() == 0).all()


def test_realizadas_no_superan_las_disponibles(sim_disposicion):
    a = sim_disposicion.accounts
    assert (a["G_r"] <= a["G_r"] + a["G_p"] + 1e-9).all()
    assert (a["sells_from_gain"] <= a["position_days_in_gain"]).all()
    assert (a["sells_from_loss"] <= a["position_days_in_loss"]).all()


def test_los_dias_posicion_superan_a_los_conteos_de_odean(sim_disposicion):
    """Los conteos de Odean usan solo dias con venta; la hazard usa todos."""
    a = sim_disposicion.accounts
    assert a["position_days_in_gain"].sum() > (a["G_r"] + a["G_p"]).sum()


def test_posiciones_neutras_son_practicamente_inexistentes(sim_disposicion):
    """Con precios continuos la igualdad exacta con el costo casi nunca ocurre."""
    frac = sim_disposicion.integrity["neutral_unit_days"] / sim_disposicion.integrity["total_unit_days"]
    assert frac < 1e-4


# ---------------------------------------------------------------------------
# Lotes multiples
# ---------------------------------------------------------------------------
def test_existen_lotes_multiples_de_verdad(sim_disposicion):
    """Anti-patron 6: declarar lotes y no permitir mas de uno por activo."""
    a = sim_disposicion.accounts
    assert (a["num_lots_opened"] > a["capacity"]).mean() > 0.5


# ---------------------------------------------------------------------------
# Cero fuga de informacion
# ---------------------------------------------------------------------------
def test_sin_fuga_los_activos_comprados_no_predicen_el_futuro(sim_grande):
    """Prueba de no-fuga a nivel de cuenta.

    Para cada compra se calcula el retorno del activo a 20 dias hacia adelante,
    en exceso del promedio transversal de ese dia (para quitar el factor de
    mercado, que es comun a todos). Si hubiera fuga, las cuentas de mayor
    turnover compraria sistematicamente activos con retorno futuro anormal.
    """
    res = sim_grande
    prices = res.market.prices
    log = res.buy_log
    horizon = 20
    ok = log["day"].astype(int) + horizon <= res.market.n_days
    agents = log["agent"][ok].astype(int)
    assets = log["asset"][ok].astype(int)
    days = log["day"][ok].astype(int)

    fwd = prices[days + horizon, assets] / prices[days, assets] - 1.0
    mercado = (prices[days + horizon, :] / prices[days, :] - 1.0).mean(axis=1)
    exceso = fwd - mercado

    n = len(res.accounts)
    suma = np.bincount(agents, weights=exceso, minlength=n)
    cuenta = np.bincount(agents, minlength=n)
    validos = cuenta >= 10
    medio = suma[validos] / cuenta[validos]
    turnover = res.accounts["turnover"].to_numpy()[validos]

    r = np.corrcoef(turnover, medio)[0, 1]
    dof = validos.sum() - 2
    t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
    assert abs(t) < 3.0, f"posible fuga: corr={r:.4f}, t={t:.2f}"


def _exceso_futuro_de_las_compras(res, horizon: int = 20) -> float:
    """Retorno futuro medio de los activos comprados, en exceso del transversal."""
    prices = res.market.prices
    log = res.buy_log
    ok = log["day"].astype(int) + horizon <= res.market.n_days
    assets = log["asset"][ok].astype(int)
    days = log["day"][ok].astype(int)
    fwd = prices[days + horizon, :] / prices[days, :] - 1.0
    return float((fwd[np.arange(len(days)), assets] - fwd.mean(axis=1)).mean())


def test_sin_fuga_las_compras_no_baten_al_mercado_entre_trayectorias():
    """Prueba de no-fuga con trayectorias independientes.

    Dentro de UNA sola trayectoria de precios el estadistico tiene un efecto
    aleatorio de trayectoria: por azar, los activos que la poblacion tiende a
    no tener en cartera pueden comportarse mejor o peor en esa realizacion
    concreta. Ese efecto no es fuga (los agentes no pueden ver el futuro: el
    candado de PriceView lo impide estructuralmente) pero domina la varianza.
    La prueba correcta promedia sobre trayectorias independientes.
    """
    vals = []
    for k in range(8):
        cfg = SimConfig(
            market=MarketConfig(n_assets=40, n_days=200),
            population=PopulationConfig(n_agents=300, kappa_target=0.6, kappa_dist="beta"),
        )
        res = run_simulation(cfg, derive_rngs(f"nofuga_{k}"), f"nofuga_{k}")
        vals.append(_exceso_futuro_de_las_compras(res))
    v = np.array(vals)
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    assert abs(t) < 3.0, f"posible fuga: exceso medio {v.mean():.5f}, t={t:.2f}"
    assert abs(v.mean()) < 5e-4


# ---------------------------------------------------------------------------
# Comportamiento emergente
# ---------------------------------------------------------------------------
def test_el_turnover_emerge_y_crece_con_kappa():
    """El turnover es un OUTPUT: nunca se inyecta, emerge de la hazard."""
    lo = run_simulation(_cfg(kappa_target=0.2, kappa_dist="beta"), derive_rngs("turn_lo"), "lo")
    hi = run_simulation(_cfg(kappa_target=0.8, kappa_dist="beta"), derive_rngs("turn_lo"), "hi")
    assert hi.accounts["turnover"].mean() > 1.5 * lo.accounts["turnover"].mean()


def test_lock_in_el_delta_alto_reduce_el_turnover_realizado():
    """delta y kappa son independientes como parametros, pero delta contamina
    el turnover realizado, que es un output.

    El efecto necesita horizonte: las perdedoras tienen que acumularse. Con
    pocos meses de simulacion el signo aun no se ha formado, asi que esta
    prueba usa el horizonte completo de dos anios.
    """
    cfg = SimConfig(
        market=MarketConfig(n_assets=60, n_days=504),
        population=PopulationConfig(n_agents=500, delta_dist="uniform"),
    )
    res = run_simulation(cfg, derive_rngs("lockin"), "lockin")
    a = res.accounts
    r = np.corrcoef(a["delta"], a["turnover"])[0, 1]
    assert r < -0.10, f"se esperaba lock-in (corr negativa), se obtuvo {r:.3f}"


def test_las_perdedoras_se_retienen_mas_que_las_ganadoras(sim_disposicion):
    a = sim_disposicion.accounts
    hg = a["sells_from_gain"].sum() / a["position_days_in_gain"].sum()
    hl = a["sells_from_loss"].sum() / a["position_days_in_loss"].sum()
    assert hg > hl


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def test_determinismo_bit_a_bit():
    cfg = _cfg(delta_target=0.5, delta_dist="beta", kappa_target=0.4, kappa_dist="beta")
    a = run_simulation(cfg, derive_rngs("det_sim"), "det")
    b = run_simulation(cfg, derive_rngs("det_sim"), "det")
    cols = ["net_return", "gross_return", "turnover", "G_r", "L_p", "sells_from_gain"]
    for c in cols:
        assert np.array_equal(a.accounts[c].to_numpy(), b.accounts[c].to_numpy()), c


@pytest.mark.parametrize("base", ["average_cost", "fifo", "per_lot"])
def test_todas_las_bases_de_costo_conservan_valor(base):
    cfg = SimConfig(
        market=MarketConfig(n_assets=30, n_days=100),
        population=PopulationConfig(n_agents=150, delta_target=0.6, delta_dist="beta"),
        accounting=AccountingConfig(cost_basis=base),
    )
    res = run_simulation(cfg, derive_rngs(f"base_{base}"), base)
    assert res.integrity["conservation_error_rel"] < 1e-6


@pytest.mark.parametrize("confound", ["rebalancing", "mean_reversion"])
def test_los_confounds_corren_y_conservan_valor(confound):
    cfg = _cfg(confound=confound)
    res = run_simulation(cfg, derive_rngs(f"conf_{confound}"), confound)
    assert res.integrity["conservation_error_rel"] < 1e-6
    assert res.accounts["trade_count"].sum() > 0, "el confound debe generar operaciones"
    assert np.all(res.accounts["delta"] == 0.0) and np.all(res.accounts["kappa"] == 0.0)
