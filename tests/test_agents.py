"""Pruebas de la poblacion de agentes y de las reglas de decision."""

import numpy as np
import pytest

from src.agents import (
    H_GAIN_CAP,
    H_LOSS_FLOOR,
    beta_params_from_mean,
    build_population,
    disposition_hazard,
    draw_bias_parameter,
    hazard_rates,
    mean_reversion_hazard,
    rebalancing_trim,
)
from src.config import PopulationConfig, derive_rngs


def _pop(**kw):
    cfg = PopulationConfig(n_agents=4000, **kw)
    return build_population(cfg, derive_rngs("test_agentes")["population"])


# ---------------------------------------------------------------------------
# Distribuciones de los parametros inyectados
# ---------------------------------------------------------------------------
def test_beta_reparametrizada_media_y_concentracion():
    a, b = beta_params_from_mean(0.8, 20.0)
    assert a == pytest.approx(16.0) and b == pytest.approx(4.0)
    rng = np.random.default_rng(0)
    x = rng.beta(a, b, size=200_000)
    assert abs(x.mean() - 0.8) < 0.01
    assert x.var() > 0.0


def test_beta_rechaza_media_cero():
    with pytest.raises(ValueError):
        beta_params_from_mean(0.0, 20.0)


def test_delta_y_kappa_tienen_varianza_positiva_cuando_el_objetivo_no_es_cero():
    pop = _pop(delta_target=0.8, delta_dist="beta", kappa_target=0.3, kappa_dist="beta")
    assert np.var(pop.delta) > 0.0
    assert np.var(pop.kappa) > 0.0
    assert abs(pop.delta.mean() - 0.8) < 0.01
    assert abs(pop.kappa.mean() - 0.3) < 0.01
    assert not pop.is_degenerate_delta and not pop.is_degenerate_kappa


def test_objetivo_cero_es_punto_exacto_y_degenerado():
    """delta = 0 exacto para todos: la ausencia del sesgo es el punto del escenario."""
    pop = _pop(delta_target=0.0, delta_dist="point")
    assert np.all(pop.delta == 0.0)
    assert pop.is_degenerate_delta, "con varianza cero la correlacion no esta definida"


def test_correlacion_delta_kappa_practicamente_nula():
    """Streams independientes -> |Corr| del orden de 1/sqrt(N)."""
    pop = _pop(delta_target=0.5, delta_dist="beta", kappa_target=0.5, kappa_dist="beta")
    c = np.corrcoef(pop.delta, pop.kappa)[0, 1]
    assert abs(c) < 0.05, f"Corr(delta, kappa) = {c:.4f} demasiado grande"


def test_correlacion_delta_kappa_nula_en_poblacion_heterogenea():
    pop = _pop(delta_dist="uniform", kappa_dist="uniform")
    c = np.corrcoef(pop.delta, pop.kappa)[0, 1]
    assert abs(c) < 0.05


def test_distribucion_desconocida_falla():
    with pytest.raises(ValueError):
        draw_bias_parameter(0.5, "gaussiana", 10, np.random.default_rng(0), 20.0)


def test_riqueza_y_capacidad_en_rango():
    pop = _pop()
    lo, hi = pop.config.wealth_bounds
    assert pop.wealth.min() >= lo - 1e-6 and pop.wealth.max() <= hi + 1e-6
    cl, ch = pop.config.capacity_bounds
    assert pop.capacity.min() >= cl and pop.capacity.max() <= ch


def test_poblacion_determinista():
    a = _pop(delta_target=0.6, delta_dist="beta")
    b = _pop(delta_target=0.6, delta_dist="beta")
    assert np.array_equal(a.delta, b.delta) and np.array_equal(a.wealth, b.wealth)


# ---------------------------------------------------------------------------
# Tasa de riesgo: el parametro esta UN NIVEL POR DEBAJO de lo estimado
# ---------------------------------------------------------------------------
def test_razon_de_hazards_es_exactamente_la_formula_estructural():
    cfg = PopulationConfig()
    delta = np.array([0.0, 0.2, 0.5, 0.8])
    kappa = np.array([0.0, 0.3, 0.6, 1.0])
    h0, hg, hl = hazard_rates(delta, kappa, cfg)
    assert np.allclose(h0, cfg.h_base + cfg.c_kappa * kappa)
    hr = hg / hl
    assert np.allclose((hr - 1.0) / (hr + 1.0), delta), "HR = (1+d)/(1-d) debe invertirse a delta"


def test_piso_y_techo_de_la_hazard():
    cfg = PopulationConfig()
    _, hg, hl = hazard_rates(np.array([0.99999]), np.array([0.0]), cfg)
    assert hl[0] == H_LOSS_FLOOR
    cfg_alto = PopulationConfig(h_base=5.0, c_kappa=0.0)
    _, hg2, _ = hazard_rates(np.array([0.5]), np.array([0.0]), cfg_alto)
    assert hg2[0] == H_GAIN_CAP


def test_disposition_hazard_asigna_la_tasa_correcta_por_estado():
    h0 = np.array([0.02]); hg = np.array([0.04]); hl = np.array([0.005])
    is_gain = np.array([[[True, False]]])
    is_loss = np.array([[[False, True]]])
    h = disposition_hazard(is_gain, is_loss, h0, hg, hl)
    assert h[0, 0, 0] == 0.04 and h[0, 0, 1] == 0.005
    # Sin clasificar (neutral) -> tasa base
    h2 = disposition_hazard(np.zeros((1, 1, 1), bool), np.zeros((1, 1, 1), bool), h0, hg, hl)
    assert h2[0, 0, 0] == 0.02


# ---------------------------------------------------------------------------
# Confounds: condicionan en variables DISTINTAS del precio de compra
# ---------------------------------------------------------------------------
def test_reversion_condiciona_en_retorno_pasado_y_esta_acotada():
    cfg = PopulationConfig()
    r = np.array([-0.5, -0.10, 0.0, 0.05, 0.10, 0.5])
    h = mean_reversion_hazard(r, cfg)
    assert np.all(np.diff(h) >= -1e-15), "debe ser monotona creciente en el retorno pasado"
    assert h[0] == pytest.approx(cfg.h_base * (1 - cfg.mr_lambda)) or h[0] == 0.0
    assert h[-1] == pytest.approx(cfg.h_base * (1 + cfg.mr_lambda))
    assert h[2] == pytest.approx(cfg.h_base)


def test_rebalanceo_recorta_exactamente_hasta_el_peso_objetivo():
    target = np.array([[0.10, 0.10, 0.10]])
    w = np.array([[0.20, 0.11, 0.05]])   # 0.20 fuera de banda, 0.11 dentro, 0.05 abajo
    trim, frac = rebalancing_trim(w, target, band=0.15)
    assert trim.tolist() == [[True, False, False]]
    w_post = w * (1.0 - frac)
    assert w_post[0, 0] == pytest.approx(0.10)
    assert frac[0, 1] == 0.0 and frac[0, 2] == 0.0


def test_rebalanceo_nunca_mira_el_precio_de_compra():
    """Prueba de contrato: la firma no admite precios, solo pesos."""
    import inspect

    params = set(inspect.signature(rebalancing_trim).parameters)
    assert params == {"weights", "target_weight", "band"}
