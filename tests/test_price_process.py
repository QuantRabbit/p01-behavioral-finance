"""Pruebas del proceso de precios y del candado anti-lookahead."""

import numpy as np
import pytest

from src.config import MarketConfig, derive_rngs
from src.price_process import (
    LookAheadError,
    PriceView,
    autocorrelation_null_band,
    generate_market,
    pooled_autocorrelation,
)


@pytest.fixture(scope="module")
def market():
    cfg = MarketConfig()
    rng = derive_rngs("test_precios")["prices"]
    return generate_market(cfg, rng)


# ---------------------------------------------------------------------------
# Integridad estructural
# ---------------------------------------------------------------------------
def test_dimensiones_y_positividad(market):
    cfg = market.config
    assert market.prices.shape == (cfg.n_days + 1, cfg.n_assets)
    assert market.log_returns.shape == (cfg.n_days, cfg.n_assets)
    assert np.all(market.prices > 0.0), "los precios lognormales no pueden ser <= 0"
    assert np.all(np.isfinite(market.prices))


def test_precios_iniciales_en_rango(market):
    lo, hi = market.config.initial_price_bounds
    assert np.all(market.prices[0] >= lo) and np.all(market.prices[0] <= hi)


def test_determinismo_de_precios():
    cfg = MarketConfig(n_assets=10, n_days=60)
    a = generate_market(cfg, derive_rngs("det")["prices"]).prices
    b = generate_market(cfg, derive_rngs("det")["prices"]).prices
    assert np.array_equal(a, b), "misma semilla debe dar precios identicos bit a bit"


def test_estructura_de_factores_genera_correlacion_transversal(market):
    """El modelo multifactorial debe producir correlacion transversal positiva.

    Es la razon de usarlo en vez de GBM independiente: las carteras tienen
    riesgo no diversificable y el control de volatilidad en la regresion de
    Barber-Odean tiene contenido.
    """
    corr = np.corrcoef(market.log_returns.T)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert off.mean() > 0.10, f"correlacion media entre activos demasiado baja: {off.mean():.3f}"


# ---------------------------------------------------------------------------
# Ausencia de estructura temporal (requisito del disenio)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lag", list(range(1, 11)))
def test_sin_autocorrelacion_rezagos_1_a_10(market, lag):
    """Los retornos son i.i.d. en el tiempo: ni momentum ni reversion real.

    La banda nula se construye permutando el orden de los dias de forma
    identica para todos los activos, lo que preserva la correlacion transversal
    y por tanto la varianza real del estimador agrupado.
    """
    rng = np.random.default_rng(1234 + lag)
    obs = pooled_autocorrelation(market.log_returns, lag)
    lo, hi = autocorrelation_null_band(market.log_returns, lag, rng, n_draws=200, alpha=0.01)
    assert lo <= obs <= hi, f"lag {lag}: acf={obs:.5f} fuera de la banda nula [{lo:.5f}, {hi:.5f}]"


def test_autocorrelacion_economicamente_despreciable(market):
    for lag in range(1, 11):
        assert abs(pooled_autocorrelation(market.log_returns, lag)) < 0.05


# ---------------------------------------------------------------------------
# Candado anti-lookahead
# ---------------------------------------------------------------------------
def test_leer_precio_futuro_lanza_excepcion(market):
    view = PriceView(market, current_day=10)
    view.get_price(0, 10)          # hoy: permitido
    view.get_price(0, 3)           # pasado: permitido
    with pytest.raises(LookAheadError):
        view.get_price(0, 11)      # manana: PROHIBIDO


def test_prices_on_futuro_lanza_excepcion(market):
    view = PriceView(market, current_day=5)
    with pytest.raises(LookAheadError):
        view.prices_on(6)


def test_past_returns_futuro_lanza_excepcion(market):
    view = PriceView(market, current_day=5)
    with pytest.raises(LookAheadError):
        view.past_returns_on(6, 3)


def test_dia_negativo_lanza_excepcion(market):
    view = PriceView(market, current_day=5)
    with pytest.raises(LookAheadError):
        view.get_price(0, -1)


def test_reloj_no_retrocede(market):
    view = PriceView(market, current_day=0)
    view.advance_to(10)
    view.advance_to(10)
    with pytest.raises(LookAheadError):
        view.advance_to(9)


def test_advance_mas_alla_del_horizonte(market):
    view = PriceView(market, current_day=0)
    with pytest.raises(LookAheadError):
        view.advance_to(market.n_days + 1)


def test_vector_de_precios_es_solo_lectura(market):
    view = PriceView(market, current_day=3)
    p = view.prices_on(3)
    with pytest.raises(ValueError):
        p[0] = 1.0


def test_get_past_return_es_correcto_y_causal(market):
    view = PriceView(market, current_day=30)
    r = view.get_past_return(7, 30, 10)
    esperado = market.prices[30, 7] / market.prices[20, 7] - 1.0
    assert abs(r - esperado) < 1e-12
    # Sin historia suficiente -> 0.0 (no hay senal, no hay excepcion)
    view2 = PriceView(market, current_day=3)
    assert view2.get_past_return(7, 3, 10) == 0.0


def test_past_returns_on_coincide_con_version_escalar(market):
    view = PriceView(market, current_day=40)
    vec = view.past_returns_on(40, 10)
    for asset in (0, 5, 17, 59):
        assert abs(vec[asset] - view.get_past_return(asset, 40, 10)) < 1e-12
