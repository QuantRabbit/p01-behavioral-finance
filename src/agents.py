"""
Poblacion de agentes, parametros inyectados y reglas de decision.

El punto central del disenio
----------------------------
Los parametros inyectados ``delta_i`` (efecto disposicion) y ``kappa_i``
(sobreprecision / churn) NO son probabilidades de venta ni tasas de rotacion.
Modulan una **tasa de riesgo diaria** que esta un nivel por debajo de las
cantidades que despues se estiman:

    h_0(kappa_i)  = h_base + c_kappa * kappa_i
    h_gain        = min(0.99, h_0 * (1 + delta_i))
    h_loss        = max(1e-4, h_0 * (1 - delta_i))
    h_neutral     = h_0

PGR, PLR, el turnover, el numero de operaciones y los horizontes de tenencia
son resultados **emergentes** de la interaccion entre esta tasa, la trayectoria
de precios y el tiempo que la posicion lleva abierta. En ningun punto del
codigo existe una sentencia de la forma ``if ganancia: vender con prob p``.

``h_base = 0.015`` corresponde a un horizonte medio de tenencia de
``1/0.015 ~ 67`` dias habiles (unos tres meses) cuando ``delta = kappa = 0``,
del orden de los horizontes de cuentas minoristas reportados por Odean (1998).

Distribuciones, no escalares
----------------------------
Salvo cuando el objetivo es exactamente 0 (donde la ausencia del sesgo es el
punto del escenario y ``delta_i = 0`` para todos), ``delta_i`` y ``kappa_i`` se
sortean de distribuciones con varianza estrictamente positiva. Sin esa
varianza, ``Corr(delta, kappa)`` saldria 0.0000 por varianza cero y no por
independencia, los errores estandar bootstrap por cuenta colapsarian y la
regresion de Barber-Odean se quedaria sin variacion transversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import PopulationConfig

H_LOSS_FLOOR = 1e-4
H_GAIN_CAP = 0.99


# ---------------------------------------------------------------------------
# Sorteo de parametros
# ---------------------------------------------------------------------------
def beta_params_from_mean(mean: float, concentration: float) -> Tuple[float, float]:
    """Reparametriza la Beta por media y concentracion: a = m*nu, b = (1-m)*nu."""
    if not (0.0 < mean < 1.0):
        raise ValueError(f"la Beta requiere 0 < media < 1, se recibio {mean}")
    return mean * concentration, (1.0 - mean) * concentration


def draw_bias_parameter(
    target: float,
    dist: str,
    n: int,
    rng: np.random.Generator,
    concentration: float,
) -> np.ndarray:
    """Sortea un parametro de sesgo para toda la poblacion.

    - ``point``: valor constante. Solo se usa con ``target = 0``, donde la
      ausencia del sesgo es precisamente lo que el escenario quiere probar.
      La consecuencia (correlaciones no definidas) se reporta como ``n/d``.
    - ``beta``: Beta reparametrizada con media ``target`` y varianza positiva.
    - ``uniform``: U[0,1], para el escenario de poblacion heterogenea completa.
    """
    if dist == "point":
        return np.full(n, float(target))
    if dist == "uniform":
        return rng.uniform(0.0, 1.0, size=n)
    if dist == "beta":
        a, b = beta_params_from_mean(target, concentration)
        return rng.beta(a, b, size=n)
    raise ValueError(f"distribucion desconocida: {dist}")


@dataclass(frozen=True)
class Population:
    """Poblacion inmutable de agentes con sus parametros y hazards base."""

    delta: np.ndarray       # (N,) parametro inyectado de disposicion
    kappa: np.ndarray       # (N,) parametro inyectado de churn
    wealth: np.ndarray      # (N,) riqueza inicial
    capacity: np.ndarray    # (N,) int, numero maximo de posiciones
    h0: np.ndarray          # (N,) tasa base
    h_gain: np.ndarray      # (N,) tasa si la posicion esta en ganancia
    h_loss: np.ndarray      # (N,) tasa si la posicion esta en perdida
    config: PopulationConfig

    @property
    def n_agents(self) -> int:
        return self.delta.shape[0]

    @property
    def is_degenerate_delta(self) -> bool:
        return float(np.var(self.delta)) <= 0.0

    @property
    def is_degenerate_kappa(self) -> bool:
        return float(np.var(self.kappa)) <= 0.0

    def summary(self) -> dict:
        return {
            "delta_mean": float(self.delta.mean()),
            "delta_sd": float(self.delta.std(ddof=1)),
            "kappa_mean": float(self.kappa.mean()),
            "kappa_sd": float(self.kappa.std(ddof=1)),
            "wealth_mean": float(self.wealth.mean()),
            "capacity_mean": float(self.capacity.mean()),
            "h0_mean": float(self.h0.mean()),
            "h_gain_mean": float(self.h_gain.mean()),
            "h_loss_mean": float(self.h_loss.mean()),
            "frac_h_loss_at_floor": float(np.mean(self.h0 * (1 - self.delta) < H_LOSS_FLOOR)),
        }


def hazard_rates(delta: np.ndarray, kappa: np.ndarray, cfg: PopulationConfig):
    """Tasas de riesgo diarias. Ver el encabezado del modulo."""
    h0 = cfg.h_base + cfg.c_kappa * kappa
    h_gain = np.minimum(H_GAIN_CAP, h0 * (1.0 + delta))
    h_loss = np.maximum(H_LOSS_FLOOR, h0 * (1.0 - delta))
    return h0, h_gain, h_loss


def build_population(cfg: PopulationConfig, rng: np.random.Generator) -> Population:
    """Construye la poblacion completa.

    ``delta`` y ``kappa`` se sortean de **streams independientes** derivados del
    generador de poblacion, de modo que su correlacion muestral sea ruido de
    orden 1/sqrt(N) y no un artefacto de compartir la secuencia aleatoria.
    """
    n = cfg.n_agents
    s_delta, s_kappa, s_rest = rng.spawn(3)

    delta = draw_bias_parameter(cfg.delta_target, cfg.delta_dist, n, s_delta, cfg.beta_concentration)
    kappa = draw_bias_parameter(cfg.kappa_target, cfg.kappa_dist, n, s_kappa, cfg.beta_concentration)

    lo, hi = cfg.wealth_bounds
    wealth = np.exp(s_rest.uniform(np.log(lo), np.log(hi), size=n))
    cap_lo, cap_hi = cfg.capacity_bounds
    capacity = s_rest.integers(cap_lo, cap_hi + 1, size=n).astype(np.int32)

    h0, h_gain, h_loss = hazard_rates(delta, kappa, cfg)
    return Population(
        delta=delta,
        kappa=kappa,
        wealth=wealth,
        capacity=capacity,
        h0=h0,
        h_gain=h_gain,
        h_loss=h_loss,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Reglas de decision
# ---------------------------------------------------------------------------
def disposition_hazard(
    is_gain: np.ndarray,
    is_loss: np.ndarray,
    h0: np.ndarray,
    h_gain: np.ndarray,
    h_loss: np.ndarray,
) -> np.ndarray:
    """Hazard del agente conductual. Condiciona en el **precio de compra**.

    ``is_gain`` / ``is_loss`` tienen forma ``(N, K, D)``; las tasas por agente
    se difunden sobre las dimensiones de posicion y lote.
    """
    h = np.broadcast_to(h0[:, None, None], is_gain.shape).copy()
    np.copyto(h, np.broadcast_to(h_gain[:, None, None], h.shape), where=is_gain)
    np.copyto(h, np.broadcast_to(h_loss[:, None, None], h.shape), where=is_loss)
    return h


def mean_reversion_hazard(past_return: np.ndarray, cfg: PopulationConfig) -> np.ndarray:
    """Hazard del agente que cree en la reversion. Condiciona en el **retorno
    reciente** del activo, NO en el precio de compra.

        h = h_base * (1 + lambda * clip(r_pasado / escala, -1, +1))

    La creencia es falsa por construccion: los retornos del mercado son i.i.d.
    Esta distincion es la que permite separar este mecanismo del conductual en
    las celdas discriminantes.
    """
    z = np.clip(past_return / cfg.mr_scale, -1.0, 1.0)
    return np.maximum(0.0, cfg.h_base * (1.0 + cfg.mr_lambda * z))


def rebalancing_trim(
    weights: np.ndarray,
    target_weight: np.ndarray,
    band: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Regla del agente que rebalancea. Condiciona en el **peso relativo**.

    Devuelve ``(trim_mask, trim_fraction)``: si el peso actual supera el
    objetivo en mas de ``band`` (relativo), se recorta la posicion hasta
    devolverla exactamente al peso objetivo, lo que es una venta **parcial**.
    El agente no tiene psicologia alguna: es perfectamente racional y nunca
    mira el precio al que compro.
    """
    upper = target_weight * (1.0 + band)
    trim = weights > upper
    frac = np.where(trim & (weights > 0.0), 1.0 - target_weight / np.maximum(weights, 1e-12), 0.0)
    return trim, np.clip(frac, 0.0, 1.0)


def rebalancing_topup_need(
    weights: np.ndarray,
    target_weight: np.ndarray,
    band: float,
    active: np.ndarray,
) -> np.ndarray:
    """Deficit relativo de peso de cada posicion (0 si esta dentro de la banda)."""
    lower = target_weight * (1.0 - band)
    need = np.where(active & (weights < lower), lower - weights, 0.0)
    return need
