"""
Proceso de precios multifactorial y candado anti-lookahead.

Diseno
------
La matriz completa de precios ``P[t, m]`` se genera **antes** de instanciar un
solo agente, con un generador aleatorio propio (``rng_prices``). Los agentes
nunca reciben la matriz: reciben un ``PriceView`` que expone lecturas y lanza
``LookAheadError`` ante cualquier intento de consultar un dia mayor que el dia
actual fijado por el motor de simulacion.

Modelo de retornos logaritmicos diarios (``dt = 1/252``):

    r_{m,t} = (mu_m - 0.5*sigma_m^2)*dt
              + beta_{m,1}*F_{1,t} + beta_{m,2}*F_{2,t} + beta_{m,3}*F_{3,t}
              + eps_{m,t}

    P_{m,t} = P_{m,t-1} * exp(r_{m,t})

donde ``sigma_m^2`` es la varianza anual TOTAL del activo (factores mas
idiosincratica), de modo que la correccion de Ito deja la deriva del retorno
simple en ``beta_1 * mu_mercado``.

Los tres factores y el termino idiosincratico son i.i.d. en el tiempo. En
consecuencia **no hay momentum ni reversion real en el mercado**: cualquier
patron temporal que un agente crea ver es una creencia falsa, no una propiedad
del proceso. Esto es lo que convierte al escenario 8 en un confound genuino.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .config import MarketConfig

ArrayLike = Union[int, np.ndarray]


class LookAheadError(RuntimeError):
    """Se intento leer un precio (o un retorno) con informacion del futuro."""


@dataclass(frozen=True)
class MarketPaths:
    """Resultado inmutable del generador de precios."""

    prices: np.ndarray        # (n_days+1, n_assets) precios medios ("mid")
    log_returns: np.ndarray   # (n_days, n_assets)
    factors: np.ndarray       # (n_days, 3)
    betas: np.ndarray         # (n_assets, 3)
    idio_vol_annual: np.ndarray   # (n_assets,)
    total_vol_annual: np.ndarray  # (n_assets,)
    config: MarketConfig

    @property
    def n_days(self) -> int:
        return self.prices.shape[0] - 1

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]

    def equal_weight_index(self) -> np.ndarray:
        """Indice equiponderado normalizado a 100 en t = 0."""
        rel = self.prices / self.prices[0]
        return 100.0 * rel.mean(axis=1)


def generate_market(cfg: MarketConfig, rng: np.random.Generator) -> MarketPaths:
    """Genera la matriz completa de precios.

    Se llama UNA vez por escenario, con ``rng`` = stream ``prices``, antes de
    crear la poblacion de agentes.
    """
    m, t_days, dt = cfg.n_assets, cfg.n_days, cfg.dt
    sqrt_dt = np.sqrt(dt)

    betas = np.empty((m, 3))
    betas[:, 0] = rng.uniform(*cfg.beta1_bounds, size=m)
    betas[:, 1] = rng.uniform(*cfg.beta2_bounds, size=m)
    betas[:, 2] = rng.uniform(*cfg.beta3_bounds, size=m)

    idio_vol = rng.uniform(*cfg.idio_vol_bounds, size=m)
    p0 = rng.uniform(*cfg.initial_price_bounds, size=m)

    # Factores: el primero lleva la deriva del mercado, los otros dos son puro
    # riesgo (media cero). Todos i.i.d. en el tiempo.
    f1 = rng.normal(cfg.mu_market_annual * dt, cfg.sigma_market_annual * sqrt_dt, size=t_days)
    f2 = rng.normal(0.0, cfg.sigma_f2_annual * sqrt_dt, size=t_days)
    f3 = rng.normal(0.0, cfg.sigma_f3_annual * sqrt_dt, size=t_days)
    factors = np.column_stack([f1, f2, f3])

    eps = rng.normal(0.0, 1.0, size=(t_days, m)) * (idio_vol * sqrt_dt)

    total_var_annual = (
        (betas[:, 0] * cfg.sigma_market_annual) ** 2
        + (betas[:, 1] * cfg.sigma_f2_annual) ** 2
        + (betas[:, 2] * cfg.sigma_f3_annual) ** 2
        + idio_vol ** 2
    )
    drift = (cfg.mu_asset_annual - 0.5 * total_var_annual) * dt

    log_returns = drift[None, :] + factors @ betas.T + eps

    prices = np.empty((t_days + 1, m))
    prices[0] = p0
    prices[1:] = p0[None, :] * np.exp(np.cumsum(log_returns, axis=0))

    return MarketPaths(
        prices=prices,
        log_returns=log_returns,
        factors=factors,
        betas=betas,
        idio_vol_annual=idio_vol,
        total_vol_annual=np.sqrt(total_var_annual),
        config=cfg,
    )


class PriceView:
    """Ventana causal sobre ``MarketPaths``.

    El motor de simulacion fija el dia actual con :meth:`advance_to`. Toda
    lectura con ``day > current_day`` levanta :class:`LookAheadError`. El dia
    actual solo puede avanzar; retroceder tambien levanta excepcion, para que
    un bug de indices no pueda "desbloquear" informacion ya consumida.
    """

    __slots__ = ("_paths", "_current_day")

    def __init__(self, paths: MarketPaths, current_day: int = 0) -> None:
        self._paths = paths
        self._current_day = int(current_day)

    # -- control del reloj -------------------------------------------------
    @property
    def current_day(self) -> int:
        return self._current_day

    @property
    def n_days(self) -> int:
        return self._paths.n_days

    @property
    def n_assets(self) -> int:
        return self._paths.n_assets

    def advance_to(self, day: int) -> None:
        day = int(day)
        if day < self._current_day:
            raise LookAheadError(
                f"El reloj no puede retroceder: current_day={self._current_day}, pedido={day}"
            )
        if day > self._paths.n_days:
            raise LookAheadError(f"Dia {day} fuera del horizonte {self._paths.n_days}")
        self._current_day = day

    def reset(self) -> None:
        """Reinicia el reloj a t = 0 (solo para reutilizar la vista en tests)."""
        self._current_day = 0

    # -- lecturas ----------------------------------------------------------
    def _check_day(self, day: int) -> int:
        day = int(day)
        if day < 0:
            raise LookAheadError(f"Dia negativo: {day}")
        if day > self._current_day:
            raise LookAheadError(
                f"Lectura del futuro bloqueada: dia {day} > current_day {self._current_day}"
            )
        return day

    def get_price(self, asset: int, day: int) -> float:
        """Precio medio de un activo en un dia. Estrictamente causal."""
        day = self._check_day(day)
        return float(self._paths.prices[day, int(asset)])

    def prices_on(self, day: int) -> np.ndarray:
        """Vector (n_assets,) de precios medios del dia. Copia de solo lectura."""
        day = self._check_day(day)
        out = self._paths.prices[day].copy()
        out.flags.writeable = False
        return out

    def get_past_return(self, asset: int, day: int, lookback: int) -> float:
        """Retorno simple del activo entre ``day - lookback`` y ``day``.

        Estrictamente causal: ambos extremos se validan contra el reloj. Si no
        hay historia suficiente devuelve 0.0 (no hay informacion, no hay senal).
        """
        day = self._check_day(day)
        start = day - int(lookback)
        if start < 0:
            return 0.0
        self._check_day(start)
        p_now = self._paths.prices[day, int(asset)]
        p_then = self._paths.prices[start, int(asset)]
        return float(p_now / p_then - 1.0)

    def past_returns_on(self, day: int, lookback: int) -> np.ndarray:
        """Version vectorizada de :meth:`get_past_return` sobre todos los activos."""
        day = self._check_day(day)
        start = day - int(lookback)
        if start < 0:
            return np.zeros(self._paths.n_assets)
        self._check_day(start)
        return self._paths.prices[day] / self._paths.prices[start] - 1.0


def pooled_autocorrelation(log_returns: np.ndarray, lag: int) -> float:
    """Autocorrelacion promedio entre activos al rezago dado.

    Se calcula por activo (desmediado por activo) y se promedia en la seccion
    transversal, que es lo que se quiere probar: ausencia de estructura
    temporal en el proceso generador.
    """
    x = log_returns - log_returns.mean(axis=0, keepdims=True)
    num = (x[lag:] * x[:-lag]).sum(axis=0)
    den = (x * x).sum(axis=0)
    return float(np.mean(num / den))


def autocorrelation_null_band(
    log_returns: np.ndarray,
    lag: int,
    rng: np.random.Generator,
    n_draws: int = 200,
    alpha: float = 0.01,
) -> tuple:
    """Banda nula de la autocorrelacion agrupada por remuestreo temporal.

    Se permuta el ORDEN DE LOS DIAS de forma identica para todos los activos.
    Eso destruye cualquier dependencia temporal pero preserva exactamente la
    correlacion transversal inducida por los factores comunes, que es la que
    infla la varianza del estimador agrupado. Es una banda de confianza sin
    supuestos asintoticos.
    """
    n_t = log_returns.shape[0]
    draws = np.empty(n_draws)
    for i in range(n_draws):
        perm = rng.permutation(n_t)
        draws[i] = pooled_autocorrelation(log_returns[perm], lag)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
