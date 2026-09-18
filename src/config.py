"""
Configuracion central del simulador: dataclasses de parametros y derivacion
determinista de semillas.

Principio de reproducibilidad
-----------------------------
Existe una unica semilla maestra (``MASTER_SEED``). De ella se derivan, por
escenario, cinco *streams* de numeros aleatorios estrictamente separados:

    prices      -> trayectoria de precios (se genera ANTES de crear agentes)
    population  -> sorteo de delta_i, kappa_i, riqueza, capacidad
    decisions   -> sorteos de venta / fraccion / seleccion de activos
    bootstrap   -> remuestreo por cuenta de los errores estandar
    placebo     -> permutaciones de la prueba placebo

La separacion no es cosmetica: garantiza que consumir mas numeros aleatorios
en las decisiones de los agentes no altere ni un solo precio, que es la
condicion necesaria para que los retornos futuros sean independientes de las
decisiones (ver seccion "cero fuga de informacion" del reporte).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Metadatos academicos
# --------------------------------------------------------------------------
STUDENT_NAME = "[TU NOMBRE]"
PROFESSOR_NAME = "[NOMBRE DEL PROFESOR]"
COURSE = "Comportamiento en las Finanzas y Toma de Decisiones"
INSTITUTION = "ITESO, Universidad Jesuita de Guadalajara"
PROJECT = "P01 - Simulador de finanzas conductuales con ground truth"

# --------------------------------------------------------------------------
# Semillas
# --------------------------------------------------------------------------
MASTER_SEED = 42

#: Orden fijo de los streams. No reordenar: cambia la reproducibilidad.
RNG_STREAM_NAMES: Tuple[str, ...] = (
    "prices",
    "population",
    "decisions",
    "bootstrap",
    "placebo",
)


def scenario_entropy(scenario_key: str) -> int:
    """Entropia entera, estable entre procesos, derivada del nombre del escenario.

    No se usa ``hash()`` de Python porque esta aleatorizado por proceso
    (PYTHONHASHSEED) y romperia la reproducibilidad entre corridas.
    """
    digest = hashlib.sha256(scenario_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def derive_rngs(scenario_key: str, master_seed: int = MASTER_SEED) -> Dict[str, np.random.Generator]:
    """Devuelve un diccionario nombre -> ``np.random.Generator`` independiente."""
    root = np.random.SeedSequence([master_seed, scenario_entropy(scenario_key)])
    children = root.spawn(len(RNG_STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(RNG_STREAM_NAMES, children)
    }


# --------------------------------------------------------------------------
# Mercado
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketConfig:
    """Parametros del proceso de precios multifactorial."""

    n_assets: int = 60
    n_days: int = 504              # dias habiles simulados (2 anios)
    dt: float = 1.0 / 252.0

    mu_market_annual: float = 0.08     # deriva anual del factor de mercado
    sigma_market_annual: float = 0.16  # volatilidad anual del factor de mercado
    sigma_f2_annual: float = 0.12      # factor sectorial / estilo
    sigma_f3_annual: float = 0.08      # segundo factor de estilo, ortogonal

    beta1_bounds: Tuple[float, float] = (0.65, 1.35)
    beta2_bounds: Tuple[float, float] = (-0.60, 0.60)
    beta3_bounds: Tuple[float, float] = (-0.40, 0.40)

    idio_vol_bounds: Tuple[float, float] = (0.12, 0.28)   # vol idiosincratica anual
    initial_price_bounds: Tuple[float, float] = (25.0, 150.0)

    #: Deriva idiosincratica anual por activo. Cero por diseno: toda la deriva
    #: esperada entra por el factor de mercado, de modo que no exista un
    #: activo intrinsecamente superior que un agente pudiera descubrir.
    mu_asset_annual: float = 0.0


# --------------------------------------------------------------------------
# Costos de transaccion
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CostConfig:
    """Comision fija por orden y bid-ask spread."""

    commission: float = 1.00      # USD por orden ejecutada
    spread_bps: float = 10.0      # spread TOTAL en puntos base

    @property
    def half_spread(self) -> float:
        """Medio spread en fraccion de precio (10 bps totales -> 0.0005)."""
        return self.spread_bps / 2.0 / 10_000.0

    @property
    def ask_mult(self) -> float:
        return 1.0 + self.half_spread

    @property
    def bid_mult(self) -> float:
        return 1.0 - self.half_spread


# --------------------------------------------------------------------------
# Contabilidad
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AccountingConfig:
    """Las cuatro decisiones contables que el enunciado exige declarar."""

    #: 'average_cost' | 'fifo' | 'per_lot'
    cost_basis: str = "average_cost"

    #: 'partial_as_full' | 'partial_as_fraction'
    partial_counting: str = "partial_as_full"

    #: 'mid' | 'ask_paid' -- precio de referencia para clasificar ganancia/perdida
    reference: str = "mid"

    #: Fracciones posibles de liquidacion cuando se dispara una venta
    partial_fractions: Tuple[float, ...] = (0.25, 0.50, 1.00)
    partial_probs: Tuple[float, ...] = (0.25, 0.25, 0.50)

    #: Profundidad maxima de lotes por posicion. Compras adicionales mas alla
    #: de esta profundidad se fusionan con el lote mas reciente (ponderando
    #: por acciones); se reporta la frecuencia de esa fusion.
    max_lots: int = 4

    #: Tolerancia relativa para declarar una posicion "exactamente al costo".
    #: Esas posiciones son NEUTRAS y quedan fuera de los cuatro conteos.
    neutral_tol: float = 1e-10

    def validate(self) -> None:
        assert self.cost_basis in ("average_cost", "fifo", "per_lot"), self.cost_basis
        assert self.partial_counting in ("partial_as_full", "partial_as_fraction")
        assert self.reference in ("mid", "ask_paid")
        assert abs(sum(self.partial_probs) - 1.0) < 1e-12
        assert len(self.partial_probs) == len(self.partial_fractions)


# --------------------------------------------------------------------------
# Poblacion de agentes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PopulationConfig:
    """Poblacion de agentes y parametrizacion de los sesgos inyectados."""

    n_agents: int = 1000

    wealth_bounds: Tuple[float, float] = (10_000.0, 500_000.0)   # log-uniforme
    capacity_bounds: Tuple[int, int] = (5, 30)                   # uniforme entera

    # --- Sesgo 1: efecto disposicion -------------------------------------
    delta_target: float = 0.0
    #: 'point' (todos iguales), 'beta' (media = target, varianza > 0), 'uniform'
    delta_dist: str = "point"

    # --- Sesgo 2: sobreprecision / churn ---------------------------------
    kappa_target: float = 0.0
    kappa_dist: str = "point"

    #: Concentracion de la Beta reparametrizada: a = m*nu, b = (1-m)*nu.
    beta_concentration: float = 20.0

    # --- Tasa de riesgo base ---------------------------------------------
    h_base: float = 0.015     # ~1/0.015 = 67 dias habiles de tenencia media
    c_kappa: float = 0.060    # sensibilidad de la hazard al churn

    # --- Confounds (agentes con delta = 0 y kappa = 0) --------------------
    #: None | 'rebalancing' | 'mean_reversion'
    confound: Optional[str] = None

    rebalance_band: float = 0.15        # umbral de deriva relativa
    rebalance_idle_hazard: float = 0.0  # hazard dentro de la banda de no-operar

    mr_lookback: int = 10               # dias de retorno pasado
    mr_lambda: float = 4.0              # sensibilidad de la hazard
    mr_scale: float = 0.10              # normalizacion del retorno pasado

    def validate(self) -> None:
        assert self.delta_dist in ("point", "beta", "uniform")
        assert self.kappa_dist in ("point", "beta", "uniform")
        assert self.confound in (None, "rebalancing", "mean_reversion")
        assert 0.0 <= self.delta_target <= 1.0
        assert 0.0 <= self.kappa_target <= 1.0


# --------------------------------------------------------------------------
# Simulacion
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SimConfig:
    """Parametros del motor diario."""

    market: MarketConfig = field(default_factory=MarketConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    accounting: AccountingConfig = field(default_factory=AccountingConfig)

    #: Valor minimo de una orden. Por debajo, el efectivo espera.
    min_order_value: float = 250.0

    #: Probabilidad de que, teniendo capacidad libre, una orden adicional del
    #: dia se dirija a un activo YA en cartera (genera lotes multiples).
    p_add_to_existing: float = 0.30

    #: Registrar el log de operaciones (necesario para el test de no-fuga).
    record_trades: bool = True

    def validate(self) -> None:
        self.accounting.validate()
        self.population.validate()


# --------------------------------------------------------------------------
# Estimacion
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EstimationConfig:
    n_bootstrap: int = 1000
    alpha: float = 0.05
    n_placebo: int = 200


def config_to_dict(cfg) -> dict:
    """Serializa cualquier dataclass de configuracion a dict (para el JSON)."""
    return asdict(cfg)
