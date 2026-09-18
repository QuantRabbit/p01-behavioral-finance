"""
Contabilidad de carteras: lotes, bases de costo, ventas parciales y
clasificacion ganancia / perdida / neutral.

Las cuatro decisiones contables que el enunciado exige declarar estan todas
implementadas de verdad y son seleccionables por configuracion:

1. **Ventas parciales.** Cuando se dispara una venta se liquida una fraccion
   ``f`` sorteada de ``{0.25, 0.50, 1.00}``. Para los conteos de Odean se
   ofrecen dos convenciones: ``partial_as_full`` (cuenta 1) y
   ``partial_as_fraction`` (cuenta ``f``). Ambas se reportan.
2. **Base de costo.** ``fifo``, ``average_cost`` y ``per_lot``, con lista real
   de lotes ``(acciones, precio de compra, dia de compra)`` por posicion y
   compras adicionales del mismo activo permitidas.
3. **Posicion exactamente al precio de referencia.** Se clasifica como
   NEUTRAL y queda fuera de los cuatro conteos de Odean.
4. **Precio de referencia.** ``mid`` compara el precio medio de hoy contra el
   precio medio del dia de compra; ``ask_paid`` lo compara contra el ask
   efectivamente pagado. La segunda introduce un sesgo sistematico de medio
   spread hacia "perdida".

Unidad de decision
------------------
Bajo ``fifo`` y ``average_cost`` la unidad de decision es la **posicion**
(el activo): existe un unico precio de referencia por posicion y por lo tanto
una unica hazard y un unico sorteo de venta. Bajo ``per_lot`` la unidad de
decision es el **lote**: cada lote tiene su propio precio de referencia, su
propia hazard y su propio sorteo. Los conteos de Odean y el estimador
estructural de hazards se calculan sobre la unidad de decision vigente, que es
la unica definicion internamente consistente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

EMPTY = -1
_EPS = 1e-12


@dataclass
class PortfolioState:
    """Estado vectorizado de las carteras de toda la poblacion.

    Formas de los arreglos: ``N`` agentes, ``K`` ranuras de posicion,
    ``D`` lotes por posicion.
    """

    asset_idx: np.ndarray     # (N, K) int32, EMPTY = ranura vacia
    lot_shares: np.ndarray    # (N, K, D) float64
    lot_mid: np.ndarray       # (N, K, D) precio MEDIO del dia de compra
    lot_ask: np.ndarray       # (N, K, D) precio efectivamente pagado (ask)
    lot_day: np.ndarray       # (N, K, D) int32, dia de compra
    cash_net: np.ndarray      # (N,) efectivo del libro neto
    cash_gross: np.ndarray    # (N,) efectivo del libro bruto
    merge_events: int = 0     # compras que excedieron la profundidad de lotes

    @classmethod
    def empty(cls, n_agents: int, max_positions: int, max_lots: int, cash: np.ndarray):
        shape3 = (n_agents, max_positions, max_lots)
        return cls(
            asset_idx=np.full((n_agents, max_positions), EMPTY, dtype=np.int32),
            lot_shares=np.zeros(shape3),
            lot_mid=np.zeros(shape3),
            lot_ask=np.zeros(shape3),
            lot_day=np.zeros(shape3, dtype=np.int32),
            cash_net=cash.astype(float).copy(),
            cash_gross=cash.astype(float).copy(),
        )

    # -- consultas basicas -------------------------------------------------
    @property
    def n_agents(self) -> int:
        return self.asset_idx.shape[0]

    @property
    def max_positions(self) -> int:
        return self.asset_idx.shape[1]

    @property
    def max_lots(self) -> int:
        return self.lot_shares.shape[2]

    def active_positions(self) -> np.ndarray:
        """(N, K) bool: ranuras ocupadas."""
        return self.asset_idx != EMPTY

    def position_shares(self) -> np.ndarray:
        """(N, K) acciones totales por posicion."""
        return self.lot_shares.sum(axis=2)

    def n_held(self) -> np.ndarray:
        """(N,) numero de posiciones abiertas."""
        return self.active_positions().sum(axis=1)

    def held_mask(self, n_assets: int) -> np.ndarray:
        """(N, M) bool: que activos tiene cada agente en cartera."""
        out = np.zeros((self.n_agents, n_assets), dtype=bool)
        rows, slots = np.nonzero(self.active_positions())
        out[rows, self.asset_idx[rows, slots]] = True
        return out

    def position_prices(self, prices_today: np.ndarray) -> np.ndarray:
        """(N, K) precio medio de hoy del activo de cada ranura (0 si vacia)."""
        safe_idx = np.where(self.asset_idx == EMPTY, 0, self.asset_idx)
        p = prices_today[safe_idx]
        return np.where(self.asset_idx == EMPTY, 0.0, p)

    def position_values(self, prices_today: np.ndarray) -> np.ndarray:
        return self.position_shares() * self.position_prices(prices_today)


# ---------------------------------------------------------------------------
# Base de costo y unidad de decision
# ---------------------------------------------------------------------------
def decision_units(state: PortfolioState, cost_basis: str, reference: str):
    """Construye la unidad de decision vigente.

    Devuelve ``(unit_active, unit_shares, unit_ref)``, todos de forma
    ``(N, K, D)``.

    - ``per_lot``: una unidad por lote con acciones positivas.
    - ``fifo`` / ``average_cost``: una unidad por posicion, alojada en el
      indice de lote 0, con las acciones agregadas de la posicion y el precio
      de referencia de la convencion correspondiente.
    """
    lots = state.lot_shares
    price_field = state.lot_mid if reference == "mid" else state.lot_ask
    lot_alive = lots > _EPS

    if cost_basis == "per_lot":
        unit_active = lot_alive
        unit_shares = np.where(unit_active, lots, 0.0)
        unit_ref = np.where(unit_active, price_field, 0.0)
        return unit_active, unit_shares, unit_ref

    pos_shares = lots.sum(axis=2)
    pos_active = state.active_positions() & (pos_shares > _EPS)

    if cost_basis == "fifo":
        # Referencia = precio del lote vivo mas antiguo. Los lotes se escriben
        # en orden cronologico dentro de la posicion, asi que el lote vivo de
        # menor indice es el mas antiguo.
        first_alive = np.argmax(lot_alive, axis=2)
        ref_pos = np.take_along_axis(price_field, first_alive[..., None], axis=2)[..., 0]
    elif cost_basis == "average_cost":
        num = (lots * price_field).sum(axis=2)
        ref_pos = np.where(pos_shares > _EPS, num / np.maximum(pos_shares, _EPS), 0.0)
    else:  # pragma: no cover - validado en AccountingConfig
        raise ValueError(f"base de costo desconocida: {cost_basis}")

    unit_active = np.zeros_like(lot_alive)
    unit_active[:, :, 0] = pos_active
    unit_shares = np.zeros_like(lots)
    unit_shares[:, :, 0] = np.where(pos_active, pos_shares, 0.0)
    unit_ref = np.zeros_like(lots)
    unit_ref[:, :, 0] = np.where(pos_active, ref_pos, 0.0)
    return unit_active, unit_shares, unit_ref


def classify_units(
    unit_active: np.ndarray,
    unit_ref: np.ndarray,
    prices_pos: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clasifica cada unidad de decision en ganancia / perdida / neutral.

    ``prices_pos`` tiene forma ``(N, K)`` (precio medio de hoy por ranura) y se
    difunde sobre la dimension de lotes. La banda neutral es RELATIVA al precio
    de referencia: ``|P/ref - 1| <= tol``.
    """
    p = prices_pos[:, :, None]
    upper = unit_ref * (1.0 + tol)
    lower = unit_ref * (1.0 - tol)
    is_gain = unit_active & (p > upper)
    is_loss = unit_active & (p < lower)
    is_neutral = unit_active & ~is_gain & ~is_loss
    return is_gain, is_loss, is_neutral


# ---------------------------------------------------------------------------
# Ejecucion de ventas
# ---------------------------------------------------------------------------
def allocate_removal(lot_shares: np.ndarray, to_remove_pos: np.ndarray, cost_basis: str) -> np.ndarray:
    """Reparte entre lotes las acciones a vender de cada posicion.

    - ``fifo``: consume los lotes del mas antiguo al mas nuevo.
    - ``average_cost``: consume proporcionalmente (todos los lotes comparten la
      misma base, asi que el reparto es irrelevante para la base pero se hace
      explicito para que las acciones cuadren).
    """
    if cost_basis == "fifo":
        cum = np.cumsum(lot_shares, axis=2)
        prev = cum - lot_shares
        return np.clip(to_remove_pos[:, :, None] - prev, 0.0, lot_shares)
    pos = lot_shares.sum(axis=2)
    frac = np.where(pos > _EPS, to_remove_pos / np.maximum(pos, _EPS), 0.0)
    return lot_shares * frac[:, :, None]


def apply_sales(
    state: PortfolioState,
    sell_unit: np.ndarray,
    fraction: np.ndarray,
    cost_basis: str,
) -> np.ndarray:
    """Aplica las ventas sobre el estado y devuelve las acciones vendidas.

    ``sell_unit`` y ``fraction`` tienen forma ``(N, K, D)`` en la unidad de
    decision vigente. Devuelve ``(N, K)`` con acciones vendidas por posicion.
    """
    lots = state.lot_shares
    if cost_basis == "per_lot":
        removed = np.where(sell_unit, lots * fraction, 0.0)
    else:
        to_remove_pos = np.where(sell_unit[:, :, 0], lots.sum(axis=2) * fraction[:, :, 0], 0.0)
        removed = allocate_removal(lots, to_remove_pos, cost_basis)

    state.lot_shares = lots - removed
    # Limpieza numerica: acciones residuales por debajo del epsilon se anulan.
    state.lot_shares[state.lot_shares < _EPS] = 0.0
    _clear_empty_lots(state)
    _clear_closed_positions(state)
    return removed.sum(axis=2)


def _clear_empty_lots(state: PortfolioState) -> None:
    dead = state.lot_shares <= 0.0
    state.lot_mid[dead] = 0.0
    state.lot_ask[dead] = 0.0
    state.lot_day[dead] = 0


def _clear_closed_positions(state: PortfolioState) -> None:
    closed = state.active_positions() & (state.lot_shares.sum(axis=2) <= 0.0)
    if not closed.any():
        return
    state.asset_idx[closed] = EMPTY
    state.lot_shares[closed] = 0.0
    state.lot_mid[closed] = 0.0
    state.lot_ask[closed] = 0.0
    state.lot_day[closed] = 0


# ---------------------------------------------------------------------------
# Ejecucion de compras
# ---------------------------------------------------------------------------
def apply_purchases(
    state: PortfolioState,
    rows: np.ndarray,
    slots: np.ndarray,
    assets: np.ndarray,
    shares: np.ndarray,
    mid_price: np.ndarray,
    ask_price: np.ndarray,
    day: int,
) -> None:
    """Inserta lotes. Cada par ``(row, slot)`` debe aparecer a lo sumo una vez.

    Si la posicion ya tiene la profundidad maxima de lotes, el lote nuevo se
    **fusiona** con el mas reciente ponderando por acciones (se conserva el dia
    de compra original para no rejuvenecer artificialmente la posicion). Se
    lleva la cuenta de esas fusiones para reportarlas.
    """
    if rows.size == 0:
        return
    lots = state.lot_shares[rows, slots]              # (B, D)
    free = lots <= _EPS
    has_free = free.any(axis=1)
    d = np.argmax(free, axis=1)
    d_merge = state.max_lots - 1

    # --- caso normal: hay ranura de lote libre --------------------------
    r_new, s_new, d_new = rows[has_free], slots[has_free], d[has_free]
    state.lot_shares[r_new, s_new, d_new] = shares[has_free]
    state.lot_mid[r_new, s_new, d_new] = mid_price[has_free]
    state.lot_ask[r_new, s_new, d_new] = ask_price[has_free]
    state.lot_day[r_new, s_new, d_new] = day

    # --- caso de fusion: profundidad agotada ----------------------------
    merge = ~has_free
    if merge.any():
        state.merge_events += int(merge.sum())
        r_m, s_m = rows[merge], slots[merge]
        old_sh = state.lot_shares[r_m, s_m, d_merge]
        add_sh = shares[merge]
        tot = old_sh + add_sh
        state.lot_mid[r_m, s_m, d_merge] = (
            old_sh * state.lot_mid[r_m, s_m, d_merge] + add_sh * mid_price[merge]
        ) / tot
        state.lot_ask[r_m, s_m, d_merge] = (
            old_sh * state.lot_ask[r_m, s_m, d_merge] + add_sh * ask_price[merge]
        ) / tot
        state.lot_shares[r_m, s_m, d_merge] = tot

    state.asset_idx[rows, slots] = assets


# ---------------------------------------------------------------------------
# Conteos de Odean
# ---------------------------------------------------------------------------
def odean_counts(
    is_gain: np.ndarray,
    is_loss: np.ndarray,
    sell_unit: np.ndarray,
    fraction: np.ndarray,
    agent_traded: np.ndarray,
    partial_counting: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Los cuatro conteos de Odean del dia, por agente.

    Se computan **solo** para los agentes que vendieron al menos una posicion
    hoy (``agent_traded``). Los dias sin ventas no aportan a ningun conteo.

    Bajo ``partial_as_fraction`` una realizacion cuenta ``f`` en el lado
    realizado y ``1 - f`` sigue contando como papel, de modo que las columnas
    ``realizado + papel`` sigan sumando el numero de posiciones elegibles.
    """
    act = agent_traded[:, None, None]
    gain = is_gain & act
    loss = is_loss & act

    if partial_counting == "partial_as_full":
        w_real = np.where(sell_unit, 1.0, 0.0)
    else:
        w_real = np.where(sell_unit, fraction, 0.0)
    w_paper = 1.0 - w_real

    g_r = (gain * w_real).sum(axis=(1, 2))
    g_p = (gain * w_paper).sum(axis=(1, 2))
    l_r = (loss * w_real).sum(axis=(1, 2))
    l_p = (loss * w_paper).sum(axis=(1, 2))
    return g_r, g_p, l_r, l_p
