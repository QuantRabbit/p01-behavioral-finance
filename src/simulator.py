"""
Motor de simulacion diario, vectorizado por agente-dia.

Orden de un dia ``t``
---------------------
1. El reloj de ``PriceView`` avanza a ``t``; se acumula el P&L de mercado de la
   noche sobre las posiciones que se traian abiertas.
2. Se valora la cartera a precios de hoy y se construyen las unidades de
   decision segun la base de costo vigente.
3. Se calcula la hazard de cada unidad y se sortea la venta (y su fraccion).
4. **Si y solo si el agente vendio al menos una posicion hoy**, se ejecuta la
   contabilidad de Odean sobre TODAS sus posiciones abiertas de hoy. Los dias
   sin ventas no aportan a ningun conteo.
5. Se ejecutan las ventas al bid, cobrando comision por orden.
6. Se reinvierte el efectivo **el mismo dia** en las ranuras libres (y, con
   cierta probabilidad, en un activo ya en cartera, lo que genera lotes
   multiples). Se registra la fraccion de efectivo de cada cuenta para poder
   controlar el *cash drag* despues.
7. Se registra el valor de cierre.

Al final del dia ``T`` se liquida todo al bid.

Dos libros contables en paralelo
--------------------------------
Dentro del mismo bucle se llevan dos libros con **las mismas decisiones, las
mismas fechas, los mismos activos y las mismas cantidades en unidades**:

- **neto**: compra al ask, vende al bid, paga comision.
- **bruto**: las mismas operaciones ejecutadas al precio medio y sin comision.

Asi ``r_gross - r_net`` es exactamente el costo de transaccion acumulado y
``beta_gross`` queda limpio de la definicion de costos. Como el libro bruto
conserva las cantidades en unidades, el efectivo que ahorra queda ocioso; por
eso se reporta ademas ``gross_return_comp``, el retorno bruto **capitalizado**
que se obtiene de componer los retornos diarios sin costo
``R_t + c_t / V_{t-1}``. La diferencia entre ambas definiciones se cuantifica
en el reporte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .agents import (
    Population,
    build_population,
    disposition_hazard,
    mean_reversion_hazard,
    rebalancing_topup_need,
    rebalancing_trim,
)
from .config import SimConfig
from .portfolio import (
    EMPTY,
    PortfolioState,
    apply_purchases,
    apply_sales,
    classify_units,
    decision_units,
    odean_counts,
    unit_buy_days,
)
from .price_process import MarketPaths, PriceView, generate_market

CELL_NAMES = ("A_rebote_bajo_agua", "B_ganadora_infraponderada", "C_perdedora_sobreponderada")


@dataclass
class SimulationResult:
    """Salida completa de una corrida."""

    accounts: pd.DataFrame
    population: Population
    market: MarketPaths
    config: SimConfig
    daily_value: np.ndarray                 # (N, T+1) valor neto de la cuenta
    cells: Dict[str, Dict[str, np.ndarray]] # celdas discriminantes por agente
    buy_log: Dict[str, np.ndarray]          # agente, activo, dia
    integrity: Dict[str, float]             # invariantes y contadores de control
    scenario_key: str = ""


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _draw_fractions(rng: np.random.Generator, shape, fractions, probs) -> np.ndarray:
    """Sortea la fraccion liquidada de cada venta."""
    u = rng.random(shape)
    cuts = np.cumsum(probs)
    out = np.full(shape, fractions[-1])
    for i in range(len(fractions) - 1, -1, -1):
        out = np.where(u < cuts[i], fractions[i], out)
    return out


def _execute_buys(
    state: PortfolioState,
    cfg: SimConfig,
    prices: np.ndarray,
    new_count: np.ndarray,
    existing_slots: np.ndarray,
    existing_valid: np.ndarray,
    rng: np.random.Generator,
    day: int,
    log: Optional[Dict[str, List[np.ndarray]]],
):
    """Ejecuta las ordenes de compra del dia.

    ``new_count`` es el numero de ordenes dirigidas a activos NUEVOS por agente;
    ``existing_slots`` / ``existing_valid`` son ordenes dirigidas a posiciones
    ya abiertas (generan lotes adicionales). Todos los pares ``(agente, ranura)``
    resultantes son distintos por construccion: las ordenes nuevas usan ranuras
    vacias y las adicionales usan ranuras ocupadas.

    Devuelve ``(costos, volumen_mid, n_ordenes)`` por agente.
    """
    n_agents, n_assets = state.n_agents, prices.shape[0]
    costs = np.zeros(n_agents)
    volume = np.zeros(n_agents)
    n_orders_done = np.zeros(n_agents, dtype=np.int32)

    cash = state.cash_net
    afford = np.floor(np.maximum(cash, 0.0) / cfg.min_order_value).astype(np.int64)
    n_ex = existing_valid.sum(axis=1)
    want = new_count.astype(np.int64) + n_ex
    total = np.minimum(want, afford)
    if not np.any(total > 0):
        return costs, volume, n_orders_done

    k_new = np.minimum(new_count.astype(np.int64), total)
    k_ex = total - k_new
    rank_ex = np.cumsum(existing_valid, axis=1) - 1
    keep_ex = existing_valid & (rank_ex < k_ex[:, None])

    budget = np.where(total > 0, cash / np.maximum(total, 1), 0.0)

    rows_l, slots_l, assets_l = [], [], []

    # --- ordenes a activos nuevos ---------------------------------------
    g_max = int(k_new.max())
    if g_max > 0:
        held = state.held_mask(n_assets)
        score = rng.random((n_agents, n_assets)) + held * 10.0
        kth = min(g_max, n_assets - 1)
        cand = np.argpartition(score, kth, axis=1)[:, :g_max]
        active = state.active_positions()
        free_slots = np.argsort(active.astype(np.int8), axis=1, kind="stable")[:, :g_max]
        cols = np.arange(g_max)[None, :]
        valid = cols < k_new[:, None]
        r, c = np.nonzero(valid)
        rows_l.append(r)
        slots_l.append(free_slots[r, c])
        assets_l.append(cand[r, c])

    # --- ordenes adicionales sobre posiciones ya abiertas ----------------
    if keep_ex.any():
        r, c = np.nonzero(keep_ex)
        s = existing_slots[r, c]
        rows_l.append(r)
        slots_l.append(s)
        assets_l.append(state.asset_idx[r, s])

    if not rows_l:
        return costs, volume, n_orders_done

    rows = np.concatenate(rows_l)
    slots = np.concatenate(slots_l).astype(np.int64)
    assets = np.concatenate(assets_l).astype(np.int64)

    mid = prices[assets]
    ask = mid * cfg.costs.ask_mult
    b = budget[rows]
    shares = (b - cfg.costs.commission) / ask
    ok = shares > 0.0
    rows, slots, assets, mid, ask, shares = (
        rows[ok], slots[ok], assets[ok], mid[ok], ask[ok], shares[ok],
    )
    if rows.size == 0:
        return costs, volume, n_orders_done

    notional_mid = shares * mid
    cost_i = notional_mid * cfg.costs.half_spread + cfg.costs.commission
    np.add.at(state.cash_net, rows, -(shares * ask + cfg.costs.commission))
    np.add.at(state.cash_gross, rows, -notional_mid)
    np.add.at(costs, rows, cost_i)
    np.add.at(volume, rows, notional_mid)
    np.add.at(n_orders_done, rows, 1)

    apply_purchases(state, rows, slots, assets, shares, mid, ask, day)

    if log is not None:
        log["agent"].append(rows.astype(np.int32))
        log["asset"].append(assets.astype(np.int16))
        log["day"].append(np.full(rows.size, day, dtype=np.int16))
        log["shares_net"].append(shares.copy())
        log["shares_gross"].append(shares.copy())
    return costs, volume, n_orders_done


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------
def run_simulation(
    cfg: SimConfig,
    rngs: Dict[str, np.random.Generator],
    scenario_key: str = "",
    market: Optional[MarketPaths] = None,
    population: Optional[Population] = None,
) -> SimulationResult:
    """Corre la simulacion completa y devuelve metricas por cuenta.

    ``market`` y ``population`` pueden inyectarse para las pruebas; en uso
    normal se generan aqui, y el mercado **siempre** se genera antes que la
    poblacion, con su propio generador.
    """
    cfg.validate()
    acc = cfg.accounting
    pcfg = cfg.population

    # 1) Precios primero, con su propio RNG. Los agentes aun no existen.
    if market is None:
        market = generate_market(cfg.market, rngs["prices"])
    if population is None:
        population = build_population(pcfg, rngs["population"])

    rng = rngs["decisions"]
    view = PriceView(market, current_day=0)

    n = population.n_agents
    n_assets = market.n_assets
    t_total = market.n_days
    k_slots = int(population.capacity.max())
    d_lots = acc.max_lots

    state = PortfolioState.empty(n, k_slots, d_lots, cash=population.wealth)

    # --- acumuladores por cuenta ----------------------------------------
    g_r = np.zeros(n); g_p = np.zeros(n); l_r = np.zeros(n); l_p = np.zeros(n)
    days_gain = np.zeros(n); days_loss = np.zeros(n); days_neutral = np.zeros(n)
    sells_gain = np.zeros(n); sells_loss = np.zeros(n); sells_neutral = np.zeros(n)
    volume_mid = np.zeros(n)
    costs_paid = np.zeros(n)
    trade_count = np.zeros(n, dtype=np.int64)
    lots_opened = np.zeros(n, dtype=np.int64)
    market_pnl = np.zeros(n)
    sum_value = np.zeros(n)
    sum_cash_w = np.zeros(n)
    sum_hhi = np.zeros(n)
    n_value_obs = 0
    sale_days = np.zeros(n, dtype=np.int64)

    cell_opp = {name: np.zeros(n) for name in CELL_NAMES}
    cell_sell = {name: np.zeros(n) for name in CELL_NAMES}

    hold_g_sum = np.zeros(n); hold_g_cnt = np.zeros(n)
    hold_l_sum = np.zeros(n); hold_l_cnt = np.zeros(n)
    sale_events: List[tuple] = []   # (agente, dias de tenencia, era ganancia)

    daily_value = np.zeros((n, t_total + 1))
    log = {k: [] for k in ("agent", "asset", "day", "shares_net", "shares_gross")} if cfg.record_trades else None

    target_w = 1.0 / population.capacity.astype(float)
    gross_comp = np.ones(n)     # producto acumulado de (1 + R_t + c_t/V_{t-1})

    # ------------------------------------------------------------------
    # Dia 0: compra inicial de n_i activos distintos, reparto equitativo
    # ------------------------------------------------------------------
    prices0 = view.prices_on(0)
    c0, v0, o0 = _execute_buys(
        state, cfg, prices0,
        new_count=population.capacity.astype(np.int64),
        existing_slots=np.zeros((n, 1), dtype=np.int64),
        existing_valid=np.zeros((n, 1), dtype=bool),
        rng=rng, day=0, log=log,
    )
    costs_paid += c0
    volume_mid += v0
    trade_count += o0
    lots_opened += o0
    value0 = state.cash_net + state.position_values(prices0).sum(axis=1)
    daily_value[:, 0] = value0
    prev_value = value0.copy()

    # ------------------------------------------------------------------
    # Dias 1..T
    # ------------------------------------------------------------------
    for t in range(1, t_total + 1):
        shares_overnight = state.position_shares()
        assets_overnight = state.asset_idx
        view.advance_to(t)
        prices = view.prices_on(t)
        prices_prev = view.prices_on(t - 1)

        safe = np.where(assets_overnight == EMPTY, 0, assets_overnight)
        d_price = np.where(assets_overnight == EMPTY, 0.0, prices[safe] - prices_prev[safe])
        market_pnl += (shares_overnight * d_price).sum(axis=1)

        pos_price = state.position_prices(prices)
        pos_value = state.position_shares() * pos_price
        holdings = pos_value.sum(axis=1)
        account_value = state.cash_net + holdings

        unit_active, unit_shares, unit_ref = decision_units(state, acc.cost_basis, acc.reference)
        is_gain, is_loss, is_neutral = classify_units(unit_active, unit_ref, pos_price, acc.neutral_tol)

        weights = np.where(account_value[:, None] > 0.0, pos_value / np.maximum(account_value[:, None], 1e-12), 0.0)
        past_ret = view.past_returns_on(t, pcfg.mr_lookback)
        pos_past_ret = np.where(assets_overnight == EMPTY, 0.0, past_ret[safe])

        # ---- hazard y sorteo de venta ---------------------------------
        unit_shape = unit_active.shape
        if pcfg.confound == "rebalancing":
            # Regla determinista: condiciona en el PESO, no en el precio de compra.
            trim, frac_trim = rebalancing_trim(weights, target_w[:, None], pcfg.rebalance_band)
            sell_unit = unit_active & trim[:, :, None]
            fraction = np.broadcast_to(frac_trim[:, :, None], unit_shape).copy()
            if pcfg.rebalance_idle_hazard > 0.0:
                extra = unit_active & (rng.random(unit_shape) < pcfg.rebalance_idle_hazard)
                sell_unit = sell_unit | extra
                fraction = np.where(
                    fraction > 0.0,
                    fraction,
                    _draw_fractions(rng, unit_shape, acc.partial_fractions, acc.partial_probs),
                )
        else:
            if pcfg.confound == "mean_reversion":
                # Condiciona en el RETORNO RECIENTE, no en el precio de compra.
                hazard = np.broadcast_to(
                    mean_reversion_hazard(pos_past_ret, pcfg)[:, :, None], unit_shape
                )
            else:
                hazard = disposition_hazard(
                    is_gain, is_loss, population.h0, population.h_gain, population.h_loss
                )
            sell_unit = unit_active & (rng.random(unit_shape) < hazard)
            fraction = _draw_fractions(rng, unit_shape, acc.partial_fractions, acc.partial_probs)

        fraction = np.where(sell_unit, fraction, 0.0)
        sell_unit &= fraction > 0.0

        # ---- metricas de hazard: TODOS los dias-posicion ---------------
        days_gain += is_gain.sum(axis=(1, 2))
        days_loss += is_loss.sum(axis=(1, 2))
        days_neutral += is_neutral.sum(axis=(1, 2))
        sells_gain += (sell_unit & is_gain).sum(axis=(1, 2))
        sells_loss += (sell_unit & is_loss).sum(axis=(1, 2))
        sells_neutral += (sell_unit & is_neutral).sum(axis=(1, 2))

        # ---- celdas discriminantes -------------------------------------
        underweight = weights < target_w[:, None] * (1.0 - pcfg.rebalance_band)
        overweight = weights > target_w[:, None] * (1.0 + pcfg.rebalance_band)
        rebound = pos_past_ret > 0.02
        cell_masks = {
            CELL_NAMES[0]: is_loss & rebound[:, :, None],
            CELL_NAMES[1]: is_gain & underweight[:, :, None],
            CELL_NAMES[2]: is_loss & overweight[:, :, None],
        }
        for name, mask in cell_masks.items():
            cell_opp[name] += mask.sum(axis=(1, 2))
            cell_sell[name] += (mask & sell_unit).sum(axis=(1, 2))

        # ---- contabilidad de Odean: solo dias con al menos una venta ----
        agent_traded = sell_unit.any(axis=(1, 2))
        sale_days += agent_traded
        d_gr, d_gp, d_lr, d_lp = odean_counts(
            is_gain, is_loss, sell_unit, fraction, agent_traded, acc.partial_counting
        )
        g_r += d_gr; g_p += d_gp; l_r += d_lr; l_p += d_lp

        # ---- horizontes de tenencia de las ventas ----------------------
        if agent_traded.any():
            buy_days = unit_buy_days(state, acc.cost_basis)
            hold = (t - buy_days)
            sg = sell_unit & is_gain
            sl = sell_unit & is_loss
            hold_g_sum += (hold * sg).sum(axis=(1, 2)); hold_g_cnt += sg.sum(axis=(1, 2))
            hold_l_sum += (hold * sl).sum(axis=(1, 2)); hold_l_cnt += sl.sum(axis=(1, 2))
            for mask, flag in ((sg, True), (sl, False)):
                if mask.any():
                    rows_m = np.nonzero(mask)[0].astype(np.int32)
                    sale_events.append((rows_m, hold[mask].astype(np.float32), flag))

        # ---- ejecucion de ventas ---------------------------------------
        shares_sold = apply_sales(state, sell_unit, fraction, acc.cost_basis)
        sold_any = shares_sold > 0.0
        n_sell_orders = sold_any.sum(axis=1)
        notional_sold = (shares_sold * pos_price).sum(axis=1)
        sell_costs = notional_sold * cfg.costs.half_spread + cfg.costs.commission * n_sell_orders
        state.cash_net += notional_sold * cfg.costs.bid_mult - cfg.costs.commission * n_sell_orders
        state.cash_gross += notional_sold
        costs_paid += sell_costs
        volume_mid += notional_sold
        trade_count += n_sell_orders

        # ---- reinversion el mismo dia -----------------------------------
        active = state.active_positions()
        n_held = active.sum(axis=1)
        free_cap = np.maximum(population.capacity - n_held, 0)
        free_slots_avail = state.max_positions - n_held
        new_count = np.minimum(free_cap, free_slots_avail)

        if pcfg.confound == "rebalancing":
            need = rebalancing_topup_need(weights, target_w[:, None], pcfg.rebalance_band, active)
            n_top = 3
            order = np.argsort(-need, axis=1)[:, :n_top]
            ex_slots = order
            ex_valid = np.take_along_axis(need, order, axis=1) > 0.0
        else:
            score = rng.random((n, state.max_positions)) + (~active) * 10.0
            ex_slots = score.argmin(axis=1)[:, None]
            add = (rng.random(n) < cfg.p_add_to_existing) & (n_held > 0)
            ex_valid = add[:, None]

        cb, vb, ob = _execute_buys(
            state, cfg, prices, new_count.astype(np.int64), ex_slots.astype(np.int64),
            ex_valid, rng, t, log,
        )
        costs_paid += cb
        volume_mid += vb
        trade_count += ob
        lots_opened += ob

        # ---- cierre del dia ---------------------------------------------
        close_holdings = state.position_values(prices).sum(axis=1)
        value_t = state.cash_net + close_holdings
        daily_value[:, t] = value_t
        cash_w = np.where(value_t > 0.0, state.cash_net / np.maximum(value_t, 1e-12), 0.0)
        sum_cash_w += cash_w
        sum_value += value_t
        w_close = np.where(value_t[:, None] > 0.0, state.position_values(prices) / np.maximum(value_t[:, None], 1e-12), 0.0)
        sum_hhi += (w_close ** 2).sum(axis=1)
        n_value_obs += 1

        day_costs = sell_costs + cb
        gross_comp *= 1.0 + np.where(
            prev_value > 0.0, (value_t - prev_value + day_costs) / np.maximum(prev_value, 1e-12), 0.0
        )
        prev_value = value_t.copy()

    # ------------------------------------------------------------------
    # Verificacion de conservacion de valor ANTES de la liquidacion final
    # ------------------------------------------------------------------
    prices_T = view.prices_on(t_total)
    holdings_T = state.position_values(prices_T).sum(axis=1)
    lhs = state.cash_net + holdings_T + costs_paid
    rhs = population.wealth + market_pnl
    conservation_error = float(np.max(np.abs(lhs - rhs) / np.maximum(population.wealth, 1.0)))

    # ------------------------------------------------------------------
    # Liquidacion final al bid
    # ------------------------------------------------------------------
    pos_shares_T = state.position_shares()
    n_liq_orders = (pos_shares_T > 0.0).sum(axis=1)
    notional_T = (pos_shares_T * state.position_prices(prices_T)).sum(axis=1)
    final_net = state.cash_net + notional_T * cfg.costs.bid_mult - cfg.costs.commission * n_liq_orders
    final_gross = state.cash_gross + notional_T
    liq_costs = notional_T * cfg.costs.half_spread + cfg.costs.commission * n_liq_orders
    costs_paid_total = costs_paid + liq_costs
    volume_mid += notional_T
    trade_count += n_liq_orders

    gross_comp_final = gross_comp * (
        1.0 + np.where(prev_value > 0.0, (final_net - prev_value + liq_costs) / np.maximum(prev_value, 1e-12), 0.0)
    )

    # ------------------------------------------------------------------
    # Metricas por cuenta
    # ------------------------------------------------------------------
    w0 = population.wealth
    net_return = final_net / w0 - 1.0
    gross_return = final_gross / w0 - 1.0
    gross_return_comp = gross_comp_final - 1.0

    mean_value = sum_value / max(n_value_obs, 1)
    turnover = np.where(
        mean_value > 0.0,
        (volume_mid / 2.0) / np.maximum(mean_value, 1e-12) * (252.0 / t_total),
        0.0,
    )

    daily_ret = np.diff(daily_value, axis=1) / np.maximum(daily_value[:, :-1], 1e-12)
    port_vol = daily_ret.std(axis=1, ddof=1) * np.sqrt(252.0)

    med_g, med_l = _median_holding_by_agent(sale_events, n)

    accounts = pd.DataFrame({
        "agent_id": np.arange(n),
        "delta": population.delta,
        "kappa": population.kappa,
        "initial_wealth": w0,
        "capacity": population.capacity,
        "final_net": final_net,
        "final_gross": final_gross,
        "net_return": net_return,
        "gross_return": gross_return,
        "gross_return_comp": gross_return_comp,
        "turnover": turnover,
        "trade_count": trade_count,
        "num_lots_opened": lots_opened,
        "mean_cash_weight": sum_cash_w / max(n_value_obs, 1),
        "portfolio_vol": port_vol,
        "hhi": sum_hhi / max(n_value_obs, 1),
        "costs_paid": costs_paid_total,
        "G_r": g_r, "G_p": g_p, "L_r": l_r, "L_p": l_p,
        "position_days_in_gain": days_gain,
        "position_days_in_loss": days_loss,
        "position_days_neutral": days_neutral,
        "sells_from_gain": sells_gain,
        "sells_from_loss": sells_loss,
        "sells_from_neutral": sells_neutral,
        "sale_days": sale_days,
        "mean_holding_winners": np.where(hold_g_cnt > 0, hold_g_sum / np.maximum(hold_g_cnt, 1), np.nan),
        "mean_holding_losers": np.where(hold_l_cnt > 0, hold_l_sum / np.maximum(hold_l_cnt, 1), np.nan),
    })
    accounts["median_holding_winners"] = med_g
    accounts["median_holding_losers"] = med_l
    for name in CELL_NAMES:
        accounts[f"cell_{name}_opp"] = cell_opp[name]
        accounts[f"cell_{name}_sell"] = cell_sell[name]

    buy_log = {}
    if log is not None and log["agent"]:
        buy_log = {k: np.concatenate(v) for k, v in log.items()}

    integrity = {
        "conservation_error_rel": conservation_error,
        "merge_events": float(state.merge_events),
        "neutral_unit_days": float(days_neutral.sum()),
        "total_unit_days": float(days_gain.sum() + days_loss.sum() + days_neutral.sum()),
        "mean_lots_per_buy_merged": float(state.merge_events) / max(float(lots_opened.sum()), 1.0),
        "gross_minus_net_equals_costs": float(
            np.max(np.abs((final_gross - final_net) - costs_paid_total) / np.maximum(w0, 1.0))
        ),
        "runtime_days": float(t_total),
    }

    return SimulationResult(
        accounts=accounts,
        population=population,
        market=market,
        config=cfg,
        daily_value=daily_value,
        cells={"opportunities": cell_opp, "sells": cell_sell},
        buy_log=buy_log,
        integrity=integrity,
        scenario_key=scenario_key,
    )


def _median_holding_by_agent(sale_events: List[tuple], n_agents: int):
    """Mediana por cuenta de los dias de tenencia de ganadoras y perdedoras.

    Se calcula sobre el registro completo de eventos de venta (una fila por
    unidad de decision vendida). Las cuentas sin ventas de un lado quedan NaN,
    que es la respuesta honesta: no hay dato, no un cero.
    """
    med = {True: np.full(n_agents, np.nan), False: np.full(n_agents, np.nan)}
    for flag in (True, False):
        rows = [r for r, _, f in sale_events if f is flag]
        if not rows:
            continue
        agents = np.concatenate(rows)
        holds = np.concatenate([h for _, h, f in sale_events if f is flag])
        s = pd.Series(holds).groupby(agents).median()
        med[flag][s.index.to_numpy()] = s.to_numpy()
    return med[True], med[False]
