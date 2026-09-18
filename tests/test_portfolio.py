"""Pruebas de la contabilidad de lotes, bases de costo y conteos de Odean."""

import numpy as np
import pytest

from src.portfolio import (
    EMPTY,
    PortfolioState,
    apply_purchases,
    apply_sales,
    classify_units,
    decision_units,
    odean_counts,
)


def _estado_con_dos_lotes():
    """Un agente, una posicion en el activo 7, dos lotes construidos a mano.

    Lote 0: 100 acciones a 10.00 (dia 0)   Lote 1: 100 acciones a 20.00 (dia 5)
    FIFO -> base 10.00 ; costo promedio -> base 15.00 ; per_lot -> 10 y 20.
    """
    st = PortfolioState.empty(1, 3, 4, cash=np.array([0.0]))
    st.asset_idx[0, 0] = 7
    st.lot_shares[0, 0, 0] = 100.0
    st.lot_mid[0, 0, 0] = 10.0
    st.lot_ask[0, 0, 0] = 10.0 * 1.0005
    st.lot_day[0, 0, 0] = 0
    st.lot_shares[0, 0, 1] = 100.0
    st.lot_mid[0, 0, 1] = 20.0
    st.lot_ask[0, 0, 1] = 20.0 * 1.0005
    st.lot_day[0, 0, 1] = 5
    return st


# ---------------------------------------------------------------------------
# Bases de costo
# ---------------------------------------------------------------------------
def test_fifo_y_promedio_dan_bases_distintas_con_lotes_multiples():
    st = _estado_con_dos_lotes()
    _, _, ref_fifo = decision_units(st, "fifo", "mid")
    _, _, ref_avg = decision_units(st, "average_cost", "mid")
    assert ref_fifo[0, 0, 0] == pytest.approx(10.0)
    assert ref_avg[0, 0, 0] == pytest.approx(15.0)
    assert ref_fifo[0, 0, 0] != ref_avg[0, 0, 0]


def test_per_lot_produce_una_unidad_por_lote():
    st = _estado_con_dos_lotes()
    act_pl, sh_pl, ref_pl = decision_units(st, "per_lot", "mid")
    assert act_pl[0, 0].tolist() == [True, True, False, False]
    assert ref_pl[0, 0, 0] == pytest.approx(10.0) and ref_pl[0, 0, 1] == pytest.approx(20.0)
    assert sh_pl[0, 0, 0] == 100.0 and sh_pl[0, 0, 1] == 100.0

    act_fi, sh_fi, _ = decision_units(st, "fifo", "mid")
    assert act_fi[0, 0].tolist() == [True, False, False, False]
    assert sh_fi[0, 0, 0] == 200.0, "bajo fifo la unidad agrega toda la posicion"


def test_referencia_ask_es_mayor_que_mid_por_medio_spread():
    st = _estado_con_dos_lotes()
    _, _, ref_mid = decision_units(st, "average_cost", "mid")
    _, _, ref_ask = decision_units(st, "average_cost", "ask_paid")
    assert ref_ask[0, 0, 0] > ref_mid[0, 0, 0]
    assert ref_ask[0, 0, 0] / ref_mid[0, 0, 0] == pytest.approx(1.0005)


# ---------------------------------------------------------------------------
# Clasificacion ganancia / perdida / neutral
# ---------------------------------------------------------------------------
def test_posicion_exactamente_al_costo_es_neutral_y_no_entra_en_ningun_conteo():
    st = _estado_con_dos_lotes()
    act, _, ref = decision_units(st, "average_cost", "mid")
    precios = np.array([[15.0, 0.0, 0.0]])   # exactamente la base promedio
    g, l, n = classify_units(act, ref, precios, tol=1e-10)
    assert not g.any() and not l.any()
    assert n[0, 0, 0]

    sell = np.zeros_like(g)
    frac = np.zeros_like(ref)
    gr, gp, lr, lp = odean_counts(g, l, sell, frac, np.array([True]), "partial_as_full")
    assert (gr + gp + lr + lp)[0] == 0.0


def test_clasificacion_ganancia_y_perdida():
    st = _estado_con_dos_lotes()
    act, _, ref = decision_units(st, "per_lot", "mid")
    precios = np.array([[15.0, 0.0, 0.0]])   # > 10 (lote 0) y < 20 (lote 1)
    g, l, _ = classify_units(act, ref, precios, tol=1e-10)
    assert g[0, 0, 0] and not g[0, 0, 1]
    assert l[0, 0, 1] and not l[0, 0, 0]


# ---------------------------------------------------------------------------
# Ventas: FIFO consume el lote mas antiguo
# ---------------------------------------------------------------------------
def test_venta_fifo_consume_primero_el_lote_mas_antiguo():
    st = _estado_con_dos_lotes()
    sell = np.zeros((1, 3, 4), bool); sell[0, 0, 0] = True
    frac = np.zeros((1, 3, 4)); frac[0, 0, 0] = 0.25      # 25% de 200 = 50 acciones
    vendidas = apply_sales(st, sell, frac, "fifo")
    assert vendidas[0, 0] == pytest.approx(50.0)
    assert st.lot_shares[0, 0, 0] == pytest.approx(50.0)
    assert st.lot_shares[0, 0, 1] == pytest.approx(100.0), "el lote nuevo no se toca"


def test_venta_promedio_consume_proporcionalmente():
    st = _estado_con_dos_lotes()
    sell = np.zeros((1, 3, 4), bool); sell[0, 0, 0] = True
    frac = np.zeros((1, 3, 4)); frac[0, 0, 0] = 0.25
    apply_sales(st, sell, frac, "average_cost")
    assert st.lot_shares[0, 0, 0] == pytest.approx(75.0)
    assert st.lot_shares[0, 0, 1] == pytest.approx(75.0)


def test_venta_per_lot_afecta_solo_al_lote_elegido():
    st = _estado_con_dos_lotes()
    sell = np.zeros((1, 3, 4), bool); sell[0, 0, 1] = True
    frac = np.zeros((1, 3, 4)); frac[0, 0, 1] = 0.5
    apply_sales(st, sell, frac, "per_lot")
    assert st.lot_shares[0, 0, 0] == pytest.approx(100.0)
    assert st.lot_shares[0, 0, 1] == pytest.approx(50.0)


def test_venta_total_libera_la_ranura():
    st = _estado_con_dos_lotes()
    sell = np.zeros((1, 3, 4), bool); sell[0, 0, 0] = True
    frac = np.zeros((1, 3, 4)); frac[0, 0, 0] = 1.0
    apply_sales(st, sell, frac, "fifo")
    assert st.asset_idx[0, 0] == EMPTY
    assert st.lot_shares[0, 0].sum() == 0.0
    assert st.n_held()[0] == 0


# ---------------------------------------------------------------------------
# Compras y lotes multiples
# ---------------------------------------------------------------------------
def test_compra_adicional_del_mismo_activo_abre_un_lote_nuevo():
    st = _estado_con_dos_lotes()
    apply_purchases(
        st,
        rows=np.array([0]), slots=np.array([0]), assets=np.array([7]),
        shares=np.array([50.0]), mid_price=np.array([30.0]),
        ask_price=np.array([30.015]), day=9,
    )
    assert st.lot_shares[0, 0, 2] == 50.0 and st.lot_mid[0, 0, 2] == 30.0
    assert st.lot_day[0, 0, 2] == 9
    assert (st.lot_shares[0, 0] > 0).sum() == 3
    assert st.merge_events == 0


def test_compra_mas_alla_de_la_profundidad_fusiona_el_ultimo_lote():
    st = PortfolioState.empty(1, 1, 2, cash=np.array([0.0]))
    st.asset_idx[0, 0] = 3
    st.lot_shares[0, 0, :] = [10.0, 10.0]
    st.lot_mid[0, 0, :] = [10.0, 20.0]
    st.lot_ask[0, 0, :] = [10.0, 20.0]
    apply_purchases(
        st, np.array([0]), np.array([0]), np.array([3]),
        np.array([10.0]), np.array([30.0]), np.array([30.0]), day=4,
    )
    assert st.merge_events == 1
    assert st.lot_shares[0, 0, 1] == pytest.approx(20.0)
    assert st.lot_mid[0, 0, 1] == pytest.approx(25.0)   # (10*20 + 10*30)/20


def test_held_mask_refleja_los_activos_en_cartera():
    st = _estado_con_dos_lotes()
    m = st.held_mask(n_assets=10)
    assert m[0, 7] and m[0].sum() == 1


# ---------------------------------------------------------------------------
# Conteos de Odean
# ---------------------------------------------------------------------------
def _escenario_conteos():
    """2 agentes x 3 posiciones. Agente 0 vende hoy, agente 1 no."""
    st = PortfolioState.empty(2, 3, 2, cash=np.zeros(2))
    st.asset_idx[:, :] = np.array([[0, 1, 2], [0, 1, 2]])
    st.lot_shares[:, :, 0] = 10.0
    st.lot_mid[:, :, 0] = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
    st.lot_ask[:, :, 0] = st.lot_mid[:, :, 0]
    return st


def test_dia_sin_ventas_no_incrementa_ningun_conteo():
    st = _escenario_conteos()
    act, _, ref = decision_units(st, "average_cost", "mid")
    precios = np.array([[12.0, 8.0, 12.0], [12.0, 8.0, 12.0]])
    g, l, _ = classify_units(act, ref, precios, 1e-10)
    sell = np.zeros_like(g)
    frac = np.zeros_like(ref)
    agente_vendio = np.array([False, False])
    gr, gp, lr, lp = odean_counts(g, l, sell, frac, agente_vendio, "partial_as_full")
    assert gr.sum() == gp.sum() == lr.sum() == lp.sum() == 0.0


def test_gr_mas_gp_cuenta_las_ganancias_abiertas_solo_en_dias_con_venta():
    st = _escenario_conteos()
    act, _, ref = decision_units(st, "average_cost", "mid")
    precios = np.array([[12.0, 8.0, 12.0], [12.0, 8.0, 12.0]])  # 2 ganancias, 1 perdida
    g, l, _ = classify_units(act, ref, precios, 1e-10)
    sell = np.zeros_like(g); sell[0, 0, 0] = True
    frac = np.zeros_like(ref); frac[0, 0, 0] = 1.0
    gr, gp, lr, lp = odean_counts(g, l, sell, frac, np.array([True, False]), "partial_as_full")
    assert gr[0] == 1.0 and gp[0] == 1.0 and lr[0] == 0.0 and lp[0] == 1.0
    assert (gr + gp)[0] == 2.0, "las dos posiciones en ganancia del agente que opero"
    assert gr[1] == gp[1] == lr[1] == lp[1] == 0.0, "el agente que no opero no aporta nada"


def test_convenciones_de_venta_parcial():
    st = _escenario_conteos()
    act, _, ref = decision_units(st, "average_cost", "mid")
    precios = np.array([[12.0, 8.0, 12.0], [12.0, 8.0, 12.0]])
    g, l, _ = classify_units(act, ref, precios, 1e-10)
    sell = np.zeros_like(g); sell[0, 0, 0] = True
    frac = np.zeros_like(ref); frac[0, 0, 0] = 0.25
    gr_f, gp_f, _, _ = odean_counts(g, l, sell, frac, np.array([True, False]), "partial_as_full")
    gr_p, gp_p, _, _ = odean_counts(g, l, sell, frac, np.array([True, False]), "partial_as_fraction")
    assert gr_f[0] == 1.0 and gp_f[0] == 1.0
    assert gr_p[0] == pytest.approx(0.25) and gp_p[0] == pytest.approx(1.75)
    assert (gr_f + gp_f)[0] == (gr_p + gp_p)[0], "realizado + papel debe conservarse"
