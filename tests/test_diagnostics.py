"""Pruebas de los diagnosticos: independencia, causalidad reversa y celdas."""

import numpy as np
import pandas as pd
import pytest

from src.config import MarketConfig, PopulationConfig, SimConfig, derive_rngs
from src.diagnostics import (
    CELL_PREDICTIONS,
    backward_regression,
    cash_drag_isolation,
    discriminating_cells,
    independence_diagnostics,
    match_a_priori,
    nearest_reference,
    ols_vs_iv,
    pattern_str,
    permutation_placebo,
    signature_distances,
    signature_matrix,
    spread_decomposition,
)
from src.simulator import CELL_NAMES, run_simulation


def _run(tag, **pop_kw):
    cfg = SimConfig(
        market=MarketConfig(n_assets=40, n_days=252),
        population=PopulationConfig(n_agents=400, **pop_kw),
    )
    return run_simulation(cfg, derive_rngs(tag), tag)


@pytest.fixture(scope="module")
def het():
    return _run("diag_het", delta_dist="uniform", kappa_dist="uniform")


@pytest.fixture(scope="module")
def nulo():
    return _run("diag_nulo")


# ---------------------------------------------------------------------------
# (D1) Independencia
# ---------------------------------------------------------------------------
def test_independencia_en_poblacion_heterogenea(het):
    d = independence_diagnostics(het)
    assert abs(d["corr_delta_kappa"]) < 0.06, "delta y kappa deben ser independientes"
    assert d["corr_delta_turnover"] < -0.2, "lock-in: delta reduce el turnover realizado"
    assert d["corr_kappa_turnover"] > 0.5, "kappa aumenta el turnover"
    assert d["delta_sd"] > 0 and d["kappa_sd"] > 0


def test_correlaciones_no_definidas_se_reportan_como_nd(nulo):
    """Anti-patron 3: nunca imprimir 0.0000 cuando la varianza es cero."""
    d = independence_diagnostics(nulo)
    assert d["delta_degenerado"] and d["kappa_degenerado"]
    assert d["corr_delta_kappa"] is None
    assert d["corr_delta_turnover"] is None
    assert d["corr_kappa_turnover"] is None


# ---------------------------------------------------------------------------
# (D2) Causalidad reversa
# ---------------------------------------------------------------------------
def test_regresion_hacia_atras_es_positiva_con_disposicion():
    """Con delta > 0 la suerte causa turnover, no al reves."""
    res = _run("diag_disp", delta_target=0.8, delta_dist="beta")
    out = backward_regression(res.accounts, "gross_return")
    assert out["beta"] > 0.0
    assert out["t"] > 2.0, f"t = {out['t']:.2f}"


def test_placebo_de_permutacion_colapsa_la_pendiente(het):
    out = permutation_placebo(het.accounts, np.random.default_rng(0), "net_return", n_placebo=100)
    assert abs(out["placebo_media"]) < 3.0 * out["placebo_sd"] / np.sqrt(out["n_placebo"]) + 1e-4
    assert out["placebo_ci_lo"] < 0.0 < out["placebo_ci_hi"], "la banda placebo debe contener el cero"


def test_ols_vs_iv_reporta_la_brecha(het):
    out = ols_vs_iv(het.accounts)
    assert out["identificado_net"] is True
    assert out["first_stage_F"] > 10.0
    assert np.isfinite(out["brecha_net"])
    assert out["brecha_net"] > 0.0, "el OLS debe estar sesgado al alza por causalidad reversa"


def test_cash_drag_se_aisla(het):
    out = cash_drag_isolation(het.accounts)
    assert np.isfinite(out["desplazamiento"])
    assert 0.0 <= out["cash_weight_medio"] < 0.5


# ---------------------------------------------------------------------------
# (D3) Descomposicion del spread
# ---------------------------------------------------------------------------
def test_identidad_de_costos_es_exacta(het):
    out = spread_decomposition(het.accounts, het.config.costs, het.config.market.n_days)
    assert out["identidad_max_error"] < 1e-9


def test_la_pendiente_del_spread_coincide_con_la_teorica(het):
    out = spread_decomposition(het.accounts, het.config.costs, het.config.market.n_days)
    assert out["error_relativo_spread"] < 0.05, out
    assert out["pendiente_total"] > 0.0


# ---------------------------------------------------------------------------
# (D4) Celdas discriminantes
# ---------------------------------------------------------------------------
def test_las_celdas_tienen_oportunidades_y_tasas_validas(het):
    cells = discriminating_cells(het, 200, np.random.default_rng(0))
    for cell in CELL_NAMES:
        est = cells[cell]
        assert est["opportunities"] > 1000, cell
        assert 0.0 <= est["estimate"] <= 1.0
        assert est["ci_lo"] <= est["estimate"] <= est["ci_hi"]


def test_la_firma_de_la_disposicion_es_la_predicha():
    """A baja, B alta, C baja: la firma registrada del mecanismo conductual."""
    res = _run("firma_disp", delta_target=0.8, delta_dist="beta")
    cells = discriminating_cells(res, 200, np.random.default_rng(0))
    mat = signature_matrix({"disposicion_alta": cells})
    assert match_a_priori(mat.loc["disposicion_alta"]) == "disposicion", pattern_str(
        mat.loc["disposicion_alta"]
    )


def test_los_tres_mecanismos_tienen_firmas_separables():
    """Indistinguibles por PGR-PLR, separables por el vector de tres celdas."""
    mat = signature_matrix({
        "disposicion": discriminating_cells(
            _run("firma_dis2", delta_target=0.8, delta_dist="beta"), 200, np.random.default_rng(1)),
        "rebalanceo": discriminating_cells(
            _run("firma_reb", confound="rebalancing"), 200, np.random.default_rng(1)),
        "reversion": discriminating_cells(
            _run("firma_rev", confound="mean_reversion"), 200, np.random.default_rng(1)),
    })
    # La celda C separa el rebalanceo: recorta perdedoras sobreponderadas.
    assert mat.loc["rebalanceo", CELL_NAMES[2]] > mat.loc["disposicion", CELL_NAMES[2]] + 1.0
    # La celda A separa la reversion de la disposicion: signos opuestos.
    assert mat.loc["reversion", CELL_NAMES[0]] > mat.loc["disposicion", CELL_NAMES[0]] + 1.0
    d = signature_distances(mat)
    fuera = d.to_numpy()[~np.eye(3, dtype=bool)]
    assert fuera.min() > 1.0, "las tres firmas deben estar bien separadas"


def test_nearest_reference_identifica_la_firma_correcta():
    ref = pd.DataFrame(
        {CELL_NAMES[0]: [-1.0, 0.0, 1.5], CELL_NAMES[1]: [1.0, -1.5, 0.8], CELL_NAMES[2]: [-1.0, 2.0, -0.5]},
        index=["disposicion", "rebalanceo", "reversion"],
    )
    fila = pd.Series({CELL_NAMES[0]: -0.9, CELL_NAMES[1]: 1.1, CELL_NAMES[2]: -1.1})
    assert nearest_reference(fila, ref)["mas_cercano"] == "disposicion"


def test_match_a_priori_reconoce_los_patrones_registrados():
    for mecanismo, pred in CELL_PREDICTIONS.items():
        valores = {"alta": 0.8, "baja": -0.8, "base": 0.0}
        fila = pd.Series({c: valores[pred[c]] for c in CELL_NAMES})
        assert match_a_priori(fila) == mecanismo


def test_match_a_priori_devuelve_none_si_no_hay_patron_registrado():
    fila = pd.Series({CELL_NAMES[0]: 0.9, CELL_NAMES[1]: 0.9, CELL_NAMES[2]: 0.9})
    assert match_a_priori(fila) is None
