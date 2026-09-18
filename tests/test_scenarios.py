"""Pruebas de los escenarios, barridos, replicas y exportacion."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import EstimationConfig, MarketConfig, PopulationConfig, SimConfig
from src.scenarios import (
    SCENARIOS,
    SCENARIO_BY_KEY,
    check_monotonicity,
    main_table_row,
    make_config,
    run_accounting_grid,
    run_null_replicates,
    run_scenario,
    run_sweep,
    save_json,
    to_markdown_table,
)

CFG = SimConfig(
    market=MarketConfig(n_assets=30, n_days=150),
    population=PopulationConfig(n_agents=300),
)
EST = EstimationConfig(n_bootstrap=300, n_placebo=30)


@pytest.fixture(scope="module")
def analisis():
    """Corre los escenarios 1, 2, 3, 7, 8 y 9 en configuracion reducida."""
    claves = ["esc1_nulo", "esc2_disposicion_baja", "esc3_disposicion_alta",
              "esc7_confound_rebalanceo", "esc8_confound_reversion", "esc9_heterogeneo"]
    out = {}
    for k in claves:
        an, _ = run_scenario(SCENARIO_BY_KEY[k], CFG, EST, full_diagnostics=True)
        out[k] = an
    return out


# ---------------------------------------------------------------------------
# Estructura
# ---------------------------------------------------------------------------
def test_hay_nueve_escenarios_y_el_heterogeneo_esta_incluido():
    assert len(SCENARIOS) == 9
    assert SCENARIOS[-1].key == "esc9_heterogeneo"
    assert SCENARIOS[-1].delta_dist == "uniform" and SCENARIOS[-1].kappa_dist == "uniform"


def test_la_tabla_principal_tiene_nueve_filas(analisis):
    filas = [main_table_row(an) for an in analisis.values()]
    tabla = pd.DataFrame(filas)
    assert set(["PGR", "PLR", "PGR_menos_PLR", "delta_hat", "beta_gross", "beta_net",
                "corr_delta_kappa", "corr_delta_turnover", "corr_kappa_turnover"]).issubset(tabla.columns)
    assert len(tabla) == len(analisis)


def test_make_config_propaga_los_parametros_del_escenario():
    cfg = make_config(SCENARIO_BY_KEY["esc3_disposicion_alta"], CFG)
    assert cfg.population.delta_target == 0.8 and cfg.population.delta_dist == "beta"
    assert cfg.population.kappa_target == 0.0 and cfg.population.confound is None
    assert cfg.market.n_assets == CFG.market.n_assets, "la base comun no se toca"


# ---------------------------------------------------------------------------
# Recuperacion y ordenamiento
# ---------------------------------------------------------------------------
def test_el_escenario_nulo_no_rechaza_al_cinco_por_ciento(analisis):
    od = analisis["esc1_nulo"]["odean"]["PGR_menos_PLR"]
    assert od["ci_lo"] <= 0.0 <= od["ci_hi"], f"IC = [{od['ci_lo']:.5f}, {od['ci_hi']:.5f}]"
    assert od["p"] > 0.05, f"p = {od['p']:.4f}"


def test_delta_recuperado_ordena_los_escenarios(analisis):
    d1 = analisis["esc1_nulo"]["hazard"]["delta_hat"]["estimate"]
    d2 = analisis["esc2_disposicion_baja"]["hazard"]["delta_hat"]["estimate"]
    d3 = analisis["esc3_disposicion_alta"]["hazard"]["delta_hat"]["estimate"]
    assert d3 > d2 > d1


def test_delta_recuperado_esta_cerca_del_inyectado(analisis):
    for clave in ("esc1_nulo", "esc2_disposicion_baja", "esc3_disposicion_alta"):
        an = analisis[clave]
        assert abs(an["error_recuperacion"]) < 0.06, (clave, an["error_recuperacion"])


def test_los_confounds_disparan_un_falso_positivo_de_disposicion(analisis):
    """delta inyectado = 0 y aun asi PGR - PLR sale significativamente positivo."""
    for clave in ("esc7_confound_rebalanceo", "esc8_confound_reversion"):
        an = analisis[clave]
        assert an["delta_inyectado_media"] == 0.0
        od = an["odean"]["PGR_menos_PLR"]
        assert od["estimate"] > 0.0, clave
        assert od["ci_lo"] > 0.0, f"{clave}: IC = [{od['ci_lo']:.5f}, {od['ci_hi']:.5f}]"


def test_el_heterogeneo_reporta_las_tres_correlaciones(analisis):
    ind = analisis["esc9_heterogeneo"]["independence"]
    assert ind["corr_delta_kappa"] is not None and abs(ind["corr_delta_kappa"]) < 0.10
    assert ind["corr_delta_turnover"] is not None and ind["corr_delta_turnover"] < 0.0
    assert ind["corr_kappa_turnover"] is not None and ind["corr_kappa_turnover"] > 0.5


def test_los_escenarios_degenerados_reportan_nd(analisis):
    ind = analisis["esc3_disposicion_alta"]["independence"]
    assert ind["kappa_degenerado"] is True
    assert ind["corr_delta_kappa"] is None
    assert analisis["esc3_disposicion_alta"]["iv_net"]["identificado"] is False


# ---------------------------------------------------------------------------
# Determinismo
# ---------------------------------------------------------------------------
def test_dos_corridas_con_la_misma_semilla_son_identicas():
    spec = SCENARIO_BY_KEY["esc2_disposicion_baja"]
    a, _ = run_scenario(spec, CFG, EST, full_diagnostics=False)
    b, _ = run_scenario(spec, CFG, EST, full_diagnostics=False)
    assert a["odean"]["PGR_menos_PLR"] == b["odean"]["PGR_menos_PLR"]
    assert a["hazard"]["delta_hat"] == b["hazard"]["delta_hat"]
    assert a["reg_net"]["beta"] == b["reg_net"]["beta"]


def test_semillas_maestras_distintas_dan_resultados_distintos():
    spec = SCENARIO_BY_KEY["esc2_disposicion_baja"]
    a, _ = run_scenario(spec, CFG, EST, master_seed=42, full_diagnostics=False)
    b, _ = run_scenario(spec, CFG, EST, master_seed=43, full_diagnostics=False)
    assert a["odean"]["PGR"]["estimate"] != b["odean"]["PGR"]["estimate"]


# ---------------------------------------------------------------------------
# Barrido y replicas
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def barrido():
    cfg = SimConfig(market=MarketConfig(n_assets=30, n_days=150),
                    population=PopulationConfig(n_agents=250))
    return run_sweep(cfg, EstimationConfig(n_bootstrap=200))


def test_el_barrido_tiene_diez_puntos(barrido):
    assert len(barrido) == 10
    assert set(barrido["parametro"]) == {"delta", "kappa"}


def test_el_barrido_en_delta_es_monotono(barrido):
    for columna in ("PGR_menos_PLR", "delta_hat"):
        chk = check_monotonicity(barrido, "delta", columna)
        assert chk["monotona_creciente"], (columna, chk["valores"])


def test_el_turnover_crece_con_kappa_en_el_barrido(barrido):
    chk = check_monotonicity(barrido, "kappa", "turnover_medio")
    assert chk["monotona_creciente"], chk["valores"]


def test_el_barrido_en_kappa_deja_callado_al_estimador_de_disposicion(barrido):
    sub = barrido[barrido["parametro"] == "kappa"]
    assert sub["delta_hat"].abs().max() < 0.10, sub[["valor_inyectado", "delta_hat"]]


def test_check_monotonicity_detecta_la_ruptura():
    df = pd.DataFrame({"parametro": ["x"] * 4, "valor_inyectado": [0, 1, 2, 3], "y": [0.0, 1.0, 0.5, 2.0]})
    chk = check_monotonicity(df, "x", "y")
    assert not chk["monotona_creciente"] and not chk["monotona_decreciente"]
    assert len(chk["rupturas"]) >= 1


def test_replicas_del_nulo_devuelven_las_columnas_necesarias():
    cfg = SimConfig(market=MarketConfig(n_assets=30, n_days=120),
                    population=PopulationConfig(n_agents=200))
    df = run_null_replicates(cfg, EstimationConfig(n_bootstrap=200), n_reps=4)
    assert len(df) == 4
    assert {"PGR_menos_PLR", "p", "rechaza_5pct", "delta_hat", "beta_net"}.issubset(df.columns)
    assert df["PGR_menos_PLR"].std() > 0, "las replicas deben usar semillas distintas"


# ---------------------------------------------------------------------------
# Sensibilidad contable
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rejilla():
    cfg = SimConfig(market=MarketConfig(n_assets=30, n_days=150),
                    population=PopulationConfig(n_agents=250))
    return run_accounting_grid(cfg, EstimationConfig(n_bootstrap=200),
                               SCENARIO_BY_KEY["esc3_disposicion_alta"])


def test_la_rejilla_contable_tiene_doce_combinaciones(rejilla):
    assert len(rejilla) == 12
    assert set(rejilla["base_costo"]) == {"fifo", "average_cost", "per_lot"}
    assert set(rejilla["referencia"]) == {"mid", "ask_paid"}
    assert set(rejilla["conteo_parcial"]) == {"partial_as_full", "partial_as_fraction"}


def test_pgr_menos_plr_depende_de_las_convenciones(rejilla):
    rango = rejilla["PGR_menos_PLR"].max() - rejilla["PGR_menos_PLR"].min()
    assert rango > 0.01, "la medida estandar deberia moverse con las convenciones"


def test_delta_hat_es_mucho_mas_estable_que_pgr_menos_plr(rejilla):
    """El estimador estructural es casi invariante a las convenciones contables."""
    cv_pgr = rejilla["PGR_menos_PLR"].std() / abs(rejilla["PGR_menos_PLR"].mean())
    cv_delta = rejilla["delta_hat"].std() / abs(rejilla["delta_hat"].mean())
    assert cv_delta < cv_pgr / 3.0, (cv_delta, cv_pgr)


# ---------------------------------------------------------------------------
# Exportacion
# ---------------------------------------------------------------------------
def test_markdown_maneja_nan_y_none():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [None, "x"], "c": [True, False]})
    txt = to_markdown_table(df)
    assert txt.count("\n") == 3
    assert "n/d" in txt and "si" in txt and "no" in txt


def test_save_json_convierte_nan_en_null(tmp_path):
    ruta = tmp_path / "x.json"
    save_json({"a": float("nan"), "b": np.float64(1.5), "c": np.array([1, 2]),
               "d": {"e": None}, "f": np.bool_(True)}, ruta)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert datos["a"] is None and datos["b"] == 1.5
    assert datos["c"] == [1, 2] and datos["f"] is True
