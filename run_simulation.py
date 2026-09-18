#!/usr/bin/env python
"""
CLI principal del proyecto P01.

Un solo comando reproduce todo desde cero:

    python run_simulation.py --agents 1000 --days 504 --assets 60 \
        --bootstrap 1000 --seed 42 --all

Genera ``outputs/`` completo: tablas principales y de diagnosticos en markdown
y CSV, el JSON de resultados, el barrido de monotonicidad, las replicas del
escenario nulo, la rejilla de sensibilidad contable, la matriz de firmas y las
figuras.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import __version__
from src.config import (
    AccountingConfig,
    CostConfig,
    EstimationConfig,
    MASTER_SEED,
    MarketConfig,
    PopulationConfig,
    SimConfig,
    config_to_dict,
    derive_rngs,
)
from src.diagnostics import (
    classify_by_signature,
    match_a_priori,
    nearest_reference,
    pattern_str,
    signature_distances,
    signature_matrix,
)
from src.scenarios import (
    SCENARIOS,
    SCENARIO_BY_KEY,
    bootstrap_comparison,
    check_monotonicity,
    diagnostics_table_row,
    main_table_row,
    make_config,
    run_accounting_grid,
    replicate_summary,
    run_null_replicates,
    run_replicates,
    run_scenario,
    run_sweep,
    save_json,
    save_table,
    to_markdown_table,
)
from src.simulator import CELL_NAMES

#: Escenarios cuyas cuentas se exportan (los necesitan las figuras y el reporte).
ESCENARIOS_CON_CUENTAS = (
    "esc1_nulo", "esc3_disposicion_alta", "esc7_confound_rebalanceo",
    "esc8_confound_reversion", "esc9_heterogeneo",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulador de finanzas conductuales - Proyecto P01")
    p.add_argument("--agents", type=int, default=1000, help="numero de agentes")
    p.add_argument("--days", type=int, default=504, help="dias habiles simulados")
    p.add_argument("--assets", type=int, default=60, help="numero de activos")
    p.add_argument("--bootstrap", type=int, default=1000, help="replicas bootstrap por cuenta")
    p.add_argument("--placebo", type=int, default=200, help="permutaciones del placebo")
    p.add_argument("--seed", type=int, default=MASTER_SEED, help="semilla maestra")
    p.add_argument("--commission", type=float, default=1.00, help="comision fija por orden (USD)")
    p.add_argument("--spread-bps", type=float, default=10.0, help="spread bid-ask total en bps")
    p.add_argument("--cost-basis", default="average_cost",
                   choices=["average_cost", "fifo", "per_lot"], help="base de costo por defecto")
    p.add_argument("--partial-counting", default="partial_as_full",
                   choices=["partial_as_full", "partial_as_fraction"])
    p.add_argument("--reference", default="mid", choices=["mid", "ask_paid"])
    p.add_argument("--null-reps", type=int, default=20, help="replicas del escenario nulo")
    p.add_argument("--disp-reps", type=int, default=10,
                   help="replicas del escenario de disposicion alta (variabilidad entre trayectorias)")
    p.add_argument("--wide-spread-bps", type=float, default=200.0,
                   help="spread ancho para la prueba de frontera de la referencia de costo")
    p.add_argument("--outdir", default="outputs", help="directorio de salida")
    p.add_argument("--all", action="store_true", help="corre todo: escenarios, barridos, replicas y figuras")
    p.add_argument("--scenarios-only", action="store_true", help="solo los 9 escenarios")
    p.add_argument("--no-figures", action="store_true", help="no generar figuras")
    p.add_argument("--figures-only", action="store_true", help="regenerar figuras desde outputs/")
    p.add_argument("--quick", action="store_true",
                   help="corrida rapida de humo (pocos agentes, dias y replicas)")
    return p


def base_config(args) -> SimConfig:
    return SimConfig(
        market=MarketConfig(n_assets=args.assets, n_days=args.days),
        costs=CostConfig(commission=args.commission, spread_bps=args.spread_bps),
        population=PopulationConfig(n_agents=args.agents),
        accounting=AccountingConfig(
            cost_basis=args.cost_basis,
            partial_counting=args.partial_counting,
            reference=args.reference,
        ),
    )


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.agents, args.days, args.assets = 250, 120, 30
        args.bootstrap, args.placebo, args.null_reps, args.disp_reps = 200, 50, 4, 3
    outdir = Path(args.outdir)
    figdir = outdir / "figuras"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        return regenerate_figures(args, outdir, figdir)

    correr_todo = args.all or not args.scenarios_only
    cfg = base_config(args)
    est = EstimationConfig(n_bootstrap=args.bootstrap, n_placebo=args.placebo)
    t_inicio = time.perf_counter()

    # ------------------------------------------------------------------
    # 1) Los nueve escenarios
    # ------------------------------------------------------------------
    log(f"Escenarios: {args.agents} agentes, {args.days} dias, {args.assets} activos, "
        f"B={args.bootstrap}, semilla={args.seed}")
    analyses = {}
    resultados = {}
    for spec in SCENARIOS:
        t0 = time.perf_counter()
        keep = spec.key in ESCENARIOS_CON_CUENTAS
        an, res = run_scenario(spec, cfg, est, args.seed, full_diagnostics=True, keep_result=keep)
        analyses[spec.key] = an
        if res is not None:
            resultados[spec.key] = res
            res.accounts.to_csv(outdir / f"cuentas_{spec.key}.csv", index=False, encoding="utf-8")
        od = an["odean"]["PGR_menos_PLR"]
        log(f"  [{spec.number}] {spec.name:<24} PGR-PLR={od['estimate']:+.4f} "
            f"(z={od['z']:6.2f})  delta_hat={an['hazard']['delta_hat']['estimate']:+.4f} "
            f"(iny={an['delta_inyectado_media']:.3f})  b_net={an['reg_net']['beta']:+.5f} "
            f"[{time.perf_counter()-t0:.1f}s]")

    main_table = pd.DataFrame([main_table_row(analyses[s.key]) for s in SCENARIOS])
    diag_table = pd.DataFrame([diagnostics_table_row(analyses[s.key]) for s in SCENARIOS])
    save_table(main_table, outdir / "tabla_principal")
    save_table(diag_table, outdir / "tabla_diagnosticos")
    log(f"Tabla principal: {len(main_table)} filas -> outputs/tabla_principal.md")

    # ------------------------------------------------------------------
    # 2) Matriz de firmas de las celdas discriminantes
    # ------------------------------------------------------------------
    cells_by_scenario = {analyses[s.key]["name"]: analyses[s.key]["cells"] for s in SCENARIOS}
    firmas = signature_matrix(cells_by_scenario)
    save_table(firmas, outdir / "matriz_firmas", index=True)
    distancias = signature_distances(firmas)
    save_table(distancias, outdir / "distancias_firmas", index=True)

    referencia = firmas.loc[["Disposicion alta", "Confound rebalanceo", "Confound reversion"]]
    referencia.index = ["disposicion", "rebalanceo", "reversion"]
    clasificacion = []
    for nombre, fila in firmas.iterrows():
        vecino = nearest_reference(fila, referencia)
        coseno = classify_by_signature(fila, referencia)
        clasificacion.append({
            "escenario": nombre,
            "patron": pattern_str(fila),
            "coincide_pre_analisis": match_a_priori(fila) or "ninguno",
            "mecanismo_por_direccion": coseno["mecanismo"],
            "coseno": coseno["coseno"],
            "coseno_segundo": coseno["coseno_segundo"],
            "intensidad_norma": coseno["norma"],
            "firma_mas_cercana_euclidiana": vecino["mas_cercano"],
            "distancia": vecino["distancia"],
        })
    save_table(pd.DataFrame(clasificacion), outdir / "clasificacion_firmas")

    # ------------------------------------------------------------------
    # 3) Bootstrap agrupado vs por transaccion (escenario 3)
    # ------------------------------------------------------------------
    comp = bootstrap_comparison(
        resultados["esc3_disposicion_alta"].accounts, est, derive_rngs("comp_boot", args.seed)["bootstrap"]
    )
    log(f"Bootstrap: SE por cuenta {comp['se_agrupado_por_cuenta']:.5f} vs por transaccion "
        f"{comp['se_por_transaccion']:.5f} (factor {comp['factor_subestimacion']:.1f}x)")

    sweep = pd.DataFrame()
    nulos = pd.DataFrame()
    disp_reps = pd.DataFrame()
    grid = pd.DataFrame()
    grid_ancho = pd.DataFrame()
    efecto_ref = {}
    monot = []
    resumen_reps = {}

    if correr_todo:
        # --------------------------------------------------------------
        # 4) Barrido de monotonicidad
        # --------------------------------------------------------------
        log("Barrido de monotonicidad (5 puntos en delta, 5 en kappa)...")
        sweep = run_sweep(cfg, est, args.seed)
        save_table(sweep, outdir / "barrido_monotonicidad", floatfmt="{:.5f}")
        for parametro, columna in (("delta", "PGR_menos_PLR"), ("delta", "delta_hat"),
                                   ("kappa", "beta_net"), ("kappa", "PGR_menos_PLR"),
                                   ("kappa", "turnover_medio")):
            chk = check_monotonicity(sweep, parametro, columna)
            monot.append(chk)
            estado = "monotona" if (chk["monotona_creciente"] or chk["monotona_decreciente"]) else "NO monotona"
            log(f"  {parametro} -> {columna}: {estado}")

        # --------------------------------------------------------------
        # 5) Replicas del escenario nulo
        # --------------------------------------------------------------
        log(f"Replicas del escenario nulo ({args.null_reps} semillas)...")
        nulos = run_null_replicates(cfg, est, args.null_reps, args.seed)
        save_table(nulos, outdir / "replicas_nulo", floatfmt="{:.5f}")
        tasa = float(nulos["rechaza_5pct"].mean())
        log(f"  tasa de rechazo de PGR-PLR al 5%: {tasa:.2%} "
            f"(nominal 5%), delta_hat: {nulos['rechaza_delta_5pct'].mean():.2%}, "
            f"beta_net: {nulos['rechaza_beta_net_5pct'].mean():.2%}")

        # --------------------------------------------------------------
        # 5b) Replicas del escenario 3: variabilidad ENTRE trayectorias
        # --------------------------------------------------------------
        log(f"Replicas del escenario de disposicion alta ({args.disp_reps} semillas)...")
        disp_reps = run_replicates(SCENARIO_BY_KEY["esc3_disposicion_alta"], cfg, est,
                                   args.disp_reps, args.seed, "disp_rep")
        save_table(disp_reps, outdir / "replicas_disposicion_alta", floatfmt="{:.5f}")
        resumen_reps = {
            "nulo_PGR_menos_PLR": replicate_summary(nulos, "PGR_menos_PLR", "SE"),
            "nulo_delta_hat": replicate_summary(nulos, "delta_hat", "SE_delta_hat"),
            "disp_PGR_menos_PLR": replicate_summary(disp_reps, "PGR_menos_PLR", "SE"),
            "disp_delta_hat": replicate_summary(disp_reps, "delta_hat", "SE_delta_hat"),
        }
        for k, v in resumen_reps.items():
            log(f"  {k}: sd entre trayectorias {v['sd_entre_trayectorias']:.5f} vs "
                f"SE bootstrap {v['se_bootstrap_medio']:.5f} (factor {v['factor_sd_sobre_se']:.2f})")
        log(f"  error de recuperacion de delta en las replicas: "
            f"media {disp_reps['error_recuperacion'].mean():+.5f}, "
            f"max |.| {disp_reps['error_recuperacion'].abs().max():.5f}")

        # --------------------------------------------------------------
        # 6) Sensibilidad contable sobre el escenario 3
        # --------------------------------------------------------------
        log("Rejilla de sensibilidad contable (3 bases x 2 conteos x 2 referencias)...")
        grid = run_accounting_grid(cfg, est, SCENARIO_BY_KEY["esc3_disposicion_alta"], args.seed)
        save_table(grid, outdir / "sensibilidad_contable", floatfmt="{:.5f}")
        rango = grid["PGR_menos_PLR"].max() - grid["PGR_menos_PLR"].min()
        log(f"  PGR-PLR varia entre {grid['PGR_menos_PLR'].min():.4f} y "
            f"{grid['PGR_menos_PLR'].max():.4f} (rango {rango:.4f}); "
            f"delta_hat entre {grid['delta_hat'].min():.4f} y {grid['delta_hat'].max():.4f}")

        # --------------------------------------------------------------
        # 6b) Frontera: la misma rejilla con un spread ancho
        # --------------------------------------------------------------
        # Con 10 bps el sesgo de medio spread de la referencia 'ask_paid' es
        # despreciable frente a los movimientos diarios de precio. Se repite la
        # rejilla con un spread de mercado ilquido para ubicar la frontera a
        # partir de la cual la eleccion de referencia si importa.
        cfg_ancho = replace(cfg, costs=CostConfig(commission=args.commission,
                                                  spread_bps=args.wide_spread_bps))
        log(f"Rejilla contable con spread ancho ({args.wide_spread_bps:.0f} bps)...")
        grid_ancho = run_accounting_grid(cfg_ancho, est,
                                         SCENARIO_BY_KEY["esc3_disposicion_alta"], args.seed)
        save_table(grid_ancho, outdir / "sensibilidad_contable_spread_ancho", floatfmt="{:.5f}")
        base_mid = grid[(grid.referencia == "mid") & (grid.conteo_parcial == "partial_as_full")]
        base_ask = grid[(grid.referencia == "ask_paid") & (grid.conteo_parcial == "partial_as_full")]
        anc_mid = grid_ancho[(grid_ancho.referencia == "mid") & (grid_ancho.conteo_parcial == "partial_as_full")]
        anc_ask = grid_ancho[(grid_ancho.referencia == "ask_paid") & (grid_ancho.conteo_parcial == "partial_as_full")]
        efecto_ref = {
            "spread_bps_base": args.spread_bps,
            "spread_bps_ancho": args.wide_spread_bps,
            "delta_PGR_menos_PLR_base": float(base_ask["PGR_menos_PLR"].mean() - base_mid["PGR_menos_PLR"].mean()),
            "delta_PGR_menos_PLR_ancho": float(anc_ask["PGR_menos_PLR"].mean() - anc_mid["PGR_menos_PLR"].mean()),
            "delta_hat_base_mid": float(base_mid["delta_hat"].mean()),
            "delta_hat_ancho_mid": float(anc_mid["delta_hat"].mean()),
            "delta_hat_ancho_ask": float(anc_ask["delta_hat"].mean()),
        }
        log(f"  efecto de la referencia sobre PGR-PLR: {efecto_ref['delta_PGR_menos_PLR_base']:+.5f} "
            f"a {args.spread_bps:.0f} bps vs {efecto_ref['delta_PGR_menos_PLR_ancho']:+.5f} "
            f"a {args.wide_spread_bps:.0f} bps")

    # ------------------------------------------------------------------
    # 7) JSON completo
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_inicio
    for an in analyses.values():
        an.pop("placebo_betas", None)
    completo = {
        "meta": {
            "version": __version__,
            "semilla_maestra": args.seed,
            "argumentos": vars(args),
            "config": config_to_dict(cfg),
            "python": sys.version.split()[0],
            "plataforma": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tiempo_total_s": elapsed,
        },
        "escenarios": analyses,
        "bootstrap_comparacion": comp,
        "monotonicidad": monot,
        "matriz_firmas": firmas.reset_index().to_dict(orient="records"),
        "clasificacion_firmas": clasificacion,
        "barrido": sweep.to_dict(orient="records") if len(sweep) else [],
        "replicas_nulo": nulos.to_dict(orient="records") if len(nulos) else [],
        "replicas_disposicion_alta": disp_reps.to_dict(orient="records") if len(disp_reps) else [],
        "resumen_replicas": resumen_reps,
        "sensibilidad_contable": grid.to_dict(orient="records") if len(grid) else [],
        "sensibilidad_contable_spread_ancho": grid_ancho.to_dict(orient="records") if len(grid_ancho) else [],
        "efecto_referencia_por_spread": efecto_ref,
    }
    save_json(completo, outdir / "resultados_completos.json")
    log(f"JSON completo -> outputs/resultados_completos.json ({elapsed:.1f}s de computo)")

    # ------------------------------------------------------------------
    # 8) Figuras
    # ------------------------------------------------------------------
    if not args.no_figures:
        from src import plots
        log("Generando figuras...")
        plots.generate_all(
            cfg=cfg, args=args, analyses=analyses, resultados=resultados,
            main_table=main_table, sweep=sweep, nulos=nulos, firmas=firmas, outdir=figdir,
        )
        log(f"Figuras -> {figdir}")

    log(f"LISTO en {time.perf_counter() - t_inicio:.1f} s")
    return 0


def regenerate_figures(args, outdir: Path, figdir: Path) -> int:
    """Regenera las figuras a partir de los artefactos guardados en ``outputs/``."""
    from src import plots

    cfg = base_config(args)
    main_table = pd.read_csv(outdir / "tabla_principal.csv")
    sweep = pd.read_csv(outdir / "barrido_monotonicidad.csv")
    nulos = pd.read_csv(outdir / "replicas_nulo.csv")
    firmas = pd.read_csv(outdir / "matriz_firmas.csv").set_index("escenario")
    cuentas = {
        key: pd.read_csv(outdir / f"cuentas_{key}.csv")
        for key in ESCENARIOS_CON_CUENTAS
        if (outdir / f"cuentas_{key}.csv").exists()
    }
    plots.generate_all(
        cfg=cfg, args=args, analyses=None, resultados=None, main_table=main_table,
        sweep=sweep, nulos=nulos, firmas=firmas, outdir=figdir, cuentas=cuentas,
    )
    log(f"Figuras regeneradas -> {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
