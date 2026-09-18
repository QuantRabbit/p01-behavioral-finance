"""
Figuras del reporte. PNG a 150 dpi, ejes etiquetados en espaniol.

Todas las funciones son puras respecto del estado global de matplotlib: fijan
su propio estilo, escriben el archivo y cierran la figura.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import derive_rngs
from .estimators import (
    barber_odean_regression,
    iv_2sls,
    odean_bootstrap_draws,
)
from .price_process import generate_market
from .simulator import CELL_NAMES

DPI = 150
COLOR_A = "#1f4e79"
COLOR_B = "#c0392b"
COLOR_C = "#2e8b57"
COLOR_GRIS = "#7f8c8d"


def _style(ax, titulo: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(titulo, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _save(fig, outdir: Path, nombre: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ruta = outdir / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# 1. Trayectorias de precios
# ---------------------------------------------------------------------------
def fig_precios(cfg, semilla: int, outdir: Path) -> Path:
    market = generate_market(cfg.market, derive_rngs("esc1_nulo", semilla)["prices"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    dias = np.arange(market.n_days + 1)
    rng = np.random.default_rng(0)
    muestra = rng.choice(market.n_assets, size=min(10, market.n_assets), replace=False)
    for m in muestra:
        ax1.plot(dias, market.prices[:, m] / market.prices[0, m] * 100.0, linewidth=0.9, alpha=0.85)
    _style(ax1, "Trayectorias de 10 activos (base 100)", "dia habil", "precio normalizado")

    ax2.plot(dias, market.equal_weight_index(), color=COLOR_A, linewidth=1.6)
    ax2.axhline(100.0, color=COLOR_GRIS, linestyle="--", linewidth=0.8)
    _style(ax2, "Indice equiponderado del mercado", "dia habil", "nivel (base 100)")
    fig.suptitle("Mercado multifactorial: retornos i.i.d. en el tiempo, correlacion transversal positiva",
                 fontsize=10.5)
    return _save(fig, outdir, "fig01_trayectorias_precios.png")


# ---------------------------------------------------------------------------
# 2. Distribucion conjunta de delta y kappa
# ---------------------------------------------------------------------------
def fig_delta_kappa(cuentas: pd.DataFrame, outdir: Path) -> Path:
    d = cuentas["delta"].to_numpy(float)
    k = cuentas["kappa"].to_numpy(float)
    corr = np.corrcoef(d, k)[0, 1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(d, bins=30, color=COLOR_A, alpha=0.85)
    _style(axes[0], f"Efecto disposicion inyectado\nmedia={d.mean():.3f}, sd={d.std(ddof=1):.3f}",
           r"$\delta_i$", "cuentas")
    axes[1].hist(k, bins=30, color=COLOR_B, alpha=0.85)
    _style(axes[1], f"Sobreprecision inyectada\nmedia={k.mean():.3f}, sd={k.std(ddof=1):.3f}",
           r"$\kappa_i$", "cuentas")
    axes[2].scatter(d, k, s=6, alpha=0.35, color=COLOR_C, edgecolors="none")
    _style(axes[2], f"Independencia por construccion\nCorr = {corr:+.4f}", r"$\delta_i$", r"$\kappa_i$")
    fig.suptitle("Los parametros inyectados son distribuciones, no escalares", fontsize=10.5)
    return _save(fig, outdir, "fig02_distribucion_delta_kappa.png")


# ---------------------------------------------------------------------------
# 3. PGR y PLR por escenario
# ---------------------------------------------------------------------------
def fig_pgr_plr(main_table: pd.DataFrame, outdir: Path) -> Path:
    t = main_table
    x = np.arange(len(t))
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(x - ancho / 2, t["PGR"], ancho, yerr=1.96 * t["SE_PGR"], capsize=3,
           color=COLOR_A, label="PGR (ganancias realizadas)")
    ax.bar(x + ancho / 2, t["PLR"], ancho, yerr=1.96 * t["SE_PLR"], capsize=3,
           color=COLOR_B, label="PLR (perdidas realizadas)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}. {e}" for n, e in zip(t["#"], t["escenario"])],
                       rotation=30, ha="right", fontsize=8)
    _style(ax, "Estimador de Odean (1998) por escenario, con barras de error bootstrap al 95%",
           "", "proporcion realizada")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, outdir, "fig03_pgr_plr_por_escenario.png")


# ---------------------------------------------------------------------------
# 4. delta recuperado vs delta inyectado  (la figura central)
# ---------------------------------------------------------------------------
def fig_recuperacion_delta(main_table: pd.DataFrame, sweep: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = [-0.05, 1.0]
    ax.plot(lim, lim, color=COLOR_GRIS, linestyle="--", linewidth=1.0, label="diagonal de 45 grados")

    if sweep is not None and len(sweep):
        s = sweep[sweep["parametro"] == "delta"]
        ax.errorbar(s["media_realizada"], s["delta_hat"], yerr=1.96 * s["SE_delta_hat"],
                    fmt="o", color=COLOR_C, markersize=5, capsize=3, linewidth=1.0,
                    label="barrido de monotonicidad")

    sin_confound = main_table[main_table["#"].isin([1, 2, 3, 4, 5, 6, 9])]
    ax.errorbar(sin_confound["delta_iny_media"], sin_confound["delta_hat"],
                yerr=1.96 * sin_confound["SE_delta_hat"], fmt="s", color=COLOR_A,
                markersize=7, capsize=3, linewidth=1.2, label="escenarios canonicos")
    for _, r in sin_confound.iterrows():
        ax.annotate(str(int(r["#"])), (r["delta_iny_media"], r["delta_hat"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=8, color=COLOR_A)

    confound = main_table[main_table["#"].isin([7, 8])]
    ax.scatter(confound["delta_iny_media"], confound["delta_hat"], marker="^", s=70,
               color=COLOR_B, label="confounds (delta inyectado = 0)")
    for _, r in confound.iterrows():
        ax.annotate(str(int(r["#"])), (r["delta_iny_media"], r["delta_hat"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=8, color=COLOR_B)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _style(ax, "Recuperacion del parametro: el estimador estructural si lo recupera",
           r"$\delta$ inyectado (media realizada)", r"$\hat{\delta}$ recuperado por razon de hazards")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return _save(fig, outdir, "fig04_delta_recuperado_vs_inyectado.png")


# ---------------------------------------------------------------------------
# 5. PGR - PLR vs delta inyectado (no linealidad)
# ---------------------------------------------------------------------------
def fig_pgr_menos_plr_vs_delta(sweep: pd.DataFrame, outdir: Path) -> Path:
    s = sweep[sweep["parametro"] == "delta"].sort_values("valor_inyectado")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.errorbar(s["media_realizada"], s["PGR_menos_PLR"], yerr=1.96 * s["SE_PGR_menos_PLR"],
                 fmt="o-", color=COLOR_B, capsize=3, linewidth=1.2)
    if len(s) > 1:
        x0, x1 = s["media_realizada"].iloc[0], s["media_realizada"].iloc[-1]
        y0, y1 = s["PGR_menos_PLR"].iloc[0], s["PGR_menos_PLR"].iloc[-1]
        ax1.plot([x0, x1], [y0, y1], color=COLOR_GRIS, linestyle=":", linewidth=1.0,
                 label="recta entre los extremos")
        ax1.legend(fontsize=8, frameon=False)
    _style(ax1, "PGR - PLR es monotona pero NO lineal en el parametro",
           r"$\delta$ inyectado", "PGR - PLR")

    ax2.errorbar(s["media_realizada"], s["delta_hat"], yerr=1.96 * s["SE_delta_hat"],
                 fmt="o-", color=COLOR_A, capsize=3, linewidth=1.2, label=r"$\hat{\delta}$")
    ax2.plot([0, 1], [0, 1], color=COLOR_GRIS, linestyle="--", linewidth=1.0, label="45 grados")
    _style(ax2, "El estimador estructural sigue la diagonal", r"$\delta$ inyectado", r"$\hat{\delta}$")
    ax2.legend(fontsize=8, frameon=False)
    return _save(fig, outdir, "fig05_pgr_menos_plr_vs_delta.png")


# ---------------------------------------------------------------------------
# 6. Turnover vs retorno neto, con OLS e IV
# ---------------------------------------------------------------------------
def fig_turnover_retorno(cuentas: pd.DataFrame, outdir: Path) -> Path:
    x = cuentas["turnover"].to_numpy(float)
    y = cuentas["net_return"].to_numpy(float)
    ols = barber_odean_regression(cuentas, "net_return")
    iv = iv_2sls(cuentas, "net_return")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(x, y, s=7, alpha=0.30, color=COLOR_GRIS, edgecolors="none", label="cuentas")
    xx = np.linspace(x.min(), x.max(), 50)
    centro_y, centro_x = y.mean(), x.mean()
    ax.plot(xx, centro_y + ols["beta_turnover"] * (xx - centro_x), color=COLOR_B, linewidth=2.0,
            label=f"OLS: beta = {ols['beta_turnover']:+.5f} (t={ols['t_turnover']:.2f})")
    if iv["identified"]:
        ax.plot(xx, centro_y + iv["beta_turnover"] * (xx - centro_x), color=COLOR_A, linewidth=2.0,
                linestyle="--",
                label=f"IV (kappa): beta = {iv['beta_turnover']:+.5f} (t={iv['t_turnover']:.2f})")
    _style(ax, "Barber-Odean: la brecha OLS-IV es la magnitud de la causalidad reversa",
           "turnover anualizado", "retorno neto a dos anios")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, outdir, "fig06_turnover_vs_retorno.png")


# ---------------------------------------------------------------------------
# 7. Placebo de permutacion
# ---------------------------------------------------------------------------
def fig_placebo(cuentas: pd.DataFrame, semilla: int, n_placebo: int, outdir: Path) -> Path:
    rng = derive_rngs("placebo_figura", semilla)["placebo"]
    observado = barber_odean_regression(cuentas, "net_return")["beta_turnover"]
    df = cuentas.copy()
    turn = cuentas["turnover"].to_numpy(float)
    betas = np.empty(n_placebo)
    for b in range(n_placebo):
        df["turnover"] = rng.permutation(turn)
        betas[b] = barber_odean_regression(df, "net_return")["beta_turnover"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(betas, bins=30, color=COLOR_GRIS, alpha=0.8, label="permutaciones del turnover")
    ax.axvline(observado, color=COLOR_B, linewidth=2.2,
               label=f"beta observado = {observado:+.5f}")
    ax.axvline(0.0, color=COLOR_A, linestyle="--", linewidth=1.0, label="cero")
    _style(ax, f"Placebo de permutacion ({n_placebo} repeticiones): la pendiente colapsa al azar",
           "beta del turnover", "frecuencia")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, outdir, "fig07_placebo_permutacion.png")


# ---------------------------------------------------------------------------
# 8. Mapa de calor de la matriz de firmas
# ---------------------------------------------------------------------------
def fig_firmas(firmas: pd.DataFrame, outdir: Path) -> Path:
    datos = firmas[list(CELL_NAMES)].to_numpy(float)
    etiquetas = ["A: perdida con\nrebote reciente", "B: ganancia\ninfraponderada",
                 "C: perdida\nsobreponderada"]
    # La firma del rebalanceo es un orden de magnitud mas extrema que las demas
    # (nunca vende ganadoras infraponderadas y SIEMPRE recorta perdedoras
    # sobreponderadas). Si la escala de color se ajusta a ese maximo, el resto
    # de la matriz queda plano. Se satura la escala y el numero exacto queda
    # impreso en cada celda.
    vmax = max(1.0, float(np.nanpercentile(np.abs(datos), 85)))
    saturadas = int(np.nansum(np.abs(datos) > vmax))
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(np.clip(datos, -vmax, vmax), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, fontsize=8)
    ax.set_yticks(range(len(firmas)))
    ax.set_yticklabels(firmas.index, fontsize=8)
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            v = datos[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * vmax else "black")
    ax.set_title(
        "Matriz de firmas: log-razon de la tasa de venta condicional contra la tasa base\n"
        f"(escala de color saturada en +/-{vmax:.1f}; {saturadas} celdas exceden esa magnitud)",
        fontsize=10.5, pad=12,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(tasa de la celda / tasa base)")
    return _save(fig, outdir, "fig08_matriz_firmas.png")


# ---------------------------------------------------------------------------
# 9. Distribucion bootstrap del escenario nulo
# ---------------------------------------------------------------------------
def fig_bootstrap_nulo(cuentas: pd.DataFrame, nulos: pd.DataFrame, semilla: int,
                       n_boot: int, outdir: Path) -> Path:
    rng = derive_rngs("bootstrap_figura", semilla)["bootstrap"]
    draws = odean_bootstrap_draws(cuentas, n_boot, rng)["PGR_menos_PLR"]
    punto = float(
        cuentas["G_r"].sum() / (cuentas["G_r"].sum() + cuentas["G_p"].sum())
        - cuentas["L_r"].sum() / (cuentas["L_r"].sum() + cuentas["L_p"].sum())
    )
    lo, hi = np.quantile(draws, [0.025, 0.975])

    n_paneles = 2 if (nulos is not None and len(nulos)) else 1
    fig, axes = plt.subplots(1, n_paneles, figsize=(5.8 * n_paneles, 4.2))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    ax.hist(draws, bins=40, color=COLOR_GRIS, alpha=0.85)
    ax.axvline(0.0, color=COLOR_A, linewidth=1.8, label="cero")
    ax.axvline(punto, color=COLOR_B, linewidth=2.0, label=f"estimacion = {punto:+.5f}")
    ax.axvline(lo, color=COLOR_C, linestyle="--", linewidth=1.0)
    ax.axvline(hi, color=COLOR_C, linestyle="--", linewidth=1.0, label="IC bootstrap 95%")
    _style(ax, "Escenario nulo: el cero cae dentro del intervalo", "PGR - PLR remuestreado", "frecuencia")
    ax.legend(fontsize=8, frameon=False)

    if n_paneles == 2:
        ax2 = axes[1]
        ax2.hist(nulos["PGR_menos_PLR"], bins=12, color=COLOR_A, alpha=0.8)
        ax2.axvline(0.0, color=COLOR_B, linewidth=1.8, label="cero")
        tasa = float(nulos["rechaza_5pct"].mean())
        _style(ax2, f"{len(nulos)} replicas del nulo con semillas distintas\n"
                    f"tasa de rechazo al 5% = {tasa:.0%}", "PGR - PLR", "replicas")
        ax2.legend(fontsize=8, frameon=False)
    return _save(fig, outdir, "fig09_bootstrap_nulo.png")


# ---------------------------------------------------------------------------
# 10. Sensibilidad contable (figura extra)
# ---------------------------------------------------------------------------
def fig_sensibilidad_contable(grid: pd.DataFrame, delta_iny: float, outdir: Path) -> Path:
    g = grid.copy()
    g["etiqueta"] = (g["base_costo"] + "\n" + g["conteo_parcial"].str.replace("partial_as_", "")
                     + " / " + g["referencia"])
    orden = g.sort_values(["referencia", "base_costo", "conteo_parcial"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(orden))
    ax1.bar(x, orden["PGR_menos_PLR"], color=COLOR_B, alpha=0.85)
    _style(ax1, "PGR - PLR depende de convenciones contables que casi ningun paper declara",
           "", "PGR - PLR")
    ax2.bar(x, orden["delta_hat"], color=COLOR_A, alpha=0.85)
    ax2.axhline(delta_iny, color=COLOR_C, linestyle="--", linewidth=1.4,
                label=f"delta inyectado = {delta_iny:.3f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(orden["etiqueta"], rotation=45, ha="right", fontsize=7)
    _style(ax2, "El estimador estructural es practicamente invariante", "", r"$\hat{\delta}$")
    ax2.legend(fontsize=8, frameon=False)
    return _save(fig, outdir, "fig10_sensibilidad_contable.png")


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def generate_all(
    cfg,
    args,
    analyses: Optional[Dict],
    resultados: Optional[Dict],
    main_table: pd.DataFrame,
    sweep: pd.DataFrame,
    nulos: pd.DataFrame,
    firmas: pd.DataFrame,
    outdir: Path,
    cuentas: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Path]:
    """Genera todas las figuras y devuelve un diccionario nombre -> ruta."""
    if cuentas is None:
        cuentas = {k: r.accounts for k, r in (resultados or {}).items()}

    rutas: Dict[str, Path] = {}
    rutas["precios"] = fig_precios(cfg, args.seed, outdir)
    if "esc9_heterogeneo" in cuentas:
        het = cuentas["esc9_heterogeneo"]
        rutas["delta_kappa"] = fig_delta_kappa(het, outdir)
        rutas["turnover_retorno"] = fig_turnover_retorno(het, outdir)
        rutas["placebo"] = fig_placebo(het, args.seed, args.placebo, outdir)
    rutas["pgr_plr"] = fig_pgr_plr(main_table, outdir)
    rutas["recuperacion"] = fig_recuperacion_delta(main_table, sweep, outdir)
    if sweep is not None and len(sweep):
        rutas["no_linealidad"] = fig_pgr_menos_plr_vs_delta(sweep, outdir)
    rutas["firmas"] = fig_firmas(firmas, outdir)
    if "esc1_nulo" in cuentas:
        rutas["bootstrap_nulo"] = fig_bootstrap_nulo(
            cuentas["esc1_nulo"], nulos, args.seed, args.bootstrap, outdir
        )
    grid_path = Path(args.outdir) / "sensibilidad_contable.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        rutas["sensibilidad"] = fig_sensibilidad_contable(
            grid, float(grid["delta_inyectado"].iloc[0]), outdir
        )
    return rutas
