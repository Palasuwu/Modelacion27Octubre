"""Figuras compartidas por la aplicación y por el informe."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

PUNTOS_DISPERSION = 3000   # submuestra para que los diagramas sigan siendo legibles
COLOR = "#2b5d8a"
COLOR_TEORICO = "#b5462f"


def _nombres(p):
    return [f"X{i + 1}" for i in range(p)]


def histogramas(res):
    x, mu, sd = res.x, res.parametros.medias, res.parametros.sigmas
    p = x.shape[1]
    columnas = min(p, 4)
    filas = int(np.ceil(p / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(3.2 * columnas, 2.6 * filas), squeeze=False)
    for k, eje in enumerate(ejes.ravel()):
        if k >= p:
            eje.axis("off")
            continue
        eje.hist(x[:, k], bins=60, density=True, color=COLOR, alpha=0.75)
        malla = np.linspace(mu[k] - 4 * sd[k], mu[k] + 4 * sd[k], 300)
        eje.plot(malla, norm.pdf(malla, mu[k], sd[k]), color=COLOR_TEORICO, lw=1.6)
        eje.set_title(f"X{k + 1}: N({mu[k]:g}, {sd[k]:g}²)", fontsize=10)
        eje.tick_params(labelsize=8)
    fig.legend(["Densidad teórica", "Muestra"], loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def matriz_dispersion(res, semilla_submuestra=0):
    x = res.x
    n, p = x.shape
    idx = np.random.default_rng(semilla_submuestra).choice(n, size=min(n, PUNTOS_DISPERSION), replace=False)
    xs = x[idx]
    nombres = _nombres(p)
    tam = 1.9 if p <= 4 else 1.35
    fig, ejes = plt.subplots(p, p, figsize=(tam * p, tam * p), squeeze=False)
    for i in range(p):
        for j in range(p):
            eje = ejes[i, j]
            if i == j:
                eje.hist(x[:, i], bins=40, color=COLOR, alpha=0.75)
            else:
                eje.scatter(xs[:, j], xs[:, i], s=2, alpha=0.25, color=COLOR, linewidths=0)
                eje.text(0.04, 0.9, f"r={res.corr_muestral[i, j]:.2f}", transform=eje.transAxes,
                         fontsize=7, color=COLOR_TEORICO)
            eje.tick_params(labelsize=6)
            if i == p - 1:
                eje.set_xlabel(nombres[j], fontsize=8)
            else:
                eje.set_xticklabels([])
            if j == 0:
                eje.set_ylabel(nombres[i], fontsize=8)
            else:
                eje.set_yticklabels([])
    fig.tight_layout()
    return fig


def mapas_correlacion(res):
    p = res.r.shape[0]
    nombres = _nombres(p)
    fig, ejes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for eje, m, titulo in zip(ejes, (res.r, res.corr_muestral), ("Correlación objetivo R", "Correlación muestral")):
        im = eje.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
        eje.set_xticks(range(p), nombres, fontsize=8)
        eje.set_yticks(range(p), nombres, fontsize=8)
        eje.set_title(titulo, fontsize=10)
        if p <= 6:
            for i in range(p):
                for j in range(p):
                    eje.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(m[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ejes, shrink=0.8)
    return fig


def disco_polar(rng_semilla=0, pares=1500):
    """Ilustra la región de aceptación del método polar."""
    rng = np.random.Generator(np.random.PCG64(rng_semilla))
    v = rng.uniform(-1, 1, size=(pares, 2))
    s = (v ** 2).sum(axis=1)
    ok = (s > 0) & (s < 1)
    fig, eje = plt.subplots(figsize=(4.2, 4.2))
    eje.scatter(v[ok, 0], v[ok, 1], s=4, color=COLOR, label=f"Aceptados ({ok.mean():.3f})")
    eje.scatter(v[~ok, 0], v[~ok, 1], s=4, color=COLOR_TEORICO, label="Rechazados")
    t = np.linspace(0, 2 * np.pi, 400)
    eje.plot(np.cos(t), np.sin(t), color="black", lw=1)
    eje.set_aspect("equal")
    eje.set_xlim(-1.02, 1.02)
    eje.set_ylim(-1.02, 1.02)
    eje.set_xlabel("V1")
    eje.set_ylabel("V2")
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def convergencia(errores_por_n):
    """errores_por_n: lista de (n, error_max_medias, error_max_cov)."""
    n, em, ec = zip(*errores_por_n)
    fig, eje = plt.subplots(figsize=(6, 3.6))
    eje.loglog(n, em, "o-", color=COLOR, label="Error máximo de medias")
    eje.loglog(n, ec, "s-", color=COLOR_TEORICO, label="Error máximo de covarianzas")
    ref = np.array(n, dtype=float)
    eje.loglog(ref, ec[0] * np.sqrt(ref[0] / ref), "--", color="gray", lw=1, label="Referencia 1/√n")
    eje.set_xlabel("Tamaño de muestra n")
    eje.set_ylabel("Error absoluto máximo")
    eje.legend(fontsize=8)
    eje.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig
