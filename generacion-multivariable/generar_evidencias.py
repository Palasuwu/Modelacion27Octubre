"""Ejecuta el experimento de referencia y guarda figuras y resultados para el informe.

Uso:  python generar_evidencias.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from nucleo import Parametros, simular
from nucleo import graficas

RAIZ = Path(__file__).resolve().parent
FIGURAS = RAIZ / "informe" / "figuras"
RESULTADOS = RAIZ / "resultados"


def guardar(fig, nombre):
    fig.savefig(FIGURAS / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURAS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    res = simular(Parametros(**config.REFERENCIA))

    guardar(graficas.histogramas(res), "histogramas.png")
    guardar(graficas.matriz_dispersion(res), "dispersion.png")
    guardar(graficas.mapas_correlacion(res), "correlaciones.png")
    guardar(graficas.disco_polar(config.REFERENCIA["semilla"]), "disco_polar.png")

    convergencia = []
    for n in (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000):
        r = simular(Parametros(**{**config.REFERENCIA, "n": n}))
        convergencia.append((n, r.error_max_medias, r.error_max_cov))
    guardar(graficas.convergencia(convergencia), "convergencia.png")

    pd.DataFrame(res.x, columns=[f"X{i + 1}" for i in range(res.x.shape[1])]).to_csv(
        RESULTADOS / "muestra_referencia.csv", index=False, float_format="%.6f")

    resumen = {
        "parametros": config.REFERENCIA,
        "generador_uniforme": f"numpy.random.Generator(PCG64), NumPy {np.__version__}",
        "R": res.r.tolist(),
        "Sigma": res.sigma.tolist(),
        "L": res.l.tolist(),
        "media_muestral": res.media_muestral.tolist(),
        "cov_muestral": res.cov_muestral.tolist(),
        "corr_muestral": res.corr_muestral.tolist(),
        "menor_valor_propio": res.menor_valor_propio,
        "error_reconstruccion": res.error_reconstruccion,
        "error_max_medias": res.error_max_medias,
        "error_max_cov": res.error_max_cov,
        "error_max_corr": res.error_max_corr,
        "tasa_aceptacion": res.tasa_aceptacion,
        "tasa_teorica": res.tasa_teorica,
        "pares_propuestos": res.pares_propuestos,
        "pares_aceptados": res.pares_aceptados,
        "media_z": res.media_z,
        "varianza_z": res.varianza_z,
        "convergencia": [{"n": n, "error_medias": a, "error_cov": b} for n, a, b in convergencia],
    }
    (RESULTADOS / "resumen_referencia.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    print(f"Tasa de aceptación: {res.tasa_aceptacion:.4f} (teórica {res.tasa_teorica:.4f})")
    print(f"Medias muestrales: {np.round(res.media_muestral, 3)}")
    print(f"Varianzas muestrales: {np.round(np.diag(res.cov_muestral), 3)}")
    print(f"Error máximo de covarianza: {res.error_max_cov:.4f}")
    print(f"Figuras en {FIGURAS}")


if __name__ == "__main__":
    main()
