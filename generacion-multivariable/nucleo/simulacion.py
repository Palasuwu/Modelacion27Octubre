"""Simulación de X = mu + L Z y cálculo de diagnósticos."""

from dataclasses import dataclass

import numpy as np

from .matrices import (
    ParametroInvalido,
    cholesky,
    correlacion_equicorrelacion,
    correlacion_factor_comun,
    covarianza,
    validar_covarianza,
)
from .polar import TASA_TEORICA, generar_normales

P_MIN, P_MAX = 3, 8
N_MIN, N_MAX = 10, 500_000


@dataclass
class Parametros:
    n: int
    semilla: int
    medias: list[float]
    sigmas: list[float]
    modelo: str                      # "factor" o "equicorrelacion"
    cargas: list[float] | None = None
    rho: float | None = None

    @property
    def p(self) -> int:
        return len(self.medias)


@dataclass
class Resultado:
    parametros: Parametros
    x: np.ndarray
    z: np.ndarray
    r: np.ndarray
    sigma: np.ndarray
    l: np.ndarray
    media_muestral: np.ndarray
    cov_muestral: np.ndarray
    corr_muestral: np.ndarray
    menor_valor_propio: float
    error_reconstruccion: float      # max |L L^T - Sigma|
    error_max_medias: float
    error_max_cov: float
    error_max_corr: float
    tasa_aceptacion: float
    pares_propuestos: int
    pares_aceptados: int
    media_z: float
    varianza_z: float

    @property
    def tasa_teorica(self) -> float:
        return TASA_TEORICA


def construir_correlacion(par: Parametros) -> np.ndarray:
    if par.modelo == "factor":
        if par.cargas is None or len(par.cargas) != par.p:
            raise ParametroInvalido("Debe indicar una carga de factor por variable.")
        return correlacion_factor_comun(par.cargas)
    if par.modelo == "equicorrelacion":
        if par.rho is None:
            raise ParametroInvalido("Debe indicar la correlación común rho.")
        return correlacion_equicorrelacion(par.p, par.rho)
    raise ParametroInvalido(f"Modelo de correlación desconocido: {par.modelo!r}.")


def validar_parametros(par: Parametros) -> None:
    if not P_MIN <= par.p <= P_MAX:
        raise ParametroInvalido(f"El número de variables debe estar entre {P_MIN} y {P_MAX}.")
    if len(par.sigmas) != par.p:
        raise ParametroInvalido("Debe indicar una desviación estándar por variable.")
    if not N_MIN <= par.n <= N_MAX:
        raise ParametroInvalido(f"El tamaño de muestra debe estar entre {N_MIN} y {N_MAX:,}.")
    if par.semilla < 0:
        raise ParametroInvalido("La semilla debe ser un entero no negativo.")
    if not np.all(np.isfinite(par.medias)):
        raise ParametroInvalido("Las medias deben ser números finitos.")


def simular(par: Parametros) -> Resultado:
    validar_parametros(par)
    r = construir_correlacion(par)
    sigma = covarianza(par.sigmas, r)

    val = validar_covarianza(sigma)
    if not val.valida:
        raise ParametroInvalido(
            "La matriz de covarianza no es válida "
            f"(finita={val.finita}, cuadrada={val.cuadrada}, simétrica={val.simetrica}, "
            f"definida positiva={val.definida_positiva}, menor valor propio={val.menor_valor_propio:.3g})."
        )
    l = cholesky(sigma)

    rng = np.random.Generator(np.random.PCG64(par.semilla))
    polar = generar_normales(par.n * par.p, rng)
    z = polar.z.reshape(par.n, par.p)

    mu = np.asarray(par.medias, dtype=float)
    x = mu + z @ l.T                 # cada fila: x = mu + L z

    media = x.mean(axis=0)
    cov = np.cov(x, rowvar=False, ddof=1)
    corr = np.corrcoef(x, rowvar=False)

    return Resultado(
        parametros=par,
        x=x,
        z=z,
        r=r,
        sigma=sigma,
        l=l,
        media_muestral=media,
        cov_muestral=cov,
        corr_muestral=corr,
        menor_valor_propio=val.menor_valor_propio,
        error_reconstruccion=float(np.max(np.abs(l @ l.T - sigma))),
        error_max_medias=float(np.max(np.abs(media - mu))),
        error_max_cov=float(np.max(np.abs(cov - sigma))),
        error_max_corr=float(np.max(np.abs(corr - r))),
        tasa_aceptacion=polar.tasa_aceptacion,
        pares_propuestos=polar.pares_propuestos,
        pares_aceptados=polar.pares_aceptados,
        media_z=float(polar.z.mean()),
        varianza_z=float(polar.z.var(ddof=1)),
    )
