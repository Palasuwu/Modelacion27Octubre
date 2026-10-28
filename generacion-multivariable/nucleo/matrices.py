"""Construcción y validación de matrices de correlación y covarianza."""

from dataclasses import dataclass

import numpy as np

TOLERANCIA_SIMETRIA = 1e-10


class ParametroInvalido(ValueError):
    """Parámetro fuera de su dominio o matriz no válida."""


def correlacion_factor_comun(cargas) -> np.ndarray:
    """R = b b^T + diag(1 - b_i^2), con |b_i| < 1."""
    b = np.asarray(cargas, dtype=float)
    if b.ndim != 1 or len(b) < 3:
        raise ParametroInvalido("Se necesitan al menos tres cargas de factor.")
    if not np.all(np.isfinite(b)):
        raise ParametroInvalido("Las cargas de factor deben ser números finitos.")
    fuera = np.flatnonzero(np.abs(b) >= 1)
    if fuera.size:
        nombres = ", ".join(f"b{i + 1} = {b[i]:g}" for i in fuera)
        raise ParametroInvalido(f"Cada carga debe cumplir |b_i| < 1. Revise: {nombres}.")
    r = np.outer(b, b)
    np.fill_diagonal(r, 1.0)
    return r


def limites_equicorrelacion(p: int) -> tuple[float, float]:
    """Intervalo abierto (-1/(p-1), 1) en el que R es definida positiva."""
    return -1.0 / (p - 1), 1.0


def correlacion_equicorrelacion(p: int, rho: float) -> np.ndarray:
    """R_ii = 1, R_ij = rho, con -1/(p-1) < rho < 1."""
    if p < 3:
        raise ParametroInvalido("Se necesitan al menos tres variables.")
    if not np.isfinite(rho):
        raise ParametroInvalido("La correlación común debe ser un número finito.")
    inferior, superior = limites_equicorrelacion(p)
    if not inferior < rho < superior:
        raise ParametroInvalido(
            f"Con p = {p} la correlación común debe cumplir "
            f"{inferior:.4f} < rho < {superior:g}; se recibió rho = {rho:g}."
        )
    r = np.full((p, p), float(rho))
    np.fill_diagonal(r, 1.0)
    return r


def covarianza(sigmas, r) -> np.ndarray:
    """Sigma = D R D con D = diag(sigma)."""
    s = np.asarray(sigmas, dtype=float)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ParametroInvalido("Todas las desviaciones estándar deben ser finitas y mayores que 0.")
    r = np.asarray(r, dtype=float)
    if r.shape != (len(s), len(s)):
        raise ParametroInvalido("La matriz de correlación no coincide con el número de variables.")
    d = np.diag(s)
    return d @ r @ d


@dataclass
class Validacion:
    finita: bool
    cuadrada: bool
    simetrica: bool
    definida_positiva: bool
    menor_valor_propio: float

    @property
    def valida(self) -> bool:
        return self.finita and self.cuadrada and self.simetrica and self.definida_positiva


def validar_covarianza(sigma) -> Validacion:
    """Comprueba finitud, forma cuadrada, simetría y definición positiva."""
    m = np.asarray(sigma, dtype=float)
    finita = bool(np.all(np.isfinite(m)))
    cuadrada = m.ndim == 2 and m.shape[0] == m.shape[1]
    if not (finita and cuadrada):
        return Validacion(finita, cuadrada, False, False, float("nan"))
    escala = max(1.0, float(np.max(np.abs(m))))
    simetrica = bool(np.max(np.abs(m - m.T)) <= TOLERANCIA_SIMETRIA * escala)
    menor = float(np.min(np.linalg.eigvalsh((m + m.T) / 2)))
    return Validacion(finita, cuadrada, simetrica, menor > 0, menor)


def cholesky(sigma) -> np.ndarray:
    """Factor triangular inferior L con L L^T = Sigma (algoritmo de Cholesky-Banachiewicz)."""
    a = np.asarray(sigma, dtype=float)
    p = a.shape[0]
    l = np.zeros_like(a)
    for i in range(p):
        for j in range(i + 1):
            suma = a[i, j] - l[i, :j] @ l[j, :j]
            if i == j:
                if suma <= 0:
                    raise ParametroInvalido(
                        "La matriz de covarianza no es definida positiva; no admite factor de Cholesky."
                    )
                l[i, i] = np.sqrt(suma)
            else:
                l[i, j] = suma / l[j, j]
    return l
