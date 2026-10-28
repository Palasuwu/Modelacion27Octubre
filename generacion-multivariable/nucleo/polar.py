"""Método polar de Marsaglia para generar normales estándar independientes.

La única fuente aleatoria es una sucesión de uniformes U(-1, 1) producida por
numpy.random.Generator con el generador de bits PCG64. La transformación polar
(rechazo y escalamiento) está implementada aquí; no se usa ninguna función que
genere normales directamente.
"""

from dataclasses import dataclass

import numpy as np

TASA_TEORICA = np.pi / 4


@dataclass
class ResultadoPolar:
    """Normales generadas y contadores del proceso de aceptación-rechazo."""

    z: np.ndarray            # valores normales estándar, longitud solicitada
    pares_propuestos: int    # pares (V1, V2) evaluados
    pares_aceptados: int     # pares con 0 < S < 1

    @property
    def tasa_aceptacion(self) -> float:
        return self.pares_aceptados / self.pares_propuestos


def transformar_pares(v1: np.ndarray, v2: np.ndarray):
    """Aplica el paso polar a pares (V1, V2) ya generados.

    Devuelve la máscara de aceptación y los pares (Z1, Z2) aceptados.
    Se rechaza el par si S = 0 o S >= 1.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    s = v1 * v1 + v2 * v2
    aceptado = (s > 0.0) & (s < 1.0)
    s_ok = s[aceptado]
    factor = np.sqrt(-2.0 * np.log(s_ok) / s_ok)
    return aceptado, v1[aceptado] * factor, v2[aceptado] * factor


def generar_normales(cantidad: int, rng: np.random.Generator) -> ResultadoPolar:
    """Genera `cantidad` normales estándar independientes con el método polar.

    Se necesitan ceil(cantidad / 2) pares aceptados. Si `cantidad` es impar se
    genera un par adicional y se descarta solo el último valor sobrante.
    Las uniformes se piden por bloques; los contadores se truncan en el par
    exacto que completa la cuota, de modo que la tasa reportada corresponde a
    los pares que realmente se necesitaron.
    """
    if cantidad < 1:
        raise ValueError("La cantidad de normales debe ser al menos 1.")

    pares_necesarios = (cantidad + 1) // 2
    z1_partes, z2_partes = [], []
    aceptados = 0
    propuestos = 0

    while aceptados < pares_necesarios:
        faltan = pares_necesarios - aceptados
        bloque = int(np.ceil(faltan / TASA_TEORICA * 1.05)) + 16
        v = rng.uniform(-1.0, 1.0, size=(bloque, 2))
        mascara, z1, z2 = transformar_pares(v[:, 0], v[:, 1])

        if len(z1) >= faltan:
            # Posición del par que completa la cuota dentro del bloque.
            ultimo = np.flatnonzero(mascara)[faltan - 1]
            propuestos += ultimo + 1
            z1, z2 = z1[:faltan], z2[:faltan]
        else:
            propuestos += bloque

        aceptados += len(z1)
        z1_partes.append(z1)
        z2_partes.append(z2)

    z1 = np.concatenate(z1_partes)
    z2 = np.concatenate(z2_partes)
    # Intercalar (Z1, Z2) de cada par para conservar el orden de generación.
    z = np.column_stack([z1, z2]).ravel()[:cantidad]
    return ResultadoPolar(z=z, pares_propuestos=int(propuestos), pares_aceptados=int(aceptados))
