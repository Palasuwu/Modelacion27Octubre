import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cholesky as cholesky_scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from nucleo import ParametroInvalido, Parametros, TASA_TEORICA, generar_normales, simular, transformar_pares
from nucleo.matrices import (
    cholesky,
    correlacion_equicorrelacion,
    correlacion_factor_comun,
    covarianza,
    validar_covarianza,
)


def rng(semilla=config.REFERENCIA["semilla"]):
    return np.random.Generator(np.random.PCG64(semilla))


@pytest.fixture(scope="module")
def referencia():
    return simular(Parametros(**config.REFERENCIA))


# --- Método polar -----------------------------------------------------------

def test_rechaza_s_cero_y_s_mayor_o_igual_a_uno():
    v1 = np.array([0.0, 1.0, 0.6, 0.8, 0.3])
    v2 = np.array([0.0, 0.0, 0.8, 0.8, 0.4])   # S = 0, 1, 1, 1.28, 0.25
    mascara, z1, z2 = transformar_pares(v1, v2)
    assert mascara.tolist() == [False, False, False, False, True]
    factor = np.sqrt(-2 * np.log(0.25) / 0.25)
    assert z1[0] == pytest.approx(0.3 * factor)
    assert z2[0] == pytest.approx(0.4 * factor)


def test_cantidad_impar_descarta_solo_el_sobrante():
    impar = generar_normales(7, rng(5))
    par = generar_normales(8, rng(5))
    assert len(impar.z) == 7
    assert impar.pares_aceptados == 4
    np.testing.assert_array_equal(impar.z, par.z[:7])


def test_tasa_de_aceptacion_cercana_a_pi_cuartos():
    res = generar_normales(400_000, rng())
    # Error estándar de una proporción con 2.5e5 ensayos aprox. 8.2e-4.
    assert abs(res.tasa_aceptacion - TASA_TEORICA) < 4e-3


def test_normales_con_media_cero_y_varianza_uno():
    z = generar_normales(400_000, rng()).z
    assert abs(z.mean()) < 0.01
    assert abs(z.var(ddof=1) - 1) < 0.01
    assert abs(np.corrcoef(z[0::2], z[1::2])[0, 1]) < 0.01


def test_reproducible_con_la_misma_semilla():
    a = generar_normales(1001, rng(42)).z
    b = generar_normales(1001, rng(42)).z
    c = generar_normales(1001, rng(43)).z
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --- Matrices ---------------------------------------------------------------

def test_factor_comun_de_referencia():
    r = correlacion_factor_comun([0.8, 0.7, 0.6])
    esperado = np.array([[1, 0.56, 0.48], [0.56, 1, 0.42], [0.48, 0.42, 1]])
    np.testing.assert_allclose(r, esperado)
    sigma = covarianza([2, 3, 4], r)
    np.testing.assert_allclose(sigma, [[4, 3.36, 3.84], [3.36, 9, 5.04], [3.84, 5.04, 16]])


@pytest.mark.parametrize("cargas", [[1.0, 0.5, 0.5], [0.5, -1.2, 0.3], [0.2, np.nan, 0.3], [0.5, 0.5]])
def test_factor_comun_rechaza_cargas_invalidas(cargas):
    with pytest.raises(ParametroInvalido):
        correlacion_factor_comun(cargas)


@pytest.mark.parametrize("p", range(3, 9))
def test_equicorrelacion_limites(p):
    inferior = -1 / (p - 1)
    for rho in (inferior + 1e-3, 0.0, 0.999):
        assert validar_covarianza(correlacion_equicorrelacion(p, rho)).valida
    for rho in (inferior, 1.0, -0.99 if p > 3 else -0.6):
        with pytest.raises(ParametroInvalido):
            correlacion_equicorrelacion(p, rho)


def test_validacion_detecta_matrices_invalidas():
    assert not validar_covarianza(np.array([[1, 0.5], [0.4, 1]])).simetrica
    assert not validar_covarianza(np.ones((2, 3))).cuadrada
    assert not validar_covarianza(np.array([[1, np.inf], [np.inf, 1]])).finita
    v = validar_covarianza(np.array([[1, 2], [2, 1]]))
    assert not v.definida_positiva and v.menor_valor_propio == pytest.approx(-1)


def test_cholesky_reproduce_sigma_y_coincide_con_scipy():
    gen = rng(7)
    for p in range(3, 9):
        b = gen.uniform(-0.95, 0.95, size=p)
        sigma = covarianza(gen.uniform(0.5, 5, size=p), correlacion_factor_comun(b))
        l = cholesky(sigma)
        assert np.allclose(np.triu(l, 1), 0)
        np.testing.assert_allclose(l @ l.T, sigma, atol=1e-12)
        np.testing.assert_allclose(l, cholesky_scipy(sigma, lower=True), atol=1e-12)


def test_cholesky_rechaza_matriz_no_definida_positiva():
    with pytest.raises(ParametroInvalido):
        cholesky(np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]], dtype=float))


# --- Simulación multivariable ----------------------------------------------

def test_experimento_de_referencia(referencia):
    res = referencia
    assert res.x.shape == (20_000, 3)
    assert res.error_reconstruccion < 1e-12
    assert res.error_max_medias < 0.1
    assert res.error_max_cov < 0.5
    assert res.error_max_corr < 0.03
    assert abs(res.tasa_aceptacion - TASA_TEORICA) < 0.01
    assert res.menor_valor_propio > 0


def test_referencia_reproducible(referencia):
    otra = simular(Parametros(**config.REFERENCIA))
    np.testing.assert_array_equal(referencia.x, otra.x)


def test_error_de_covarianza_disminuye_al_crecer_n():
    base = dict(config.REFERENCIA)
    errores = []
    for n in (1_000, 10_000, 100_000):
        base["n"] = n
        errores.append(simular(Parametros(**base)).error_max_cov)
    assert errores[2] < errores[0]


def test_equicorrelacion_negativa_con_ocho_variables():
    par = Parametros(n=50_000, semilla=1, medias=[0.0] * 8, sigmas=[1.0] * 8,
                     modelo="equicorrelacion", rho=-0.1)
    res = simular(par)
    fuera = ~np.eye(8, dtype=bool)
    assert np.all(res.corr_muestral[fuera] < 0)
    assert res.error_max_corr < 0.03


def test_signo_de_correlaciones_en_factor_comun():
    par = Parametros(n=30_000, semilla=3, medias=[0.0] * 4, sigmas=[1.0] * 4,
                     modelo="factor", cargas=[0.8, -0.7, 0.6, -0.5])
    res = simular(par)
    assert np.all(np.sign(res.corr_muestral) == np.sign(res.r))


@pytest.mark.parametrize("cambios", [
    {"medias": [0.0, 0.0]},
    {"sigmas": [1.0, 0.0, 1.0]},
    {"n": 1},
    {"semilla": -1},
    {"modelo": "otro"},
])
def test_parametros_fuera_de_dominio(cambios):
    base = dict(config.REFERENCIA)
    base.update(cambios)
    if "medias" in cambios:
        base["sigmas"] = [1.0, 1.0]
        base["cargas"] = [0.5, 0.5]
    with pytest.raises(ParametroInvalido):
        simular(Parametros(**base))
