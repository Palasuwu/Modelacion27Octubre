"""Comprobaciones de la interfaz con el ejecutor sin navegador de Streamlit."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def entrada(at, prefijo):
    return next(w for w in at.number_input if w.label.startswith(prefijo))


def metrica(at, etiqueta):
    return next(m.value for m in at.metric if m.label == etiqueta)


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=120).run()


def test_inicia_sin_errores_con_la_referencia(app):
    assert not app.exception
    assert not app.error
    assert float(metrica(app, "Tasa de aceptación polar")) == pytest.approx(0.7835, abs=5e-4)
    assert len(app.tabs) == 6


def test_los_controles_modifican_la_simulacion(app):
    antes = metrica(app, "Error máximo de covarianzas")
    entrada(app, "Semilla").set_value(1).run()
    assert metrica(app, "Error máximo de covarianzas") != antes
    entrada(app, "Tamaño").set_value(5000).run()
    assert "5,000 observaciones" in " ".join(m.value for m in app.markdown)
    entrada(app, "mu1").set_value(100.0).run()
    assert not app.exception


def test_equicorrelacion_fuera_de_dominio_muestra_error(app):
    app.radio[0].set_value("equicorrelacion").run()
    entrada(app, "Correlación común").set_value(-0.6).run()
    assert app.error and "rho" in app.error[0].value
    entrada(app, "Correlación común").set_value(0.3).run()
    assert not app.error and not app.exception


def test_carga_invalida_muestra_error(app):
    entrada(app, "b2").set_value(1.0).run()
    assert app.error and "|b_i| < 1" in app.error[0].value


def test_ocho_variables(app):
    entrada(app, "Número de variables").set_value(8).run()
    assert not app.exception and not app.error
    assert any(w.label == "b8" for w in app.number_input)
