"""Aplicación interactiva: generación de variables dependientes con el método polar.

Uso:  streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from nucleo import ParametroInvalido, Parametros, simular
from nucleo import graficas
from nucleo.matrices import limites_equicorrelacion
from nucleo.simulacion import N_MAX, N_MIN, P_MAX, P_MIN

RAIZ = Path(__file__).resolve().parent

st.set_page_config(page_title="Variables dependientes - Método polar", layout="wide")


def valores_por_defecto(p):
    ref = config.REFERENCIA
    medias = [ref["medias"][i] if i < 3 else 10.0 * (i + 1) for i in range(p)]
    sigmas = [ref["sigmas"][i] if i < 3 else float(i + 2) for i in range(p)]
    cargas = [ref["cargas"][i] if i < 3 else round(0.8 - 0.1 * i, 2) for i in range(p)]
    return medias, sigmas, cargas


@st.cache_data(show_spinner="Simulando...")
def simular_cacheado(n, semilla, medias, sigmas, modelo, cargas, rho):
    return simular(Parametros(n=n, semilla=semilla, medias=list(medias), sigmas=list(sigmas),
                              modelo=modelo, cargas=list(cargas) if cargas else None, rho=rho))


def tabla_matriz(m, decimales=4):
    nombres = [f"X{i + 1}" for i in range(m.shape[0])]
    return pd.DataFrame(m, index=nombres, columns=nombres).round(decimales)


def matriz_latex(m, decimales=3):
    filas = [" & ".join(f"{v:.{decimales}f}" for v in fila) for fila in np.atleast_2d(m)]
    return r"\begin{bmatrix}" + r"\\".join(filas) + r"\end{bmatrix}"


def mostrar_figura(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# --- Entradas ---------------------------------------------------------------

with st.sidebar:
    st.header("Parámetros")
    p = st.number_input("Número de variables p", P_MIN, P_MAX, 3, step=1)
    n = st.number_input("Tamaño de muestra n", N_MIN, N_MAX, config.REFERENCIA["n"], step=1000)
    semilla = st.number_input("Semilla", 0, 2**32 - 1, config.REFERENCIA["semilla"], step=1)
    modelo = st.radio("Modelo de correlación", ["factor", "equicorrelacion"],
                      format_func={"factor": "Factor común", "equicorrelacion": "Equicorrelación"}.get)

    medias_def, sigmas_def, cargas_def = valores_por_defecto(int(p))
    rho = None
    if modelo == "equicorrelacion":
        inferior, superior = limites_equicorrelacion(int(p))
        st.caption(f"Dominio válido: {inferior:.4f} < rho < {superior:g}")
        rho = st.number_input("Correlación común rho", -1.0, 1.0, 0.5, step=0.05, format="%.4f")

    st.subheader("Variables")
    medias, sigmas, cargas = [], [], []
    for i in range(int(p)):
        columnas = st.columns(3 if modelo == "factor" else 2)
        medias.append(columnas[0].number_input(f"mu{i + 1}", value=medias_def[i], key=f"mu{i}_{p}"))
        sigmas.append(columnas[1].number_input(f"sigma{i + 1}", value=sigmas_def[i], key=f"s{i}_{p}",
                                               format="%.4f"))
        if modelo == "factor":
            cargas.append(columnas[2].number_input(f"b{i + 1}", -1.0, 1.0, cargas_def[i], step=0.05,
                                                   key=f"b{i}_{p}", format="%.4f"))
    if modelo == "factor":
        st.caption("Cada carga debe cumplir |b_i| < 1.")

st.title("Generación de variables aleatorias dependientes")
st.write(
    f"{config.CURSO['codigo']} {config.CURSO['nombre']}, sección {config.CURSO['seccion']}. "
    "Normales estándar generadas con el método polar de Marsaglia y transformadas con "
    "X = mu + L Z, donde L es el factor de Cholesky de la matriz de covarianza."
)

try:
    res = simular_cacheado(int(n), int(semilla), tuple(medias), tuple(sigmas), modelo,
                           tuple(cargas) if modelo == "factor" else None, rho)
except ParametroInvalido as error:
    st.error(f"No se puede simular con estos parámetros. {error}")
    st.stop()

pestanas = st.tabs(["Resultados", "Gráficas", "Datos", "Pizarra", "Método", "Referencias"])

# --- Resultados -------------------------------------------------------------

with pestanas[0]:
    c = st.columns(4)
    c[0].metric("Tasa de aceptación polar", f"{res.tasa_aceptacion:.4f}",
                f"{res.tasa_aceptacion - res.tasa_teorica:+.4f} frente a pi/4", delta_color="off")
    c[1].metric("Error máximo de medias", f"{res.error_max_medias:.4f}")
    c[2].metric("Error máximo de covarianzas", f"{res.error_max_cov:.4f}")
    c[3].metric("Menor valor propio de Sigma", f"{res.menor_valor_propio:.4f}")
    st.caption(
        f"Pares propuestos: {res.pares_propuestos:,}; aceptados: {res.pares_aceptados:,}; "
        f"pi/4 = {res.tasa_teorica:.4f}. Normales generadas: media {res.media_z:.4f}, "
        f"varianza {res.varianza_z:.4f}. Error de reconstrucción max|LL^T - Sigma| = "
        f"{res.error_reconstruccion:.2e}."
    )

    st.subheader("Medias")
    st.dataframe(pd.DataFrame({
        "Objetivo": res.parametros.medias,
        "Muestral": res.media_muestral,
        "Diferencia": res.media_muestral - np.array(res.parametros.medias),
    }, index=[f"X{i + 1}" for i in range(res.parametros.p)]).round(4))

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.subheader("Covarianza objetivo Sigma = DRD")
        st.dataframe(tabla_matriz(res.sigma))
        st.subheader("Correlación objetivo R")
        st.dataframe(tabla_matriz(res.r))
        st.subheader("Factor de Cholesky L")
        st.latex(r"L=" + matriz_latex(res.l))
    with derecha:
        st.subheader("Covarianza muestral")
        st.dataframe(tabla_matriz(res.cov_muestral))
        st.subheader("Correlación muestral")
        st.dataframe(tabla_matriz(res.corr_muestral))
        st.subheader("Comprobación")
        st.latex(r"LL^{\mathsf T}=" + matriz_latex(res.l @ res.l.T))

# --- Gráficas ---------------------------------------------------------------

with pestanas[1]:
    st.subheader("Histogramas marginales")
    mostrar_figura(graficas.histogramas(res))
    st.subheader("Mapas de calor de correlación")
    mostrar_figura(graficas.mapas_correlacion(res))
    st.subheader("Matriz de diagramas de dispersión")
    st.caption(f"Se grafica una submuestra de hasta {graficas.PUNTOS_DISPERSION:,} observaciones; "
               "las correlaciones indicadas usan la muestra completa.")
    mostrar_figura(graficas.matriz_dispersion(res))

# --- Datos ------------------------------------------------------------------

with pestanas[2]:
    datos = pd.DataFrame(res.x, columns=[f"X{i + 1}" for i in range(res.parametros.p)])
    st.write(f"Vista previa de las primeras 20 de {len(datos):,} observaciones.")
    st.dataframe(datos.head(20).round(4))
    st.download_button(
        "Descargar muestra en CSV",
        datos.to_csv(index=False, float_format="%.6f").encode("utf-8"),
        file_name=f"muestra_p{res.parametros.p}_n{res.parametros.n}_semilla{res.parametros.semilla}.csv",
        mime="text/csv",
    )

# --- Pizarra ----------------------------------------------------------------

with pestanas[3]:
    izquierda, derecha = st.columns([2, 3])
    izquierda.image(str(RAIZ / "assets" / "pizarra.png"),
                    caption="Pizarra de la clase de Modelización y Simulación (fotografía del curso).")
    with derecha:
        st.markdown(
            "**Matriz de varianza-covarianza.** En la pizarra, V marca las varianzas de la "
            "diagonal y CV las covarianzas fuera de ella. Las filas y columnas 1 y 2 indican las variables:"
        )
        st.latex(r"\Sigma=\begin{bmatrix}\operatorname{Var}(X_1)&\operatorname{Cov}(X_1,X_2)\\"
                 r"\operatorname{Cov}(X_2,X_1)&\operatorname{Var}(X_2)\end{bmatrix}")
        st.markdown(
            "**Combinación lineal.** La expresión 11 Z1 + 12 Z2 + 13 Z3 escribe a la izquierda de "
            "cada Z el índice del coeficiente; corresponde a la primera fila de X = mu + A Z:"
        )
        st.latex(r"X_1=\mu_1+\ell_{11}Z_1+\ell_{12}Z_2+\ell_{13}Z_3")
        st.markdown(
            "Si A = L es el factor de Cholesky, los coeficientes sobre la diagonal valen cero "
            "(en la primera fila, l12 = l13 = 0):"
        )
        st.latex(r"\begin{aligned}X_1&=\mu_1+\ell_{11}Z_1\\X_2&=\mu_2+\ell_{21}Z_1+\ell_{22}Z_2\\"
                 r"X_3&=\mu_3+\ell_{31}Z_1+\ell_{32}Z_2+\ell_{33}Z_3\end{aligned}")
        st.markdown(
            "**Método polar para generar 3 o más variables dependientes.** El método polar produce "
            "las Z independientes; la dependencia la introduce la matriz L."
        )

# --- Método -----------------------------------------------------------------

with pestanas[4]:
    st.subheader("1. Método polar de Marsaglia")
    st.markdown("Se generan V1, V2 ~ U(-1, 1) independientes y se calcula")
    st.latex(r"S=V_1^2+V_2^2.")
    st.markdown("El par se rechaza si S = 0 o S >= 1. Si se acepta:")
    st.latex(r"Z_1=V_1\sqrt{\frac{-2\ln S}{S}},\qquad Z_2=V_2\sqrt{\frac{-2\ln S}{S}}.")
    st.markdown("La probabilidad de aceptar un par es el área del disco unitario entre el área del cuadrado:")
    st.latex(r"P(0<S<1)=\frac{\pi}{4}\approx 0.7854.")
    mostrar_figura(graficas.disco_polar(int(semilla)))

    st.subheader("2. Matriz de covarianza")
    st.latex(r"\Sigma=DRD,\qquad D=\operatorname{diag}(\sigma_1,\ldots,\sigma_p).")
    st.markdown("Factor común, con |b_i| < 1:")
    st.latex(r"R=\mathbf b\mathbf b^{\mathsf T}+\operatorname{diag}(1-b_1^2,\ldots,1-b_p^2).")
    st.markdown("Equicorrelación:")
    st.latex(r"R_{ii}=1,\quad R_{ij}=\rho\ (i\neq j),\qquad -\frac{1}{p-1}<\rho<1.")

    st.subheader("3. Transformación")
    st.latex(r"\Sigma=LL^{\mathsf T},\qquad \mathbf X=\boldsymbol\mu+L\mathbf Z.")
    st.latex(r"E[\mathbf X]=\boldsymbol\mu+L\,E[\mathbf Z]=\boldsymbol\mu,\qquad "
             r"\operatorname{Cov}(\mathbf X)=L\operatorname{Cov}(\mathbf Z)L^{\mathsf T}=LIL^{\mathsf T}=\Sigma.")
    st.caption("Generador uniforme: numpy.random.Generator con PCG64. La transformación polar y "
               "la factorización de Cholesky están implementadas en el paquete nucleo.")

# --- Referencias ------------------------------------------------------------

with pestanas[5]:
    st.markdown("""
- Johnson, R. A., & Wichern, D. W. (2007). *Applied multivariate statistical analysis* (6th ed.). Pearson Prentice Hall.
- Knuth, D. E. (1997). *The art of computer programming: Vol. 2. Seminumerical algorithms* (3rd ed.). Addison-Wesley.
- Marsaglia, G., & Bray, T. A. (1964). A convenient method for generating normal variables. *SIAM Review, 6*(3), 260-264. https://doi.org/10.1137/1006063
- National Institute of Standards and Technology. (s. f.). *The multivariate normal distribution*. https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc542.htm
- NumPy Developers. (s. f.). *numpy.random.Generator.multivariate_normal*. https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html
- Ross, S. M. (2013). *Simulation* (5th ed.). Academic Press.
- SciPy Community. (s. f.). *scipy.linalg.cholesky*. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html
""")
