# Generación de tres o más variables dependientes con el método polar

CC3074 Modelización y Simulación, sección 30. Universidad del Valle de Guatemala.

Herramienta interactiva e informe académico que generan normales estándar con el método polar de
Marsaglia y las transforman en `p` variables dependientes (3 a 8) mediante `X = mu + L Z`, donde
`L` es el factor de Cholesky de `Sigma = D R D`.

## Estructura

```
config.py                  Datos del estudiante y experimento de referencia (semilla 231385)
nucleo/polar.py            Método polar de Marsaglia (a partir de uniformes PCG64)
nucleo/matrices.py         Factor común, equicorrelación, Sigma = DRD, validación, Cholesky
nucleo/simulacion.py       Transformación X = mu + L Z y diagnósticos
nucleo/graficas.py         Figuras compartidas por la app y el informe
app.py                     Aplicación Streamlit
generar_evidencias.py      Experimento de referencia: figuras y resultados para el informe
informe/plantilla_informe.md   Fuente editable del informe (Markdown con ecuaciones LaTeX)
informe/construir_informe.py   Genera el DOCX (y el PDF si LibreOffice está instalado)
informe/figuras/           Figuras generadas
informe/capturas/          Capturas de la app incrustadas en el informe (excluidas de git)
assets/pizarra.png         Fotografía de la pizarra
resultados/                Resumen JSON y muestra CSV del experimento de referencia
tests/                     Pruebas del núcleo matemático y de la interfaz
```

## Instalación

Requiere Python 3.11 o posterior.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para regenerar el informe se necesita además [pandoc](https://pandoc.org/installing.html)
(macOS: `brew install pandoc`). Para exportar el PDF automáticamente, LibreOffice
(macOS: `brew install --cask libreoffice`); sin él, el DOCX puede exportarse a PDF desde Word.

## Ejecutar la aplicación

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. En la barra lateral se eligen el número de variables, el tamaño
de muestra, la semilla, el modelo de correlación y los parámetros de cada variable. Las pestañas
muestran resultados, gráficas, datos (con descarga CSV), la pizarra, el método y las referencias.

## Pruebas

```bash
python -m pytest
```

## Regenerar resultados e informe

```bash
python generar_evidencias.py           # figuras y resultados del experimento de referencia
python informe/construir_informe.py    # informe/Informe_Generacion_Multivariable.docx (y .pdf)
```

El informe incluye capturas de la aplicación (`app_resultados.png`, `app_graficas.png`,
`app_datos.png`, `app_pizarra.png`, `app_error.png`), tomadas con la app en ejecución. Se leen de
`informe/capturas/`, quedan incrustadas en el DOCX y esa carpeta está excluida de git.

Los datos del estudiante, el curso y el experimento de referencia se cambian en `config.py`.

## Experimento de referencia

`n = 20,000`, semilla `231385`, `mu = (10, 20, 30)`, `sigma = (2, 3, 4)`, cargas `b = (0.8, 0.7, 0.6)`.
Generador uniforme: `numpy.random.Generator(PCG64)` con NumPy 2.5.1. Resultados: tasa de
aceptación 0.7835 (pi/4 = 0.7854), error máximo de medias 0.0249, error máximo de covarianzas 0.1421.
Con otra versión de NumPy u otro generador los números pueden cambiar, pero deben conservar
esas propiedades estadísticas.
