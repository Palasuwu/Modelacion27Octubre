"""Datos académicos y experimento de referencia compartidos por la app, las pruebas y el informe."""

ESTUDIANTE = {
    "nombre": "Jorge Palacios",
    "carne": "231385",
}

CURSO = {
    "universidad": "Universidad del Valle de Guatemala",
    "facultad": "Facultad de Ingeniería",
    "codigo": "CC3074",
    "nombre": "Modelización y Simulación",
    "seccion": "30",
    "fecha": "27 de octubre de 2026",
    "titulo": "Generación de tres o más variables aleatorias dependientes mediante el método polar de Marsaglia",
}

REFERENCIA = {
    "n": 20_000,
    "semilla": 231385,
    "medias": [10.0, 20.0, 30.0],
    "sigmas": [2.0, 3.0, 4.0],
    "modelo": "factor",
    "cargas": [0.8, 0.7, 0.6],
}
