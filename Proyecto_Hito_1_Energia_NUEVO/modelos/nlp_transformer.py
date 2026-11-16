
import re
import unicodedata

CATEGORIAS = {
    "consumo_anormal": [
        "consumo", "aumento", "anormal", "excesivo",
        "sobrecarga", "pico", "demanda"
    ],
    "mantenimiento": [
        "falla", "revision", "revisar", "mantenimiento",
        "equipo", "averia", "reparacion"
    ],
    "eficiencia_energetica": [
        "ahorro", "eficiencia", "optimizar",
        "reduccion", "energia", "sostenibilidad"
    ]
}

def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )
    return texto

def analizar(texto):
    if not isinstance(texto, str) or not texto.strip():
        return {
            "estado": "error",
            "mensaje": "Debe ingresar un texto valido."
        }

    texto_normalizado = normalizar(texto)
    palabras = set(re.findall(r"\b\w+\b", texto_normalizado))

    puntuaciones = {
        categoria: sum(
            1 for palabra in claves
            if palabra in palabras
        )
        for categoria, claves in CATEGORIAS.items()
    }

    categoria = max(puntuaciones, key=puntuaciones.get)
    coincidencias = puntuaciones[categoria]

    if coincidencias == 0:
        categoria = "sin_clasificar"
        prioridad = "baja"
        recomendacion = "Revisar el texto manualmente."
    elif categoria == "consumo_anormal":
        prioridad = "alta"
        recomendacion = (
            "Verificar medidores, horarios y equipos "
            "con mayor consumo electrico."
        )
    elif categoria == "mantenimiento":
        prioridad = "media"
        recomendacion = (
            "Programar una inspeccion de los equipos."
        )
    else:
        prioridad = "baja"
        recomendacion = (
            "Evaluar medidas de eficiencia energetica."
        )

    return {
        "estado": "ok",
        "texto": texto,
        "categoria": categoria,
        "prioridad": prioridad,
        "coincidencias": coincidencias,
        "recomendacion": recomendacion,
        "metodo": "clasificacion_por_reglas"
    }
