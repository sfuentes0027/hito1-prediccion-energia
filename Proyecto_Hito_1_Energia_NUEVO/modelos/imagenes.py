
from pathlib import Path
from PIL import Image, ImageStat

def clasificar(ruta_imagen):
    ruta = Path(ruta_imagen)

    with Image.open(ruta) as imagen:
        imagen = imagen.convert("RGB")
        ancho, alto = imagen.size

        estadisticas = ImageStat.Stat(imagen)
        brillo = sum(estadisticas.mean) / 3

    return {
        "estado": "ok",
        "archivo": ruta.name,
        "ancho": ancho,
        "alto": alto,
        "formato_procesado": "RGB",
        "brillo_promedio": round(brillo, 2),
        "clasificacion": "no_disponible",
        "metodo": "analisis_basico_imagen",
        "mensaje": (
            "Imagen procesada correctamente. "
            "No se ha realizado clasificacion mediante IA."
        )
    }
