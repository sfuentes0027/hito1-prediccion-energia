from pathlib import Path
from PIL import Image, ImageStat, UnidentifiedImageError


def clasificar(ruta_imagen):
    ruta = Path(ruta_imagen)

    if not ruta.is_file():
        raise FileNotFoundError("La imagen no existe")

    try:
        with Image.open(ruta) as imagen:
            imagen = imagen.convert("RGB")

            # Reducir tamaño para ahorrar memoria
            imagen.thumbnail((512, 512))

            ancho, alto = imagen.size
            estadisticas = ImageStat.Stat(imagen)

            brillo = sum(estadisticas.mean) / 3
            contraste = sum(estadisticas.stddev) / 3

        # Clasificación visual por reglas
        if brillo < 65:
            clasificacion = "imagen_oscura"
            recomendacion = "Revisar iluminación de la imagen."

        elif brillo > 210:
            clasificacion = "imagen_muy_clara"
            recomendacion = "Revisar exposición y reflejos."

        elif contraste < 25:
            clasificacion = "contraste_bajo"
            recomendacion = "Mejorar contraste para observar detalles."

        else:
            clasificacion = "condiciones_visuales_adecuadas"
            recomendacion = "Imagen apta para revisión visual."

        return {
            "estado": "ok",
            "archivo": ruta.name,
            "ancho_procesado": ancho,
            "alto_procesado": alto,
            "brillo_promedio": round(brillo, 2),
            "contraste_promedio": round(contraste, 2),
            "clasificacion": clasificacion,
            "recomendacion": recomendacion,
            "metodo": "analisis_visual_por_reglas",
            "nota": (
                "Clasificación de calidad visual. "
                "No identifica equipos ni detecta fallas eléctricas."
            )
        }

    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Archivo de imagen inválido") from e
