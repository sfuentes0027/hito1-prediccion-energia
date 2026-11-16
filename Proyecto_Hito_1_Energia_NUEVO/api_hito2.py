
"""API Hito 2 - Energia.
Optimizada para Render Free (512 MB).
Los modelos pesados se importan solo cuando se necesitan.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import csv
import math
import os
import tempfile
import gc

ROOT = Path(__file__).resolve().parent
DATOS = ROOT / "datos"

app = FastAPI(
    title="Sistema integrado Hito 2 Energia",
    version="2.1"
)


class Solicitud(BaseModel):
    archivo: str = "escenario_1_normal.csv"


class Texto(BaseModel):
    texto: str


def ruta_segura(nombre: str) -> Path:
    if Path(nombre).name != nombre:
        raise HTTPException(
            status_code=400,
            detail="Nombre de archivo no permitido"
        )

    if not nombre.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos CSV"
        )

    ruta = DATOS / nombre

    if not ruta.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Archivo no encontrado: {nombre}"
        )

    return ruta


@app.get("/")
def inicio():
    return {
        "status": "ok",
        "mensaje": "API de prediccion de energia activa",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "hito2"
    }


@app.post("/predict")
def predict(s: Solicitud):
    try:
        # Importacion diferida: evita cargar sklearn
        # cuando solo se consulta /health o /docs.
        from pipeline_energia import ejecutar

        resultado = ejecutar(ruta_segura(s.archivo))
        return resultado

    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(
            status_code=503,
            detail="Memoria insuficiente para prediccion"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error en prediccion: {str(e)}"
        ) from e
    finally:
        gc.collect()


def calcular_metricas(real, predicho):
    if not real or len(real) != len(predicho):
        raise ValueError("No hay datos suficientes para evaluar")

    n = len(real)
    errores = [
        a - b for a, b in zip(real, predicho)
    ]

    mae = sum(abs(e) for e in errores) / n
    rmse = math.sqrt(
        sum(e * e for e in errores) / n
    )

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }


def evaluar(ruta: Path):
    """Compara dos baselines livianos sin pandas ni sklearn.

    Persistencia: predice el valor de la hora anterior.
    Estacional: predice el valor de la misma hora
    del dia anterior (24 horas antes).
    """

    valores = []

    with ruta.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as archivo:
        lector = csv.DictReader(archivo)

        if "energy_kwh" not in (lector.fieldnames or []):
            raise ValueError(
                "El CSV necesita la columna energy_kwh"
            )

        for fila in lector:
            try:
                valor = float(fila["energy_kwh"])
            except (ValueError, TypeError):
                continue

            if math.isfinite(valor) and valor >= 0:
                valores.append(valor)

            if len(valores) > 100_000:
                raise ValueError(
                    "Archivo demasiado grande para esta API"
                )

    if len(valores) < 50:
        raise ValueError(
            "Se necesitan al menos 50 registros validos"
        )

    # Division cronologica: ultimos 20% para prueba.
    inicio_test = max(
        24,
        int(len(valores) * 0.80)
    )

    reales = valores[inicio_test:]

    pred_persistencia = [
        valores[i - 1]
        for i in range(inicio_test, len(valores))
    ]

    pred_estacional = [
        valores[i - 24]
        for i in range(inicio_test, len(valores))
    ]

    met_persistencia = calcular_metricas(
        reales, pred_persistencia
    )
    met_estacional = calcular_metricas(
        reales, pred_estacional
    )

    ganador = (
        "persistencia"
        if met_persistencia["MAE"] <= met_estacional["MAE"]
        else "estacional_24h"
    )

    return {
        "archivo": ruta.name,
        "estado": "ok",
        "n_registros": len(valores),
        "n_test": len(reales),
        "metodologia": "Division cronologica 80/20",
        "modelos": {
            "persistencia": met_persistencia,
            "estacional_24h": met_estacional
        },
        "mejor_baseline": ganador,
        "nota": (
            "Comparacion de dos modelos base. "
            "No incluye Random Forest ni Transformer."
        )
    }


@app.post("/series/compare")
def compare(s: Solicitud):
    try:
        return evaluar(ruta_segura(s.archivo))
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error en comparacion: {str(e)}"
        ) from e


@app.post("/nlp/classify")
def nlp(s: Texto):
    try:
        # El modelo se importa solo al usar este endpoint.
        from modelos.nlp_transformer import analizar
        return analizar(s.texto)

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo NLP no disponible. "
                "Revisar dependencias y memoria."
            )
        ) from e
    finally:
        gc.collect()


@app.post("/image/classify")
async def image_classify(file: UploadFile = File(...)):
    nombre = file.filename or "imagen.jpg"
    suffix = Path(nombre).suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten JPG y PNG"
        )

    # Limitar la lectura para evitar archivos enormes.
    contenido = await file.read(4_000_001)

    if len(contenido) > 4_000_000:
        raise HTTPException(
            status_code=413,
            detail="Imagen supera los 4 MB"
        )

    temporal = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as f:
            f.write(contenido)
            temporal = f.name

        # Importacion diferida del modelo de imagenes.
        from modelos.imagenes import clasificar
        return clasificar(temporal)

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo de imagenes no disponible. "
                "Revisar pesos, dependencias y memoria."
            )
        ) from e

    finally:
        if temporal and os.path.exists(temporal):
            os.unlink(temporal)
        await file.close()
        gc.collect()
