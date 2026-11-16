
"""
API Hito 2 - Sistema de predicción de consumo eléctrico.

Incluye:
- Predicción con Random Forest (Hito 1).
- Comparación de modelos: persistencia, Random Forest y GRU.
- Clasificación de textos con Transformer.
- Clasificación de imágenes con ResNet18.

Las acciones críticas requieren aprobación humana mediante n8n.
Los modelos NLP e imágenes necesitan dependencias adicionales.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os

from pipeline_energia import ejecutar

ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="Sistema integrado Hito 2 Energia",
    version="2.1"
)


# -----------------------------------------
# MODELOS DE ENTRADA
# -----------------------------------------

class Solicitud(BaseModel):
    archivo: str = "escenario_1_normal.csv"


class Texto(BaseModel):
    texto: str


# -----------------------------------------
# VALIDACIÓN DE ARCHIVOS
# -----------------------------------------

def ruta_segura(nombre):
    p = ROOT / "datos" / Path(nombre).name

    if not p.is_file():
        raise HTTPException(
            status_code=404,
            detail="Archivo no encontrado"
        )

    return p


# -----------------------------------------
# ESTADO DEL SERVIDOR
# -----------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "hito2"
    }


# -----------------------------------------
# HITO 1: PREDICCIÓN RANDOM FOREST
# -----------------------------------------

@app.post("/predict")
def predict(s: Solicitud):
    try:
        return ejecutar(ruta_segura(s.archivo))

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error al realizar predicción: {str(e)}"
        ) from e


# -----------------------------------------
# HITO 2: COMPARACIÓN DE MODELOS
# -----------------------------------------

@app.post("/series/compare")
def compare(s: Solicitud):

    try:
        # Importación diferida para evitar cargar
        # PyTorch durante el inicio de la API.
        from modelos.serie_gru import evaluar

        # Reducimos el entrenamiento de 14 a 5 épocas.
        resultado = evaluar(
            ruta_segura(s.archivo),
            epochs=5
        )

        return resultado

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo ejecutar la comparación: {str(e)}"
        ) from e


# -----------------------------------------
# HITO 2: CLASIFICACIÓN DE TEXTO
# -----------------------------------------

@app.post("/nlp/classify")
def nlp(s: Texto):

    try:
        from modelos.nlp_transformer import analizar

        return analizar(s.texto)

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo NLP no disponible. "
                "Revisar dependencias y descarga del modelo."
            )
        ) from e


# -----------------------------------------
# HITO 2: CLASIFICACIÓN DE IMÁGENES
# -----------------------------------------

@app.post("/image/classify")
async def image_classify(file: UploadFile = File(...)):

    suffix = Path(file.filename or "imagen.jpg").suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes JPG o PNG"
        )

    contents = await file.read()

    if len(contents) > 8_000_000:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera el límite de 8 MB"
        )

    name = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as f:
            f.write(contents)
            name = f.name

        from modelos.imagenes import clasificar

        return clasificar(name)

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo de imágenes no disponible. "
                "Revisar pesos y dependencias."
            )
        ) from e

    finally:
        if name and os.path.exists(name):
            os.unlink(name)
