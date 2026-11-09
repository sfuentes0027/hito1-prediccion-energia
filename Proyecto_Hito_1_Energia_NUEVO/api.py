from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from pipeline_energia import ejecutar

app = FastAPI(title="API Hito 1 Energia")

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"

class Solicitud(BaseModel):
    archivo: str = "escenario_1_normal.csv"

@app.post("/predict")
def predict(s: Solicitud):
    nombre = Path(s.archivo).name
    ruta = DATOS_DIR / nombre

    if not ruta.is_file():
        disponibles = sorted(
            p.name for p in DATOS_DIR.glob("*.csv")
        ) if DATOS_DIR.is_dir() else []

        raise HTTPException(
            status_code=404,
            detail={
                "error": "Archivo no encontrado",
                "archivo_solicitado": nombre,
                "archivos_disponibles": disponibles
            }
        )

    return ejecutar(ruta)
