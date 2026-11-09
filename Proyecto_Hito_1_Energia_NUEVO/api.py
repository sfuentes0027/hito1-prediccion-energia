from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from pipeline_energia import ejecutar
app=FastAPI(title="API Hito 1 Energía")
class Solicitud(BaseModel):
    archivo: str = "escenario_1_normal.csv"
@app.post("/predict")
def predict(s: Solicitud):
    ruta=Path(__file__).resolve().parent/"datos"/Path(s.archivo).name
    if not ruta.is_file():
        raise HTTPException(status_code=404,detail="Archivo no encontrado")
    return ejecutar(ruta)
