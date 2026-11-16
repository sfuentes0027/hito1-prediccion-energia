"""API Hito 2: reutiliza endpoint /predict del Hito 1.
Los endpoints de imagen y NLP necesitan instalar dependencias opcionales y descargar modelos.
La aprobación humana se implementa en n8n, nunca se omite para acciones críticas.
"""
from fastapi import FastAPI,HTTPException,UploadFile,File
from pydantic import BaseModel
from pathlib import Path
import tempfile,os
from pipeline_energia import ejecutar
from modelos.serie_gru import evaluar
ROOT=Path(__file__).resolve().parent
app=FastAPI(title='Sistema integrado Hito 2 Energia')
class Solicitud(BaseModel):
 archivo:str='escenario_1_normal.csv'
class Texto(BaseModel):
 texto:str

def ruta_segura(nombre):
 p=ROOT/'datos'/Path(nombre).name
 if not p.is_file():raise HTTPException(status_code=404,detail='Archivo no encontrado')
 return p
@app.get('/health')
def health():return {'status':'ok','version':'hito2'}
@app.post('/predict')
def predict(s:Solicitud):return ejecutar(ruta_segura(s.archivo))
@app.post('/series/compare')
def compare(s:Solicitud):return evaluar(ruta_segura(s.archivo))
@app.post('/nlp/classify')
def nlp(s:Texto):
 try:
  from modelos.nlp_transformer import analizar
  return analizar(s.texto)
 except Exception as e:raise HTTPException(status_code=503,detail='Modelo NLP no disponible; revisar dependencias y descarga') from e
@app.post('/image/classify')
async def image_classify(file:UploadFile=File(...)):
 suffix=Path(file.filename or 'imagen.jpg').suffix.lower()
 if suffix not in {'.jpg','.jpeg','.png'}:raise HTTPException(400,'Solo JPG/PNG')
 contents=await file.read()
 if len(contents)>8_000_000:raise HTTPException(413,'Imagen supera 8 MB')
 name=None
 try:
  with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f:f.write(contents);name=f.name
  from modelos.imagenes import clasificar
  return clasificar(name)
 except Exception as e:raise HTTPException(503,'Modelo de imágenes no disponible; revisar pesos y dependencias') from e
 finally:
  if name and os.path.exists(name):os.unlink(name)
