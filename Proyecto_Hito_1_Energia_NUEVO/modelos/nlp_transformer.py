"""Clasificador zero-shot basado en Transformer multilingüe.
Modelo público: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli; descarga inicial requiere internet.
No sustituye la revisión humana de informes técnicos.
"""
from pathlib import Path
import json
LABELS=['riesgo de sobreconsumo','mantenimiento preventivo','falla de equipo','operación normal']
def analizar(texto):
 from transformers import pipeline
 clasificador=pipeline('zero-shot-classification',model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
 r=clasificador(texto,candidate_labels=LABELS,multi_label=False)
 return {'modelo':'mDeBERTa-v3-base-mnli-xnli','clasificacion':r['labels'][0],
 'confianza':round(float(r['scores'][0]),4),'alternativas':[{'etiqueta':a,'score':round(float(b),4)} for a,b in zip(r['labels'],r['scores'])],
 'texto_original':texto[:2000], 'advertencia':'Clasificación automática, requiere revisión de un responsable.'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--texto',required=True);a=p.parse_args();print(json.dumps(analizar(a.texto),ensure_ascii=False,indent=2))
