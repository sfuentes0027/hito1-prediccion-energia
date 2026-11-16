"""Clasificación de fotografías de equipos eléctricos con ResNet18 preentrenada ImageNet.
Es una clasificación visual general, NO diagnóstico eléctrico ni detector de fallas.
En primer uso descarga pesos públicos; necesita conexión. Probar con imágenes reales etiquetadas.
"""
from pathlib import Path
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights

def clasificar(ruta):
 weights=ResNet18_Weights.DEFAULT
 model=resnet18(weights=weights).eval()
 img=Image.open(ruta).convert('RGB');x=weights.transforms()(img).unsqueeze(0)
 with torch.no_grad():p=torch.softmax(model(x),dim=1)[0]
 v,i=torch.topk(p,3)
 return {'modelo':'ResNet18 pretrained ImageNet','imagen':Path(ruta).name,
 'top3':[{'clase':weights.meta['categories'][int(k)],'probabilidad':round(float(q),4)} for q,k in zip(v,i)],
 'advertencia':'Etiquetas ImageNet generales; no certifica estado de seguridad ni falla eléctrica.'}
if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser();p.add_argument('imagen');a=p.parse_args();print(json.dumps(clasificar(a.imagen),ensure_ascii=False,indent=2))
