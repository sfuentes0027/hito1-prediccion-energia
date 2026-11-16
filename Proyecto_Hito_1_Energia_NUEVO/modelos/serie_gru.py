"""Comparación cronológica: persistencia, Random Forest (Hito 1), GRU (Hito 2).
Entrena sobre datos sintéticos; no usar para decisiones reales.
"""
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
class GRU(nn.Module):
 def __init__(self,dim):
  super().__init__();self.gru=nn.GRU(dim,24,batch_first=True);self.head=nn.Linear(24,1)
 def forward(self,x):
  out,_=self.gru(x);return self.head(out[:,-1,:]).squeeze(-1)

def evaluar(archivo,epochs=14):
 torch.manual_seed(42);np.random.seed(42)
 df=pd.read_csv(archivo,parse_dates=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp')
 df=df.dropna(subset=['energy_kwh','temperature_c','humidity_pct','is_workday'])
 df=df[df.energy_kwh>=0].reset_index(drop=True)
 if len(df)<160:raise ValueError('Se necesitan al menos 160 filas')
 df['hour_sin']=np.sin(2*np.pi*df.timestamp.dt.hour/24)
 df['hour_cos']=np.cos(2*np.pi*df.timestamp.dt.hour/24)
 cols=['energy_kwh','temperature_c','humidity_pct','is_workday','hour_sin','hour_cos']
 raw=df[cols].to_numpy(dtype='float32'); cut=int(len(raw)*.8)
 scaler=StandardScaler().fit(raw[:cut]); z=scaler.transform(raw)
 lookback=24; X=[];Y=[];indices=[]
 for t in range(lookback,len(raw)):
  X.append(z[t-lookback:t]);Y.append(z[t,0]);indices.append(t)
 X=np.array(X,dtype='float32');Y=np.array(Y,dtype='float32');indices=np.array(indices)
 tr=indices<cut;te=~tr
 model=GRU(X.shape[-1]);opt=torch.optim.Adam(model.parameters(),lr=.008);loss=nn.MSELoss()
 xt=torch.from_numpy(X[tr]);yt=torch.from_numpy(Y[tr]);xv=torch.from_numpy(X[te])
 for _ in range(epochs):
  model.train();opt.zero_grad();pred=model(xt);l=loss(pred,yt);l.backward();opt.step()
 model.eval()
 with torch.no_grad():pred_gru=model(xv).numpy()
 sd=scaler.scale_[0];mean=scaler.mean_[0]
 ytrue=Y[te]*sd+mean;gru=pred_gru*sd+mean
 persist=raw[indices[te]-1,0]
 # RF uses same past 24 hours and same temporal split, to compare fairly.
 rf=RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=2)
 rf.fit(X[tr].reshape(sum(tr),-1),Y[tr]);rf_pred=rf.predict(X[te].reshape(sum(te),-1))*sd+mean
 def metric(y):return {'MAE_kwh':round(float(mean_absolute_error(ytrue,y)),3),'RMSE_kwh':round(float(np.sqrt(mean_squared_error(ytrue,y))),3)}
 result={'archivo':Path(archivo).name,'datos':'sinteticos','n_train':int(sum(tr)),'n_test':int(sum(te)),
 'baseline_persistencia':metric(persist),'random_forest':metric(rf_pred),'gru':metric(gru),
 'comparacion':'Menor MAE/RMSE es mejor; el aprendizaje profundo no garantiza mejorar el baseline.'}
 out=ROOT/'resultados'/('comparacion_'+Path(archivo).stem+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archivo',default=str(ROOT/'datos'/'escenario_1_normal.csv'));p.add_argument('--epochs',type=int,default=14)
 a=p.parse_args();print(json.dumps(evaluar(a.archivo,a.epochs),ensure_ascii=False,indent=2))
