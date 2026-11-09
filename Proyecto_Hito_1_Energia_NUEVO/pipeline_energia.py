"""
Hito 1 - Caso F: Energía
Predicción horaria de consumo eléctrico con Random Forest.
Los CSV de ejemplo son sintéticos y sirven para probar el flujo; reemplazar por
mediciones reales antes de usar el resultado en una decisión operacional.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datos"
OUT = ROOT / "resultados"
OUT.mkdir(exist_ok=True)

def cargar(ruta):
    df = pd.read_csv(ruta, parse_dates=["timestamp"])
    requeridas = {"timestamp","temperature_c","humidity_pct","is_workday","energy_kwh"}
    faltan = requeridas - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    df = df.dropna(subset=list(requeridas)).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])
    for c in ["temperature_c","humidity_pct","is_workday","energy_kwh"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[df["energy_kwh"] >= 0]
    if len(df) < 100:
        raise ValueError("Se requieren al menos 100 registros válidos para entrenar.")
    return df.reset_index(drop=True)

def preparar(df):
    x = df.copy()
    x["hour"] = x.timestamp.dt.hour
    x["dayofweek"] = x.timestamp.dt.dayofweek
    x["month"] = x.timestamp.dt.month
    x["lag_1"] = x.energy_kwh.shift(1)
    x["lag_24"] = x.energy_kwh.shift(24)
    x["rolling_24"] = x.energy_kwh.shift(1).rolling(24).mean()
    x["target"] = x.energy_kwh.shift(-1)
    x = x.dropna()
    features = ["temperature_c","humidity_pct","is_workday","hour","dayofweek","month","lag_1","lag_24","rolling_24"]
    return x, features

def entrenar(df):
    x, features = preparar(df)
    # División cronológica: el conjunto de prueba contiene las observaciones más recientes.
    corte = int(len(x)*0.8)
    X_train, X_test = x[features].iloc[:corte], x[features].iloc[corte:]
    y_train, y_test = x.target.iloc[:corte], x.target.iloc[corte:]
    modelo = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=2,
                                   random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    baseline = x["energy_kwh"].iloc[corte:].to_numpy()  # persistencia: siguiente hora = hora actual
    metricas = {
        "MAE_random_forest": round(float(mean_absolute_error(y_test,pred)),3),
        "RMSE_random_forest": round(float(np.sqrt(mean_squared_error(y_test,pred))),3),
        "MAE_baseline_persistencia": round(float(mean_absolute_error(y_test,baseline)),3),
        "RMSE_baseline_persistencia": round(float(np.sqrt(mean_squared_error(y_test,baseline))),3),
        "n_train": int(len(X_train)), "n_test": int(len(X_test))
    }
    return modelo, features, metricas

def pronosticar(df, modelo, features, horas=24):
    # Pronóstico recursivo: utiliza los últimos valores conocidos y repite condiciones
    # ambientales/operacionales de cada fila futura disponible.
    hist=df.copy().reset_index(drop=True)
    filas=[]
    for i in range(min(horas, len(hist)-24)):
        j=len(hist)-1
        ts=hist.timestamp.iloc[j]+pd.Timedelta(hours=1)
        row={
            "temperature_c": float(hist.temperature_c.iloc[j]),
            "humidity_pct": float(hist.humidity_pct.iloc[j]),
            "is_workday": int(hist.is_workday.iloc[j]),
            "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month,
            "lag_1": float(hist.energy_kwh.iloc[-1]),
            "lag_24": float(hist.energy_kwh.iloc[-24]),
            "rolling_24": float(hist.energy_kwh.tail(24).mean())
        }
        y=float(modelo.predict(pd.DataFrame([row])[features])[0])
        filas.append({"timestamp":ts.isoformat(),"predicted_kwh":round(max(y,0),2)})
        hist.loc[len(hist)]={"timestamp":ts,"temperature_c":row["temperature_c"],
            "humidity_pct":row["humidity_pct"],"is_workday":row["is_workday"],"energy_kwh":max(y,0)}
    return filas

def ejecutar(archivo):
    df=cargar(archivo)
    modelo, features, metricas=entrenar(df)
    # Se generan predicciones de validación con el último bloque para diagnóstico.
    x, feats=preparar(df)
    cut=int(len(x)*0.8)
    pred=modelo.predict(x[feats].iloc[cut:])
    promedio=float(np.mean(pred))
    p95=float(np.percentile(pred,95))
    promedio_real=float(df.energy_kwh.mean())
    desviacion=(promedio-promedio_real)/promedio_real*100 if promedio_real else 0
    estado="ALERTA" if promedio > promedio_real*1.15 or p95 > promedio_real*1.35 else "NORMAL"
    recomendacion=("Revisar equipos de alto consumo, horarios de operación y posibles cargas simultáneas. "
                   "Confirmar la lectura del medidor antes de tomar acciones.") if estado=="ALERTA" else (
                   "Mantener seguimiento horario y comparar el consumo real con el pronóstico. "
                   "Revisar desviaciones antes de modificar la operación.")
    resultado={
        "caso":Path(archivo).stem, "estado":estado,
        "consumo_promedio_predicho_kwh":round(promedio,2),
        "percentil_95_predicho_kwh":round(p95,2),
        "consumo_promedio_historico_kwh":round(promedio_real,2),
        "variacion_porcentual":round(desviacion,2),
        "metricas":metricas, "recomendacion":recomendacion,
        "nota":"Datos de demostración sintéticos; no usar como medición real."
    }
    destino=OUT/f"resultado_{Path(archivo).stem}.json"
    destino.write_text(json.dumps(resultado,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(resultado,indent=2,ensure_ascii=False))
    return resultado

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--archivo",default=str(DATA/"escenario_1_normal.csv"))
    args=ap.parse_args()
    ejecutar(args.archivo)
