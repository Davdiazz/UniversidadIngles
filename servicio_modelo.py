"""
Servicio del modelo ML de nivel de inglés.

Carga el modelo ENTRENADO por train_model.py (models/modelo_nivel.joblib, scikit-learn) y lo expone por HTTP
para que n8n le pregunte el nivel del estudiante. Si el modelo no existe (no se ha entrenado) o no se puede
cargar, lo dice claramente y n8n NO sigue con Groq: el chat muestra que el modelo no está disponible.

Ejecutar (desde la carpeta universidadIA):
    uvicorn servicio_modelo:app --host 0.0.0.0 --port 8001

Endpoints:
    GET  /estado    -> si el modelo está cargado, cuándo se entrenó, exactitud, etc.
    POST /predecir  -> body: {"respuestas": {"spoon": "cuchara", "gale": "No la conozco", ...}}
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).parent
RUTA_MODELO = BASE / "models" / "modelo_nivel.joblib"
RUTA_INFO = BASE / "models" / "modelo_nivel.json"
CEFR = ["A1", "A2", "B1", "B2", "C1", "C2"]
NO_LA_CONOZCO = "No la conozco"

app = FastAPI(title="Modelo ML de nivel de inglés")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

with open(BASE / "data" / "banco_vocabulario.csv", encoding="utf-8") as f:
    BANCO = {fila["palabra"]: fila for fila in csv.DictReader(f)}

modelo = None
info = {}
error_carga = None


def cargar_modelo():
    """Carga el modelo entrenado. Si no existe, deja el motivo en error_carga."""
    global modelo, info, error_carga
    modelo, info, error_carga = None, {}, None
    if not RUTA_MODELO.exists():
        error_carga = "El modelo no está entrenado: no existe models/modelo_nivel.joblib. Ejecuta: python train_model.py"
        return
    try:
        modelo = joblib.load(RUTA_MODELO)
        info = json.loads(RUTA_INFO.read_text()) if RUTA_INFO.exists() else {}
        if "entrenado_el" not in info:
            info["entrenado_el"] = datetime.fromtimestamp(RUTA_MODELO.stat().st_mtime).isoformat(timespec="seconds")
    except Exception as e:  # p. ej. versión de scikit-learn distinta a la del entrenamiento
        modelo = None
        error_carga = f"No se pudo cargar el modelo ({e}). Vuelve a entrenarlo: python train_model.py"


cargar_modelo()
print("Modelo cargado:" if modelo else "MODELO NO DISPONIBLE:", info.get("entrenado_el") if modelo else error_carga)


def estado():
    if modelo is None:
        return {"modelo_cargado": False, "error": error_carga}
    return {
        "modelo_cargado": True,
        "tipo": type(modelo).__name__,
        "archivo": str(RUTA_MODELO.relative_to(BASE)),
        "entrenado_el": info.get("entrenado_el"),
        "estudiantes_entrenamiento": info.get("n_estudiantes"),
        "exactitud_prueba": info.get("exactitud_prueba"),
        "clases": list(modelo.classes_),
        "features": list(modelo.feature_names_in_) if hasattr(modelo, "feature_names_in_") else info.get("features"),
    }


@app.get("/estado")
def ver_estado():
    return estado()


@app.post("/recargar")
def recargar():
    """Vuelve a leer el modelo del disco (útil después de reentrenar sin reiniciar el servicio)."""
    cargar_modelo()
    return estado()


class Encuesta(BaseModel):
    respuestas: dict[str, str | None]


@app.post("/predecir")
def predecir(encuesta: Encuesta):
    if modelo is None:
        raise HTTPException(status_code=503, detail=error_carga)
    if not encuesta.respuestas:
        # Todavía no hizo la encuesta: el modelo está listo pero no hay datos para predecir
        return {"nivel": None, "modelo": estado()}

    por_nivel = info.get("preguntas_por_nivel", {"A1": 2, "A2": 2, "B1": 2, "B2": 2, "C1": 1, "C2": 1})
    total_preguntas = sum(por_nivel.values())
    aciertos = {n: 0 for n in CEFR}
    no_la_conozco = 0
    desconocidas = []

    for palabra, respuesta in encuesta.respuestas.items():
        p = BANCO.get(palabra)
        if not p:
            continue
        if respuesta == p["traduccion"]:
            aciertos[p["cefr"]] += 1
        else:
            no_la_conozco += (not respuesta) or respuesta == NO_LA_CONOZCO
            desconocidas.append({"palabra": p["palabra"], "traduccion": p["traduccion"], "cefr": p["cefr"]})

    # Las mismas 7 variables con las que se entrenó el modelo (ver train_model.py)
    fila = {f"acierto_{n}": aciertos[n] / por_nivel[n] for n in CEFR}
    fila["pct_no_la_conozco"] = no_la_conozco / total_preguntas
    X = pd.DataFrame([fila])[list(modelo.feature_names_in_)] if hasattr(modelo, "feature_names_in_") else pd.DataFrame([fila])

    # Aquí es donde el modelo entrenado hace la predicción
    probs = modelo.predict_proba(X)[0]
    probabilidades = {clase: round(float(p), 3) for clase, p in zip(modelo.classes_, probs)}
    nivel = str(modelo.predict(X)[0])

    return {
        "nivel": nivel,
        "probabilidades": probabilidades,
        "aciertos": f"{sum(aciertos.values())}/{total_preguntas}",
        "palabrasDesconocidas": desconocidas,
        "modelo": {"tipo": type(modelo).__name__, "entrenado_el": info.get("entrenado_el"),
                   "exactitud_prueba": info.get("exactitud_prueba")},
    }


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": modelo is not None}
