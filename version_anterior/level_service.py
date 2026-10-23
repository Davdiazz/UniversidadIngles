"""
Servicio que detecta el nivel de inglés del estudiante (lo llama n8n).

Combina dos fuentes:
  1. Preguntas de opción múltiple (placement_test.json), ligadas a puntos gramaticales del CEFR-J.
  2. Respuestas escritas, evaluadas con el modelo entrenado en train_model.py.

Ejecutar:
    venv/bin/uvicorn level_service:app --host 0.0.0.0 --port 8001

Endpoints:
    GET  /placement-test   -> preguntas para mostrar en el frontend (sin las respuestas correctas)
    POST /evaluate         -> recibe las respuestas y devuelve nivel + temas a repasar
    POST /predict-level    -> nivel CEFR de un texto suelto (útil para seguir el progreso en el chat)
    GET  /health
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import CEFR_A_NIVEL_APP, CEFR_ORDEN, DATA_DIR, cargar_vocabulario, perfil_vocabulario, tokenizar

BASE = Path(__file__).parent
modelo = joblib.load(BASE / "models" / "nivel_model.joblib")
TEST = json.loads((BASE / "placement_test.json").read_text())
PREGUNTAS = {p["id"]: p for p in TEST["opcion_multiple"]}
VOCAB = cargar_vocabulario()

# Palabras muy comunes en español: si el estudiante responde en español, no podemos evaluar su inglés escrito
STOPWORDS_ES = {"el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es", "mi", "yo", "con", "por",
                "para", "pero", "muy", "fue", "fui", "tengo", "hice", "se", "lo", "del", "al", "mas", "más", "porque"}


def _cargar_temario() -> dict[str, list[str]]:
    """Puntos gramaticales del CEFR-J Grammar Profile agrupados por nivel (A1..B2)."""
    g = pd.read_csv(DATA_DIR / "cefrj-grammar.csv")
    nivel = g["CEFR-J Level"].fillna(g["GSELO"]).astype(str).str[:2]
    temario: dict[str, list[str]] = {}
    for n, item in zip(nivel, g["Grammatical Item"]):
        if n in CEFR_ORDEN and isinstance(item, str) and item not in temario.setdefault(n, []):
            temario[n].append(item)
    return temario


TEMARIO = _cargar_temario()

app = FastAPI(title="Detector de nivel de inglés")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Evaluacion(BaseModel):
    # {"q1": "is", "q2": "drinks", ...}
    opcion_multiple: dict[str, str] = {}
    # {"w1": "I am Marlon...", "w2": "...", "w3": "..."}
    abiertas: dict[str, str] = {}


class Texto(BaseModel):
    text: str


def nivel_por_test(respuestas: dict[str, str]) -> tuple[float | None, list[dict]]:
    """Sube nivel por nivel mientras el estudiante acierte al menos la mitad de las preguntas de ese nivel."""
    if not respuestas:
        return None, []
    falladas = []
    aciertos_por_nivel: dict[str, list[bool]] = {}
    for id_, pregunta in PREGUNTAS.items():
        dada = (respuestas.get(id_) or "").strip().lower()
        correcta = dada == pregunta["correcta"].lower()
        aciertos_por_nivel.setdefault(pregunta["nivel_cefr"], []).append(correcta)
        if not correcta:
            falladas.append({
                "id": id_, "nivel_cefr": pregunta["nivel_cefr"], "tema": pregunta["tema"],
                "pregunta": pregunta["pregunta"], "respuesta_dada": dada or "(sin responder)",
                "respuesta_correcta": pregunta["correcta"],
            })
    nivel = 0.0  # A1 por defecto
    for i, cefr in enumerate(CEFR_ORDEN):
        aciertos = aciertos_por_nivel.get(cefr)
        if not aciertos:
            continue
        proporcion = sum(aciertos) / len(aciertos)
        if proporcion >= 0.5:
            nivel = i + (proporcion - 0.5)  # un poco más alto si acertó todas
        else:
            break
    return nivel, falladas


def nivel_por_escritura(textos: list[str]) -> dict:
    texto = " ".join(t.strip() for t in textos if t and t.strip())
    tokens = tokenizar(texto)
    if len(tokens) < 5:
        return {"indice": 0.0, "cefr": "A1", "motivo": "Respuestas muy cortas o vacías"}
    if sum(t in STOPWORDS_ES for t in tokens) / len(tokens) > 0.25:
        return {"indice": 0.0, "cefr": "A1", "motivo": "Respondió mayormente en español"}

    frases = [f for f in re.split(r"(?<=[.!?])\s+|\n+", texto) if len(tokenizar(f)) >= 2] or [texto]
    probas = modelo.predict_proba(frases)
    clases = list(modelo.classes_)
    # Nivel esperado por frase (promedio ponderado por probabilidad), luego promedio de todas las frases
    indices = probas @ np.array([CEFR_ORDEN.index(c) for c in clases])
    indice = float(np.mean(indices))
    perfil = perfil_vocabulario(texto, VOCAB)
    return {
        "indice": indice,
        "cefr": CEFR_ORDEN[int(round(indice))],
        "frases_evaluadas": len(frases),
        "vocabulario_por_nivel": perfil["conteo"],
        "palabras_avanzadas_usadas": sorted(set(perfil["palabras"]["B2"] + perfil["palabras"]["C1"] + perfil["palabras"]["C2"])),
    }


@app.get("/placement-test")
def placement_test():
    return {
        "opcion_multiple": [{k: v for k, v in p.items() if k not in ("correcta", "gramatica_cefrj")} for p in TEST["opcion_multiple"]],
        "abiertas": TEST["abiertas"],
    }


@app.post("/evaluate")
def evaluate(datos: Evaluacion):
    indice_test, falladas = nivel_por_test(datos.opcion_multiple)
    escritura = nivel_por_escritura(list(datos.abiertas.values()))

    # Mitad test, mitad escritura. Si falta uno, usamos el otro.
    if indice_test is None:
        indice = escritura["indice"]
    else:
        indice = 0.5 * indice_test + 0.5 * escritura["indice"]
    cefr = CEFR_ORDEN[min(int(round(indice)), 5)]
    nivel_app = CEFR_A_NIVEL_APP[cefr]

    # Temas a repasar: primero lo que falló en su nivel o por debajo (son bases que le faltan)
    falladas.sort(key=lambda f: CEFR_ORDEN.index(f["nivel_cefr"]))
    temas_a_repasar = list(dict.fromkeys(f["tema"] for f in falladas if CEFR_ORDEN.index(f["nivel_cefr"]) <= CEFR_ORDEN.index(cefr) + 1))

    # Lo que sigue: puntos gramaticales del CEFR-J del siguiente nivel
    siguiente = CEFR_ORDEN[min(CEFR_ORDEN.index(cefr) + 1, 3)]  # el Grammar Profile llega hasta B2
    return {
        "nivel": nivel_app,
        "cefr_estimado": cefr,
        "detalle": {
            "indice_test": None if indice_test is None else round(indice_test, 2),
            "aciertos_test": f"{len(PREGUNTAS) - len(falladas)}/{len(PREGUNTAS)}" if datos.opcion_multiple else None,
            "escritura": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in escritura.items()},
        },
        "preguntas_falladas": falladas,
        "temas_a_repasar": temas_a_repasar,
        "siguientes_contenidos_cefrj": TEMARIO.get(siguiente, [])[:8],
        "respuestas_abiertas": datos.abiertas,
    }


@app.post("/predict-level")
def predict_level(datos: Texto):
    escritura = nivel_por_escritura([datos.text])
    return {"cefr": escritura["cefr"], "nivel": CEFR_A_NIVEL_APP[escritura["cefr"]], "detalle": escritura}


@app.get("/health")
def health():
    return {"status": "ok"}
