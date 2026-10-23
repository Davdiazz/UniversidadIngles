"""
Prueba el modelo ML de nivel sin n8n ni Groq: haces la encuesta en la terminal y ves la predicción.

Usa models/modelo_nivel.json y hace exactamente el mismo cálculo que el nodo "Predecir nivel (modelo ML)" de n8n.

Uso:
    python predecir.py               -> encuesta interactiva de 10 palabras
    python predecir.py --ejemplos    -> predicciones para perfiles típicos (sabe hasta A1, A2, ...)
"""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
CEFR = ["A1", "A2", "B1", "B2", "C1", "C2"]
NO_LA_CONOZCO = "No la conozco"

MODELO = json.loads((BASE / "models" / "modelo_nivel.json").read_text())
with open(BASE / "data" / "banco_vocabulario.csv", encoding="utf-8") as f:
    BANCO = list(csv.DictReader(f))


def predecir(aciertos: dict, no_la_conozco: int) -> dict:
    """aciertos = {"A1": 2, "A2": 1, ...} -> {nivel: probabilidad}, igual que en n8n."""
    por_nivel = MODELO["preguntas_por_nivel"]
    total = sum(por_nivel.values())
    x = np.array([aciertos.get(n, 0) / por_nivel[n] for n in CEFR] + [no_la_conozco / total])
    puntajes = np.array(MODELO["coeficientes"]) @ x + np.array(MODELO["interceptos"])
    exps = np.exp(puntajes - puntajes.max())
    probs = exps / exps.sum()
    return dict(sorted(zip(MODELO["clases"], probs.round(3)), key=lambda kv: -kv[1]))


def mostrar(probs: dict):
    for nivel, p in probs.items():
        print(f"  {nivel:<13} {p:6.1%}  {'█' * int(p * 30)}")


def encuesta():
    """Elige palabras del banco por nivel (como la página) y pregunta con 4 opciones + "No la conozco"."""
    aciertos = {n: 0 for n in CEFR}
    no_la_conozco = 0
    desconocidas = []
    preguntas = [p for n, k in MODELO["preguntas_por_nivel"].items()
                 for p in random.sample([b for b in BANCO if b["cefr"] == n], k)]
    for i, p in enumerate(preguntas, 1):
        distractores = random.sample([b["traduccion"] for b in BANCO if b["palabra"] != p["palabra"]], 3)
        opciones = random.sample(distractores + [p["traduccion"]], 4) + [NO_LA_CONOZCO]
        print(f"\n{i}/10  ¿Qué significa \"{p['palabra']}\"?")
        for j, o in enumerate(opciones, 1):
            print(f"   {j}) {o}")
        while (r := input("   > ").strip()) not in {str(j) for j in range(1, 6)}:
            print("   Escribe un número del 1 al 5")
        respuesta = opciones[int(r) - 1]
        if respuesta == p["traduccion"]:
            aciertos[p["cefr"]] += 1
        else:
            no_la_conozco += respuesta == NO_LA_CONOZCO
            desconocidas.append(f"{p['palabra']} ({p['traduccion']}, {p['cefr']})")

    print(f"\nAciertos: {sum(aciertos.values())}/10  por nivel: {aciertos}  'No la conozco': {no_la_conozco}")
    print("Predicción del modelo:")
    mostrar(predecir(aciertos, no_la_conozco))
    if desconocidas:
        print("Palabras a reforzar:", ", ".join(desconocidas))


def ejemplos():
    """Estudiante que sabe todas las palabras hasta cierto nivel y marca "No la conozco" en el resto."""
    for hasta in [None] + CEFR:
        sabe = CEFR[:CEFR.index(hasta) + 1] if hasta else []
        aciertos = {n: MODELO["preguntas_por_nivel"][n] for n in sabe}
        no_sabe = 10 - sum(aciertos.values())
        print(f"\nSabe hasta {hasta or 'ninguna'} ({10 - no_sabe}/10):")
        mostrar(predecir(aciertos, no_sabe))


if __name__ == "__main__":
    ejemplos() if "--ejemplos" in sys.argv else encuesta()
