"""
Utilidades compartidas entre el entrenamiento (train_model.py) y el servicio (level_service.py).

Aquí vive todo lo que convierte un texto en "señales" que el modelo puede entender:
- Qué tan largas son las frases y las palabras.
- Qué porcentaje de las palabras pertenece a cada nivel CEFR según la lista CEFR-J.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DATA_DIR = Path(__file__).parent / "data"

CEFR_ORDEN = ["A1", "A2", "B1", "B2", "C1", "C2"]

# Los 4 niveles de la app (los mismos que usa el workflow de n8n) agrupando los 6 del CEFR.
CEFR_A_NIVEL_APP = {
    "A1": "principiante",
    "A2": "basico",
    "B1": "intermedio",
    "B2": "intermedio",
    "C1": "avanzado",
    "C2": "avanzado",
}
NIVELES_APP = ["principiante", "basico", "intermedio", "avanzado"]

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenizar(texto: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(texto)]


def cargar_vocabulario() -> dict[str, str]:
    """Diccionario palabra -> nivel CEFR (A1..C2) usando CEFR-J (A1-B2) + Octanove (C1-C2)."""
    vocab: dict[str, str] = {}
    for archivo in ["cefrj-vocabulary.csv", "octanove-c1c2.csv"]:
        df = pd.read_csv(DATA_DIR / archivo)
        for headword, nivel in zip(df["headword"], df["CEFR"]):
            if not isinstance(headword, str) or nivel not in CEFR_ORDEN:
                continue
            # Algunas entradas traen variantes: "a.m./A.M./am/AM"
            for variante in headword.split("/"):
                palabra = variante.strip().lower()
                # Si una palabra aparece en varios niveles, nos quedamos con el más bajo
                if palabra and (palabra not in vocab or CEFR_ORDEN.index(nivel) < CEFR_ORDEN.index(vocab[palabra])):
                    vocab[palabra] = nivel
    return vocab


def _lema_simple(palabra: str, vocab: dict[str, str]) -> str | None:
    """Busca la palabra en el vocabulario quitando terminaciones comunes (plurales, -ed, -ing)."""
    if palabra in vocab:
        return palabra
    for sufijo, reemplazo in [("ies", "y"), ("es", ""), ("s", ""), ("ied", "y"), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("er", ""), ("est", "")]:
        if palabra.endswith(sufijo):
            candidato = palabra[: -len(sufijo)] + reemplazo
            if candidato in vocab:
                return candidato
    return None


def perfil_vocabulario(texto: str, vocab: dict[str, str]) -> dict:
    """Cuenta cuántas palabras del texto hay de cada nivel CEFR."""
    conteo = {nivel: 0 for nivel in CEFR_ORDEN}
    desconocidas = 0
    palabras_por_nivel: dict[str, list[str]] = {nivel: [] for nivel in CEFR_ORDEN}
    for palabra in tokenizar(texto):
        lema = _lema_simple(palabra, vocab)
        if lema is None:
            desconocidas += 1
            continue
        nivel = vocab[lema]
        conteo[nivel] += 1
        palabras_por_nivel[nivel].append(palabra)
    return {"conteo": conteo, "desconocidas": desconocidas, "palabras": palabras_por_nivel}


class RasgosLinguisticos(BaseEstimator, TransformerMixin):
    """Transformer de scikit-learn que calcula rasgos numéricos para cada texto."""

    def __init__(self):
        self.vocab = None

    def fit(self, X, y=None):
        self.vocab = cargar_vocabulario()
        return self

    def transform(self, X):
        filas = []
        for texto in X:
            tokens = tokenizar(texto)
            n = max(len(tokens), 1)
            perfil = perfil_vocabulario(texto, self.vocab)
            proporciones = [perfil["conteo"][nivel] / n for nivel in CEFR_ORDEN]
            nivel_max = max((i for i, nivel in enumerate(CEFR_ORDEN) if perfil["conteo"][nivel] > 0), default=0)
            filas.append([
                np.log1p(len(tokens)),                         # longitud de la frase
                np.mean([len(t) for t in tokens]) if tokens else 0,  # largo promedio de palabra
                len(set(tokens)) / n,                           # variedad de vocabulario
                perfil["desconocidas"] / n,                     # palabras fuera de la lista (suelen ser raras)
                nivel_max / 5,                                  # palabra más difícil usada
                texto.count(",") / n,                           # frases con subordinadas
                *proporciones,
            ])
        return np.array(filas)
