"""
Entrena el modelo que estima el nivel de inglés (CEFR) de un texto escrito por el estudiante.

Dataset principal: edesaras/CEFR-Sentence-Level-Annotations (Hugging Face, licencia MIT)
    10.004 frases en inglés etiquetadas del 1 (A1) al 6 (C2) por dos profesores.
Dataset de apoyo: CEFR-J Vocabulary Profile + Octanove C1/C2 (lista de palabras por nivel).

Uso:
    venv/bin/python train_model.py
Genera:
    models/nivel_model.joblib   -> el modelo entrenado
    models/reporte.txt          -> métricas de evaluación
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from features import CEFR_A_NIVEL_APP, CEFR_ORDEN, DATA_DIR, NIVELES_APP, RasgosLinguisticos

MODELS_DIR = Path(__file__).parent / "models"


def cargar_split(nombre: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"sentences_{nombre}.parquet")
    # Promediamos a los dos profesores y redondeamos: 1..6 -> A1..C2
    promedio = ((df["Annotator I"] + df["Annotator II"]) / 2).round().astype(int).clip(1, 6)
    df["cefr"] = promedio.map(lambda n: CEFR_ORDEN[n - 1])
    # El texto viene tokenizado con espacios ("you 're") -> lo dejamos más natural
    df["text"] = df["text"].str.replace(r" (?=[.,!?;:'])", "", regex=True).str.replace(" n't", "n't")
    return df


def construir_modelo() -> Pipeline:
    rasgos = FeatureUnion([
        ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("letras", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)),
        ("linguisticos", Pipeline([("calc", RasgosLinguisticos()), ("escala", StandardScaler())])),
    ])
    clasificador = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")
    return Pipeline([("rasgos", rasgos), ("clf", clasificador)])


def a_nivel_app(etiquetas) -> list[str]:
    return [CEFR_A_NIVEL_APP[e] for e in etiquetas]


def main():
    train = pd.concat([cargar_split("train"), cargar_split("validation")])
    test = cargar_split("test")
    print(f"Entrenando con {len(train)} frases, evaluando con {len(test)}")
    print(train["cefr"].value_counts().sort_index().to_string())

    modelo = construir_modelo()
    modelo.fit(train["text"], train["cefr"])

    pred = modelo.predict(test["text"])
    real = test["cefr"].tolist()

    # Exactitud "a un nivel de distancia": útil porque A2 vs B1 es una frontera difusa
    idx = {nivel: i for i, nivel in enumerate(CEFR_ORDEN)}
    adyacente = np.mean([abs(idx[p] - idx[r]) <= 1 for p, r in zip(pred, real)])

    reporte = []
    reporte.append(f"Exactitud CEFR exacta (6 niveles): {accuracy_score(real, pred):.3f}")
    reporte.append(f"Exactitud CEFR +-1 nivel:          {adyacente:.3f}")
    reporte.append(f"Exactitud en los 4 niveles de la app: {accuracy_score(a_nivel_app(real), a_nivel_app(pred)):.3f}")
    reporte.append("\nReporte por nivel de la app:")
    reporte.append(classification_report(a_nivel_app(real), a_nivel_app(pred), labels=NIVELES_APP, zero_division=0))
    reporte.append("Matriz de confusión (filas = real, columnas = predicho) " + str(NIVELES_APP))
    reporte.append(str(confusion_matrix(a_nivel_app(real), a_nivel_app(pred), labels=NIVELES_APP)))
    texto_reporte = "\n".join(reporte)
    print(texto_reporte)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(modelo, MODELS_DIR / "nivel_model.joblib")
    (MODELS_DIR / "reporte.txt").write_text(texto_reporte)
    print(f"\nModelo guardado en {MODELS_DIR / 'nivel_model.joblib'}")


if __name__ == "__main__":
    main()
