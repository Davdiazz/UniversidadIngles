"""
Entrena un modelo de ML sencillo que predice el nivel de inglés a partir de la encuesta de vocabulario.

La encuesta tiene 10 preguntas sacadas de data/banco_vocabulario.csv (palabras del CEFR-J Vocabulary Profile):
    A1: 2   A2: 2   B1: 2   B2: 2   C1: 1   C2: 1
En cada pregunta el estudiante elige la traducción correcta o "No la conozco".

Como todavía no hay respuestas de usuarios reales, se generan estudiantes simulados con un modelo
de Teoría de Respuesta al Ítem (TRI): cada estudiante tiene una habilidad, y la probabilidad de conocer
una palabra baja cuanto más difícil es la palabra respecto a esa habilidad.

Uso:
    venv/bin/python train_model.py
Genera:
    data/estudiantes_simulados.csv  -> el dataset de entrenamiento
    models/modelo_nivel.json        -> pesos del modelo (los usa n8n en JavaScript)
    models/modelo_nivel.joblib      -> el mismo modelo en formato scikit-learn
    models/reporte.txt              -> métricas
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

CEFR = ["A1", "A2", "B1", "B2", "C1", "C2"]
PREGUNTAS_POR_NIVEL = {"A1": 2, "A2": 2, "B1": 2, "B2": 2, "C1": 1, "C2": 1}
CEFR_A_NIVEL_APP = {"A1": "principiante", "A2": "basico", "B1": "intermedio", "B2": "intermedio",
                    "C1": "avanzado", "C2": "avanzado"}
NIVELES_APP = ["principiante", "basico", "intermedio", "avanzado"]
FEATURES = [f"acierto_{c}" for c in CEFR] + ["pct_no_la_conozco"]

N_ESTUDIANTES = 6000
rng = np.random.default_rng(42)


def validar_banco() -> pd.DataFrame:
    """Comprueba que cada palabra del banco tenga en CEFR-J el mismo nivel que le asignamos."""
    banco = pd.read_csv(DATA_DIR / "banco_vocabulario.csv")
    vocab = pd.concat([pd.read_csv(DATA_DIR / "cefrj-vocabulary.csv"), pd.read_csv(DATA_DIR / "octanove-c1c2.csv")])
    for _, fila in banco.iterrows():
        niveles = set(vocab.loc[(vocab["headword"] == fila["palabra"]) & (vocab["pos"] == fila["pos"]), "CEFR"])
        assert fila["cefr"] in niveles, f"{fila['palabra']} no es {fila['cefr']} en CEFR-J ({niveles})"
    for nivel, n in PREGUNTAS_POR_NIVEL.items():
        assert (banco["cefr"] == nivel).sum() >= n
    return banco


def simular_estudiante(habilidad: float) -> list[float]:
    """Simula cómo responde la encuesta un estudiante con esa habilidad (0 = inicio de A1, 6 = C2 completo)."""
    aciertos = {c: [] for c in CEFR}
    no_la_conozco = 0
    for i, nivel in enumerate(CEFR):
        for _ in range(PREGUNTAS_POR_NIVEL[nivel]):
            # Dificultad de la palabra: inicio de su nivel + algo de variación (un estudiante de un nivel
            # conoce la mayoría de las palabras de ese nivel; no todas las B1 son igual de difíciles)
            dificultad = i + rng.normal(0, 0.35)
            p_conoce = 1 / (1 + np.exp(-1.8 * (habilidad - dificultad)))
            if rng.random() < p_conoce:
                correcta = rng.random() < 0.97          # a veces se equivoca aunque la sepa
            elif rng.random() < 0.55:
                correcta = False                        # marca "No la conozco"
                no_la_conozco += 1
            else:
                correcta = rng.random() < 0.25          # adivina entre 4 opciones
            aciertos[nivel].append(correcta)
    total = sum(PREGUNTAS_POR_NIVEL.values())
    return [float(np.mean(aciertos[c])) for c in CEFR] + [no_la_conozco / total]


def generar_dataset() -> pd.DataFrame:
    habilidades = rng.uniform(0, 6, N_ESTUDIANTES)
    filas = [simular_estudiante(h) for h in habilidades]
    df = pd.DataFrame(filas, columns=FEATURES)
    df["habilidad"] = habilidades.round(3)
    df["cefr_real"] = [CEFR[min(int(h), 5)] for h in habilidades]
    df["nivel"] = df["cefr_real"].map(CEFR_A_NIVEL_APP)
    return df


def main():
    validar_banco()
    df = generar_dataset()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "estudiantes_simulados.csv", index=False)
    print(f"Dataset: {len(df)} estudiantes simulados")
    print(df["nivel"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["nivel"], test_size=0.2, random_state=42, stratify=df["nivel"])

    # Modelo principal: regresión logística (sencilla y fácil de pasar a JavaScript para n8n)
    modelo = LogisticRegression(max_iter=2000, class_weight="balanced")
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    # Comparaciones para el informe
    arbol = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42).fit(X_train, y_train)
    total_aciertos = X_test[FEATURES[:6]].mul(list(PREGUNTAS_POR_NIVEL.values())).sum(axis=1)
    # Regla sin ML: 0-3 aciertos principiante, 4-5 básico, 6-8 intermedio, 9-10 avanzado
    regla = pd.cut(total_aciertos, [-1, 3, 5, 8, 10], labels=NIVELES_APP).astype(str)

    reporte = [
        f"Dataset: {len(df)} estudiantes simulados (80% entrenamiento / 20% prueba)",
        "",
        "Exactitud en el set de prueba:",
        f"  Regresión logística (modelo usado): {accuracy_score(y_test, pred):.3f}",
        f"  Árbol de decisión (profundidad 4):  {accuracy_score(y_test, arbol.predict(X_test)):.3f}",
        f"  Regla simple contando aciertos:     {accuracy_score(y_test, regla):.3f}",
        "",
        classification_report(y_test, pred, labels=NIVELES_APP, zero_division=0),
        "Matriz de confusión (filas = real, columnas = predicho) " + str(NIVELES_APP),
        str(confusion_matrix(y_test, pred, labels=NIVELES_APP)),
        "",
        "Reglas del árbol de decisión (para explicar el modelo):",
        export_text(arbol, feature_names=FEATURES),
    ]
    texto = "\n".join(reporte)
    print(texto)

    # Exportar pesos para usarlos en un nodo Code de n8n (JavaScript), sin servidor de Python
    MODELS_DIR.mkdir(exist_ok=True)
    pesos = {
        "tipo": "regresion_logistica_multinomial",
        "features": FEATURES,
        "clases": modelo.classes_.tolist(),
        "coeficientes": modelo.coef_.round(5).tolist(),
        "interceptos": modelo.intercept_.round(5).tolist(),
        "preguntas_por_nivel": PREGUNTAS_POR_NIVEL,
        "exactitud_prueba": round(float(accuracy_score(y_test, pred)), 3),
        "n_estudiantes": len(df),
        "entrenado_el": datetime.now().isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "modelo_nivel.json").write_text(json.dumps(pesos, indent=2))
    joblib.dump(modelo, MODELS_DIR / "modelo_nivel.joblib")
    (MODELS_DIR / "reporte.txt").write_text(texto)
    print(f"Modelo exportado a {MODELS_DIR / 'modelo_nivel.json'}")


if __name__ == "__main__":
    main()
