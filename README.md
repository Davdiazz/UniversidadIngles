# Tutor de inglés con IA + modelo ML que predice el nivel

1. Tu página (`Universidad/universidad.html`) le hace al usuario una **encuesta de 10 palabras** sacadas del dataset de vocabulario CEFR-J.
2. Las respuestas viajan junto a cada mensaje del chat al webhook `tutor-ingles`. En n8n, el nodo **"Predecir nivel (modelo ML)"**
   (regresión logística entrenada por nosotros) calcula el nivel: principiante, básico, intermedio o avanzado.
3. Ese nivel se pasa a **Groq**, que adapta cuánto inglés usa, el vocabulario y las correcciones, y refuerza las palabras que el usuario no supo.

```
universidad.html ──POST tutor-ingles { message, conversationHistory, encuesta: { respuestas } }──►
  n8n: Webhook1 → Predecir nivel (modelo ML) → Preparar mensajes y nivel → Armar prompt tutor → Llamar a Groq → Parsear respuesta
  ◄── { response, nivel, prediccion: { nivel, probabilidades, aciertos, palabrasDesconocidas } }
```

No hace falta un servidor de Python: el modelo se entrena en Python y sus pesos quedan dentro del nodo Code de n8n.

## Dataset

| Archivo | Origen | Uso |
|---|---|---|
| `data/cefrj-vocabulary.csv` (7.798 palabras A1–B2) | [CEFR-J Vocabulary Profile 1.5](https://github.com/openlanguageprofiles/olp-en-cefrj), Tono Lab, Tokyo Univ. of Foreign Studies | Nivel oficial de cada palabra |
| `data/octanove-c1c2.csv` (2.136 palabras C1–C2) | [Octanove Vocabulary Profile](https://github.com/openlanguageprofiles/olp-en-cefrj) (CC BY-SA 4.0) | Nivel de palabras avanzadas |
| `data/banco_vocabulario.csv` (72 palabras, 12 por nivel) | Seleccionadas del CEFR-J + traducción al español | Preguntas de la encuesta. Se evitaron palabras parecidas al español (*planet*, *intelligent*) porque serían demasiado fáciles. `train_model.py` verifica que el nivel de cada palabra coincida con CEFR-J |
| `data/estudiantes_simulados.csv` (6.000 filas) | Generado por `train_model.py` | Entrenamiento del modelo |

### ¿Por qué estudiantes simulados?
Para entrenar se necesitan encuestas respondidas por personas de nivel conocido, y todavía no existen.
Se simulan con **Teoría de Respuesta al Ítem**: cada estudiante tiene una habilidad (de A1 a C2) y la
probabilidad de conocer una palabra baja cuanto más difícil es respecto a esa habilidad.
Si no la conoce, a veces marca "No la conozco" y a veces adivina (25% de acertar).
Cuando haya usuarios reales, sus respuestas (con un nivel verificado) reemplazan a las simuladas y se reentrena igual.

## El modelo

**Entradas (7 variables):** % de aciertos en palabras A1, A2, B1, B2, C1, C2 y % de "No la conozco".
**Salida:** el nivel en la app, con su probabilidad.

Resultados en el 20% de prueba (`models/reporte.txt`):

| Modelo | Exactitud |
|---|---|
| **Regresión logística (el que se usa)** | **71.7%** |
| Árbol de decisión (profundidad 4) | 65.4% |
| Regla sin ML (contar aciertos) | 69.8% |

Los errores casi siempre son entre niveles vecinos (por ejemplo básico ↔ intermedio). Con solo 10 preguntas hay un
límite a lo que se puede saber; por eso el tutor también tiene la instrucción de ajustarse si nota otro nivel en la conversación.

Ejemplos (salida del modelo):

| El usuario sabe las palabras hasta… | Aciertos | Nivel predicho |
|---|---|---|
| ninguna | 0/10 | principiante (98%) |
| A1 | 2/10 | principiante (85%) |
| A2 | 4/10 | básico (61%) |
| B1 | 6/10 | intermedio (62%) |
| B2 | 8/10 | intermedio (62%) |
| C1 | 9/10 | avanzado (77%) |
| C2 | 10/10 | avanzado (94%) |

## Cómo entrenar

```bash
cd universidadIA
python3 -m venv venv
venv/bin/pip install -r requirements.txt
./download_data.sh                    # descarga el vocabulario CEFR-J
venv/bin/python train_model.py        # simula estudiantes, entrena, guarda models/modelo_nivel.json
venv/bin/python n8n/build_workflows.py  # mete el banco y los pesos del modelo en los workflows
```

Si cambias el banco de palabras o reentrenas, corre los dos últimos comandos y vuelve a importar `n8n/workflow_tutor.json`.
(Si cambias el banco, también hay que actualizar la constante `BANCO_VOCAB` en `universidad.html`.)

## Cómo ponerlo en n8n

1. **Workflows → Import from file → `n8n/workflow_tutor.json`**.
2. **Desactiva o borra tu workflow viejo del tutor**, porque los dos usan la ruta `tutor-ingles`.
3. En el nodo *Llamar a Groq*, elige tu credencial **Groq account**.
4. Pruebas: clic en **Execute workflow** y pega en la página la URL de prueba (`.../webhook-test/tutor-ingles`).
   Cada clic sirve para **un solo mensaje**.
   Uso normal: **activa** el workflow y pega la URL de producción (`.../webhook/tutor-ingles`).

## Versión anterior
`version_anterior/` tiene el enfoque previo (modelo entrenado con frases y test de gramática). Ya no se usa.
