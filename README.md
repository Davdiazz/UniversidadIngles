# Tutor de inglés con IA + modelo ML que predice el nivel

1. Tu página (`Universidad/universidad.html`) le hace al usuario una **encuesta de 10 palabras** sacadas del dataset de vocabulario CEFR-J.
2. Las respuestas viajan junto a cada mensaje del chat al webhook `tutor-ingles`. En n8n, el nodo **"Predecir nivel (modelo ML)"**
   se las manda a `servicio_modelo.py`, que carga el modelo entrenado (`models/modelo_nivel.joblib`, regresión logística de
   scikit-learn) y devuelve el nivel: principiante, básico, intermedio o avanzado.
   **Si el modelo no está entrenado o el servicio está apagado, n8n no llama a Groq** y el chat muestra el motivo.
3. Ese nivel se pasa a **Groq**, que adapta cuánto inglés usa, el vocabulario y las correcciones, y refuerza las palabras que el usuario no supo.

```
universidad.html ──POST tutor-ingles { message, conversationHistory, encuesta: { respuestas } }──►
  n8n: Webhook1 → Predecir nivel (modelo ML) ──HTTP──► servicio_modelo.py (puerto 8001, modelo .joblib)
       → ¿Modelo disponible? ─ sí → Preparar mensajes y nivel → Armar prompt tutor → Llamar a Groq → Parsear respuesta
                              └ no → Avisar: modelo no disponible (no se llama a Groq)
  ◄── { response, nivel, prediccion: { nivel, probabilidades, aciertos, palabrasDesconocidas, modelo } }
```

Para ver si el modelo está cargado y cuándo se entrenó: abre `http://localhost:8001/estado`.

## Cómo correrlo (tu computador como servidor)

Todo corre en **un solo servidor** (`servidor.py`, puerto 8080) que carga una vez el modelo de nivel y Whisper:

| Ruta | Qué es |
|---|---|
| `/` | La página del tutor, ya configurada (no hay que pegar webhook) |
| `/api/tutor` | El chat: reenvía cada mensaje a n8n. **n8n nunca queda expuesto a internet** |
| `/modelo/estado`, `/modelo/predecir` | El modelo ML de nivel (`servicio_modelo.py`) |
| `/stt/transcribe` | El micrófono, voz a texto (`stt_service.py`) |
| `/api/estado` | Revisa de un vistazo modelo + n8n |

```bash
bash iniciar.sh           # local + internet (si configuraste NGROK_DOMAIN en .env)
bash iniciar.sh --local   # solo en este computador: http://localhost:8080
```
n8n se abre aparte, con el workflow **activo** (el servidor usa la URL de producción `/webhook/tutor-ingles`).

### Publicar en internet con ngrok (URL fija gratis)
1. Crea una cuenta en https://ngrok.com e instala ngrok (`sudo snap install ngrok`).
2. Copia tu authtoken del dashboard y ejecuta una sola vez: `ngrok config add-authtoken TU_TOKEN`
3. En el dashboard ve a **Domains** y reclama tu dominio gratis (algo como `xxxx.ngrok-free.app`).
4. `cp .env.example .env` y pon ese dominio en `NGROK_DOMAIN=`.
5. `bash iniciar.sh` → comparte `https://TU_DOMINIO`. Funciona en celulares (con micrófono, porque es https).

### Cada estudiante tiene su propia prueba
La encuesta, el nivel y el identificador de sesión se guardan en el `localStorage` **del navegador de cada persona**
y viajan con cada mensaje. El servidor y n8n no guardan estado compartido, así que varias personas pueden usarlo al
mismo tiempo sin mezclarse. El botón **Encuesta de nivel** permite repetirla.
Para no gastar tu cuota de Groq, cada visitante puede mandar máximo `LIMITE_MENSAJES_POR_MINUTO` mensajes (20 por defecto).

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
venv/bin/python n8n/build_workflows.py  # genera n8n/workflow_tutor.json
venv/bin/uvicorn servicio_modelo:app --port 8001   # sirve el modelo entrenado a n8n
```

Prueba de que el tutor depende del modelo: apaga `servicio_modelo.py` (o borra `models/modelo_nivel.joblib`)
y escribe en el chat: aparece "⚠️ El modelo de machine learning que calcula tu nivel NO está disponible…".

Si reentrenas, reinicia `servicio_modelo.py` (o llama a `POST /recargar`). No hace falta reimportar el workflow.
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
