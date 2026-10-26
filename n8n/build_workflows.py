"""
Genera el workflow del tutor para n8n. El nivel lo calcula el modelo entrenado (servicio_modelo.py):
n8n le manda las respuestas de la encuesta por HTTP y, si el modelo no está disponible, avisa y no llama a Groq.
Correr después de train_model.py (si reentrenas, vuelve a correr esto y reimporta):
    venv/bin/python n8n/build_workflows.py
"""

import json
import os
import uuid
from pathlib import Path

BASE = Path(__file__).parent.parent
AQUI = Path(__file__).parent

# Desde qué páginas se puede llamar al webhook (CORS). "*" = cualquiera, así funciona abriendo
# la página en local (http://localhost:5500) y también en GitHub Pages.
ORIGEN_FRONTEND = "*"
# Dónde corre el modelo ML entrenado. Por defecto, dentro de servidor.py (puerto 8080, ruta /modelo).
# Si lo corres solo (uvicorn servicio_modelo:app --port 8001) usa MODELO_URL=http://localhost:8001
# Si n8n corre en Docker usa: MODELO_URL=http://host.docker.internal:8080/modelo python n8n/build_workflows.py
MODELO_URL = os.environ.get("MODELO_URL", "http://localhost:8080/modelo")
GROQ_CREDENCIAL = {"groqApi": {"id": "MIar86booTIHpgaI", "name": "Groq account"}}


def nodo(nombre, tipo, version, x, params, y=0, **extra):
    return {"parameters": params, "type": tipo, "typeVersion": version, "position": [x, y],
            "id": str(uuid.uuid4()), "name": nombre, **extra}


def webhook(nombre, path, metodo, y=0):
    return nodo(nombre, "n8n-nodes-base.webhook", 2.1, 0, {
        "httpMethod": metodo, "path": path, "responseMode": "responseNode",
        "options": {"allowedOrigins": ORIGEN_FRONTEND},
    }, y=y, webhookId=str(uuid.uuid4()))


def code(nombre, x, js, y=0):
    return nodo(nombre, "n8n-nodes-base.code", 2, x, {"jsCode": js.strip()}, y=y)


def responder(nombre, x, y=0):
    return nodo(nombre, "n8n-nodes-base.respondToWebhook", 1.4, x,
                {"respondWith": "json", "responseBody": "={{ $json }}", "options": {}}, y=y)


def cadena(*nombres):
    return {a: {"main": [[{"node": b, "type": "main", "index": 0}]]} for a, b in zip(nombres, nombres[1:])}


# ═══════════════════════════════════════════════════════════════════
# MODELO ML: n8n le pregunta el nivel a servicio_modelo.py (Python + scikit-learn)
# ═══════════════════════════════════════════════════════════════════

# Cuerpo que se manda al modelo: solo las respuestas de la encuesta
BODY_MODELO = "={{ JSON.stringify({ respuestas: ($json.body && $json.body.encuesta && $json.body.encuesta.respuestas) || {} }) }}"

JS_RESULTADO_MODELO = r"""
// Revisa qué contestó el servicio del modelo ML (servicio_modelo.py)
const r = $input.first().json;
const body = $('Webhook1').first().json.body || $('Webhook1').first().json;

const respondio = r && Object.prototype.hasOwnProperty.call(r, 'nivel') && !r.error && !r.detail;
if (!respondio) {
  let motivo = r && (r.detail || (r.error && (r.error.message || r.error.description || JSON.stringify(r.error))));
  if (!motivo || /ECONNREFUSED|connect/i.test(motivo)) {
    motivo = 'El servidor del modelo no está encendido (servidor.py en el puerto 8080). ' + (motivo || '');
  }
  return [{ json: { body, prediccion: null, modeloOk: false, motivo } }];
}

// r.nivel === null -> el usuario todavía no hizo la encuesta
return [{ json: { body, prediccion: r.nivel ? r : null, modeloOk: true } }];
"""

JS_MODELO_NO_DISPONIBLE = r"""
const motivo = $input.first().json.motivo;
return [{
  json: {
    response: "⚠️ El modelo de machine learning que calcula tu nivel NO está disponible, así que no puedo empezar la tutoría.\n\n" +
              "Motivo: " + motivo + "\n\n" +
              "Solución: en la carpeta universidadIA ejecuta\n  python train_model.py   (si no está entrenado)\n  bash iniciar.sh",
    nivel: null,
    prediccion: null,
    modeloError: motivo
  }
}];
"""

SI_MODELO_OK = {
    "conditions": {
        "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "loose"},
        "conditions": [{
            "id": str(uuid.uuid4()), "leftValue": "={{ $json.modeloOk }}", "rightValue": "",
            "operator": {"type": "boolean", "operation": "true", "singleValue": True},
        }],
        "combinator": "and",
    },
    "options": {},
}

# ═══════════════════════════════════════════════════════════════════
# WORKFLOW: Tutor (tu workflow original + modelo ML de la encuesta)
# ═══════════════════════════════════════════════════════════════════

JS_PREPARAR = r"""
const inputData = $input.all();

// Niveles válidos que el frontend puede enviar en body.level
const NIVELES_VALIDOS = ['principiante', 'basico', 'intermedio', 'avanzado'];

return inputData.map(item => {
  const body = item.json.body || item.json;
  const prediccion = item.json.prediccion;  // viene del nodo "Resultado del modelo" (servicio_modelo.py)
  const userMessage = body.message || "";
  const history = body.conversationHistory || [];

  // Nivel: el que predijo el modelo ML con la encuesta. Si no hay encuesta, el que mande la página
  // en body.level, y si tampoco, 'principiante' para no asumir un nivel que no tiene.
  let nivel = ((prediccion && prediccion.nivel) || body.level || '').toLowerCase().trim();
  if (!NIVELES_VALIDOS.includes(nivel)) {
    nivel = 'principiante';
  }

  // Palabras que el estudiante no supo en la encuesta
  const palabrasDesconocidas = ((prediccion && prediccion.palabrasDesconocidas) || []).slice(0, 10);

  const messages = [];

  const recentHistory = history.slice(-10).filter(msg => {
    return msg.content &&
           !msg.content.includes('={{ $json') &&
           !msg.content.includes('={{') &&
           msg.content.trim().length > 0;
  });

  recentHistory.forEach(msg => {
    messages.push({
      role: msg.role === 'user' ? 'user' : 'assistant',
      content: msg.content
    });
  });

  messages.push({
    role: 'user',
    content: userMessage
  });

  return {
    json: {
      messages: messages,
      nivel: nivel,
      aciertosEncuesta: prediccion ? prediccion.aciertos : null,
      palabrasDesconocidas: palabrasDesconocidas,
      esPrimerMensaje: recentHistory.length === 0
    }
  };
});
"""

JS_ARMAR_PROMPT = r"""
const inputData = $input.first().json;
const messages = inputData.messages || [];
const nivel = inputData.nivel;
const esPrimerMensaje = inputData.esPrimerMensaje;
const palabrasDesconocidas = inputData.palabrasDesconocidas || [];
const aciertosEncuesta = inputData.aciertosEncuesta;

const now = new Date();
const fecha = now.toLocaleDateString('es-CO', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });

// Instrucciones específicas por nivel: cuánto inglés meter, qué tan
// simple debe ser el vocabulario, y qué tanto se explica en español.
const PERFILES_NIVEL = {
  principiante: `NIVEL: PRINCIPIANTE (A1)
- Habla principalmente en ESPAÑOL. Introduce inglés en frases muy cortas y simples (2-5 palabras), siempre traducidas de inmediato.
- Vocabulario: solo palabras y frases del día a día (saludos, números, colores, familia, comida).
- Corrige errores con mucha suavidad y siempre en español, mostrando la forma correcta.
- Repite palabras clave en inglés varias veces a lo largo de la conversación para reforzar.
- Nunca uses tiempos verbales complejos ni vocabulario abstracto.`,
  basico: `NIVEL: BÁSICO (A2)
- Mezcla español e inglés en proporción aprox. 60% español / 40% inglés.
- Usa frases completas simples en inglés (presente simple, presente continuo), siempre con la traducción al lado la primera vez que aparecen.
- Introduce vocabulario de temas cotidianos: rutinas, trabajo, viajes básicos, compras.
- Corrige errores explicando brevemente la regla en español.`,
  intermedio: `NIVEL: INTERMEDIO (B1-B2)
- Mezcla equilibrada 50% español / 50% inglés. Puedes sostener partes completas de la conversación en inglés.
- Usa tiempos verbales variados (pasado, futuro, condicional simple) y vocabulario de temas generales, trabajo, opiniones.
- Corrige errores en inglés cuando sea posible, y aclara en español solo si el punto gramatical es complejo.
- Anima al usuario a construir oraciones más largas y a dar su opinión en inglés.`,
  avanzado: `NIVEL: AVANZADO (C1-C2)
- Habla PRINCIPALMENTE en inglés (80-90%). Usa español solo para aclarar matices muy específicos o si el usuario se traba mucho.
- Vocabulario amplio: modismos, phrasal verbs, registro formal e informal, temas abstractos.
- Corrige errores sutiles de manera directa en inglés (naturalidad, tono, matices), no solo gramática básica.
- Empuja al usuario a debatir, argumentar y usar estructuras más sofisticadas.`
};

// Palabras que el estudiante no supo en la encuesta: el tutor las va metiendo en la conversación
let seccionVocabulario = '';
if (palabrasDesconocidas.length > 0) {
  seccionVocabulario = `
═══════════════════════════════
VOCABULARIO A REFORZAR (de la encuesta)
═══════════════════════════════
El estudiante NO conocía estas palabras: ${palabrasDesconocidas.map(p => `${p.palabra} (${p.traduccion}, ${p.cefr})`).join(', ')}.
- Empieza por las de nivel más bajo y usa máximo 1-2 por respuesta, de forma natural dentro de la conversación, con su traducción entre paréntesis.
- Más adelante, haz preguntas que lo animen a usarlas él mismo.
`;
}

const systemPrompt = `Eres un tutor de inglés conversacional, cálido y paciente, que enseña mezclando español e inglés según el nivel del estudiante. Tu objetivo es que la persona practique inglés real en una conversación natural, no en ejercicios mecánicos.

FECHA: ${fecha}

${aciertosEncuesta ? `El nivel del estudiante lo predijo un modelo de machine learning a partir de una encuesta de vocabulario (acertó ${aciertosEncuesta} palabras). Si durante la conversación notas claramente que su nivel real es más alto o más bajo, ajústate poco a poco.` : ''}

${PERFILES_NIVEL[nivel]}
${seccionVocabulario}
═══════════════════════════════
CÓMO ENSEÑAS
═══════════════════════════════
- Mantén una conversación real y fluida sobre temas cotidianos o de interés del estudiante — no repitas listas de vocabulario sin contexto.
- Cuando el estudiante cometa un error, no lo interrumpas de forma brusca: primero responde su idea, luego corrige con delicadeza mostrando la forma correcta.
- Cuando introduzcas una palabra o expresión nueva en inglés, dala junto a su traducción entre paréntesis la primera vez, así: "Let's go! (¡Vamos!)".
- Haz preguntas de seguimiento para que el estudiante siga hablando/escribiendo, no des respuestas que cierren la conversación.
- Si el estudiante pide explícitamente subir o bajar de nivel, ajústate de inmediato al nuevo nivel indicado.
- Sé alentador: reconoce el esfuerzo antes de corregir, nunca hagas sentir mal al estudiante por un error.

═══════════════════════════════
PRIMER MENSAJE DE LA CONVERSACIÓN
═══════════════════════════════
${esPrimerMensaje ? 'Este es el primer mensaje del estudiante. Salúdalo cálidamente, cuéntale en una frase el nivel que salió en su encuesta, y propón un tema sencillo de conversación para arrancar.' : 'Ya llevan conversación previa — continúa el hilo de forma natural, no vuelvas a presentarte ni a repetir el saludo inicial.'}

═══════════════════════════════
REGLAS GENERALES
═══════════════════════════════
- Máximo 4-6 líneas por respuesta — es una conversación, no una clase magistral.
- Nunca respondas solo en un idioma si el nivel pide mezcla; respeta la proporción español/inglés indicada arriba.
- Si el estudiante escribe en inglés con errores, no te limites a corregir — sigue la conversación y dale una razón para seguir practicando.`;

const allMessages = [
  { role: 'system', content: systemPrompt },
  ...messages
];

return [{
  json: {
    model: "openai/gpt-oss-120b",
    messages: allMessages,
    temperature: 0.8,
    max_tokens: 500,
    reasoning_effort: "low"
  }
}];
"""

JS_PARSEAR = r"""
const groqResponse = $input.first().json;
const nivel = $('Preparar mensajes y nivel').first().json.nivel;
// Resultado del modelo ML: la página lo usa para mostrar el nivel y las probabilidades
const prediccion = $('Resultado del modelo').first().json.prediccion;

if (!groqResponse || !groqResponse.choices || groqResponse.choices.length === 0) {
  return [{
    json: {
      response: "Lo siento, hubo un error al procesar tu mensaje. Por favor intenta de nuevo.",
      nivel: nivel,
      prediccion: prediccion
    }
  }];
}

const aiMessage = groqResponse.choices[0].message.content;

return [{
  json: {
    response: aiMessage,
    nivel: nivel,
    prediccion: prediccion
  }
}];
"""

wf_tutor = {
    "name": "Tutor inglés - Conversación + modelo ML",
    "nodes": [
        {**webhook("Webhook1", "tutor-ingles", "POST"),
         "id": "7f18568d-98f0-4a47-ba50-f6ac3fb05ae5", "webhookId": "753a56a9-6777-454f-b239-454b5fb44ca7"},
        nodo("Predecir nivel (modelo ML)", "n8n-nodes-base.httpRequest", 4.2, 224, {
            "method": "POST", "url": MODELO_URL + "/predecir",
            "sendBody": True, "specifyBody": "json", "jsonBody": BODY_MODELO,
            "options": {"timeout": 10000, "response": {"response": {"neverError": True}}},
        }, onError="continueRegularOutput"),
        code("Resultado del modelo", 448, JS_RESULTADO_MODELO),
        nodo("¿Modelo disponible?", "n8n-nodes-base.if", 2, 672, SI_MODELO_OK),
        code("Preparar mensajes y nivel", 896, JS_PREPARAR, y=-100),
        code("Armar prompt tutor", 1120, JS_ARMAR_PROMPT, y=-100),
        nodo("Llamar a Groq", "n8n-nodes-base.httpRequest", 4.3, 1344, {
            "method": "POST", "url": "https://api.groq.com/openai/v1/chat/completions",
            "authentication": "predefinedCredentialType", "nodeCredentialType": "groqApi",
            "sendHeaders": True, "headerParameters": {"parameters": [{"name": "Content-Type", "value": "application/json"}]},
            "sendBody": True, "specifyBody": "json", "jsonBody": "={{ $json }}", "options": {},
        }, y=-100, credentials=GROQ_CREDENCIAL),
        code("Parsear respuesta", 1568, JS_PARSEAR, y=-100),
        responder("Respond to Webhook", 1792, y=-100),
        code("Avisar: modelo no disponible", 896, JS_MODELO_NO_DISPONIBLE, y=150),
        responder("Responder error del modelo", 1120, y=150),
    ],
    "connections": {
        **cadena("Webhook1", "Predecir nivel (modelo ML)", "Resultado del modelo", "¿Modelo disponible?"),
        "¿Modelo disponible?": {"main": [
            [{"node": "Preparar mensajes y nivel", "type": "main", "index": 0}],
            [{"node": "Avisar: modelo no disponible", "type": "main", "index": 0}],
        ]},
        **cadena("Preparar mensajes y nivel", "Armar prompt tutor", "Llamar a Groq", "Parsear respuesta", "Respond to Webhook"),
        **cadena("Avisar: modelo no disponible", "Responder error del modelo"),
    },
    "pinData": {},
    "settings": {"executionOrder": "v1"},
}

(AQUI / "workflow_tutor.json").write_text(json.dumps(wf_tutor, indent=2, ensure_ascii=False))
print("Escrito", AQUI / "workflow_tutor.json")
