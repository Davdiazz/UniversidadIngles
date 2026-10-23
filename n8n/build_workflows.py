"""
Genera el workflow del tutor para n8n con el banco de palabras y los pesos del modelo ya incrustados.
Correr después de train_model.py (si reentrenas, vuelve a correr esto y reimporta):
    venv/bin/python n8n/build_workflows.py
"""

import csv
import json
import uuid
from pathlib import Path

BASE = Path(__file__).parent.parent
AQUI = Path(__file__).parent

# Desde qué páginas se puede llamar al webhook (CORS). "*" = cualquiera, así funciona abriendo
# la página en local (http://localhost:5500) y también en GitHub Pages.
ORIGEN_FRONTEND = "*"
GROQ_CREDENCIAL = {"groqApi": {"id": "MIar86booTIHpgaI", "name": "Groq account"}}

with open(BASE / "data" / "banco_vocabulario.csv", encoding="utf-8") as f:
    BANCO = list(csv.DictReader(f))
MODELO = json.loads((BASE / "models" / "modelo_nivel.json").read_text())


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


BANCO_JS = "const BANCO = " + json.dumps(BANCO, ensure_ascii=False, indent=0).replace("\n", "") + ";"
MODELO_JS = "const MODELO = " + json.dumps(MODELO) + ";"

# ═══════════════════════════════════════════════════════════════════
# NODO DEL MODELO ML (va dentro del workflow del tutor)
# ═══════════════════════════════════════════════════════════════════

JS_PREDECIR = BANCO_JS + "\n" + MODELO_JS + r"""

// La página envía junto a cada mensaje: { encuesta: { respuestas: { "spoon": "cuchara", "gale": "No la conozco", ... } } }
const body = $input.first().json.body || $input.first().json;
const respuestas = (body.encuesta && body.encuesta.respuestas) || {};
const NIVELES = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'];

// Si el usuario no hizo la encuesta, seguimos sin predicción (se usa body.level o 'principiante')
if (Object.keys(respuestas).length === 0) {
  return [{ json: { body, prediccion: null } }];
}

const porPalabra = Object.fromEntries(BANCO.map(p => [p.palabra, p]));
const aciertos = Object.fromEntries(NIVELES.map(n => [n, 0]));
let noLaConozco = 0;
let totalAciertos = 0;
const palabrasDesconocidas = [];

for (const [palabra, respuesta] of Object.entries(respuestas)) {
  const p = porPalabra[palabra];
  if (!p) continue;
  if (respuesta === p.traduccion) {
    aciertos[p.cefr] += 1;
    totalAciertos += 1;
  } else {
    if (!respuesta || respuesta === 'No la conozco') noLaConozco += 1;
    palabrasDesconocidas.push({ palabra: p.palabra, traduccion: p.traduccion, cefr: p.cefr });
  }
}

// Mismas variables (features) con las que se entrenó el modelo en train_model.py
const totalPreguntas = Object.values(MODELO.preguntas_por_nivel).reduce((a, b) => a + b, 0);
const x = [
  ...NIVELES.map(n => aciertos[n] / MODELO.preguntas_por_nivel[n]),
  noLaConozco / totalPreguntas
];

// Regresión logística multinomial: puntaje por clase -> softmax -> probabilidades
const puntajes = MODELO.coeficientes.map((w, k) =>
  w.reduce((suma, wi, i) => suma + wi * x[i], MODELO.interceptos[k]));
const maximo = Math.max(...puntajes);
const exps = puntajes.map(s => Math.exp(s - maximo));
const sumaExps = exps.reduce((a, b) => a + b, 0);
const probabilidades = {};
MODELO.clases.forEach((clase, k) => { probabilidades[clase] = Math.round((exps[k] / sumaExps) * 1000) / 1000; });

const nivel = Object.entries(probabilidades).sort((a, b) => b[1] - a[1])[0][0];

return [{
  json: {
    body,
    prediccion: {
      nivel,
      probabilidades,
      aciertos: `${totalAciertos}/${totalPreguntas}`,
      palabrasDesconocidas
    }
  }
}];
"""

# ═══════════════════════════════════════════════════════════════════
# WORKFLOW: Tutor (tu workflow original + modelo ML de la encuesta)
# ═══════════════════════════════════════════════════════════════════

JS_PREPARAR = r"""
const inputData = $input.all();

// Niveles válidos que el frontend puede enviar en body.level
const NIVELES_VALIDOS = ['principiante', 'basico', 'intermedio', 'avanzado'];

return inputData.map(item => {
  const body = item.json.body || item.json;
  const prediccion = item.json.prediccion;  // viene del nodo "Predecir nivel (modelo ML)"
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
const prediccion = $('Predecir nivel (modelo ML)').first().json.prediccion;

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
        code("Predecir nivel (modelo ML)", 224, JS_PREDECIR),
        code("Preparar mensajes y nivel", 448, JS_PREPARAR),
        code("Armar prompt tutor", 672, JS_ARMAR_PROMPT),
        nodo("Llamar a Groq", "n8n-nodes-base.httpRequest", 4.3, 896, {
            "method": "POST", "url": "https://api.groq.com/openai/v1/chat/completions",
            "authentication": "predefinedCredentialType", "nodeCredentialType": "groqApi",
            "sendHeaders": True, "headerParameters": {"parameters": [{"name": "Content-Type", "value": "application/json"}]},
            "sendBody": True, "specifyBody": "json", "jsonBody": "={{ $json }}", "options": {},
        }, credentials=GROQ_CREDENCIAL),
        code("Parsear respuesta", 1120, JS_PARSEAR),
        responder("Respond to Webhook", 1344),
    ],
    "connections": cadena("Webhook1", "Predecir nivel (modelo ML)", "Preparar mensajes y nivel", "Armar prompt tutor",
                          "Llamar a Groq", "Parsear respuesta", "Respond to Webhook"),
    "pinData": {},
    "settings": {"executionOrder": "v1"},
}

(AQUI / "workflow_tutor.json").write_text(json.dumps(wf_tutor, indent=2, ensure_ascii=False))
print("Escrito", AQUI / "workflow_tutor.json")
