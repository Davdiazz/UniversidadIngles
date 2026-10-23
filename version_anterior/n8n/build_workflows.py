"""
Genera los JSON de los workflows de n8n (para importarlos con "Import from file").
    python3 n8n/build_workflows.py

Escribir los workflows desde Python evita errores de comillas al meter código JavaScript dentro del JSON.
"""

import json
import uuid
from pathlib import Path

AQUI = Path(__file__).parent

# Cambia esto si n8n corre en Docker: "http://host.docker.internal:8001"
LEVEL_SERVICE_URL = "http://localhost:8001"
ORIGEN_FRONTEND = "https://davdiazz.github.io"
GROQ_CREDENCIAL = {"groqApi": {"id": "MIar86booTIHpgaI", "name": "Groq account"}}


def nodo(nombre, tipo, version, x, params, **extra):
    return {"parameters": params, "type": tipo, "typeVersion": version, "position": [x, extra.pop("y", 0)],
            "id": str(uuid.uuid4()), "name": nombre, **extra}


def webhook(nombre, path, metodo, y=0):
    return nodo(nombre, "n8n-nodes-base.webhook", 2.1, 0, {
        "httpMethod": metodo, "path": path, "responseMode": "responseNode",
        "options": {"responseHeaders": {"entries": [{"name": "Access-Control-Allow-Origin", "value": ORIGEN_FRONTEND}]}},
    }, webhookId=str(uuid.uuid4()), y=y)


def code(nombre, x, js, y=0):
    return nodo(nombre, "n8n-nodes-base.code", 2, x, {"jsCode": js.strip()}, y=y)


def llamar_groq(nombre, x, y=0):
    return nodo(nombre, "n8n-nodes-base.httpRequest", 4.3, x, {
        "method": "POST", "url": "https://api.groq.com/openai/v1/chat/completions",
        "authentication": "predefinedCredentialType", "nodeCredentialType": "groqApi",
        "sendHeaders": True, "headerParameters": {"parameters": [{"name": "Content-Type", "value": "application/json"}]},
        "sendBody": True, "specifyBody": "json", "jsonBody": "={{ $json }}", "options": {},
    }, credentials=GROQ_CREDENCIAL, y=y)


def responder(nombre, x, y=0):
    return nodo(nombre, "n8n-nodes-base.respondToWebhook", 1.4, x,
                {"respondWith": "json", "responseBody": "={{ $json }}", "options": {}}, y=y)


def cadena(*nombres):
    return {a: {"main": [[{"node": b, "type": "main", "index": 0}]]} for a, b in zip(nombres, nombres[1:])}


# ═══════════════════════════════════════════════════════════════════
# WORKFLOW 1: Test de nivel (dataset + modelo + Groq)
# ═══════════════════════════════════════════════════════════════════

JS_PROMPT_DIAGNOSTICO = r"""
// Resultado del modelo entrenado con el dataset (servicio Python /evaluate)
const modelo = $input.first().json;
const abiertas = modelo.respuestas_abiertas || {};

const respuestasTexto = Object.entries(abiertas)
  .map(([id, texto]) => `- (${id}) ${texto || '(sin respuesta)'}`)
  .join('\n');

const falladas = (modelo.preguntas_falladas || [])
  .map(f => `- [${f.nivel_cefr}] ${f.tema}: "${f.pregunta}" respondió "${f.respuesta_dada}" (correcto: "${f.respuesta_correcta}")`)
  .join('\n') || '- Ninguna';

const systemPrompt = `Eres un evaluador experto de inglés para hispanohablantes. Un modelo entrenado con un dataset CEFR ya estimó el nivel del estudiante. Tu trabajo es analizar sus errores concretos y armar un plan de repaso.

NIVEL ESTIMADO POR EL MODELO: ${modelo.cefr_estimado} (${modelo.nivel})
PREGUNTAS DE OPCIÓN MÚLTIPLE FALLADAS:
${falladas}
CONTENIDOS GRAMATICALES DEL SIGUIENTE NIVEL (CEFR-J): ${(modelo.siguientes_contenidos_cefrj || []).join('; ')}

Responde SOLO con un JSON válido con esta forma exacta:
{
  "errores": [{"original": "frase con error", "correccion": "frase corregida", "explicacion": "explicación breve en español", "tema": "tema gramatical en español"}],
  "fortalezas": ["algo que hace bien, en español"],
  "temas_a_repasar": ["máximo 4 temas en español, del más urgente al menos urgente"],
  "plan_de_repaso": ["3 pasos concretos y cortos en español"],
  "mensaje_para_estudiante": "2-3 frases cálidas en español explicando su nivel y qué vamos a trabajar"
}
No inventes errores: si las respuestas escritas no tienen errores, deja "errores" vacío.`;

return [{
  json: {
    model: 'openai/gpt-oss-120b',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Respuestas escritas del estudiante:\n${respuestasTexto}` }
    ],
    temperature: 0.2,
    max_tokens: 1200,
    reasoning_effort: 'low',
    response_format: { type: 'json_object' }
  }
}];
"""

JS_UNIR_DIAGNOSTICO = r"""
const modelo = $('Detectar nivel (modelo)').first().json;
const groq = $input.first().json;

let analisis = {};
try {
  analisis = JSON.parse(groq.choices[0].message.content);
} catch (e) {
  analisis = { errores: [], fortalezas: [], temas_a_repasar: [], plan_de_repaso: [],
               mensaje_para_estudiante: 'Ya tenemos tu nivel. ¡Empecemos a practicar!' };
}

// Unimos los temas detectados por el test (dataset) con los que encontró Groq en la escritura
const temas = [...new Set([...(modelo.temas_a_repasar || []), ...(analisis.temas_a_repasar || [])])].slice(0, 5);

// Este objeto "perfil" es el que el frontend debe guardar y reenviar en cada mensaje del tutor
return [{
  json: {
    nivel: modelo.nivel,
    cefr: modelo.cefr_estimado,
    aciertos_test: modelo.detalle.aciertos_test,
    temasARepasar: temas,
    errores: (analisis.errores || []).slice(0, 6),
    fortalezas: analisis.fortalezas || [],
    planDeRepaso: analisis.plan_de_repaso || [],
    mensaje: analisis.mensaje_para_estudiante || ''
  }
}];
"""

wf_test = {
    "name": "Tutor inglés - Test de nivel",
    "nodes": [
        webhook("Pedir preguntas", "test-nivel", "GET", y=-200),
        nodo("Obtener preguntas", "n8n-nodes-base.httpRequest", 4.3, 224,
             {"method": "GET", "url": f"{LEVEL_SERVICE_URL}/placement-test", "options": {}}, y=-200),
        responder("Responder preguntas", 448, y=-200),

        webhook("Recibir respuestas", "nivel-ingles", "POST"),
        nodo("Detectar nivel (modelo)", "n8n-nodes-base.httpRequest", 4.3, 224, {
            "method": "POST", "url": f"{LEVEL_SERVICE_URL}/evaluate",
            "sendBody": True, "specifyBody": "json",
            "jsonBody": "={{ JSON.stringify($json.body) }}", "options": {},
        }),
        code("Armar prompt diagnóstico", 448, JS_PROMPT_DIAGNOSTICO),
        llamar_groq("Groq analiza errores", 672),
        code("Unir diagnóstico", 896, JS_UNIR_DIAGNOSTICO),
        responder("Responder diagnóstico", 1120),
    ],
    "connections": {
        **cadena("Pedir preguntas", "Obtener preguntas", "Responder preguntas"),
        **cadena("Recibir respuestas", "Detectar nivel (modelo)", "Armar prompt diagnóstico",
                 "Groq analiza errores", "Unir diagnóstico", "Responder diagnóstico"),
    },
    "pinData": {},
    "settings": {"executionOrder": "v1"},
}

# ═══════════════════════════════════════════════════════════════════
# WORKFLOW 2: Tutor (tu workflow original + perfil del test de nivel)
# ═══════════════════════════════════════════════════════════════════

JS_PREPARAR = r"""
const inputData = $input.all();

// Niveles válidos que el frontend puede enviar en body.level
const NIVELES_VALIDOS = ['principiante', 'basico', 'intermedio', 'avanzado'];

return inputData.map(item => {
  const body = item.json.body || item.json;
  const userMessage = body.message || "";
  const history = body.conversationHistory || [];

  // Perfil que devolvió el test de nivel (opcional). Si el estudiante no hizo el test, viene vacío.
  const perfil = body.perfil || {};

  // Nivel: el del test tiene prioridad; si no hay, el que eligió el usuario; si no, 'principiante'.
  let nivel = (perfil.nivel || body.level || '').toLowerCase().trim();
  if (!NIVELES_VALIDOS.includes(nivel)) {
    nivel = 'principiante';
  }

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
      cefr: perfil.cefr || null,
      temasARepasar: (perfil.temasARepasar || []).slice(0, 5),
      errores: (perfil.errores || []).slice(0, 6),
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
const cefr = inputData.cefr;
const temasARepasar = inputData.temasARepasar || [];
const errores = inputData.errores || [];

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

// Sección nueva: lo que detectó el test de nivel (modelo entrenado con dataset + análisis de Groq)
let seccionDiagnostico = '';
if (temasARepasar.length > 0 || errores.length > 0) {
  const listaErrores = errores
    .map(e => `- "${e.original}" → "${e.correccion}" (${e.tema})`)
    .join('\n');
  seccionDiagnostico = `
═══════════════════════════════
DIAGNÓSTICO DEL TEST DE NIVEL
═══════════════════════════════
Nivel CEFR estimado: ${cefr || nivel}
Temas que este estudiante necesita repasar (del más urgente al menos urgente):
${temasARepasar.map((t, i) => `${i + 1}. ${t}`).join('\n')}
${listaErrores ? `Errores que cometió en el test:\n${listaErrores}` : ''}

Cómo usar este diagnóstico:
- Guía la conversación para que el estudiante tenga que USAR estos temas (por ejemplo, si debe repasar el pasado simple, pregúntale qué hizo ayer).
- Trabaja un tema a la vez, empezando por el primero. Cuando lo use bien 2-3 veces, felicítalo y pasa al siguiente.
- Si repite uno de los errores del test, corrígelo y recuérdale brevemente la regla.
- Cada 4-5 mensajes puedes proponer un mini-ejercicio de 1 línea sobre el tema actual.
`;
}

const systemPrompt = `Eres un tutor de inglés conversacional, cálido y paciente, que enseña mezclando español e inglés según el nivel del estudiante. Tu objetivo es que la persona practique inglés real en una conversación natural, no en ejercicios mecánicos.

FECHA: ${fecha}

${PERFILES_NIVEL[nivel]}
${seccionDiagnostico}
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
${esPrimerMensaje
  ? (temasARepasar.length > 0
      ? 'Este es el primer mensaje del estudiante. Salúdalo cálidamente, cuéntale en una frase su nivel según el test y el primer tema que van a practicar, y arranca con una pregunta que lo obligue a usar ese tema.'
      : 'Este es el primer mensaje del estudiante. Salúdalo cálidamente, confirma el nivel con el que van a trabajar, y propón un tema sencillo de conversación para arrancar.')
  : 'Ya llevan conversación previa — continúa el hilo de forma natural, no vuelvas a presentarte ni a repetir el saludo inicial.'}

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

if (!groqResponse || !groqResponse.choices || groqResponse.choices.length === 0) {
  return [{
    json: {
      response: "Lo siento, hubo un error al procesar tu mensaje. Por favor intenta de nuevo.",
      nivel: nivel
    }
  }];
}

const aiMessage = groqResponse.choices[0].message.content;

return [{
  json: {
    response: aiMessage,
    nivel: nivel
  }
}];
"""

wf_tutor = {
    "name": "Tutor inglés - Conversación",
    "nodes": [
        webhook("Webhook1", "tutor-ingles", "POST"),
        code("Preparar mensajes y nivel", 224, JS_PREPARAR),
        code("Armar prompt tutor", 448, JS_ARMAR_PROMPT),
        llamar_groq("Llamar a Groq", 672),
        code("Parsear respuesta", 896, JS_PARSEAR),
        responder("Respond to Webhook", 1120),
    ],
    "connections": cadena("Webhook1", "Preparar mensajes y nivel", "Armar prompt tutor",
                          "Llamar a Groq", "Parsear respuesta", "Respond to Webhook"),
    "pinData": {},
    "settings": {"executionOrder": "v1"},
}

for nombre, wf in [("workflow_test_nivel.json", wf_test), ("workflow_tutor.json", wf_tutor)]:
    (AQUI / nombre).write_text(json.dumps(wf, indent=2, ensure_ascii=False))
    print("Escrito", AQUI / nombre)
