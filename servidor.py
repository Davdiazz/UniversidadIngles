"""
Servidor único del tutor de inglés: con UN solo puerto (8080) sirve todo lo que necesita un estudiante.

    /               -> la página (universidad.html), ya configurada: no hay que pegar ningún webhook
    /api/tutor      -> el chat: reenvía el mensaje al webhook de n8n (n8n NO queda expuesto a internet)
    /modelo/...     -> el modelo ML de nivel (servicio_modelo.py): /modelo/estado, /modelo/predecir
    /stt/...        -> el micrófono, voz a texto con Whisper (stt_service.py): /stt/transcribe

Los dos modelos de Python (nivel + Whisper) se cargan una sola vez al arrancar.
Para que cualquier dispositivo entre desde internet se publica este puerto con ngrok (ver publicar.sh).

Ejecutar (desde la carpeta universidadIA, con el venv activado):
    uvicorn servidor:app --host 0.0.0.0 --port 8080

Variables opcionales (en el archivo .env, ver .env.example):
    N8N_WEBHOOK                 URL del webhook de producción de n8n
    LIMITE_MENSAJES_POR_MINUTO  mensajes por minuto que puede mandar cada visitante (protege tu cuota de Groq)
"""

import os
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

BASE = Path(__file__).parent


def leer_env():
    """Carga el archivo .env (si existe) sin depender de librerías extra."""
    archivo = BASE / ".env"
    if not archivo.exists():
        return
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


leer_env()

# Se importan después de leer .env. Al importarlos se cargan el modelo de nivel y Whisper (una sola vez).
from servicio_modelo import app as app_modelo  # noqa: E402
from stt_service import app as app_stt  # noqa: E402

N8N_WEBHOOK = os.environ.get("N8N_WEBHOOK", "http://localhost:5678/webhook/tutor-ingles")
LIMITE_POR_MINUTO = int(os.environ.get("LIMITE_MENSAJES_POR_MINUTO", "20"))
TAMANO_MAXIMO = 8 * 1024 * 1024  # 8 MB por mensaje (alcanza para una imagen)

app = FastAPI(title="Tutor de inglés con IA")
# Permite que otra copia de la página (por ejemplo la de GitHub Pages) use este servidor
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/modelo", app_modelo)
app.mount("/stt", app_stt)

# La página sabe que está detrás de este servidor y usa rutas relativas (sirve en localhost y en ngrok)
CONFIG_PAGINA = (
    "<script>window.TUTOR_SERVIDOR = { chatUrl: '/api/tutor', sttUrl: '/stt/transcribe' };</script>"
)


@app.get("/", response_class=HTMLResponse)
def pagina():
    html = (BASE / "universidad.html").read_text(encoding="utf-8")
    return html.replace("<head>", "<head>\n" + CONFIG_PAGINA, 1)


@app.get("/universidad.html", response_class=HTMLResponse)
def pagina_alias():
    return pagina()


# ── Límite de mensajes por visitante ──
_mensajes_por_ip: dict[str, deque] = defaultdict(deque)


def ip_visitante(request: Request) -> str:
    # ngrok manda la IP real del visitante en X-Forwarded-For
    reenviada = request.headers.get("x-forwarded-for")
    return reenviada.split(",")[0].strip() if reenviada else (request.client.host if request.client else "?")


def supera_limite(ip: str) -> bool:
    ahora = time.time()
    cola = _mensajes_por_ip[ip]
    while cola and ahora - cola[0] > 60:
        cola.popleft()
    if len(cola) >= LIMITE_POR_MINUTO:
        return True
    cola.append(ahora)
    return False


def respuesta_aviso(texto: str, status: int = 200) -> JSONResponse:
    # Se responde con el mismo formato que n8n para que la página lo muestre en el chat
    return JSONResponse({"response": texto, "nivel": None, "prediccion": None, "servidorError": True}, status_code=status)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# "/" y "/webhook/tutor-ingles" también reciben mensajes: así funciona aunque alguien pegue el link de ngrok
# como "Webhook de consulta" en la página de GitHub Pages
@app.post("/")
@app.post("/webhook/tutor-ingles")
@app.post("/api/tutor")
async def tutor(request: Request):
    if supera_limite(ip_visitante(request)):
        return respuesta_aviso("⏳ Estás enviando mensajes muy rápido. Espera un minuto e inténtalo de nuevo.")

    cuerpo = await request.body()
    if len(cuerpo) > TAMANO_MAXIMO:
        return respuesta_aviso("El mensaje es demasiado grande (máximo 8 MB).")

    # Cada petición viaja sola: la encuesta y el historial vienen del navegador de ESE estudiante,
    # así que dos personas conectadas al tiempo nunca se mezclan.
    try:
        async with httpx.AsyncClient(timeout=120) as cliente:
            r = await cliente.post(N8N_WEBHOOK, content=cuerpo, headers={"Content-Type": "application/json"})
    except httpx.HTTPError as e:
        print("No se pudo contactar a n8n:", repr(e))
        return respuesta_aviso("⚠️ El tutor no está disponible en este momento (n8n está apagado o el workflow no está activo). "
                               "Intenta de nuevo en un rato.")

    if r.status_code == 404:
        return respuesta_aviso("⚠️ El workflow del tutor no está ACTIVO en n8n (el webhook de producción respondió 404).")
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.get("/api/estado")
async def estado_general():
    """Resumen rápido para revisar que todo está bien."""
    from servicio_modelo import estado as estado_modelo
    try:
        async with httpx.AsyncClient(timeout=5) as cliente:
            await cliente.get(N8N_WEBHOOK.split("/webhook")[0] + "/healthz")
        n8n_ok = True
    except httpx.HTTPError:
        n8n_ok = False
    return {"modelo_nivel": estado_modelo(), "microfono": "ok", "n8n_encendido": n8n_ok, "n8n_webhook": N8N_WEBHOOK}
