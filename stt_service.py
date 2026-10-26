"""
Servicio de transcripción de voz a texto (Speech-to-Text) usando faster-whisper.
Soporta español e inglés (y mezcla de ambos) con detección automática de idioma.

Instalación:
    pip install fastapi uvicorn faster-whisper python-multipart

Ejecutar localmente:
    uvicorn stt_service:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /transcribe
    Body: multipart/form-data con un campo "audio" (archivo .webm, .mp3, .wav, etc.)
    Respuesta: {"text": "...", "language": "es" | "en"}
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# CORS: en producción, reemplaza "*" por tu dominio real (ej. "https://davdiazz.github.io")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Tamaños de modelo disponibles: tiny, base, small, medium, large-v3
# "small" es un buen balance velocidad/precisión para español-inglés conversacional.
# device="cpu" funciona en cualquier máquina; usa "cuda" si tienes GPU disponible.
model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribir(ruta: str) -> dict:
    segments, info = model.transcribe(ruta, language=None, vad_filter=True)
    texto_completo = " ".join(segment.text.strip() for segment in segments)  # aquí se transcribe de verdad
    return {
        "text": texto_completo.strip(),
        "language": info.language,
        "language_probability": round(info.language_probability, 3)
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No se recibió ningún archivo de audio")

    # Guardamos el audio recibido en un archivo temporal para que Whisper lo lea
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # language=None deja que Whisper detecte el idioma automáticamente por segmento.
        # Esto es clave para tu caso: si el estudiante mezcla español e inglés, cada
        # segmento se transcribe en su idioma real en vez de forzar uno solo.
        # Whisper tarda unos segundos: se corre en otro hilo para que el servidor siga atendiendo
        # a los demás estudiantes (chat, encuesta) mientras transcribe este audio.
        return await run_in_threadpool(transcribir, tmp_path)
    finally:
        os.remove(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
