#!/usr/bin/env bash
# IAingles: arranca TODO el tutor desde universidadIA
#   1. n8n (si está apagado)
#   2. servidor.py en el puerto 8080 = página + modelo ML de nivel + micrófono (Whisper) + puente a n8n
#   3. ngrok: publica el tutor en internet (URL fija si pusiste NGROK_DOMAIN en .env; si no, una temporal)
# Uso:  IAingles            -> local + internet
#       IAingles --local    -> solo en este computador
# Ctrl+C apaga todo lo que se encendió aquí.
set -e
cd "$(dirname "$0")"

# Un alias no carga las rutas de ~/.bashrc: se agregan a mano las rutas donde suelen estar n8n y ngrok
export PATH="$HOME/.npm-global/bin:$HOME/.local/bin:/snap/bin:/usr/local/bin:$PATH"
if [ -s "$HOME/.nvm/nvm.sh" ]; then source "$HOME/.nvm/nvm.sh" >/dev/null 2>&1 || true; fi

[ -f .env ] && set -a && source .env && set +a
PUERTO="${PUERTO:-8080}"
# Dominio fijo de ngrok: es el mismo que tiene universidad.html en URL_TUTOR (se puede cambiar en .env)
NGROK_DOMAIN="${NGROK_DOMAIN:-veneering-uniquely-feminine.ngrok-free.dev}"
puerto_ocupado() { python3 -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$1))==0 else 1)"; }

# ── Python ──
if [ ! -d venv ]; then
  echo "Creando entorno virtual (solo la primera vez)..."
  python3 -m venv venv
fi
source venv/bin/activate
python -c "import fastapi, httpx, faster_whisper, multipart, sklearn, pandas, joblib, av; assert int(av.__version__.split('.')[0]) < 19" 2>/dev/null || {
  echo "Instalando dependencias (solo la primera vez)..."
  pip install -r requirements.txt
}
if [ ! -f models/modelo_nivel.joblib ]; then
  echo "El modelo no está entrenado. Entrenando..."
  python train_model.py
fi
if puerto_ocupado "$PUERTO"; then
  echo "ERROR: el puerto $PUERTO ya está en uso (¿otro IAingles abierto?). Ciérralo con Ctrl+C y vuelve a intentar."
  exit 1
fi

SERVIDOR_PID=""; NGROK_PID=""; N8N_PID=""
trap 'echo; echo "Apagando..."; kill $SERVIDOR_PID $NGROK_PID $N8N_PID 2>/dev/null' INT TERM EXIT

# ── 1. n8n ──
if puerto_ocupado 5678; then
  echo "[n8n] ya estaba corriendo en el puerto 5678."
elif command -v n8n >/dev/null; then
  echo "[n8n] iniciando... (sus mensajes quedan en n8n.log)"
  n8n start > n8n.log 2>&1 &
  N8N_PID=$!
  for i in $(seq 1 90); do puerto_ocupado 5678 && break; sleep 1; done
  if puerto_ocupado 5678; then echo "[n8n] listo."; else echo "[n8n] AVISO: no respondió en 90 s. Revisa n8n.log"; fi
else
  echo "[n8n] AVISO: no encontré el comando n8n (busqué en ~/.npm-global/bin, ~/.local/bin, nvm...)."
  echo "      Ábrelo como lo hagas normalmente y el tutor lo usará solo."
fi

# ── 2. Servidor del tutor ──
echo "[tutor] iniciando en el puerto $PUERTO (carga el modelo de nivel y Whisper)..."
uvicorn servidor:app --host 0.0.0.0 --port "$PUERTO" &
SERVIDOR_PID=$!
for i in $(seq 1 120); do puerto_ocupado "$PUERTO" && break; sleep 1; done

# ── 3. Internet con ngrok ──
URL_PUBLICA=""
if [ "$1" != "--local" ]; then
  if ! command -v ngrok >/dev/null; then
    echo "[internet] AVISO: ngrok no está instalado, el tutor solo queda en este computador."
    echo "           Instálalo con:  sudo snap install ngrok   y luego:  ngrok config add-authtoken TU_TOKEN"
  else
    if [ -n "$NGROK_DOMAIN" ]; then
      ngrok http --url="$NGROK_DOMAIN" "$PUERTO" --log=ngrok.log > /dev/null 2>&1 &
    else
      ngrok http "$PUERTO" --log=ngrok.log > /dev/null 2>&1 &
    fi
    NGROK_PID=$!
    # Se pregunta a ngrok cuál es la URL pública
    for i in $(seq 1 20); do
      URL_PUBLICA=$(python - <<'PY' 2>/dev/null
import json, urllib.request
t = json.load(urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels"))["tunnels"]
print(next(x["public_url"] for x in t if x["public_url"].startswith("https")))
PY
) && [ -n "$URL_PUBLICA" ] && break
      sleep 1
    done
    if [ -z "$URL_PUBLICA" ]; then
      echo "[internet] ERROR: ngrok no pudo publicar. Mira ngrok.log (¿falta 'ngrok config add-authtoken TU_TOKEN'?)"
    elif [ -z "$NGROK_DOMAIN" ]; then
      echo "[internet] OJO: esta URL cambia cada vez. Para una fija pon NGROK_DOMAIN en .env (ver README)."
    fi
  fi
fi

cat <<MSG

==========================================================
  TUTOR DE INGLÉS ENCENDIDO
  En este computador:     http://localhost:$PUERTO
${URL_PUBLICA:+  Desde cualquier lugar:  $URL_PUBLICA   <- comparte ESTE link
                          (celulares, otros PCs; el micrófono funciona)
}  Estado de todo:         http://localhost:$PUERTO/api/estado
  n8n:                    http://localhost:5678  (workflow del tutor ACTIVO)
  Ctrl+C aquí apaga todo.
==========================================================

MSG
wait $SERVIDOR_PID
