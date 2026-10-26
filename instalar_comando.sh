#!/usr/bin/env bash
# Crea el comando IAingles: escríbelo en cualquier terminal y arranca todo el tutor.
# Uso (una sola vez):  bash instalar_comando.sh   y luego abre una terminal nueva.
set -e
CARPETA="$(cd "$(dirname "$0")" && pwd)"
LINEA="alias IAingles='bash \"$CARPETA/iniciar.sh\"'"

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  [ -f "$rc" ] || continue
  sed -i '/alias IAingles=/d' "$rc"          # quita una versión vieja si existía
  echo "$LINEA" >> "$rc"
  echo "Comando IAingles agregado a $rc"
done

echo
echo "Listo. Abre una terminal NUEVA (o ejecuta: source ~/.bashrc) y escribe:  IAingles"
echo "Solo en tu computador, sin internet:  IAingles --local"
