#!/usr/bin/env bash
# Lance PersonaPlex pour accès LAN (à exécuter sur la VM IA Linux).
# Prérequis : clone NVIDIA/personaplex, venv activé, HF_TOKEN exporté.
#
# Usage :
#   ./scripts/start-personaplex.sh
#   ./scripts/start-personaplex.sh --cpu-offload
#   PERSONAPLEX_DIR=/path/to/personaplex ./scripts/start-personaplex.sh

set -euo pipefail

HOST="${PERSONAPLEX_HOST:-0.0.0.0}"
PORT="${PERSONAPLEX_PORT:-8998}"
SSL_DIR="${PERSONAPLEX_SSL_DIR:-${HOME}/personaplex-ssl}"
PERSONAPLEX_DIR="${PERSONAPLEX_DIR:-${HOME}/personaplex}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Erreur: HF_TOKEN n'est pas défini." >&2
  echo "  export HF_TOKEN=hf_..." >&2
  echo "Accepte aussi la licence : https://huggingface.co/nvidia/personaplex-7b-v1" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Erreur: python introuvable. Active ton venv PersonaPlex." >&2
  exit 1
fi

mkdir -p "$SSL_DIR"

EXTRA_ARGS=()
if [[ "${1:-}" == "--cpu-offload" ]]; then
  EXTRA_ARGS+=(--cpu-offload)
fi

echo "PersonaPlex → https://${HOST}:${PORT}"
echo "SSL dir     → ${SSL_DIR}"
echo "Repo local  → ${PERSONAPLEX_DIR} (info)"
echo "Depuis le PC: https://<IP_DE_CETTE_VM>:${PORT}"
echo

exec python -m moshi.server \
  --host "$HOST" \
  --port "$PORT" \
  --ssl "$SSL_DIR" \
  "${EXTRA_ARGS[@]}"
