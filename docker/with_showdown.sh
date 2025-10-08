#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve and export runtime env so children (Python) see the same values
# Cloud Run Jobs may set PORT; prefer SHOWDOWN_PORT if provided, else fall back to PORT, else 8000
: "${PORT:=8000}"
: "${SHOWDOWN_PORT:=${PORT}}"
: "${SHOWDOWN_HOST:=127.0.0.1}"
: "${METAMON_APP_DIR:=/app}"
: "${SHOWDOWN_DIR:=${METAMON_APP_DIR}/server/pokemon-showdown}"
: "${METAMON_CACHE_DIR:=/cache}"
: "${HF_HOME:=${METAMON_CACHE_DIR%/}/hf}"
: "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}}"

export PORT SHOWDOWN_PORT SHOWDOWN_HOST METAMON_APP_DIR METAMON_CACHE_DIR HF_HOME HUGGINGFACE_HUB_CACHE

mkdir -p "${METAMON_CACHE_DIR}" "${HF_HOME}"

echo "[with-showdown] Using host ${SHOWDOWN_HOST} and port ${SHOWDOWN_PORT}"

cd "${SHOWDOWN_DIR}"
# Start Showdown in the background; battles run in the foreground Python process
node pokemon-showdown start --no-security "${SHOWDOWN_PORT}" &
PS_PID=$!

cleanup() {
  if kill -0 "${PS_PID}" 2>/dev/null; then
    kill -TERM "${PS_PID}" 2>/dev/null || true
    wait "${PS_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Wait for HTTP to respond on the target port before starting Python
for i in $(seq 1 180); do
  if curl -fsS "http://${SHOWDOWN_HOST}:${SHOWDOWN_PORT}/" -o /dev/null; then
    break
  fi
  if [[ "${i}" -eq 180 ]]; then
    echo "Pokémon Showdown failed to become ready on ${SHOWDOWN_HOST}:${SHOWDOWN_PORT}" >&2
    echo "Node version:" >&2; node -v >&2 || true
    echo "NPM version:" >&2; npm -v >&2 || true
    echo "Showdown dist tree (top):" >&2; find "${SHOWDOWN_DIR}/dist" -maxdepth 2 -type d -print 2>/dev/null >&2 || true
    echo "Expecting: ${SHOWDOWN_DIR}/dist/lib/static-server.js" >&2; ls -l "${SHOWDOWN_DIR}/dist/lib/static-server.js" 2>/dev/null >&2 || true
    exit 1
  fi
  sleep 0.5
done

cd "${METAMON_APP_DIR}"
# Execute the self-play runner in the foreground so PID 1 blocks until battles end
exec "$@"
