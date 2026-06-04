#!/usr/bin/env bash
# Launch the RF-DETR FastAPI app. The model is downloaded on first run (~349 MB)
# and inference runs on a GPU when one is usable, otherwise on CPU.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"

cd "$(dirname "$0")"

echo "Starting RF-DETR app on http://${HOST}:${PORT}"
echo "  Model:    ${RFDETR_MODEL:-nano}  (override with RFDETR_MODEL=nano|small|medium|large)"
exec uv run uvicorn backend.main:app --host "$HOST" --port "$PORT" "$@"
