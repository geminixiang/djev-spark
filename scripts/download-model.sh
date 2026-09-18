#!/usr/bin/env bash
# Fetch the NVFP4 checkpoint into ${MODELS_DIR:-./models}/dgemma. Uses a host
# `hf` when there is one, otherwise the one inside the built image.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=${MODEL_REPO:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}
MODELS_DIR=${MODELS_DIR:-./models}
NAME=${MODEL_NAME:-dgemma}
mkdir -p "$MODELS_DIR/$NAME"
DEST=$(cd "$MODELS_DIR" && pwd)

if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir "$DEST/$NAME"
else
  docker compose build -q dgemma
  docker run --rm -v "$DEST:/models" -e HF_TOKEN="${HF_TOKEN:-}" \
    --entrypoint hf djev-spark:latest download "$REPO" --local-dir "/models/$NAME"
fi
echo "model at $DEST/$NAME"
ls "$DEST/$NAME" | head
