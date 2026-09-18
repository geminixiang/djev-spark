#!/usr/bin/env bash
# Run the structured server's fake-upstream test inside the image. Needs the
# model directory for its tokenizer and nothing else: no GPU, no vLLM.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build -q dgemma
docker compose run --rm --no-deps --entrypoint python3 \
  -e SERVER_DIR=/opt/dgemma -e TOKENIZER=/models/dgemma \
  dgemma /opt/dgemma/test_structured_server.py
