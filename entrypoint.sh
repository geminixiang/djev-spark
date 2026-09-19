#!/usr/bin/env bash
# Boot vLLM with the structured-reads overlay, then the structured server in
# front of it. Refuses to start without memory headroom: on unified memory a
# CUDA overshoot is a host OOM and a hang, not a failed request, and the
# start-up profiling step runs the sampler at the full batch, about ten fp32
# copies of [MAX_SEQS x CANVAS, vocab].
set -euo pipefail

MODEL=${MODEL:-/models/dgemma}
SERVED_NAME=${SERVED_NAME:-dgemma}
CANVAS=${CANVAS:-128}
MAX_SEQS=${MAX_SEQS:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_UTIL=${GPU_UTIL:-0.40}
ATTN=${ATTN:-TRITON_ATTN}
PORT=${PORT:-8010}
STRUCTURED_PORT=${STRUCTURED_PORT:-8011}
TLS_PORT=${TLS_PORT:-0}
EXTRA_ARGS=${EXTRA_ARGS:---async-scheduling}
KV_CACHE_GB=${KV_CACHE_GB:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
HEADROOM_GB=${HEADROOM_GB:-12}
WAIT_SECS=${WAIT_SECS:-1800}

[[ -f "$MODEL/config.json" ]] || { echo "no model at $MODEL; run scripts/download-model.sh" >&2; exit 2; }

WEIGHTS_GB=19
TRANSIENT_GB=$(( MAX_SEQS * CANVAS * 262144 * 4 * 10 / 1073741824 + 1 ))
NEED_GB=$(( WEIGHTS_GB + KV_CACHE_GB + TRANSIENT_GB + HEADROOM_GB ))
AVAIL_GB=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
echo "memory: ${AVAIL_GB} GB available, need ${NEED_GB} (weights ${WEIGHTS_GB} + KV ${KV_CACHE_GB} + start-up transient ${TRANSIENT_GB} + headroom ${HEADROOM_GB})"
if (( AVAIL_GB < NEED_GB )); then
  echo "refusing to start: not enough memory; stop other models first" >&2
  exit 2
fi

healthy() {
  python3 - "$1" <<'EOF'
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2)
except Exception:
    sys.exit(1)
EOF
}

# shellcheck disable=SC2086  # EXTRA_ARGS is a flag list
vllm serve "$MODEL" --served-model-name "$SERVED_NAME" --trust-remote-code \
  --max-num-seqs "$MAX_SEQS" --max-model-len "$MAX_MODEL_LEN" \
  --attention-backend "$ATTN" --gpu-memory-utilization "$GPU_UTIL" --kv-cache-memory $(( KV_CACHE_GB * 1073741824 )) \
  ${MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"} \
  --max-logprobs 32 --enable-prefix-caching \
  --diffusion-config "{\"canvas_length\": ${CANVAS}}" \
  --override-generation-config '{"max_new_tokens": null}' \
  --port "$PORT" $EXTRA_ARGS &
VLLM_PID=$!

for (( i = 0; i < WAIT_SECS / 5; i++ )); do
  healthy "$PORT" && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm exited during start-up" >&2; exit 1; }
  sleep 5
done
healthy "$PORT" || { echo "vllm not healthy after ${WAIT_SECS}s" >&2; kill "$VLLM_PID"; exit 1; }
echo "vllm ready on :${PORT}"

# The structured server is restarted whenever it exits, so its code can be
# reloaded (docker cp the files in, then pkill -f structured_server.py)
# without touching vLLM. Only vLLM's exit ends the container.
serve_structured() {
  while kill -0 "$VLLM_PID" 2>/dev/null; do
    # A reload signals the server, and set -e would take this loop down with it.
    python3 /opt/dgemma/structured_server.py --upstream "http://127.0.0.1:${PORT}" --model "$SERVED_NAME" \
      --tokenizer "$MODEL" --canvas "$CANVAS" --port "$STRUCTURED_PORT" --tls-port "$TLS_PORT" --cert-dir /root/.cache/djev || true
    echo "structured server exited; restarting" >&2
    sleep 1
  done
}
serve_structured &
SERVER_LOOP=$!

trap 'kill "$VLLM_PID" "$SERVER_LOOP" 2>/dev/null; pkill -f structured_server.py 2>/dev/null; wait' TERM INT
wait "$VLLM_PID"
echo "vllm exited; stopping" >&2
kill "$SERVER_LOOP" 2>/dev/null
pkill -f structured_server.py 2>/dev/null
wait
exit 1
