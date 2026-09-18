#!/usr/bin/env bash
# One plain generation through vLLM and one structured decision through the
# server. Both must answer for the stack to count as up.
set -euo pipefail
PORT=${PORT:-8010}
STRUCTURED_PORT=${STRUCTURED_PORT:-8011}

echo "== vllm :${PORT}"
curl -sf "localhost:${PORT}/v1/chat/completions" -H 'content-type: application/json' -d '{
  "model": "dgemma", "max_tokens": 32,
  "messages": [{"role": "user", "content": "Name three primary colors."}],
  "chat_template_kwargs": {"enable_thinking": false}}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'

echo "== structured :${STRUCTURED_PORT}"
curl -sf "localhost:${STRUCTURED_PORT}/v1/chat/completions" -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [
        {\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"},
        {\"id\": \"bucket\", \"type\": \"choice\", \"instructions\": \"Which team owns this?\",
         \"options\": [{\"name\": \"billing\"}, {\"name\": \"outage\", \"description\": \"service down\"}, {\"name\": \"feature\"}]},
        {\"id\": \"tone\", \"type\": \"score\", \"instructions\": \"How angry is the customer?\", \"levels\": [\"calm\", \"annoyed\", \"furious\"]}],
      \"samples\": 1}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon. Fix it now.\"}"}
  ]}' | python3 -c '
import json, sys
d = json.loads(json.load(sys.stdin)["choices"][0]["message"]["content"])
for k, a in d["answers"].items():
    print(f"  {k:7s} {a["label"]:3s} confidence {a["confidence"]:.2f}")
print(f"  {d["diagnostics"]["timing"]["total_ms"]:.0f} ms, {d["diagnostics"]["samples"]["n"]} read(s)")'
