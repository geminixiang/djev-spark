#!/usr/bin/env bash
# One plain generation through vLLM and one structured decision through the
# server. Both must answer for the stack to count as up.
set -euo pipefail
PORT=${PORT:-8010} STRUCTURED_PORT=${STRUCTURED_PORT:-8011} python3 - <<'EOF'
import json, os, time, urllib.request

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d, (time.time() - t) * 1e3

port, sport = os.environ["PORT"], os.environ["STRUCTURED_PORT"]
print(f"== vllm :{port}")
d, ms = post(f"http://127.0.0.1:{port}/v1/chat/completions", {
    "model": "dgemma", "max_tokens": 32,
    "messages": [{"role": "user", "content": "Name three primary colors."}],
    "chat_template_kwargs": {"enable_thinking": False}})
print(f"  {ms:.0f} ms: {d['choices'][0]['message']['content'].strip()[:120]!r}")

print(f"== structured :{sport}")
d, ms = post(f"http://127.0.0.1:{sport}/v1/systemone", {
    "model": "jev-latest",
    "state": {"ticket": "Everything is down and we have a demo at noon. Fix it now."},
    "questions": {
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        "bucket": {"type": "choice", "instructions": "Which team owns this?",
                   "criteria": {"billing": None, "outage": "service down", "feature": None}},
        "tone": {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "furious"]}},
    "samples": 1})
for k, a in d["answers"].items():
    print(f"  {k:7s} {json.dumps({x: y for x, y in a.items() if x != 'type'})}")
print(f"  {ms:.0f} ms, {d['diagnostics']['samples']['n']} read(s), usage {d['usage']}")
EOF
