"""Size the KV cache for long states: send one structured decision over a
state of N tokens, cold and then warm (prefix cached), and report latency
and the server's KV usage. Run against a server started with MAX_MODEL_LEN
above N.

usage: long-context-probe.py [tokens ...]   (default 8000 32000 100000)
"""
import json
import sys
import time
import urllib.request

STRUCTURED = "http://127.0.0.1:8011"
VLLM = "http://127.0.0.1:8010"
SCHEMA = {"questions": [
    {"id": "topic", "type": "choice", "instructions": "What is the document about?",
     "options": ["shipping logistics", "a software incident", "a cooking recipe", "a court ruling"]},
    {"id": "urgent", "type": "noul", "instructions": "Does the document ask for action today?"},
], "samples": 1}
PARAGRAPH = ("At 03:12 the on-call engineer was paged: the checkout service was returning errors for "
             "about a third of requests. The database primary had failed over and the replica was "
             "serving stale connection pools. Restarting the pods cleared it by 03:40. ")


def tokens_of(text):
    body = {"model": "dgemma", "prompt": text}
    req = urllib.request.Request(VLLM + "/tokenize", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["count"]


def decide(state):
    body = {"messages": [{"role": "system", "content": json.dumps(SCHEMA)}, {"role": "user", "content": json.dumps(state)}]}
    req = urllib.request.Request(STRUCTURED + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    out = json.loads(d["choices"][0]["message"]["content"])
    return out, time.time() - t


def kv_usage():
    text = urllib.request.urlopen(VLLM + "/metrics", timeout=30).read().decode()
    for line in text.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc"):
            return float(line.split()[-1])
    return None


per_para = tokens_of(PARAGRAPH)
for want in [int(a) for a in sys.argv[1:]] or [8000, 32000, 100000]:
    n = max(1, want // per_para)
    doc = "\n".join(f"[{i}] " + PARAGRAPH for i in range(n))
    state = {"document": doc, "note": "the document is an incident log"}
    ntok = tokens_of(json.dumps(state))
    out, cold = decide(state)
    out2, warm = decide(state)
    a = out["answers"]
    print(f"{ntok:7d} tokens: cold {cold:6.2f} s, warm {warm:6.2f} s, KV usage {kv_usage()}, "
          f"topic={a['topic']['choice']!r} ({a['topic']['confidence']:.2f}) urgent={a['urgent']['label']} ({a['urgent']['confidence']:.2f})", flush=True)
