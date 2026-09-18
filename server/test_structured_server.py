"""End-to-end check of structured_server.py against a fake vLLM upstream.
Needs only the tokenizer: run inside the dgemma image with /models/dgemma
mounted, or set TOKENIZER to a local copy and CHAT_TEMPLATE to a jinja file
when that copy ships without one."""
import json, math, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.environ.get("SERVER_DIR", "/opt/dgemma"))
import structured_server as S

SEEN = []
CONF = [0.7]  # first-label probability the fake returns at every slot
THOUGHT = None  # token ids the fake writes when asked to think

class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        req["_path"] = self.path
        SEEN.append(req)
        if "vllm_xargs" not in req:
            # a thought: the fake writes a fixed one and closes the channel
            toks = [f"token_id:{i}" for i in THOUGHT + S.THOUGHT_CLOSE][:req["max_tokens"]]
            body = json.dumps({"choices": [{"logprobs": {"tokens": toks}, "finish_reason": "stop"}], "usage": {}}).encode()
            self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        x = req["vllm_xargs"]
        canvas = x["diffusion_seed_canvas"]
        assert len(canvas) == x["diffusion_canvas_length"] <= S.CANVAS_LEN
        assert x["diffusion_read_only"] is True and x["diffusion_max_steps"] == 1
        # Every position carries the seed id near-certain plus, for each label
        # family, the first label at CONF and the rest sharing the remainder,
        # so the fake needs no knowledge of where the slots are.
        content = []
        for pos, tid in enumerate(canvas[:req["max_tokens"]]):
            top = [{"token": f"token_id:{tid}", "logprob": -0.01}]
            for ids in FAMILIES:
                p = [CONF[0]] + [(1 - CONF[0]) / (len(ids) - 1)] * (len(ids) - 1)
                top += [{"token": f"token_id:{i}", "logprob": math.log(pi)} for i, pi in zip(ids, p)]
            content.append({"token": f"token_id:{tid}", "logprob": -0.01, "top_logprobs": top})
        if self.path.endswith("/v1/completions"):
            rows = [{tp["token"]: tp["logprob"] for tp in c["top_logprobs"]} for c in content]
            body = json.dumps({"choices": [{"logprobs": {"top_logprobs": rows}}], "usage": {}}).encode()
        else:
            body = json.dumps({"choices": [{"logprobs": {"content": content}}], "usage": {}}).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

SCHEMA = {"questions": [
    {"id": "urgent", "type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    {"id": "bucket", "type": "choice", "instructions": "Which team owns this?", "options": [{"name": "billing"}, {"name": "outage", "description": "service down"}, {"name": "feature"}]},
    {"id": "tone", "type": "score", "instructions": "How angry is the customer?", "levels": ["calm", "annoyed", "furious"]}],
    "samples": 3}

S.ARGS = type("A", (), {"upstream": "http://127.0.0.1:8998", "model": "dgemma"})()
tok = __import__("transformers").AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", os.environ.get("MODEL", "/models/dgemma")))
if tok.chat_template is None and os.environ.get("CHAT_TEMPLATE"):
    tok.chat_template = open(os.environ["CHAT_TEMPLATE"]).read()
S.init_tokenizer(tok)
THOUGHT = S.enc("The customer says everything is down, so this is urgent.")
schema = S.parse_schema(SCHEMA)
TEMPLATE, SLOTS = S.resolve_template(schema["questions"], S.SCAFFOLD, "", "lines")
FAMILIES = [s["label_ids"] for s in SLOTS]
print("template tokens", len(TEMPLATE), "slots", [(q["id"], s["pos"], [S.TOK.decode([i]) for i in s["label_ids"]]) for q, s in zip(schema["questions"], SLOTS)])
print("system prompt:\n" + S.system_text(schema))

threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", 8998), Fake).serve_forever, daemon=True).start()
threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", 8999), S.Handler).serve_forever, daemon=True).start()
time.sleep(0.3)

def post(body):
    req = urllib.request.Request("http://127.0.0.1:8999/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        r = urllib.request.urlopen(req); return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

code, d = post({"model": "x", "messages": [{"role": "system", "content": json.dumps(SCHEMA)}, {"role": "user", "content": json.dumps({"ticket": "Everything is down and I am furious, fix it now"})}]})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
print(json.dumps(out["answers"], indent=1))
assert out["answers"]["urgent"]["label"] == "yes" and abs(out["answers"]["urgent"]["noul"] - 0.7) < 1e-6
assert out["answers"]["bucket"]["choice"] == "billing" and out["answers"]["tone"]["level"] == "calm"
assert out["diagnostics"]["samples"]["n"] == 3 and out["answers"]["urgent"]["stderr"] < 1e-9 and out["answers"]["urgent"]["agreement"] == 1.0
assert len(SEEN) == 3 and len({tuple(r["vllm_xargs"]["diffusion_seed_canvas"]) for r in SEEN}) == 3, "reads must differ in their noise"
assert all(r["chat_template_kwargs"] == {"enable_thinking": False} and "ignore_eos" not in r for r in SEEN)
assert out["diagnostics"]["thought"] is None and d["usage"]["completion_tokens"] == len(TEMPLATE) + 1
print("fixed samples ok; upstream saw", len(SEEN), "reads, max_tokens", SEEN[0]["max_tokens"])

# auto policy: an uncertain first read extends to max, a confident one stops at one
auto = dict(SCHEMA); auto.pop("samples")
for conf, want in [(0.7, 4), (0.999, 1)]:
    SEEN.clear(); CONF[0] = conf
    code, d = post({"messages": [{"role": "system", "content": json.dumps(auto)}, {"role": "user", "content": "{}"}]})
    out = json.loads(d["choices"][0]["message"]["content"])
    print("auto conf", conf, out["diagnostics"]["samples"]["policy"], "reads", len(SEEN))
    assert out["diagnostics"]["samples"]["n"] == want and len(SEEN) == want
    dq = out["diagnostics"]["questions"]
    assert [dq[q]["pos"] for q in ("urgent", "bucket", "tone")] == [s["pos"] for s in SLOTS]
    assert all(len(dq[q]["entropy"]) == want for q in dq)
    assert out["diagnostics"]["samples"]["policy"]["first_read_entropy"] == {q: dq[q]["entropy"][0] for q in dq}
CONF[0] = 0.7

# think: one generation in the thought channel, then reads that carry it in their prompt
SEEN.clear()
code, d = post({"messages": [{"role": "system", "content": json.dumps(dict(SCHEMA, samples=2, think=64))}, {"role": "user", "content": "{}"}]})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
gen, reads = SEEN[0], SEEN[1:]
assert gen["_path"].endswith("/v1/completions") and gen["max_tokens"] == 64 and gen["stop_token_ids"] == S.THOUGHT_CLOSE
assert gen["prompt"][-len(S.THOUGHT_OPEN):] == S.THOUGHT_OPEN and "<|think|>" in S.TOK.decode(gen["prompt"]), "thinking on, open tag last"
assert len(reads) == 2 and all(r["prompt"] == gen["prompt"] + THOUGHT + S.THOUGHT_CLOSE for r in reads), "reads continue the closed thought"
assert all(r["vllm_xargs"]["diffusion_seed_canvas"][: len(S.SCAFFOLD)] != S.SCAFFOLD for r in reads), "no second thought block on the canvas"
th = out["diagnostics"]["thought"]
assert th["tokens"] == len(THOUGHT) and th["closed"] and "urgent" in th["text"], th
assert out["answers"]["urgent"]["label"] == "yes" and out["diagnostics"]["samples"]["n"] == 2
assert d["usage"]["completion_tokens"] == len(THOUGHT) + reads[0]["max_tokens"]
print("think ok: thought", repr(th["text"]), "then", len(reads), "reads")

# chunking: twelve yes/no questions do not fit 32 rows, so the server splits them
SEEN.clear(); CONF[0] = 0.7
many = {"questions": [{"id": f"w{i}", "type": "noul", "instructions": f"word {i}?"} for i in range(12)],
        "samples": 1, "chunk_rows": 32}
assert schema["format"] == "lines" and S.parse_schema(many)["format"] == "indexed"
code, d = post({"messages": [{"role": "system", "content": json.dumps(dict(many, chunk_prompt="shared"))}, {"role": "user", "content": "{}"}]})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
chunks = out["diagnostics"]["chunks"]
assert len(chunks) > 1 and sum(len(c) for c in chunks) == 12, chunks
assert all(out["answers"][f"w{i}"]["label"] == "yes" for i in range(12))
assert len(SEEN) == len(chunks) and len({r["messages"][0]["content"] for r in SEEN}) == 1, "chunks share one system prompt"
assert all(r["vllm_xargs"]["diffusion_canvas_length"] <= 32 for r in SEEN)
print("chunked:", chunks, "reads", len(SEEN))
SEEN.clear()
code, d = post({"messages": [{"role": "system", "content": json.dumps(many)}, {"role": "user", "content": "{}"}]})
assert code == 200 and len({r["messages"][0]["content"] for r in SEEN}) == len(chunks), "own prompts (the default) differ per chunk"
print("chunked with own prompts ok")

# parallel chunks each think for themselves
SEEN.clear()
code, d = post({"messages": [{"role": "system", "content": json.dumps(dict(many, think=64))}, {"role": "user", "content": "{}"}]})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
assert len(SEEN) == 2 * len(chunks) and sum("vllm_xargs" not in r for r in SEEN) == len(chunks)
assert [t["tokens"] for t in out["diagnostics"]["thought"]] == [len(THOUGHT)] * len(chunks)
print("chunked with a thought per chunk ok")

# sequential chunks: chunk two continues chunk one's answer in the prompt
SEEN.clear()
code, d = post({"messages": [{"role": "system", "content": json.dumps(dict(many, sequential=True))}, {"role": "user", "content": "{}"}]})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
assert out["diagnostics"]["sequential"] and len(SEEN) == len(chunks)
first, second = SEEN[0], SEEN[1]
assert first["_path"].endswith("/v1/chat/completions") and "messages" in first
assert second["_path"].endswith("/v1/completions") and isinstance(second["prompt"], list)
tail = S.TOK.decode(second["prompt"][-40:])
assert "<channel|>" in tail and tail.endswith(chunks[0][-1] + "yes") and "w0yes w1yes" in tail, tail
print("sequential ok: chunk 2 prompt ends", repr(tail[-40:]))

# sequential chunks share one thought, written under the full question list
SEEN.clear()
code, d = post({"messages": [{"role": "system", "content": json.dumps(dict(many, sequential=True, think=64))}, {"role": "user", "content": "{}"}]})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
gen, first, second = SEEN[0], SEEN[1], SEEN[2]
assert len(SEEN) == 1 + len(chunks) and "vllm_xargs" not in gen
assert first["prompt"] == gen["prompt"] + THOUGHT + S.THOUGHT_CLOSE, "chunk 1 reads right after the thought"
assert second["prompt"][: len(first["prompt"])] == first["prompt"] and S.TOK.decode(second["prompt"][-40:]).endswith(chunks[0][-1] + "yes")
assert isinstance(out["diagnostics"]["thought"], dict) and out["diagnostics"]["thought"]["tokens"] == len(THOUGHT)
assert d["usage"]["completion_tokens"] == len(THOUGHT) + sum(r["max_tokens"] for r in SEEN[1:])
print("sequential with one thought ok")

# bad requests
for body, want in [
    ({"messages": [{"role": "user", "content": "{}"}]}, "exactly two"),
    ({"messages": [{"role": "system", "content": "hello"}, {"role": "user", "content": "{}"}]}, "JSON question schema"),
    ({"messages": [{"role": "system", "content": json.dumps(SCHEMA)}, {"role": "user", "content": "not json"}]}, "JSON question schema"),
    ({"messages": [{"role": "system", "content": json.dumps({"questions": [{"id": "a", "type": "choice", "options": ["x"]}]})}, {"role": "user", "content": "{}"}]}, "at least two"),
    ({"messages": [{"role": "system", "content": json.dumps({"questions": [{"id": "a", "type": "noul"}] * 2})}, {"role": "user", "content": "{}"}]}, "duplicate"),
    ({"messages": [{"role": "system", "content": json.dumps({"questions": [{"id": "q" * 300, "type": "noul"}]})}, {"role": "user", "content": "{}"}]}, "canvas holds"),
    ({"messages": [{"role": "system", "content": json.dumps(dict(SCHEMA, think="lots"))}, {"role": "user", "content": "{}"}]}, "think must be"),
    ({"messages": [{"role": "system", "content": json.dumps(dict(SCHEMA, think=8))}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]}, "think needs a text state"),
]:
    code, d = post(body)
    assert code == 400 and want in d["error"]["message"], (code, d)
    print("400 ok:", d["error"]["message"][:90])
print("ALL OK")
