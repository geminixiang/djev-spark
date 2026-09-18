# djev-spark

DiffusionGemma as Jev, for the Spark: DiffusionGemma 26B-A4B (NVFP4) on a DGX
Spark, serving structured decisions.

This is a single container that runs vLLM with the structured-reads patches on port 8010, and the
structured decision server on port 8011 in front of it. The server speaks Jev's `POST /v1/systemone` API.

The engine patches are from [vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250).
The container builds from branch `structured-reads-spark` of
[mmastrac/vllm](https://github.com/mmastrac/vllm/tree/structured-reads-spark), which
is a combination of the open PRs (https://github.com/vllm-project/vllm/pulls/mmastrac).

## Image

- Base: `vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910`.
  vLLM 0.29.1rc1.dev347 and ships flashinfer 0.6.18.post1 with
  a prebuilt kernel cache, so the engine starts in about 90 s with no JIT.
- Overlay: the fork branch's changed `vllm/` files (9 files) are copied over
  the base's site-packages.
- `patches/link_cuda_headers.sh`: links the CUDA headers and unversioned
  `.so` names the base leaves out. FlashInfer's JIT needs them when it runs (it currently doesn't but just in case).
- `patches/raise_recompile_limit.py`: sets torch dynamo's recompile limit
  to 64 for the sampler module. Each canvas width is one specialization
  and the default of 8 is too small for normal use.
- `patches/worker_memory_cap.py`, `patches/spark_mem_trace.py`: per-worker
  memory cap, active only when `TORCH_MEM_FRACTION` is set. From
  a local, un-upstreamed patch (this one might get dropped later on).
- `server/structured_server.py`: the structured server, installed at
  `/opt/dgemma/structured_server.py`. It is the fork's example server plus
  the `/v1/systemone` route.

## Porting Notes

To move to a newer engine: rebase the fork branch onto main, set `VLLM_REF`
to its head, set `BASE` and `VLLM_BASE` to a nightly at or after that
commit.

## Requirements

This _might_ work on other hardware, reports welcome.

- DGX Spark or other GB10 box (aarch64, unified memory, CUDA 13 driver).
- Docker with the NVIDIA runtime and BuildKit.
- 25 GB disk for the image, 18 GB for the checkpoint.
- Memory: weights 19 GB, KV pool `KV_CACHE_GB`, plus a start-up transient
  that scales with `MAX_SEQS x CANVAS`.

## Run

```bash
cp .env.example .env
scripts/download-model.sh       # nvidia/diffusiongemma-26B-A4B-it-NVFP4 -> $MODELS_DIR/$MODEL_NAME
docker compose up -d --build
docker compose logs -f dgemma
scripts/smoke.sh
```

`POST /v1/systemone` accepts Jev's request body format and returns Jev's style of answers.
No API key. `model` is ignored.

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team owns this?",
             "criteria": {"billing": null, "outage": "service down", "feature": null}},
    "tone": {"type": "score", "instructions": "How angry is the customer?",
             "criteria": ["calm", "annoyed", "furious"]}
  }}'
```

### Request

| field | description |
|---|---|
| `state` | The content the questions are asked about. A string, object or array. |
| `questions` | Map of question id to question object. Answers use the same ids, in the same order. |
| `model` | Accepted for compatibility with Jev clients and ignored. |
| `seed` | Optional. Seeds the noise draws; the same request with the same seed gives the same answer. Default 42. |

Each question has a `type`, an `instructions` string (the question itself), and `criteria`, whose shape depends on the type:

| type | what it asks | `criteria` |
|---|---|---|
| `noul` | Yes or no. | Optional. `{"true": "what yes means", "false": "what no means"}`. The descriptions are rendered next to the labels in the prompt. |
| `choice` | One of several options. | Required. Object mapping each option name to a description, or `null` for none. Option order is kept. |
| `score` | A level on an ordered scale. | Required. List of level names from lowest to highest, at least two. |

### Response

| field | what it holds |
|---|---|
| `answers` | One answer per question id. Shape depends on the question type; see below. |
| `usage` | `input_tokens`: the prompt input count, images included. `output_tokens`: the canvas rows read, plus any thought tokens. |
| `diagnostics` | Server-specific, subject to change. |
| `model` | The served model name. |

| answer type | fields |
|---|---|
| `noul` | `noul`: probability of yes, 0 to 1. |
| `choice` | `choice`: the most probable option. `probabilities`: probability per option name. `confidence`: the probability of `choice`. |
| `score` | `score`: probability-weighted level index, lowest level 0. `legend`: level index to level name. `probabilities`: probability per level index. `confidence`: the probability of the most probable level. |

Validation errors return 422 with `{"error": {"message": ...}}`. A failed upstream call returns 502.

### Extensions

This server extends Jev's API.

These go at the top level of the request body, beside `state` and `questions`.

| key | default | description |
|---|---|---|
| `samples` | `"auto"` | Number of noise draws to average. `"auto"` reads once, then more only if the first read's entropy is above `auto_threshold`. |
| `auto_max` | 4 | Maximum reads under `"auto"`. |
| `auto_threshold` | 0.1 | Entropy above which `"auto"` reads again. |
| `think` | 0 | The model writes up to this many tokens of thought before the read; the read conditions on it. One extra generation per decision. With images the thought is seeded into the canvas, so the canvas bounds it. |
| `instructions` | | Context rendered ahead of the questions. Can help KV prefix reuse when multiple questions are sent with it. |
| `chunk_rows` | canvas | Splits a long question list into chunks of at most this many rows. Default is the served canvas. |
| `chunk_prompt` | `"own"` | Whether each chunk's prompt lists only its own questions (`"own"`) or every question (`"shared"`). |
| `sequential` | false | Runs chunks in order, prefilling each chunk's answers before the next, so later answers condition on earlier ones. Text-only states. |
| `ask` | | List of question ids to answer in one read; the rest are skipped. |
| `steps` | 1 | Denoise steps per read. More than 1 lets the canvas drift from the template. |

### Dependencies

By default every question in a request is read in one canvas: the answers
share the prompt and the canvas but do not see each other. Three
per-question keys change that.

| key | what it does |
|---|---|
| `depends_on` | List of question ids. This question is read in a later stage than those, with their answers in its prompt. |
| `ask_if` | Map of question id to a list of that question's answers (option names, level names, or `yes`/`no`). The question is asked only when the answer is in the list; otherwise its answer is `null`. Implies `depends_on`. |
| `alone` | `true` reads the question in a canvas of its own, beside the others in its stage. For questions that pull each other, like a direction question next to a hazard question. |

Questions run in stages by their dependencies, in declaration order within
a stage. A stage is one joint read, chunked by the canvas, with `alone`
questions in their own reads, all concurrent. Later stages continue the
earlier answers: for a text state as a prefilled continuation of the same
prompt, about one read's cost per stage; for an image state as a fresh
read with the earlier answers restated in the state text, one image
decision per stage. Cycles, unknown ids, `ask_if` values that are not
answers of the named question, and an `ask` list missing a dependency
are 422s. `diagnostics.stages` lists the ids read in each stage,
`diagnostics.skipped` the gated ones with the answer that gated them, and
`diagnostics.conditioning` is `prefill`, `restated` or null.

```json
"questions": {
  "ahead": {"type": "choice", "instructions": "...", "criteria": {"all clear ahead": null, "danger: wall ahead": null}},
  "side": {"type": "choice", "instructions": "Which half of the frame is the obstacle in?",
           "criteria": {"left half": null, "right half": null},
           "ask_if": {"ahead": ["danger: wall ahead"]}}
}
```

### Images

Jev's API has no images. This server takes them in two forms. Images go ahead of the state in the prompt.

| form | how |
|---|---|
| multipart | `multipart/form-data` with the JSON body in a part named `request` and each image as a file part. Any part name, any number of images, in order. This is what `curl -F` and browser `FormData` send. |
| JSON | An `images` array in the body, each entry a `data:image/...;base64,...` URL or an object `{"content_type": "image/png", "base64": "..."}`. |

`sequential` needs a text-only state and returns 422 with images.

```bash
curl -s localhost:8011/v1/systemone \
  -F 'request={"model": "jev-latest", "state": {"note": "the photo is from the returns desk"},
               "questions": {"damaged": {"type": "noul", "instructions": "Is the item damaged?"}}}' \
  -F 'photo=@returns/1234.jpg'
```

Playground: with `TEST_PAGE=1`, `http://<box>:8011/` serves a page with the
request JSON in a textbox and an image mode dropdown: none, image file,
webcam one-shot (capture and send), webcam live (a frame every N seconds),
webcam realtime (capture again as each answer returns). `#req=<base64 JSON>`
in the URL fills and sends a request on load. Off by default.

Webcam live image mode:

![playground in webcam realtime mode](docs/playground.jpg)

Static image tests:

![playground with image](docs/triangle.png)

### From another device

The container is on the host network and both servers listen on all
interfaces, so `http://<box-ip>:8011/` works from the LAN with no port
mapping. Browsers open a webcam only on a secure origin, so for the
playground's webcam modes from another device set `TLS_PORT` (for example
8443) and use `https://<box-ip>:8443/`; the certificate is self-signed and
the browser asks once.

### Walking demo

`https://<box-ip>:8443/walk` (with `TEST_PAGE=1` and `TLS_PORT=8443`) is a
phone page: it streams the back camera, sends a 512px frame per round trip
as one hazard question, and shows the label full width under the video,
a direction arrow over it, and the probabilities below, no scrolling. Hazard labels: all clear ahead, danger: stairs, danger: wall
ahead, danger: object ahead, danger: pet ahead, door ahead; the
instructions tell the model to judge only a 30 degree window at the
center of the frame and to say all clear when nothing is within 2 meters
there. A second question, gated with `ask_if` on the danger labels, asks
which half of the frame the obstacle is in, and the arrow points the
other way; on a clear frame it is not asked and the arrow is go ahead. Two things measured on
frames with an obstacle on a known side decided that shape. Asking the
turn directly ("which side is more open", turn left or turn right) was
biased right: p(turn left) 0.03 to 0.25 whichever side the obstacle was
on, and listing right first flipped it the other way. Asking where the
obstacle is was right at 0.95 to 1.00. And the two questions are not
asked in one read: a direction question beside the hazard question
pulled the hazard slot toward obstacles (all clear on a clear frame fell
from 0.8 to 0.04). A clear frame is one image decision, about 320 ms on
the Spark plus the upload; a blocked frame is two. The "think" box adds a 64-token thought per frame, written
with the frame in view and seeded into the read's canvas ahead of the
answer; the canvas bounds the thought, so at 128 rows a request for more
is clipped and `diagnostics.thought.budget` says to what.

To reload the server code without restarting vLLM: copy the files into
the container and kill the server process; the entrypoint restarts it.

```bash
docker cp server/. dgemma:/opt/dgemma/ && docker exec dgemma pkill -f structured_server.py
```

### Other routes

| route | what it does |
|---|---|
| `POST /v1/chat/completions` | The same decision as an OpenAI-shaped call: system message = the schema JSON, user message = the state JSON or image parts; reply `content` = the answer JSON. Schema documented at the top of `server/structured_server.py`. |
| `POST /v1/raw/chat/completions` | Passes the body to vLLM's chat completions unchanged: plain generation through the same port and, with `API_KEY`, the same token. |
| `GET /health` | Always open, no token. |

## Configuration

Environment variables, same defaults in `compose.yaml` and `.env.example`.

| variable | default | meaning |
|---|---|---|
| `MODELS_DIR`, `MODEL_NAME` | `./models`, `dgemma` | checkpoint at `$MODELS_DIR/$MODEL_NAME` |
| `CACHE_DIR` | `./cache` | flashinfer autotune and torch compile cache |
| `CANVAS` | 128 | served canvas in tokens; a read pays for its own width |
| `MAX_SEQS` | 32 | concurrent requests |
| `MAX_MODEL_LEN` | 4096 | prompt plus canvas |
| `GPU_UTIL` | 0.40 | fraction of box memory vLLM plans for |
| `KV_CACHE_GB` | 2 | KV pool, fixed |
| `MAX_NUM_BATCHED_TOKENS` | empty | prefill chunk; empty = vLLM default |
| `ATTN` | TRITON_ATTN | attention backend |
| `EXTRA_ARGS` | `--async-scheduling` | appended to `vllm serve` |
| `HEADROOM_GB` | 12 | free memory required beyond weights, KV and transient |
| `TORCH_MEM_FRACTION` | empty | per-worker cap; empty = unbounded |
| `TEST_PAGE` | empty | `1` serves the playground page at `/` on the structured port |
| `TLS_PORT` | 0 | nonzero adds an HTTPS listener with a self-signed certificate; browsers need it to open a webcam from another device |
| `API_KEY` | empty | when set, POST routes on the structured port need `Authorization: Bearer <key>` |
| `PORT`, `STRUCTURED_PORT` | 8010, 8011 | host network |

128k profile (in `.env.example`, commented): `MAX_MODEL_LEN=131072
KV_CACHE_GB=24 GPU_UTIL=0.45`.

## Benchmarks

All on one GX10 (GB10, 121 GB), 2026-09-18, this image, nothing else
running. Engine init 88 to 93 s.

KV cache at `MAX_MODEL_LEN=131072 KV_CACHE_GB=24`: 1,808,085 tokens,
13.79 x 128k requests (about 1.7 GiB per 128k request; 25 of 30 layers are
sliding-window 1024 and the hybrid allocator gives them only the window).
Host memory free with the container up: 67 GB.

Single reads, canvas width 32, sequential, medians of 15 (`vllm-patch/bench_read.py`):

| case | median ms |
|---|---|
| read, no logprobs | 98.4 |
| read, top5 logprobs | 101.7 |
| read, long state (12x) | 93.2 |
| read, same state every time (cached) | 103.9 |
| read, 2 steps | 225.1 |
| read, 3 steps | 282.0 |
| commit path (not read-only) | 203.0 |
| structured server, samples=1 | 104.3 |

Concurrency, read-only single reads, canvas 32, unique state per request,
15 s per level (`vllm-patch/curve.py`), `MAX_MODEL_LEN=131072 MAX_SEQS=32`:

| clients | req/s | decisions/s | p50 s | p95 s |
|---|---|---|---|---|
| 1 | 8.54 | 25.6 | 0.12 | 0.12 |
| 8 | 27.87 | 83.6 | 0.28 | 0.30 |
| 16 | 41.41 | 124.2 | 0.38 | 0.42 |
| 32 | 49.47 | 148.4 | 0.60 | 0.81 |

Same curve with `MAX_SEQS=16`: 32 clients 42.89 req/s, p50 0.74 s.

Previous build (vLLM 487ecf187 base, same patches, `MAX_MODEL_LEN=4096`,
2026-09-17): 1 client 8.7 req/s, 8 clients 27.8, 32 clients 53.3 to 54.0.

This image at `MAX_MODEL_LEN=4096 MAX_SEQS=32`: 1 client 8.29 req/s (p50
0.12 s), 32 clients 53.37 req/s (160.1 decisions/s, p50 0.58 s, p95 0.79 s).
The 32-client difference between the two tables comes from the model
length, not the engine version.

The first batch at a new tile width or batch size compiles once
(8 concurrent cold: 7.7 s).

Long states, 128k profile, one decision per state, cold then warm
(`scripts/long-context-probe.py`):

| state tokens | cold s | warm s |
|---|---|---|
| 8,678 | 5.42 | 0.14 |
| 35,133 | 13.37 | 0.21 |
| 110,707 | 104.94 | 0.44 |

Same with `MAX_NUM_BATCHED_TOKENS=32768`: 38,448 tokens 30.18 / 0.23 s;
110,707 tokens 147.79 / 0.51 s; KV pool 413,955 tokens (3.16 x 128k).

## Tests

- `scripts/self-test.sh`: the server's fake-upstream test inside the image.
  Needs the checkpoint for its tokenizer, no GPU.
- `scripts/smoke.sh`: one generation on 8010, one decision on 8011.
- `scripts/long-context-probe.py [tokens ...]`: cold and warm decision
  latency over states of the given sizes.

## Files

```
Dockerfile                      base + fork engine overlay + patches + server
compose.yaml                    one service, host network
entrypoint.sh                   memory guard, vllm serve, structured server
.env.example
patches/link_cuda_headers.sh
patches/overlay_vllm.py
patches/raise_recompile_limit.py
patches/worker_memory_cap.py
patches/spark_mem_trace.py
scripts/download-model.sh
scripts/smoke.sh
scripts/self-test.sh
scripts/long-context-probe.py
server/structured_server.py
server/playground.html
server/walk.html
server/test_structured_server.py
```
