# djev-spark

DiffusionGemma as Jev, for the Spark: DiffusionGemma 26B-A4B (NVFP4) on a DGX
Spark, serving structured decisions.
One container: vLLM with the structured-reads patches on port 8010, the
structured decision server on port 8011 in front of it.

The engine patches are [vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250).
The container builds from branch `structured-reads-spark` of
[mmastrac/vllm](https://github.com/mmastrac/vllm/tree/structured-reads-spark),
which is that PR plus the open dtype-cast fix.

## Image

- Base: `vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910`.
  vLLM publishes one of these per main commit, multi-arch, CUDA 13.0.2.
  This one is vLLM 0.29.1rc1.dev347 and ships flashinfer 0.6.18.post1 with
  its prebuilt kernel cache, so the engine starts in about 90 s with no JIT.
- Overlay: the fork branch's changed `vllm/` files (9 files) are copied over
  the base's site-packages. Nothing is compiled. The fork stage of the
  Dockerfile fails the build if upstream touched any of those files between
  the branch's base and the image's commit; that is when the branch needs a
  rebase onto main.
- `patches/link_cuda_headers.sh`: links the CUDA headers and unversioned
  `.so` names the base leaves out. FlashInfer's JIT needs them when it runs.
- `patches/raise_recompile_limit.py`: sets torch dynamo's recompile limit
  to 64 for the sampler module. Each canvas width is one specialization
  and the default of 8 is exceeded in normal use.
- `patches/worker_memory_cap.py`, `patches/spark_mem_trace.py`: per-worker
  memory cap, active only when `TORCH_MEM_FRACTION` is set. From
  home-infra `infra/docker/spark/glm53/patches`.
- The structured server from the fork commit is at
  `/opt/dgemma/structured_server.py`, its README at `/opt/dgemma/README.md`.

To move to a newer engine: rebase the fork branch onto main, set `VLLM_REF`
to its head, set `BASE` and `VLLM_BASE` to a nightly at or after that
commit.

## Requirements

- DGX Spark or other GB10 box (aarch64, unified memory, CUDA 13 driver).
- Docker with the NVIDIA runtime and BuildKit.
- 25 GB disk for the image, 18 GB for the checkpoint.
- Memory: weights 19 GB, KV pool `KV_CACHE_GB`, plus a start-up transient
  that scales with `MAX_SEQS x CANVAS`. The entrypoint refuses to start
  unless that plus `HEADROOM_GB` is free. An overshoot on unified memory
  hangs the host.

## Run

```bash
cp .env.example .env
scripts/download-model.sh       # nvidia/diffusiongemma-26B-A4B-it-NVFP4 -> $MODELS_DIR/$MODEL_NAME
docker compose up -d --build
docker compose logs -f dgemma
scripts/smoke.sh
```

Structured request: two messages, schema JSON as system, state JSON as
user. Reply content is one distribution per question.

```bash
curl -s localhost:8011/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"}]}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon.\"}"}
  ]}'
```

Question types: `noul`, `choice` with `options`, `score` with `levels`.
Other schema fields: `samples`, `chunk_rows`, `ask`, `sequential`,
`think`, image parts in the state. Documented at the top of
`structured_server.py`.

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

First batch at a new tile width or batch size pays a one-time compile
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
Dockerfile                      base + fork overlay + patches + server
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
server/test_structured_server.py
```
