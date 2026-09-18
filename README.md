# dgemma-spark

DiffusionGemma 26B-A4B (NVFP4) serving structured decisions on a DGX Spark.
One container runs vLLM with the structured-reads changes on port 8010 and
the structured decision server on port 8011 in front of it.

A structured decision is one denoise step over a seeded canvas: the server
turns a question schema into the canvas, reads a calibrated distribution per
question from the logprobs, and returns the answers as JSON. The engine
changes are on the `structured-reads` branch of
[mmastrac/vllm](https://github.com/mmastrac/vllm/tree/structured-reads) and
upstream as [vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250).

## What the image is

Nothing is compiled. The base is the per-model vLLM image for GLM-5.3-Flash
on arm64 with CUDA 13 (`vllm/vllm-openai:glm53-flash-arm64-cu130`), whose
vLLM is commit 487ecf187. The fork branch is that commit plus the read
changes, which touch python files only, so the build:

1. pins FlashInfer to the nightly the reads were measured on and puts
   cutlass-dsl back at the version GB10 wants;
2. links the CUDA headers and unversioned `.so` names the base leaves out,
   without which FlashInfer's JIT dies on `nvrtc.h` (`patches/link_cuda_headers.sh`);
3. checks out the fork at a pinned commit and copies its changed `vllm/`
   files over the base's site-packages, after asserting the base's vLLM is
   still commit 487ecf187 (`patches/overlay_vllm.py`);
4. raises torch dynamo's recompile limit for the sampler, which is one
   specialization per canvas width (`patches/raise_recompile_limit.py`);
5. adds an optional per-worker memory cap, inert unless `TORCH_MEM_FRACTION`
   is set (from home-infra's glm53 image);
6. installs the structured server from the same fork commit at
   `/opt/dgemma/structured_server.py`, with its README beside it.

`VLLM_REF` and `VLLM_BASE` in the Dockerfile move together: a new fork
commit must be built on the base image's vLLM commit, and the overlay step
refuses anything else.

## Requirements

- A DGX Spark or another GB10 box: aarch64, unified memory, CUDA 13 driver.
- Docker with the NVIDIA container runtime and BuildKit (any current Docker).
- About 40 GB free on disk for the base image and JIT cache, 18 GB for the
  checkpoint.
- Memory: the weights take about 19 GB and the start-up profile pass adds a
  transient that scales with `MAX_SEQS x CANVAS`. The entrypoint refuses to
  start unless that plus `HEADROOM_GB` is free. On unified memory an
  overshoot hangs the host rather than failing a request, so do not run this
  next to another large model.

## Quick start

```bash
cp .env.example .env            # optional; defaults are in compose.yaml
scripts/download-model.sh       # nvidia/diffusiongemma-26B-A4B-it-NVFP4 into ./models/dgemma
docker compose up -d --build
docker compose logs -f dgemma   # first start JIT-builds FlashInfer kernels, several minutes; later starts take about 30 s
scripts/smoke.sh                # one generation through vLLM, one decision through the server
```

The JIT output and torch's compile cache live in `./cache`, so the build
cost is paid once per host.

## Using it

vLLM's OpenAI API is on port 8010 as usual, model name `dgemma`. The
structured server is on port 8011 and speaks `/v1/chat/completions` with two
messages: the schema as the system message, the state JSON as the user
message. The reply content is one distribution per question.

```bash
curl -s localhost:8011/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"}]}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon.\"}"}
  ]}'
```

Question types are `noul` (yes/no), `choice` with `options`, and `score`
with ordered `levels`. Schema options cover noise draws (`samples`), chunking
of long question lists, sequential chunks that condition on earlier answers,
image parts in the state, and `think`, which lets the model write a thought
in its thought channel before the read. The full schema is documented at the
top of `/opt/dgemma/structured_server.py` and in `/opt/dgemma/README.md`
inside the image, both from the fork.

## Configuration

Every knob is an environment variable with the same default in
`compose.yaml` and `.env.example`.

| variable | default | meaning |
|---|---|---|
| `CANVAS` | 128 | served canvas in tokens; a read only pays for its own width |
| `MAX_SEQS` | 32 | concurrent requests |
| `MAX_MODEL_LEN` | 4096 | prompt plus canvas |
| `GPU_UTIL` | 0.40 | fraction of the box's memory vLLM plans for |
| `ATTN` | TRITON_ATTN | attention backend; FlashInfer cannot mix causal and bidirectional here |
| `EXTRA_ARGS` | `--async-scheduling` | appended to `vllm serve` |
| `HEADROOM_GB` | 12 | free memory the entrypoint insists on beyond weights and transient |
| `TORCH_MEM_FRACTION` | empty | per-worker cap; empty leaves the worker unbounded |
| `PORT`, `STRUCTURED_PORT` | 8010, 8011 | the two listeners, on the host network |

## Tests

`scripts/self-test.sh` runs the server's end-to-end test against a fake
upstream inside the image. It needs the model directory for the tokenizer
and nothing else, so it runs on any machine that can run the image.

## Updating the engine

Push a new commit to the fork's `structured-reads` branch, set `VLLM_REF` in
the Dockerfile to it, and rebuild. If the base image's vLLM moves, the fork
branch has to be rebased onto the new commit first and `VLLM_BASE` updated
to match; the build stops otherwise.

## Provenance

This recipe reproduces a working setup that ran on a Spark as a stock
`glm53-spark` image with the same files bind-mounted over site-packages. The
Dockerfile bakes those layers instead. `patches/link_cuda_headers.sh`,
`patches/spark_mem_trace.py` and `patches/worker_memory_cap.py` are copied
from home-infra's `infra/docker/spark/glm53/patches`.
