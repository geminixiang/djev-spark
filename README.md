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

Nothing is compiled. The base is one of vLLM's per-commit nightly images
(`vllm/vllm-openai:nightly-<sha>`), which are multi-arch and CUDA 13 and
ship FlashInfer's prebuilt kernel cache, so the engine starts in about 90 s
with no JIT. The fork branch is the structured-reads PR branch on upstream
main plus the open dtype-cast bugfix; its changes touch python files only,
so the build:

1. checks out the fork at a pinned commit and lists the `vllm/` files it
   changes against its own base, then refuses to continue if upstream has
   touched any of those files between that base and the image's commit
   (the fork stage of the Dockerfile);
2. links the CUDA headers and unversioned `.so` names the base leaves out,
   which FlashInfer's JIT needs whenever it does run (`patches/link_cuda_headers.sh`);
3. copies the changed files over the base's site-packages, after asserting
   the base's vLLM version names the expected commit (`patches/overlay_vllm.py`);
4. raises torch dynamo's recompile limit for the sampler, which is one
   specialization per canvas width (`patches/raise_recompile_limit.py`);
5. adds an optional per-worker memory cap, inert unless `TORCH_MEM_FRACTION`
   is set (from home-infra's glm53 image);
6. installs the structured server from the same fork commit at
   `/opt/dgemma/structured_server.py`, with its README beside it.

Keeping the container current is the same job as keeping the PR current.
When upstream changes one of the overlaid files, rebase the branch onto
main, then point `VLLM_REF` at the new head and `BASE` and `VLLM_BASE` at a
nightly at or after that commit.

## Requirements

- A DGX Spark or another GB10 box: aarch64, unified memory, CUDA 13 driver.
- Docker with the NVIDIA container runtime and BuildKit (any current Docker).
- About 25 GB free on disk for the image, 18 GB for the checkpoint.
- Memory: the weights take about 19 GB and the start-up profile pass adds a
  transient that scales with `MAX_SEQS x CANVAS`. The entrypoint refuses to
  start unless that plus `HEADROOM_GB` is free. On unified memory an
  overshoot hangs the host rather than failing a request, so do not run this
  next to another large model.

## Quick start

```bash
cp .env.example .env            # optional; defaults are in compose.yaml
scripts/download-model.sh       # nvidia/diffusiongemma-26B-A4B-it-NVFP4 into ./models/dgemma (MODELS_DIR and MODEL_NAME move it)
docker compose up -d --build
docker compose logs -f dgemma   # the engine is up in about 90 s
scripts/smoke.sh                # one generation through vLLM, one decision through the server
```

FlashInfer's autotune results and torch's compile cache live in `./cache`.

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

## Long contexts

The model has 25 sliding-window layers (1024 tokens) and 5 global layers,
and this vLLM's hybrid allocator gives the sliding layers only their window,
so a 128k request costs about 1.7 GiB of KV rather than the 14 GiB it would
at full length. The profile in `.env.example` (`MAX_MODEL_LEN=131072`,
`MAX_SEQS=16`, `KV_CACHE_GB=24`, `GPU_UTIL=0.45`) holds thirteen 128k
requests at once and leaves the box with about 67 GB free.

What a long state costs, measured on one Spark with that profile:

| state tokens | first decision (prefill) | later decisions (prefix cached) |
|---|---|---|
| 8.7k | 5.4 s | 0.14 s |
| 35k | 13 s | 0.21 s |
| 111k | 105 s | 0.44 s |

The prefill is paid once per distinct prefix, so put the long document
first and vary the questions after it. The read itself grows with context
because the global layers attend over all of it, but stays under half a
second at 111k. A larger prefill chunk (`MAX_NUM_BATCHED_TOKENS=32768`)
made both columns worse and cost the KV pool most of its capacity; leave it
at vLLM's default.

## Tests

`scripts/self-test.sh` runs the server's end-to-end test against a fake
upstream inside the image. It needs the model directory for the tokenizer
and nothing else, so it runs on any machine that can run the image.
`scripts/long-context-probe.py` sends one decision over states of the sizes
you give it, cold and then warm, and prints the table above for your box.

## Provenance

This recipe replaces a setup that ran on a Spark as a locally built
`glm53-spark` image with the same files bind-mounted over site-packages. `patches/link_cuda_headers.sh`,
`patches/spark_mem_trace.py` and `patches/worker_memory_cap.py` are copied
from home-infra's `infra/docker/spark/glm53/patches`.
