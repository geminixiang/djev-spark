# syntax=docker/dockerfile:1
# DiffusionGemma NVFP4 structured reads on a DGX Spark (GB10, aarch64, CUDA 13).
#
# Nothing is compiled. The base is the per-model vLLM image for GLM-5.3-Flash
# on arm64 + cu130, pinned by digest because the tag moves: this digest is
# the tag as pulled on 2026-09-18, vLLM 0.28.1rc1.dev580 at commit 385dce36b.
# The structured-reads-0.28 branch of mmastrac/vllm is that commit plus the
# read changes, which touch python files only, so the build overlays those
# files onto the base's site-packages and asserts at build time that the
# base's vLLM is the commit the overlay was written against.

ARG BASE=vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5

# --- the fork, at a pinned commit --------------------------------------------
FROM alpine/git:latest AS fork
ARG VLLM_FORK=https://github.com/mmastrac/vllm.git
ARG VLLM_REF=36951f122ceedceea922b898d7acb47ed0c8444e
ARG VLLM_BASE=385dce36bcee42309924a5ece951a96db3dce7f2
RUN git clone --filter=blob:none --quiet "${VLLM_FORK}" /fork \
    && cd /fork \
    && git checkout --quiet "${VLLM_REF}" \
    && git merge-base --is-ancestor "${VLLM_BASE}" HEAD \
    && git diff --name-only "${VLLM_BASE}" HEAD -- vllm > /fork/changed.txt \
    && cat /fork/changed.txt

# --- the image ---------------------------------------------------------------
FROM ${BASE}
ARG VLLM_BASE=385dce36bcee42309924a5ece951a96db3dce7f2

# The base already ships flashinfer 0.6.18 and cutlass-dsl 4.6.2, the
# versions the reads were measured on, so nothing is pinned here.

# The base ships CUDA libraries without their headers and without some
# unversioned .so symlinks. FlashInfer's JIT needs both; see the script.
COPY patches/link_cuda_headers.sh /tmp/link_cuda_headers.sh
RUN bash /tmp/link_cuda_headers.sh && rm /tmp/link_cuda_headers.sh

# The structured-reads overlay. The build stops if the base's vLLM is not the
# commit the overlay was written against or a target file is missing.
COPY patches/overlay_vllm.py /tmp/overlay_vllm.py
RUN --mount=type=bind,from=fork,source=/fork,target=/fork \
    python3 /tmp/overlay_vllm.py /fork "${VLLM_BASE}" && rm /tmp/overlay_vllm.py

# The sampler is one dynamo specialization per canvas width. Past torch's
# default of 8 every later width runs eager for the rest of the process.
COPY patches/raise_recompile_limit.py /tmp/raise_recompile_limit.py
RUN python3 /tmp/raise_recompile_limit.py && rm /tmp/raise_recompile_limit.py

# Worker memory cap, armed by TORCH_MEM_FRACTION at run time and inert
# otherwise. On unified memory an unbounded worker takes the host down rather
# than its own request. Both files come from home-infra's glm53 image.
COPY patches/spark_mem_trace.py /usr/local/lib/python3.12/dist-packages/spark_mem_trace.py
COPY patches/worker_memory_cap.py /tmp/worker_memory_cap.py
RUN python3 /tmp/worker_memory_cap.py && rm /tmp/worker_memory_cap.py

# The structured server, from the same fork commit as the engine overlay.
COPY --from=fork /fork/examples/features/diffusion_reads/structured_server.py /opt/dgemma/structured_server.py
COPY --from=fork /fork/examples/features/diffusion_reads/README.md /opt/dgemma/README.md
COPY server/test_structured_server.py /opt/dgemma/test_structured_server.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && python3 -m py_compile /opt/dgemma/structured_server.py

# 8010 vLLM (OpenAI API), 8011 structured decisions
EXPOSE 8010 8011
ENTRYPOINT ["/entrypoint.sh"]
