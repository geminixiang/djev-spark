# syntax=docker/dockerfile:1
# DiffusionGemma NVFP4 structured reads on a DGX Spark (GB10, aarch64, CUDA 13).
#
# Nothing is compiled. The base is one of vLLM's per-commit nightly images,
# which are multi-arch and CUDA 13, and the fork branch holds the read changes
# on top of upstream main, python files only. The build copies the branch's
# changed vllm/ files over the base's site-packages. That is sound as long as
# upstream has not touched those files between the branch's own base and the
# image's commit, which the fork stage checks; when it has, the branch needs
# its routine rebase onto main first.

ARG BASE=vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910

# --- the fork, at a pinned commit --------------------------------------------
FROM alpine/git:latest AS fork
ARG VLLM_FORK=https://github.com/mmastrac/vllm.git
ARG VLLM_UPSTREAM=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=bb82320bdc2eb88425c6a4a3b14780bd33bb9d03
ARG VLLM_BASE=dee37d89115db4c94a820a79a78a7828e141c910
RUN git clone --filter=blob:none --quiet "${VLLM_FORK}" /fork \
    && cd /fork \
    && git checkout --quiet "${VLLM_REF}" \
    && git fetch --filter=blob:none --quiet "${VLLM_UPSTREAM}" "${VLLM_BASE}" \
    && mb=$(git merge-base "${VLLM_BASE}" HEAD) \
    && git diff --name-only "$mb" HEAD -- vllm > /fork/changed.txt \
    && cat /fork/changed.txt \
    && if ! git diff --quiet "$mb" "${VLLM_BASE}" -- $(cat /fork/changed.txt); then \
         echo "upstream changed overlaid files between the branch base and ${VLLM_BASE}; rebase the branch first" >&2; \
         git diff --stat "$mb" "${VLLM_BASE}" -- $(cat /fork/changed.txt) >&2; exit 1; fi

# --- the image ---------------------------------------------------------------
FROM ${BASE}
ARG VLLM_BASE=dee37d89115db4c94a820a79a78a7828e141c910

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

# The structured server: this repo's copy, which is the fork's example
# server plus Jev's /v1/systemone contract.
COPY server/structured_server.py server/playground.html server/test_structured_server.py /opt/dgemma/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && python3 -m py_compile /opt/dgemma/structured_server.py

# 8010 vLLM (OpenAI API), 8011 structured decisions
EXPOSE 8010 8011
ENTRYPOINT ["/entrypoint.sh"]
