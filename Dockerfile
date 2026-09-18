# syntax=docker/dockerfile:1
# DiffusionGemma NVFP4 structured reads on a DGX Spark (GB10, aarch64, CUDA 13).
#
# Nothing is compiled. The base is the per-model vLLM image for GLM-5.3-Flash
# on arm64 + cu130, whose vLLM is commit 487ecf187. The structured-reads
# branch of mmastrac/vllm is that commit plus the read changes, which touch
# python files only, so the build overlays those files onto the base's
# site-packages and asserts at build time that the base is still the commit
# the overlay was written against.

ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130

# --- the fork, at a pinned commit --------------------------------------------
FROM alpine/git:latest AS fork
ARG VLLM_FORK=https://github.com/mmastrac/vllm.git
ARG VLLM_REF=1050fbca07f7c08f1221819fc5dc556554743ac3
ARG VLLM_BASE=487ecf187d3dfe74d2cf6119a92881dba403c219
RUN git clone --filter=blob:none --quiet "${VLLM_FORK}" /fork \
    && cd /fork \
    && git checkout --quiet "${VLLM_REF}" \
    && git merge-base --is-ancestor "${VLLM_BASE}" HEAD \
    && git diff --name-only "${VLLM_BASE}" HEAD -- vllm > /fork/changed.txt \
    && cat /fork/changed.txt

# --- the image ---------------------------------------------------------------
FROM ${BASE}
ARG VLLM_BASE=487ecf187d3dfe74d2cf6119a92881dba403c219

# FlashInfer nightly at the version the reads were measured on. The nightly
# skews cutlass-dsl, which GB10's CuTeDSL wants back. flashinfer-jit-cache is
# removed so every kernel is JIT-built for this GPU on first start; the result
# lives under /root/.cache, so that cost is paid once per host.
ARG FLASHINFER_VERSION=0.6.18.dev20260819
ARG CUTLASS_DSL_VERSION=4.6.2
RUN pip install --no-cache-dir --pre \
        "flashinfer-python==${FLASHINFER_VERSION}" \
        "flashinfer-cubin==${FLASHINFER_VERSION}" \
        --extra-index-url https://flashinfer.ai/whl/nightly/ \
    && pip uninstall -q -y flashinfer-jit-cache \
    && pip install --no-cache-dir "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
    && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

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
