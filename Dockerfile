# syntax=docker/dockerfile:1.7
ARG CUDA_DEVEL=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed
ARG CUDA_RUNTIME=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc
FROM ${CUDA_DEVEL} AS dependencies
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-dev git build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --upgrade pip==25.2 && pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements.lock
FROM dependencies AS build
# Torch 2.9 has no matching upstream FA2 wheel in this release. Compile against
# the exact installed torch/CUDA ABI, supporting Ampere and Ada (48GB targets).
ENV MAX_JOBS=1 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN pip install --no-deps --no-build-isolation "https://files.pythonhosted.org/packages/6c/b1/b77625e3d11260d2bd6a4fc249a4be9a3da97ff848e2222e810c4f0e0211/flash_attn-2.8.3.post2.tar.gz#sha256=d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b"
COPY stretto-main/stretto-main/kvpress /tmp/kvpress
RUN pip install --no-deps /tmp/kvpress && pip check

FROM ${CUDA_RUNTIME} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 \
    HF_HOME=/data/hf REASONDB_CACHE_DIR=/data/cache STRETTO_DATA_ROOT=/data \
    STRETTO_RESULTS_ROOT=/results CUDA_VISIBLE_DEVICES=0 KV_CACHE_PIN_GB=0 \
    TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=4
ARG SOURCE_REVISION=uncommitted-audited-source
LABEL org.opencontainers.image.source="https://github.com/hazcashi/SemBench-Stretto_experiment" \
      org.opencontainers.image.revision=$SOURCE_REVISION
COPY stretto-main/stretto-main /opt/stretto/source
COPY configs /opt/stretto/configs
COPY tests /opt/stretto/tests
COPY docs/source_provenance.json /opt/stretto/source_provenance.json
COPY scripts/fetch_rotowire.py /opt/stretto/fetch_rotowire.py
WORKDIR /opt/stretto/source
RUN python /opt/stretto/fetch_rotowire.py --provenance /opt/stretto/source_provenance.json --destination reasondb/evaluation/benchmarks/files && \
    pip install --no-deps -e . && mkdir -p /results/logging && ln -s /results/logging logging && \
    pip check && python -c "import duckdb; c=duckdb.connect(); c.execute('INSTALL vss;'); c.execute('LOAD vss;')" && \
    python -c "import torch,flash_attn; from kvpress import ExpectedAttentionPress,KeyRerotationPress,KVzipPress,FinchPress" && \
    python -m reasondb.reproduction.runner inspect && \
    python -m pytest /opt/stretto/tests -q
ENTRYPOINT ["python", "-m", "reasondb.reproduction.runner"]
CMD ["inspect"]
