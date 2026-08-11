# RAG Pipeline Docker Image
#
# Build:
#   docker build --build-arg DEVICE=gpu -t rag-pipeline:gpu .
#   docker build --build-arg DEVICE=cpu -t rag-pipeline:cpu .
#
# Run (GPU — requires nvidia-container-toolkit / Docker Desktop with WSL2 GPU):
#   docker run --gpus all -p 8001:8001 -v ./data/models:/app/data/models rag-pipeline:gpu
#   docker compose --profile gpu up -d
#
# NOTE (China network): base-image pull + torch CUDA wheel come from
# docker.io / pytorch.org, which are slow here. Configure a Docker registry
# mirror (Settings → Docker Engine → registry-mirrors) before building, and
# expect the torch wheel step to be the slowest (~2 GB). All other pip deps
# go through the Tsinghua mirror.
#
# GPU images bundle CUDA runtime libs via pip (torch cu126 + paddlepaddle-gpu),
# so no nvidia/cuda base image is required — python-slim + pip libs + `--gpus
# all` is enough.

FROM python:3.12-slim AS base

ARG DEVICE=gpu
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tsinghua PyPI mirror for pip installs inside the build
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_ENDPOINT=https://hf-mirror.com

# System deps for PyMuPDF, EasyOCR, OpenCV, paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/cache/apt/* /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage: dependencies (separate layer for caching)
# ---------------------------------------------------------------------------
FROM base AS deps

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Core web + parsing deps (fast, via Tsinghua mirror)
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.30.0" \
    pydantic-settings>=2.5.0 \
    "redis[hiredis]>=5.0.0" \
    pymupdf>=1.24.0 \
    python-docx>=1.1.0 \
    qdrant-client>=1.16.0 \
    rank-bm25>=0.2.2 \
    httpx>=0.27.0 \
    python-multipart>=0.0.9 \
    aiofiles>=24.0.0 \
    jieba>=0.42.1 \
    tiktoken>=0.7.0 \
    loguru>=0.7.0 \
    slowapi>=0.1.9 \
    sentence-transformers>=5.6.0 \
    optimum-onnx[onnxruntime]>=0.1.0 \
    easyocr>=1.7.0 \
    transformers>=4.40.0 \
    pix2tex==0.1.4 \
    munch \
    numpy \
    pillow

# torch CUDA — the slow step (pytorch.org wheel ~2GB, no reliable China mirror).
# If this times out, curl the cu126 wheel (as in the WSL guide) then
# `pip install /tmp/torch.whl --no-deps`.
RUN if [ "$DEVICE" = "gpu" ]; then \
      pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu126; \
    else \
      pip install --no-cache-dir torch torchvision; \
    fi

# paddle GPU (cu126, official index; bundles CUDA runtime libs)
RUN if [ "$DEVICE" = "gpu" ]; then \
      pip install --no-cache-dir paddlepaddle-gpu==3.3.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
      && pip install --no-cache-dir paddlex paddleocr; \
    else \
      pip install --no-cache-dir paddlepaddle paddlex paddleocr; \
    fi

# ---------------------------------------------------------------------------
# Stage: application
# ---------------------------------------------------------------------------
FROM deps AS app

COPY . .

# Default directories
RUN mkdir -p /app/data/uploads /app/data/vector_db /app/data/models /app/data/logs

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8001/api/v1/health'); assert r.status_code==200" || exit 1

CMD ["python", "run.py"]
