# RAG Pipeline Docker Image
#
# Build:
#   docker build -t rag-pipeline .
#   docker build --build-arg DEVICE=cpu -t rag-pipeline:cpu .
#
# Run:
#   docker-compose up
#   docker run -p 8001:8001 -v ./data:/app/data rag-pipeline
#
# NOTE: Full ML stack (PaddlePaddle + PyTorch + EasyOCR) is ~8-15 GB.
# For development, mount the models directory as a volume to avoid
# re-downloading on every build:  -v ./data/models:/app/data/models

FROM python:3.12-slim AS base

ARG DEVICE=gpu
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for PyMuPDF, EasyOCR, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage: dependencies (separate layer for caching)
# ---------------------------------------------------------------------------
FROM base AS deps

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install without ML-heavy packages first, then add the ML stack
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
    sentence-transformers>=3.0.0 \
    easyocr>=1.7.0 \
    transformers>=4.40.0 \
    tokenizers>=0.19.0

# GPU variant: install PaddlePaddle GPU
# (swap for paddlepaddle==3.0.0 for CPU-only)
RUN if [ "$DEVICE" = "gpu" ]; then \
    pip install --no-cache-dir paddlepaddle-gpu>=2.6.0 paddlex>=3.7.0 paddleocr>=3.7.0; \
    else \
    pip install --no-cache-dir paddlepaddle>=3.0.0 paddlex>=3.7.0 paddleocr>=3.7.0; \
    fi

# ---------------------------------------------------------------------------
# Stage: application
# ---------------------------------------------------------------------------
FROM deps AS app

COPY . .

# Default directories
RUN mkdir -p /app/data/uploads /app/data/vector_db /app/data/models /app/data/logs

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8001/api/v1/health'); assert r.status_code==200" || exit 1

CMD ["python", "run.py"]
