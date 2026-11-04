# ==============================================================================
# InSpace Standalone Inference Image
# Dedicated NVIDIA Blackwell (sm_120 / RTX 50-Series) Production Image
# ==============================================================================

# ----------------- Stage 1: Build & Python Dependencies -----------------------
FROM python:3.10-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_NETWORK_ATTEMPTS=5 \
    UV_CONCURRENT_INSTALLS=3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch & Torchvision (CUDA 12.8) - attention handled by native sm_120 flash_attn
RUN uv pip install \
    --python /opt/venv/bin/python \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install standard requirements
WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt
COPY third_party/ /tmp/third_party/
COPY third_party/ /third_party/
RUN cd /tmp && uv pip install \
    --python /opt/venv/bin/python \
    -r /tmp/requirements.txt

# Install universal pre-compiled CUDA wheels
COPY wheels/ /tmp/wheels/
RUN chmod +x /tmp/wheels/BUILD_FALLBACK.sh \
    && /tmp/wheels/BUILD_FALLBACK.sh \
    && uv pip install \
    --python /opt/venv/bin/python \
    --no-deps \
    /tmp/wheels/*.whl


# ----------------- Stage 2: Minimal Runtime Environment ----------------------
FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    ATTN_BACKEND=flash_attn \
    SPARSE_ATTN_BACKEND=flash_attn \
    SPARSE_CONV_BACKEND=flex_gemm \
    SPCONV_ALGO=native \
    CHECKPOINT_DIR=/checkpoints \
    INSPACE_OUTPUT=/data/outputs \
    INSPACE_DOCKER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libegl1 \
    libosmesa6 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment and source code
COPY --from=builder /opt/venv /opt/venv
COPY inspace/ /app/inspace/
COPY app.py /app/app.py

RUN chmod +x /app/app.py \
    && ln -s /app/app.py /usr/local/bin/inspace

# Create standard mount points
RUN mkdir -p /checkpoints /data /data/outputs

EXPOSE 7860

ENTRYPOINT ["python", "app.py"]
CMD []

