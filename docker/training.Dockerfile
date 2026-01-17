# MedGemma Training Dockerfile
# GPU-enabled container for distributed training with Ray and DeepSpeed
#
# Build:
#   docker build -f docker/training.Dockerfile -t medai-training .
#
# Features:
#   - CUDA 12.1 base image
#   - PyTorch 2.2 with Flash Attention
#   - Ray Train/Tune for distributed training
#   - DeepSpeed for ZeRO-3 optimization
#   - uv for fast dependency management

# =============================================================================
# Stage 1: Base CUDA Image
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY config/ ./config/

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install PyTorch with CUDA 12.1
RUN uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention (requires torch first)
RUN uv pip install flash-attn==2.5.6 --no-build-isolation

# Install training dependencies
RUN uv pip install \
    transformers>=4.40.0 \
    accelerate>=0.28.0 \
    bitsandbytes>=0.42.0 \
    peft>=0.10.0 \
    datasets>=2.17.0 \
    deepspeed>=0.13.0 \
    "ray[train,tune]==2.9.0" \
    mlflow>=2.10.0 \
    huggingface_hub>=0.20.0 \
    scipy>=1.12.0 \
    safetensors>=0.4.0 \
    sentencepiece>=0.1.99 \
    protobuf>=4.25.0

# Install additional utilities
RUN uv pip install \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0 \
    minio>=7.2.0 \
    boto3>=1.34.0 \
    psutil>=5.9.0 \
    nvidia-ml-py>=12.535.0

# =============================================================================
# Stage 3: Application
# =============================================================================
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY medai_compass/ ./medai_compass/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Create directories for data and checkpoints
RUN mkdir -p /data /checkpoints /logs

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV HF_HOME="/data/huggingface"
ENV TRANSFORMERS_CACHE="/data/huggingface/transformers"
ENV TORCH_HOME="/data/torch"

# Ray configuration
ENV RAY_ADDRESS="auto"
ENV RAY_OBJECT_STORE_MEMORY=10000000000

# DeepSpeed configuration
ENV DS_ACCELERATOR="cuda"
ENV DS_SKIP_CUDA_CHECK=1

# NCCL configuration for multi-GPU
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0
ENV NCCL_P2P_DISABLE=0

# =============================================================================
# Stage 4: Development (with dev tools)
# =============================================================================
FROM application AS development

# Install development dependencies
RUN uv pip install \
    pytest>=8.0.0 \
    pytest-cov>=4.1.0 \
    pytest-asyncio>=0.23.0 \
    pytest-mock>=3.12.0 \
    hypothesis>=6.98.0 \
    ruff>=0.2.0 \
    mypy>=1.8.0 \
    ipython>=8.0.0 \
    jupyter>=1.0.0

# Copy tests
COPY tests/ ./tests/

# Development entrypoint
CMD ["bash"]

# =============================================================================
# Stage 5: Production Training
# =============================================================================
FROM application AS production

# Create non-root user for security
RUN groupadd -r medai && useradd -r -g medai medai

# Set ownership
RUN chown -R medai:medai /app /data /checkpoints /logs

# Switch to non-root user
USER medai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Default command: Start Ray worker
CMD ["ray", "start", "--address=ray-head:6379", "--block"]

# =============================================================================
# Stage 6: Ray Head Node
# =============================================================================
FROM application AS ray-head

# Expose Ray ports
EXPOSE 6379 8265 10001 8076 8077 8078

# Create non-root user
RUN groupadd -r medai && useradd -r -g medai medai
RUN chown -R medai:medai /app /data /checkpoints /logs
USER medai

# Health check for Ray head
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ray status || exit 1

# Start Ray head node
CMD ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--block"]

# =============================================================================
# Stage 7: Ray Worker Node
# =============================================================================
FROM application AS ray-worker

# Create non-root user
RUN groupadd -r medai && useradd -r -g medai medai
RUN chown -R medai:medai /app /data /checkpoints /logs
USER medai

# Health check for Ray worker
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ray status || exit 1

# Start Ray worker (connects to head)
CMD ["ray", "start", "--address=ray-head:6379", "--block"]

# =============================================================================
# Stage 8: MLflow Server
# =============================================================================
FROM dependencies AS mlflow-server

# Install MLflow server dependencies
RUN uv pip install \
    mlflow>=2.10.0 \
    psycopg2-binary>=2.9.9 \
    boto3>=1.34.0 \
    gunicorn>=21.0.0

# Expose MLflow port
EXPOSE 5000

# Create non-root user
RUN groupadd -r mlflow && useradd -r -g mlflow mlflow

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

USER mlflow

# Start MLflow server
CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "${MLFLOW_BACKEND_URI}", \
    "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}"]
