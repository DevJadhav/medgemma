# Multi-stage Dockerfile for MedAI Compass
# Optimized for HIPAA-compliant deployment

# =============================================================================
# Stage 1: Base Python environment
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (including postgres client and redis tools for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    openssl \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 medai \
    && useradd --uid 1000 --gid 1000 -m medai

WORKDIR /app

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base as dependencies

# Install uv for fast package management
RUN pip install uv

# Copy dependency files and README
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies using uv (with optional extras for production)
RUN uv pip install --system . && \
    uv pip install --system gunicorn celery[redis] minio

# =============================================================================
# Stage 3: Application
# =============================================================================
FROM dependencies as application

# Copy application code
COPY --chown=medai:medai . .

# Install the application
RUN uv pip install --system -e .

# Copy startup scripts
COPY --chown=medai:medai scripts/generate_secrets.py scripts/verify_connections.py scripts/

# Make scripts executable
RUN chmod +x scripts/*.py

# Switch to non-root user
USER medai

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

# Default command
CMD ["python", "-m", "medai_compass.api.main"]

# =============================================================================
# Stage 4: Development (optional)
# =============================================================================
FROM application as development

USER root

# Install development dependencies
RUN uv pip install --system pytest pytest-cov pytest-asyncio httpx

USER medai

CMD ["pytest", "-v"]

# =============================================================================
# Stage 5: Production
# =============================================================================
FROM application as production

# Additional security hardening
USER medai

# Expose API port
EXPOSE 8000

# Copy entrypoint script
COPY --chown=medai:medai docker/entrypoint.sh /entrypoint.sh

USER root
RUN chmod +x /entrypoint.sh
USER medai

# Use entrypoint for startup validation
ENTRYPOINT ["/entrypoint.sh"]

# Production command with gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", "medai_compass.api.main:app"]
