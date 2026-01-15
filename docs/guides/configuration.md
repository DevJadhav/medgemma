# Configuration Guide

All configuration options for MedAI Compass.

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace access token for HAI-DEF models |
| `POSTGRES_PASSWORD` | Database password |
| `REDIS_PASSWORD` | Cache password |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development`, `staging`, `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `API_PORT` | `8000` | API server port |

## Full Configuration

```bash
# =============================================================================
# HuggingFace Configuration
# =============================================================================
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
HF_TOKEN=hf_your_token_here

# =============================================================================
# Modal Configuration (for GPU inference)
# =============================================================================
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret

# =============================================================================
# Database Configuration
# =============================================================================
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=medai
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=medai_compass

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# =============================================================================
# Storage Configuration
# =============================================================================
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=your_minio_secret
MINIO_BUCKET=medai-compass

# =============================================================================
# API Keys (Optional)
# =============================================================================
OPENAI_API_KEY=sk-your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# =============================================================================
# Application Configuration
# =============================================================================
ENVIRONMENT=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# =============================================================================
# Security Configuration
# =============================================================================
JWT_SECRET=your_jwt_secret_key_here
PHI_ENCRYPTION_KEY=your_fernet_key_here

# =============================================================================
# PhysioNet Credentials (for MIMIC access)
# =============================================================================
PHYSIONET_USERNAME=your_physionet_username
PHYSIONET_PASSWORD=your_physionet_password
```

## Generating Keys

### JWT Secret

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### PHI Encryption Key

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Docker Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.9'
services:
  api:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./:/app
```

## Logging Configuration

### Log Levels

| Level | Use Case |
|-------|----------|
| `DEBUG` | Development, detailed tracing |
| `INFO` | Production, normal operations |
| `WARNING` | Issues that don't stop execution |
| `ERROR` | Failures that need attention |

### Structured Logging

```python
from medai_compass.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing request", extra={"request_id": "123"})
```
