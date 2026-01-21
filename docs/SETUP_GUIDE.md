# MedAI Compass - Complete Setup Guide

This guide covers setting up MedAI Compass for both **development** and **production** environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Production)](#quick-start-production)
3. [Development Setup](#development-setup)
4. [Production Deployment](#production-deployment)
5. [Configuration Reference](#configuration-reference)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Backend runtime |
| Docker | 24.0+ | Container orchestration |
| Docker Compose | v2.0+ | Multi-container deployment |
| Node.js | 18+ | Frontend (development only) |
| uv | Latest | Python package manager |

### Required Accounts

| Service | Purpose | Link |
|---------|---------|------|
| HuggingFace | Model access (MedGemma) | https://huggingface.co/settings/tokens |
| Modal | Cloud GPU (H100) | https://modal.com |
| PhysioNet | MIMIC datasets (optional) | https://physionet.org |

### MedGemma Access

1. Go to https://huggingface.co/google/medgemma-4b-it
2. Accept the usage agreement
3. Create an access token at https://huggingface.co/settings/tokens

---

## Quick Start (Production)

The fastest way to get MedAI Compass running with MedGemma 27B on cloud GPUs:

```bash
# 1. Clone repository
git clone https://github.com/DevJadhav/medgemma.git
cd medgemma

# 2. Install Python dependencies
uv sync

# 3. Setup Modal (cloud GPU)
uv run modal token new
uv run modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
uv run modal volume create medgemma-model-cache
uv run modal volume create medgemma-checkpoints
uv run modal deploy medai_compass/modal/app.py

# 4. Generate secrets
python scripts/generate_secrets.py

# 5. Configure environment
# Copy Modal tokens from ~/.modal.toml to .env:
# MODAL_TOKEN_ID=ak-xxx
# MODAL_TOKEN_SECRET=as-xxx

# 6. Start all services
docker compose up -d

# 7. Verify
curl http://localhost:8000/health
```

**Access Points:**
- Frontend: http://localhost:3001
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/admin)

---

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/DevJadhav/medgemma.git
cd medgemma

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### 2. Environment Configuration

```bash
# Generate secrets
python scripts/generate_secrets.py

# This creates .env with:
# - POSTGRES_PASSWORD
# - REDIS_PASSWORD  
# - JWT_SECRET
# - PHI_ENCRYPTION_KEY
```

Edit `.env` to add your API tokens:

```bash
# Required
HF_TOKEN=hf_your_huggingface_token

# For cloud GPU (Modal)
MODAL_TOKEN_ID=ak-xxx
MODAL_TOKEN_SECRET=as-xxx

# Optional (for MIMIC datasets)
PHYSIONET_USERNAME=your_username
PHYSIONET_PASSWORD=your_password
```

### 3. Start Development Services

```bash
# Start database and cache only
docker compose up -d postgres redis

# Run API locally (hot reload)
uv run python -m medai_compass.api.main

# Or with uvicorn directly
uv run uvicorn medai_compass.api.main:app --reload --port 8000
```

### 4. Frontend Development

```bash
cd frontend
npm install
npm run dev
# Frontend at http://localhost:3000
```

### 5. Run Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test files
uv run pytest tests/test_api.py -v
uv run pytest tests/test_communication_agent.py -v

# With coverage
uv run pytest tests/ --cov=medai_compass --cov-report=html
```

---

## Production Deployment

### 1. Modal GPU Setup (Required)

```bash
# Authenticate with Modal
modal token new

# Create HuggingFace secret
modal secret create huggingface-secret HF_TOKEN=hf_xxx

# Create persistent volumes
modal volume create medgemma-model-cache
modal volume create medgemma-checkpoints

# Deploy inference service
modal deploy medai_compass/modal/app.py

# Verify deployment
modal app list
```

### 2. Docker Deployment

```bash
# Generate production secrets
python scripts/generate_secrets.py

# Build and start all services
docker compose up -d --build

# Check service health
docker compose ps
docker compose logs api --tail=50
```

### 3. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test chat API
curl -X POST http://localhost:8000/api/v1/communication/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "patient_id": "test-123"}'

# Test diagnostic API (with DICOM file)
curl -X POST http://localhost:8000/api/v1/diagnostic/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/data/mimic_cxr/sample.dcm"}'

# Check escalation stats
curl http://localhost:8000/api/v1/escalations/stats
```

### 4. SSL/TLS Configuration (Production)

For production, configure SSL in `docker/nginx/nginx.conf`:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    
    location / {
        proxy_pass http://frontend:3000;
    }
    
    location /api {
        proxy_pass http://api:8000;
        proxy_read_timeout 300s;  # For long AI inference
    }
}
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace access token |
| `MODAL_TOKEN_ID` | Yes* | Modal authentication ID |
| `MODAL_TOKEN_SECRET` | Yes* | Modal authentication secret |
| `POSTGRES_PASSWORD` | Yes | PostgreSQL password |
| `REDIS_PASSWORD` | Yes | Redis password |
| `JWT_SECRET` | Yes | JWT signing secret |
| `PHI_ENCRYPTION_KEY` | Yes | AES-256 encryption key |
| `PHYSIONET_USERNAME` | No | For MIMIC dataset access |
| `PHYSIONET_PASSWORD` | No | For MIMIC dataset access |

*Required for cloud GPU inference

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI backend |
| Frontend | 3001 | Next.js UI |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache/sessions |
| Grafana | 3000 | Monitoring UI |
| Prometheus | 9090 | Metrics |
| MinIO | 9000/9001 | Object storage |
| Nginx | 80/443 | Reverse proxy |

### Docker Compose Profiles

```bash
# Core services only
docker compose up -d

# With GPU inference (local NVIDIA GPU)
docker compose --profile gpu up -d

# With logging (ELK stack)
docker compose --profile logging up -d

# Everything
docker compose --profile all up -d
```

---

## Troubleshooting

### Common Issues

#### 1. HTTP 500 on Chat/Imaging

**Symptom:** Frontend shows "Internal Server Error"

**Causes & Solutions:**

```bash
# Check API logs
docker compose logs api --tail=100

# If "Worker timeout" - increase gunicorn timeout
# Already set to 300s in Dockerfile

# If "Modal connection failed" - verify tokens
cat ~/.modal.toml
# Ensure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env

# If "MedGemma not accessible" - check HF token
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/models/google/medgemma-4b-it
```

#### 2. Frontend Can't Reach API

**Symptom:** Network errors in browser console

```bash
# Check if API is accessible from frontend container
docker compose exec frontend wget -qO- http://api:8000/health

# If fails, check network
docker network ls
docker compose down && docker compose up -d
```

#### 3. Modal Deployment Issues

```bash
# Check Modal status
modal app list

# Redeploy
modal deploy medai_compass/modal/app.py --force

# Check logs
modal app logs medai-compass
```

#### 4. Database Connection Issues

```bash
# Check PostgreSQL
docker compose logs postgres

# Reset database
docker compose down -v  # WARNING: Deletes data
docker compose up -d
```

### Getting Help

1. Check logs: `docker compose logs <service> --tail=100`
2. Verify health: `curl http://localhost:8000/health`
3. Review [HIPAA Compliance](./HIPAA_COMPLIANCE.md) for security issues
4. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Docker version)

---

## Next Steps

- [API Documentation](./api/README.md)
- [Architecture Overview](./architecture.md)
- [HIPAA Compliance](./HIPAA_COMPLIANCE.md)
- [Security Assessment](./SECURITY_ASSESSMENT_REPORT.md)
