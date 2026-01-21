# MedAI Compass - Production Setup Guide

## Quick Start (5 Minutes)

This guide will help you deploy MedAI Compass with **MedGemma 27B** running on **NVIDIA H100 GPUs** via Modal cloud.

### Prerequisites

- **Docker & Docker Compose** - [Install Docker](https://docs.docker.com/get-docker/)
- **Python 3.11+** with `uv` - [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Modal Account** - [Sign up at modal.com](https://modal.com/)
- **HuggingFace Account** - [Sign up at huggingface.co](https://huggingface.co/)

---

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/medgemma.git
cd medgemma

# Install Python dependencies
uv sync
```

## Step 2: Configure Modal (Cloud GPU)

### 2.1 Get Modal Token

```bash
# Install Modal CLI and authenticate
uv run modal token new
```

This will open your browser to authenticate. After authentication, your tokens will be stored in `~/.modal.toml`.

### 2.2 Verify Modal Token Format

Check your `~/.modal.toml` file:

```bash
cat ~/.modal.toml
```

You should see something like:

```toml
[your-profile-name]
token_id = "ak-xxxxxxxxxxxxxxxxxxxxxxx"
token_secret = "as-xxxxxxxxxxxxxxxxxxxxxxx"
active = true
```

**Important:** Token ID starts with `ak-` and token secret starts with `as-`.

### 2.3 Create Modal Secret for HuggingFace

Get your HuggingFace token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
uv run modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2.4 Create Modal Volumes

```bash
uv run modal volume create medgemma-model-cache
uv run modal volume create medgemma-checkpoints
```

### 2.5 Deploy Modal App

```bash
uv run modal deploy medai_compass/modal/app.py
```

You should see:

```
✓ Created objects.
├── 🔨 Created function MedGemmaInference.*.
└── 🔨 Created function test_inference.
✓ App deployed! 🎉
```

### 2.6 Verify Modal Setup

```bash
# List deployed apps
uv run modal app list

# List volumes
uv run modal volume list

# List secrets
uv run modal secret list
```

---

## Step 3: Configure Environment Variables

### 3.1 Generate Secrets

```bash
python scripts/generate_secrets.py
```

This creates a `.env` file with secure random secrets.

### 3.2 Update Modal Tokens in .env

Open `.env` and update the Modal tokens (get values from `~/.modal.toml`):

```bash
# Modal GPU Configuration
MODAL_TOKEN_ID="ak-your-token-id-here"
MODAL_TOKEN_SECRET="as-your-token-secret-here"
PREFER_MODAL_GPU=true

# HuggingFace Token
HF_TOKEN="hf_your-token-here"

# Model Selection (27b recommended for production)
MEDGEMMA_MODEL=google/medgemma-27b-it
```

---

## Step 4: Start Docker Services

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps
```

You should see these services running:

| Service | Port | Description |
|---------|------|-------------|
| `medai-api` | 8000 | FastAPI Backend |
| `medai-frontend` | 3001 | Next.js Frontend |
| `medai-postgres` | 5432 | PostgreSQL Database |
| `medai-redis` | 6379 | Redis Cache |
| `medai-minio` | 9000/9001 | Object Storage |
| `medai-prometheus` | 9090 | Metrics |
| `medai-grafana` | 3000 | Dashboards |

---

## Step 5: Verify Production System

### 5.1 Check API Health

```bash
curl http://localhost:8000/health | jq .
```

Expected response:

```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "redis": "healthy"
  }
}
```

### 5.2 Check Inference Status

```bash
curl http://localhost:8000/api/v1/inference/status | jq .
```

Expected response:

```json
{
  "status": "ready",
  "backend": "modal",
  "model_source": "huggingface",
  "model_path": "google/medgemma-27b-it",
  "is_trained_model": false,
  "modal_available": true,
  "device": "cuda"
}
```

### 5.3 Test MedGemma 27B Inference

```bash
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the diagnostic criteria for Type 2 Diabetes?",
    "max_tokens": 300
  }' | jq .
```

Expected response (first request may take 60-90s for cold start):

```json
{
  "request_id": "...",
  "response": "The American Diabetes Association (ADA) provides specific diagnostic criteria...",
  "confidence": 0.85,
  "model": "google/medgemma-27b-it",
  "backend": "modal",
  "device": "H100",
  "tokens_generated": 300,
  "processing_time_ms": 25000
}
```

### 5.4 Access Frontend

Open [http://localhost:3001](http://localhost:3001) in your browser.

---

## API Endpoints Reference

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check |
| `/api/v1/inference/status` | GET | Inference service status |

### Inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/inference/generate` | POST | Text generation |
| `/api/v1/inference/analyze-image` | POST | Image analysis |

#### Generate Text Request

```json
{
  "prompt": "Your medical question",
  "max_tokens": 512,
  "temperature": 0.1,
  "system_prompt": "You are a medical AI assistant"
}
```

#### Analyze Image Request

```json
{
  "prompt": "Describe findings in this X-ray",
  "image_base64": "base64-encoded-image-data",
  "max_tokens": 512
}
```

---

## Modal Dashboard

Access your Modal dashboard at:
**https://modal.com/apps/YOUR_USERNAME/main/deployed/medai-compass**

Here you can:
- Monitor GPU usage and costs
- View logs and errors
- Scale resources
- Manage secrets and volumes

---

## Troubleshooting

### Worker Timeout Errors

If you see `WORKER TIMEOUT` errors in the API logs, the gunicorn timeout is too short. The Dockerfile is configured with 300s timeout for Modal cold starts.

### Modal Connection Errors

1. Verify tokens in `.env`:
   ```bash
   grep MODAL .env
   ```

2. Ensure tokens match `~/.modal.toml`:
   ```bash
   cat ~/.modal.toml
   ```

3. Redeploy Modal app:
   ```bash
   uv run modal deploy medai_compass/modal/app.py
   ```

### Model Not Found

Ensure `huggingface-secret` is created with valid HF_TOKEN:

```bash
uv run modal secret list
```

If missing, create it:

```bash
uv run modal secret create huggingface-secret HF_TOKEN=hf_your_token
```

### Slow First Request

The first request after deployment takes 60-90 seconds because:
1. Modal spins up a new H100 container
2. Downloads MedGemma 27B model (~54GB)
3. Loads model into GPU memory

Subsequent requests are fast (~15-25 seconds) while the container is warm.

---

## Production Checklist

- [ ] Modal tokens configured in `.env`
- [ ] HuggingFace secret created in Modal
- [ ] Modal volumes created
- [ ] Modal app deployed
- [ ] Docker services running
- [ ] Health check passing
- [ ] Inference endpoint working
- [ ] Frontend accessible

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Production Stack                          │
├──────────────────────────────────────────────────────────────────┤
│  Frontend (Next.js)          │  API (FastAPI)                    │
│  http://localhost:3001       │  http://localhost:8000            │
├──────────────────────────────┼───────────────────────────────────┤
│  PostgreSQL                  │  Redis                            │
│  Port 5432                   │  Port 6379                        │
├──────────────────────────────┴───────────────────────────────────┤
│                          Modal Cloud                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  MedGemma 27B on NVIDIA H100 (80GB)                        │ │
│  │  - Model cache volume                                       │ │
│  │  - Checkpoints volume                                       │ │
│  │  - HuggingFace secret                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Support

- **Documentation**: `/docs/`
- **Issues**: GitHub Issues
- **Modal Support**: [modal.com/docs](https://modal.com/docs)

---

## Security Notes

⚠️ **Important**: This is a research demo. For clinical use:
- Enable HTTPS/TLS
- Configure proper authentication
- Review HIPAA compliance requirements
- Implement audit logging
- Use production-grade secrets management
