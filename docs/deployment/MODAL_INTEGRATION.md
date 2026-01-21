# Modal GPU Integration

## Status: ✅ PRODUCTION READY

## Overview
MedAI Compass uses Modal H100 GPUs for MedGemma 27B inference via HuggingFace, 
with trained model fallback support.

**Current Configuration:**
- **Model**: `google/medgemma-27b-it` (default)
- **GPU**: NVIDIA H100 (80GB VRAM)
- **Backend**: Modal Cloud
- **Fallback**: `google/medgemma-4b-it` or trained models

## Quick Setup

```bash
# 1. Authenticate with Modal
uv run modal token new

# 2. Create HuggingFace secret
uv run modal secret create huggingface-secret HF_TOKEN=hf_your_token_here

# 3. Create volumes
uv run modal volume create medgemma-model-cache
uv run modal volume create medgemma-checkpoints

# 4. Deploy
uv run modal deploy medai_compass/modal/app.py

# 5. Verify
uv run modal app list
uv run modal volume list
uv run modal secret list
```

## Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   API Container │────▶│ Inference Service │────▶│ Modal H100 GPU      │
│   (Docker)      │     │                  │     │ (MedGemma 27B)      │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
        │                       │                         │
        │                       ▼                         ▼
        │              ┌──────────────────┐      ┌──────────────────┐
        │              │ Trained Model    │      │ HuggingFace      │
        │              │ Detection        │      │ MedGemma (27b)   │
        │              └──────────────────┘      └──────────────────┘
        │
        ▼
┌─────────────────┐
│ Checkpoints Vol │
│ (./model_output)│
└─────────────────┘
```

## Model Loading Priority
1. **Trained/Fine-tuned Model** - Checks checkpoint directories first
2. **Modal GPU with Trained Model** - If trained model found, uses it on Modal
3. **Modal GPU with HuggingFace** - Falls back to google/medgemma-27b-it (default)
4. **Local GPU** - Final fallback if Modal unavailable

## Implementation Summary

### Completed Components

1. **TDD Tests** (`tests/test_modal_integration.py`)
   - 25 tests covering all Modal integration functionality
   - All tests passing

2. **Modal Setup Script** (`scripts/setup_modal.py`)
   - Token verification
   - Volume creation (model-cache, checkpoints)
   - Trained model upload
   - App deployment

3. **Modal App** (`medai_compass/modal/app.py`)
   - H100 GPU configuration
   - Trained model path support (TRAINED_MODEL_PATH env)
   - LoRA adapter loading support
   - `_find_trained_model()` method
   - `_is_valid_checkpoint()` method
   - `get_model_info()` method
   - Model source tracking (trained/huggingface)

4. **Modal Client** (`medai_compass/modal/client.py`)
   - `verify_tokens()` method returning ModalTokenStatus dataclass
   - `trained_model_path` parameter
   - `get_model_info()` async method
   - Model source in results

5. **Inference Service** (`medai_compass/models/inference_service.py`)
   - `_find_trained_model()` method
   - `_is_valid_checkpoint()` method
   - `checkpoint_dirs` parameter
   - `prefer_modal=True` default
   - Model source tracking
   - `get_model_info()` async method

6. **API Endpoints** (`medai_compass/api/main.py`)
   - `POST /api/v1/inference/generate` - Text generation
   - `POST /api/v1/inference/analyze-image` - Image analysis
   - `GET /api/v1/inference/status` - Service status

7. **Docker Compose** (`docker-compose.yml`)
   - MODEL_CHECKPOINT_DIR environment variable
   - MEDGEMMA_MODEL environment variable
   - PREFER_MODAL_GPU=true by default

8. **Dependencies** (`pyproject.toml`)
   - `modal>=0.64.0`
   - `peft>=0.10.0` (for LoRA adapters)

## Configuration

### Environment Variables
```bash
# Modal GPU (get from ~/.modal.toml)
MODAL_TOKEN_ID=ak-your-token-id
MODAL_TOKEN_SECRET=as-your-token-secret
PREFER_MODAL_GPU=true

# HuggingFace
HF_TOKEN=hf-your-hf-token
MEDGEMMA_MODEL=google/medgemma-27b-it

# Checkpoints (for trained models)
MODEL_CHECKPOINT_DIR=./model_output/checkpoints
```

### Modal Volumes
- `medgemma-model-cache` - HuggingFace model cache
- `medgemma-checkpoints` - Trained model storage

## Usage

### Setup Modal
```bash
# Verify Modal tokens
uv run python scripts/setup_modal.py --verify

# Create volumes
uv run python scripts/setup_modal.py --setup-volumes

# Upload trained model (if you have one)
uv run python scripts/setup_modal.py --upload-model ./model_output/checkpoints/best

# Deploy Modal app
uv run modal deploy medai_compass/modal/app.py
```

### API Endpoints
```bash
# Check inference status
curl http://localhost:8000/api/v1/inference/status

# Generate text
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is pneumonia?", "max_tokens": 512}'

# Analyze image
curl -X POST http://localhost:8000/api/v1/inference/analyze-image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe findings", "image_base64": "..."}'
```

## Testing
```bash
# Run Modal integration tests
uv run pytest tests/test_modal_integration.py -v

# Expected: 25 passed
```

## File Structure

```
medai_compass/
├── modal/
│   ├── app.py          # Modal H100 GPU inference app
│   └── client.py       # Async client for Modal calls
├── models/
│   └── inference_service.py  # Unified inference service
├── api/
│   └── main.py         # FastAPI with inference endpoints
└── utils/
    └── gpu.py          # GPU detection utilities

scripts/
└── setup_modal.py      # Modal setup and deployment script

tests/
└── test_modal_integration.py  # TDD tests (25 tests)
```
