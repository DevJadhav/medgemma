# GPU & Inference Configuration Guide

This guide covers GPU detection, Modal cloud inference, and the unified inference service.

## Overview

MedAI Compass supports multiple GPU backends with automatic fallback:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local CUDA    │ -> │   Local MPS     │ -> │  Modal (Cloud)  │
│ (A100/H100 80GB)│    │ (Apple Silicon) │    │    (H100)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓                     ↓                      ↓
    Best Option          Dev Option           Fallback Option
```

## GPU Detection

The `medai_compass.utils.gpu` module provides automatic hardware detection:

```python
from medai_compass.utils.gpu import (
    detect_local_gpu,
    should_use_modal,
    get_inference_config
)

# Check local GPU availability
gpu_info = detect_local_gpu()
print(f"GPU Available: {gpu_info['available']}")
print(f"GPU Type: {gpu_info['type']}")  # 'cuda', 'mps', or None
print(f"Devices: {gpu_info['devices']}")

# Determine if Modal should be used
use_modal = should_use_modal(required_memory_gb=40)
print(f"Use Modal: {use_modal}")

# Get complete inference configuration
config = get_inference_config(model_name="google/medgemma-27b-it")
print(f"Backend: {config['backend']}")
print(f"Device: {config['device']}")
print(f"Precision: {config['model_precision']}")
```

### GPU Decision Matrix

| Local GPU | Memory | Model | Decision |
|-----------|--------|-------|----------|
| CUDA 80GB | ≥80GB | 27B | Local CUDA |
| CUDA 80GB | ≥40GB | 4B | Local CUDA |
| CUDA 16GB | <40GB | 27B | Modal |
| CUDA 16GB | ≥16GB | 4B | Local CUDA |
| MPS | Any | 4B | Local MPS |
| MPS | Any | 27B | Modal |
| None | N/A | Any | Modal |

## Modal Cloud GPU (Optional)

Modal provides on-demand H100 GPU access. The `medai_compass/modal/` folder is **optional** and can be deleted for fully local deployments.

### Setup

```bash
# Install Modal
uv pip install modal

# Authenticate (one-time)
modal setup

# Set environment variables
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret
```

### Deploying the MedGemma Service

```bash
# Deploy to Modal cloud
modal deploy medai_compass/modal/app.py

# Check deployment status
modal app list
```

### Using the Modal Client

```python
from medai_compass.modal.client import MedGemmaModalClient

# Initialize client
client = MedGemmaModalClient(
    model_name="google/medgemma-4b-it",
    max_retries=3
)

# Initialize connection
await client.initialize()

# Health check
health = await client.health_check()
print(f"Status: {health['status']}")

# Analyze image
import base64
with open("chest_xray.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

result = await client.analyze_image(
    image_data=image_b64,
    prompt="Analyze this chest X-ray for abnormalities",
    max_tokens=1024
)
print(result["response"])

# Generate text
result = await client.generate(
    prompt="Generate a discharge summary for a patient with pneumonia",
    max_tokens=2048,
    temperature=0.2
)
print(result["response"])
```

### Modal Configuration

The Modal app uses:
- **GPU**: NVIDIA H100 80GB
- **Memory**: 64GB system RAM
- **Timeout**: 10 minutes per request
- **Concurrency**: 5 concurrent requests

## Unified Inference Service

The `MedGemmaInferenceService` provides a unified interface regardless of backend:

```python
from medai_compass.models.inference_service import MedGemmaInferenceService

# Create service (auto-detects best backend)
service = MedGemmaInferenceService(
    model_name="google/medgemma-4b-it"
)

# Initialize (loads model or connects to Modal)
await service.initialize()

# Check which backend is being used
print(f"Backend: {service.backend}")  # 'local' or 'modal'

# Analyze image
result = await service.analyze_image(
    image_data=image_bytes,
    prompt="Describe the findings in this CT scan"
)
print(f"Response: {result.response}")
print(f"Confidence: {result.confidence}")
print(f"Processing Time: {result.processing_time_ms}ms")

# Generate text
result = await service.generate(
    prompt="Write a progress note",
    max_tokens=1024,
    temperature=0.3,
    system_prompt="You are a clinical documentation specialist."
)
print(result.response)
```

### Inference Result

```python
@dataclass
class InferenceResult:
    response: str              # Model output
    confidence: float = None   # Confidence score (0-1)
    backend: str = "local"     # 'local' or 'modal'
    processing_time_ms: float = None
    error: str = None          # Error message if failed
    metadata: dict = None      # Additional info
```

### Fallback Behavior

The service automatically handles failures:

```python
# Enable fallback (try Modal if local fails)
result = await service.generate(
    prompt="Test",
    fallback=True  # If local fails, try Modal
)
```

## Force Specific Backend

Override automatic detection:

```python
# Force local (even without GPU)
config = get_inference_config(force_local=True)

# Force Modal (even with local GPU)
config = get_inference_config(force_modal=True)

# Or via environment
export MEDAI_FORCE_MODAL=true
export MEDAI_FORCE_LOCAL=true
```

## Triton Inference Server

For production deployments, use Triton for model serving:

```bash
# Start Triton with model repository
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Check model status
curl localhost:8000/v2/models

# Inference request
curl -X POST localhost:8000/v2/models/medgemma_4b/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "text_input", "data": ["Analyze this image"]}]}'
```

See [models/README.md](../models/README.md) for Triton configuration details.

## Performance Tuning

### Local CUDA

```python
# Optimize for A100/H100
config = get_inference_config(
    model_name="google/medgemma-27b-it",
    optimization_hints={
        "use_flash_attention": True,
        "use_cuda_graphs": True,
        "max_batch_size": 4
    }
)
```

### Modal

```python
# Adjust Modal settings
client = MedGemmaModalClient(
    timeout=600,  # 10 minutes
    max_retries=3,
    retry_delay=1.0
)
```

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Use quantization (INT8/INT4)
- Switch to Modal for 27B model

### "Modal connection failed"
- Check `modal setup` was run
- Verify MODAL_TOKEN_ID and MODAL_TOKEN_SECRET
- Check Modal dashboard for service status

### "Model loading timeout"
- First load downloads model weights
- Allow 10-15 minutes for 27B model
- Use Modal for faster cold starts (cached)

### "MPS not available"
- Ensure PyTorch 2.0+ with MPS support
- Check macOS 12.3+ on Apple Silicon
- Verify with `torch.backends.mps.is_available()`
