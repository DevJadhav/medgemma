# Triton Model Repository

This directory contains the Triton Inference Server model repository configuration
for MedAI Compass. The models are organized for optimal performance with health-grade
AI inference requirements.

## Directory Structure

```
models/
├── README.md
├── medgemma_4b/           # MedGemma 4B for diagnostic imaging
│   ├── config.pbtxt
│   └── 1/                 # Version 1
│       └── (model files loaded at runtime)
├── medgemma_27b/          # MedGemma 27B for documentation
│   ├── config.pbtxt
│   └── 1/
│       └── (model files loaded at runtime)
├── path_foundation/       # Path Foundation for pathology
│   ├── config.pbtxt
│   └── 1/
│       └── (model files loaded at runtime)
├── cxr_foundation/        # CXR Foundation for chest X-rays
│   ├── config.pbtxt
│   └── 1/
│       └── (model files loaded at runtime)
└── ensemble/              # Ensemble configurations
    └── diagnostic_pipeline/
        ├── config.pbtxt
        └── 1/
```

## Model Specifications

### MedGemma 4B (Diagnostic Imaging)
- **Purpose**: Primary diagnostic analysis for medical images
- **Hardware**: NVIDIA H100/A100 GPU (80GB VRAM)
- **Batch Size**: Dynamic (1-4)
- **Precision**: FP16/BF16
- **Framework**: vLLM backend

### MedGemma 27B (Documentation)  
- **Purpose**: Clinical documentation generation
- **Hardware**: NVIDIA H100/A100 GPU (80GB VRAM)
- **Batch Size**: Dynamic (1-2)
- **Precision**: INT8 quantized
- **Framework**: vLLM backend

### Path Foundation
- **Purpose**: Pathology slide analysis
- **Hardware**: NVIDIA GPU (16GB+ VRAM)
- **Batch Size**: 1-8
- **Input**: 224x224 image patches
- **Framework**: PyTorch

### CXR Foundation
- **Purpose**: Chest X-ray analysis
- **Hardware**: NVIDIA GPU (16GB+ VRAM)
- **Batch Size**: 1-16
- **Input**: 224x224 preprocessed images
- **Framework**: PyTorch

## Setup Instructions

### 1. Download Model Weights

```bash
# Ensure HuggingFace token is set
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Download MedGemma models (requires access approval)
huggingface-cli download google/medgemma-4b-it --local-dir models/medgemma_4b/1/model
huggingface-cli download google/medgemma-27b-it --local-dir models/medgemma_27b/1/model

# Download foundation models
huggingface-cli download google/path-foundation --local-dir models/path_foundation/1/model
huggingface-cli download google/cxr-foundation --local-dir models/cxr_foundation/1/model
```

### 2. Start Triton Server

```bash
# Using Docker
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Or using docker-compose
docker-compose up triton
```

### 3. Verify Models

```bash
# Check model status
curl localhost:8000/v2/health/ready
curl localhost:8000/v2/models
```

## Configuration Details

Each model's `config.pbtxt` defines:
- Input/output tensor specifications
- Dynamic batching parameters
- Instance group configuration (GPU allocation)
- Optimization settings (TensorRT, quantization)

## Health Requirements

For HIPAA-compliant deployment:
- Models are loaded fresh on each server start
- No model caching to persistent storage
- Inference logs are encrypted
- Model weights are verified with checksums

## Performance Tuning

### For H100 GPUs
- Enable FP8 precision where available
- Use Tensor Core optimizations
- Configure CUDA memory pool

### For A100 GPUs
- Use INT8 quantization for 27B model
- Enable MIG for multi-model deployment
- Configure NVLink for multi-GPU setups

## Troubleshooting

### Out of Memory
- Reduce batch size in config.pbtxt
- Enable model quantization
- Use CPU offloading for 27B model

### Slow Inference
- Check GPU utilization with `nvidia-smi`
- Verify dynamic batching is working
- Profile with Triton's trace endpoint
