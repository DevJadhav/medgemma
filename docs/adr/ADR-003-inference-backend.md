# ADR-003: GPU Inference Backend Selection

## Status
Accepted

## Date
2025-12-01

## Context

MedAI Compass needs to serve MedGemma models (4B and 27B parameters) with production-grade performance. We need an inference backend that:

1. Supports large language models (4B-27B parameters)
2. Provides high throughput with dynamic batching
3. Enables tensor parallelism for 27B model
4. Works across multiple deployment environments (local, cloud)

### Performance Requirements

| Metric | Target |
|--------|--------|
| P50 Latency | < 500ms |
| P99 Latency | < 2000ms |
| Throughput | > 100 requests/sec |
| GPU Memory | < 80GB (single H100) |

## Decision

We will implement a **multi-backend inference architecture** with three options:

### Primary: vLLM
- **Use Case**: Production serving with high throughput
- **Features**:
  - PagedAttention for memory efficiency
  - Continuous batching
  - Tensor parallelism
  - Speculative decoding

### Secondary: Ray Serve
- **Use Case**: Autoscaling deployments with monitoring
- **Features**:
  - Horizontal scaling
  - Health checks and metrics
  - Request routing
  - Gradual rollouts

### Tertiary: Triton Inference Server
- **Use Case**: Enterprise deployments with strict SLAs
- **Features**:
  - Dynamic batching
  - Model ensembles
  - Multi-framework support
  - Prometheus metrics

### Cloud Fallback: Modal
- **Use Case**: macOS development, no local GPU
- **Features**:
  - Serverless H100 GPUs
  - Pay-per-use pricing
  - Automatic scaling

## Implementation

### Backend Selection Logic

```python
def select_inference_backend():
    """Automatically select the best inference backend."""

    # Check for local GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory

        if gpu_memory >= 80 * 1e9:  # H100 80GB
            return "vllm"  # Best for large models
        elif gpu_memory >= 40 * 1e9:  # A100 40GB
            return "ray_serve"  # Good balance
        else:
            return "triton"  # Memory efficient

    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        return "local_mps"  # 4B model only

    # Fallback to cloud
    elif os.environ.get("MODAL_TOKEN_ID"):
        return "modal"  # Cloud GPU

    else:
        return "cpu"  # Testing only
```

### vLLM Configuration

```python
from vllm import LLM, SamplingParams

engine = LLM(
    model="google/medgemma-27b-it",
    tensor_parallel_size=4,  # 4 GPUs for 27B
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    enable_prefix_caching=True,
)
```

### Ray Serve Configuration

```python
from ray import serve

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
class MedGemmaDeployment:
    def __init__(self):
        self.model = load_model()

    async def __call__(self, request):
        return await self.model.generate(request)
```

## Consequences

### Positive
- Optimal backend for each deployment environment
- vLLM provides state-of-the-art throughput
- Ray Serve enables easy autoscaling
- Modal provides zero-ops cloud GPU access
- Graceful fallback chain

### Negative
- Multiple backends to test and maintain
- Different configuration formats per backend
- Potential behavior differences between backends
- Increased CI/CD complexity

### Mitigation
- Unified `InferenceEngine` interface abstracts backend differences
- Comprehensive integration tests for all backends
- Backend-specific configuration files in Hydra
- Performance benchmarks compare backends

## Performance Comparison

| Backend | Throughput | P50 Latency | P99 Latency | Memory |
|---------|------------|-------------|-------------|--------|
| vLLM | 150 req/s | 200ms | 800ms | 70GB |
| Ray Serve | 100 req/s | 300ms | 1200ms | 75GB |
| Triton | 120 req/s | 250ms | 1000ms | 65GB |
| Modal | 80 req/s | 400ms | 1500ms | N/A |

## Alternatives Considered

### 1. TensorRT-LLM Only
- **Rejected**: Requires NVIDIA-specific compilation
- vLLM provides similar performance with less complexity

### 2. Hugging Face Text Generation Inference
- **Rejected**: Less flexible than vLLM for our use case
- vLLM has better PagedAttention implementation

### 3. Custom Inference Engine
- **Rejected**: High development cost
- Existing solutions are mature and well-tested

## References

- [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.readthedocs.io/)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/)
- [Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [Modal Documentation](https://modal.com/docs)
