# Inference Guide

This guide covers deploying and running MedGemma models for inference in production, including optimization techniques and serving configurations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Inference Backends](#inference-backends)
3. [Optimization Techniques](#optimization-techniques)
4. [Ray Serve Deployment](#ray-serve-deployment)
5. [Modal Cloud Deployment](#modal-cloud-deployment)
6. [API Usage](#api-usage)
7. [Performance Tuning](#performance-tuning)
8. [Benchmarking](#benchmarking)

---

## Quick Start

### Local Inference

```python
from medai_compass.inference import create_inference_engine

# Create engine (auto-selects best backend)
engine = create_inference_engine(model_name="medgemma-4b")

# Generate response
response = engine.generate(
    prompt="What are the symptoms of pneumonia?",
    max_tokens=256,
)
print(response.text)
```

### API Inference

```bash
# Start API server
uv run python -m medai_compass.api.main

# Test inference
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the symptoms of pneumonia?", "max_tokens": 256}'
```

### Modal Cloud Inference

```bash
# Deploy to Modal
modal deploy medai_compass/modal/app.py

# Test Modal endpoint
curl -X POST https://your-app--inference.modal.run/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the symptoms of pneumonia?"}'
```

---

## Inference Backends

### Backend Selection Logic

```python
from medai_compass.inference.strategy_selector import select_inference_backend

# Automatic selection based on available hardware
backend = select_inference_backend()
print(f"Selected backend: {backend}")
# Outputs: vllm, ray_serve, triton, modal, local_mps, or cpu
```

### Backend Comparison

| Backend | Throughput | Latency | Memory | Best For |
|---------|------------|---------|--------|----------|
| vLLM | Highest | Low | High | Production serving |
| Ray Serve | High | Medium | Medium | Autoscaling |
| Triton | High | Low | Medium | Enterprise |
| Modal | Medium | Higher | N/A | Cloud/serverless |
| Local MPS | Low | Medium | Low | Mac development |
| CPU | Lowest | High | Low | Testing only |

### vLLM Backend

```python
from medai_compass.inference import VLLMInferenceEngine

engine = VLLMInferenceEngine(
    model_name="google/medgemma-27b-it",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    enable_prefix_caching=True,
)

response = engine.generate(
    prompts=["What is hypertension?"],
    sampling_params={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 256,
    },
)
```

### Ray Serve Backend

```python
from medai_compass.inference import RayServeEngine

engine = RayServeEngine(
    model_name="google/medgemma-4b-it",
    num_replicas=2,
    max_concurrent_queries=100,
)

# Deploy
engine.deploy()

# Generate
response = await engine.generate("What are the symptoms of diabetes?")
```

### Triton Backend

```python
from medai_compass.inference import TritonInferenceEngine

engine = TritonInferenceEngine(
    model_name="medgemma_4b",
    triton_url="localhost:8001",
)

response = engine.generate(
    prompt="Explain the treatment options for hypertension",
    max_tokens=512,
)
```

### Modal Backend

```python
from medai_compass.modal.client import MedGemmaModalClient

client = MedGemmaModalClient()

# Async generation
response = await client.generate(
    prompt="What is the first-line treatment for type 2 diabetes?",
    max_tokens=256,
)
```

---

## Optimization Techniques

### Flash Attention 2

```python
from medai_compass.inference import H100InferenceConfig

config = H100InferenceConfig(
    use_flash_attention_2=True,
    use_sdpa_fallback=True,  # Fallback if flash-attn not available
)
```

**Benefits**:
- 2-4x faster attention computation
- 5-10x memory reduction
- Enables longer context (8K+ tokens)

### CUDA Graphs

```python
config = H100InferenceConfig(
    use_cuda_graphs=True,
    cuda_graph_batch_sizes=[1, 2, 4, 8, 16],
)
```

**Benefits**:
- 20-40% latency reduction
- Eliminates kernel launch overhead
- Best for fixed batch sizes

### KV Cache Optimization

```python
config = H100InferenceConfig(
    kv_cache_dtype="fp8",  # FP8 quantization (H100)
    max_kv_cache_length=8192,
    use_paged_attention=True,
    page_size=16,
)
```

**Benefits**:
- 2x memory reduction with FP8
- Efficient memory management with paging
- Enables higher batch sizes

### Dynamic Batching

```python
config = H100InferenceConfig(
    dynamic_batching=True,
    max_batch_size=32,
    max_wait_ms=50,
    pad_to_multiple=8,
)
```

**Benefits**:
- Higher throughput
- Better GPU utilization
- Automatic batch formation

### Tensor Parallelism

```python
config = H100InferenceConfig(
    use_tensor_parallelism=True,
    tensor_parallel_size=4,  # Split across 4 GPUs
)
```

**Benefits**:
- Run larger models (27B)
- Reduced per-GPU memory
- Linear speedup potential

---

## Ray Serve Deployment

### Basic Deployment

```python
from medai_compass.inference import deploy_medgemma

# Deploy with default settings
deployment = deploy_medgemma(
    model_name="google/medgemma-4b-it",
)

# Test
import requests
response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Hello"}
)
```

### Autoscaling Configuration

```python
from ray import serve

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
        "upscale_delay_s": 30,
        "downscale_delay_s": 300,
    },
    max_ongoing_requests=100,
)
class MedGemmaDeployment:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/medgemma-4b-it",
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")

    async def __call__(self, request):
        prompt = request.json().get("prompt")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return {"response": self.tokenizer.decode(outputs[0], skip_special_tokens=True)}
```

### Health Checks

```python
from ray import serve

@serve.deployment(
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class MedGemmaDeployment:
    def check_health(self):
        # Verify model is loaded
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available")
        return True
```

### Metrics Collection

```python
from ray import serve
from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter("medgemma_requests_total", "Total requests")
LATENCY_HISTOGRAM = Histogram("medgemma_latency_seconds", "Request latency")

@serve.deployment
class MedGemmaDeployment:
    async def __call__(self, request):
        REQUEST_COUNTER.inc()
        with LATENCY_HISTOGRAM.time():
            # Process request
            response = await self.generate(request)
        return response
```

---

## Modal Cloud Deployment

### Deployment Configuration

```python
# medai_compass/modal/app.py
import modal

stub = modal.Stub("medgemma-inference")

# Define volume for model caching
volume = modal.Volume.from_name("medgemma-model-cache")

# Define image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "transformers>=4.40",
    "accelerate>=0.28",
    "bitsandbytes>=0.42",
)

@stub.function(
    gpu="H100",
    image=image,
    volumes={"/models": volume},
    timeout=600,
    concurrency_limit=10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(prompt: str, max_tokens: int = 256) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-27b-it",
        cache_dir="/models",
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "google/medgemma-27b-it",
        cache_dir="/models",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Deploying to Modal

```bash
# Deploy
modal deploy medai_compass/modal/app.py

# Test
modal run medai_compass/modal/app.py::generate --prompt "What is diabetes?"

# View logs
modal app logs medgemma-inference
```

### Modal Web Endpoint

```python
@stub.function(gpu="H100")
@modal.web_endpoint(method="POST")
def generate_endpoint(request: dict) -> dict:
    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 256)

    response = generate(prompt, max_tokens)

    return {"response": response}
```

---

## API Usage

### Inference Endpoint

```bash
# POST /api/v1/inference/generate
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "prompt": "What are the symptoms of pneumonia?",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**Response**:
```json
{
  "id": "gen-abc123",
  "response": "Pneumonia symptoms typically include...",
  "tokens_generated": 150,
  "latency_ms": 450,
  "model": "medgemma-4b"
}
```

### Batch Inference

```bash
# POST /api/v1/inference/batch
curl -X POST http://localhost:8000/api/v1/inference/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "What is hypertension?",
      "What is diabetes?",
      "What is asthma?"
    ],
    "max_tokens": 256
  }'
```

### Streaming Response

```bash
# POST /api/v1/inference/stream
curl -X POST http://localhost:8000/api/v1/inference/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the pathophysiology of heart failure"}' \
  --no-buffer
```

### Status Check

```bash
# GET /api/v1/inference/status
curl http://localhost:8000/api/v1/inference/status
```

**Response**:
```json
{
  "status": "healthy",
  "model": "medgemma-4b",
  "backend": "vllm",
  "gpu_available": true,
  "gpu_memory_used_gb": 12.5,
  "queue_size": 3
}
```

---

## Performance Tuning

### Latency Optimization

```python
config = H100InferenceConfig(
    # Enable all latency optimizations
    use_flash_attention_2=True,
    use_cuda_graphs=True,
    use_tensor_cores=True,
    compute_dtype="bfloat16",

    # Smaller batches for lower latency
    max_batch_size=8,
    max_wait_ms=20,
)
```

### Throughput Optimization

```python
config = H100InferenceConfig(
    # Enable throughput optimizations
    dynamic_batching=True,
    max_batch_size=64,
    max_wait_ms=100,

    # vLLM settings
    use_vllm=True,
    vllm_gpu_memory_utilization=0.95,
    enable_prefix_caching=True,
)
```

### Memory Optimization

```python
config = H100InferenceConfig(
    # Memory optimizations
    kv_cache_dtype="fp8",
    use_paged_attention=True,
    max_kv_cache_length=4096,

    # Quantization
    compute_dtype="bfloat16",
)
```

### Model-Specific Settings

#### 4B Model

```python
config = H100InferenceConfig.for_model("medgemma-4b")
# Defaults:
# - max_batch_size: 32
# - tensor_parallel_size: 1
# - max_kv_cache_length: 8192
```

#### 27B Model

```python
config = H100InferenceConfig.for_model("medgemma-27b")
# Defaults:
# - max_batch_size: 8
# - tensor_parallel_size: 4
# - max_kv_cache_length: 4096
# - use_vllm: True
```

---

## Benchmarking

### Built-in Benchmark

```python
from medai_compass.inference import ThroughputBenchmark

benchmark = ThroughputBenchmark(
    model_name="medgemma-4b",
    num_requests=1000,
    concurrency=10,
)

results = benchmark.run()
print(f"Throughput: {results.requests_per_second:.2f} req/s")
print(f"P50 Latency: {results.p50_latency_ms:.2f} ms")
print(f"P99 Latency: {results.p99_latency_ms:.2f} ms")
```

### Command-Line Benchmark

```bash
# Run inference benchmark
python -m medai_compass.benchmarking.inference \
    --model medgemma-4b \
    --num-requests 1000 \
    --concurrency 10 \
    --output results.json
```

### Locust Load Testing

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class MedAIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def generate(self):
        self.client.post("/api/v1/inference/generate", json={
            "prompt": "What is diabetes?",
            "max_tokens": 256,
        })
```

```bash
# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### Expected Performance

| Model | Backend | P50 Latency | P99 Latency | Throughput |
|-------|---------|-------------|-------------|------------|
| 4B | vLLM | 150ms | 400ms | 150 req/s |
| 4B | Ray Serve | 200ms | 600ms | 100 req/s |
| 27B | vLLM (4 GPU) | 400ms | 1200ms | 50 req/s |
| 27B | Modal H100 | 600ms | 1800ms | 30 req/s |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM Error | Model too large | Use tensor parallelism or Modal |
| Slow inference | No GPU | Check `torch.cuda.is_available()` |
| High latency | Cold start | Enable model warming |
| Timeout | Long generation | Increase `max_tokens` limit |
| Empty response | Tokenizer issue | Check tokenizer settings |

### Debug Commands

```bash
# Check GPU status
python -c "import torch; print(torch.cuda.is_available())"

# Check model loading
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('google/medgemma-4b-it')
print('Model loaded successfully')
"

# Test inference
curl -v http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'

# Check vLLM server
curl http://localhost:8000/v1/models
```

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable vLLM logging
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
```

---

## Production Checklist

Before deploying to production:

- [ ] Benchmark latency and throughput
- [ ] Test with expected load
- [ ] Configure autoscaling
- [ ] Set up health checks
- [ ] Enable monitoring and alerting
- [ ] Test failover scenarios
- [ ] Document API endpoints
- [ ] Load test at 2x expected traffic
- [ ] Verify model output quality
- [ ] Configure rate limiting
