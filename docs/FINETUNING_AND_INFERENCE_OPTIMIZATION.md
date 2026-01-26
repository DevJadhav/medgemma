# MedGemma Fine-Tuning & Inference Optimization Guide

This comprehensive guide covers all fine-tuning techniques and inference optimizations implemented in the MedAI Compass platform for MedGemma models.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Fine-Tuning Techniques](#2-fine-tuning-techniques)
   - [Parameter-Efficient Fine-Tuning (PEFT)](#21-parameter-efficient-fine-tuning-peft)
   - [Alignment Algorithms](#22-alignment-algorithms)
   - [Distributed Training Strategies](#23-distributed-training-strategies)
3. [Inference Optimization](#3-inference-optimization)
   - [Attention Optimizations](#31-attention-optimizations)
   - [Memory Optimizations](#32-memory-optimizations)
   - [Serving Optimizations](#33-serving-optimizations)
4. [Production Deployment](#4-production-deployment)
5. [Benchmarking & Performance](#5-benchmarking--performance)
6. [Quick Reference](#6-quick-reference)

---

## 1. Overview

MedAI Compass provides a comprehensive suite of training and inference optimizations specifically designed for MedGemma models (4B and 27B parameters) running on NVIDIA H100 GPUs.

### Supported Models
| Model | Parameters | Recommended GPU | Training Strategy |
|-------|------------|-----------------|-------------------|
| MedGemma-4B | 4 billion | 1x H100 80GB | LoRA/QLoRA |
| MedGemma-27B | 27 billion | 8x H100 80GB | DeepSpeed ZeRO-3 + LoRA |

### Key Capabilities
- **10+ PEFT Methods**: LoRA, QLoRA, DoRA, IA3, Adapters
- **4 Alignment Algorithms**: DPO, KTO, GRPO, RLHF/PPO
- **5 Distributed Strategies**: DeepSpeed, FSDP, Megatron, 5D Parallelism
- **3 Production Serving Backends**: vLLM, Ray Serve, Triton

---

## 2. Fine-Tuning Techniques

### 2.1 Parameter-Efficient Fine-Tuning (PEFT)

#### 2.1.1 LoRA (Low-Rank Adaptation)

LoRA freezes the pretrained model and injects trainable rank-decomposition matrices into transformer layers.

**Implementation**: `medai_compass/training/algorithms/trainers.py::LoRATrainer`

```python
from medai_compass.training.algorithms import LoRAConfig, LoRATrainer

# Configure LoRA
config = LoRAConfig(
    r=64,                    # Rank (higher = more capacity, more params)
    lora_alpha=128,          # Scaling factor (alpha/r determines update magnitude)
    lora_dropout=0.05,       # Dropout for regularization
    target_modules=[         # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",             # Don't train bias terms
    task_type="CAUSAL_LM",
)

# Create trainer
trainer = LoRATrainer(
    model_name="google/medgemma-27b-it",
    config=config,
)

# Train
trainer.train(train_dataset, eval_dataset)
```

**Memory Savings**:
- 27B model: ~0.5% trainable parameters (466M / 27.9B)
- Memory: ~40GB instead of 200GB+ for full fine-tuning

**Best Practices**:
- Use `r=64-128` for medical domain adaptation
- Higher `lora_alpha` (2x rank) for stronger updates
- Target all linear layers for best quality

---

#### 2.1.2 QLoRA (Quantized LoRA)

QLoRA combines 4-bit quantization with LoRA for extreme memory efficiency.

**Implementation**: `medai_compass/training/algorithms/trainers.py::QLoRATrainer`

```python
from medai_compass.training.algorithms import QLoRAConfig, QLoRATrainer

config = QLoRAConfig(
    # LoRA settings
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # Quantization settings
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # Normal Float 4-bit
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,  # Nested quantization
)

trainer = QLoRATrainer(
    model_name="google/medgemma-27b-it",
    config=config,
)
```

**Memory Comparison**:
| Method | 27B Model VRAM |
|--------|----------------|
| Full Fine-Tuning | ~200GB |
| LoRA (bf16) | ~60GB |
| QLoRA (4-bit) | ~20GB |

---

#### 2.1.3 DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA decomposes pretrained weights into magnitude and direction, applying LoRA only to direction.

**Implementation**: `medai_compass/training/algorithms/trainers.py::DoRATrainer`

```python
from medai_compass.training.algorithms import DoRAConfig, DoRATrainer

config = DoRAConfig(
    r=64,
    lora_alpha=128,
    use_dora=True,  # Enable weight decomposition
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

trainer = DoRATrainer(model_name="google/medgemma-27b-it", config=config)
```

**Advantages**:
- Better convergence than standard LoRA
- More stable training dynamics
- Closer to full fine-tuning quality

---

#### 2.1.4 IA3 (Infused Adapter by Inhibiting and Amplifying)

IA3 learns vectors that rescale inner activations, requiring even fewer parameters than LoRA.

**Implementation**: `medai_compass/training/algorithms/trainers.py::IA3Trainer`

```python
from medai_compass.training.algorithms import IA3Config, IA3Trainer

config = IA3Config(
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)

trainer = IA3Trainer(model_name="google/medgemma-4b-it", config=config)
```

**Parameters**: Only ~0.01% of base model (vs 0.5% for LoRA)

---

#### 2.1.5 Adapter Modules

Adapters insert small bottleneck layers between transformer blocks.

**Implementation**: `medai_compass/training/algorithms/trainers.py::AdapterTrainer`

```python
from medai_compass.training.algorithms import AdapterConfig, AdapterTrainer

config = AdapterConfig(
    adapter_type="houlsby",  # or "pfeiffer"
    adapter_size=64,         # Bottleneck dimension
    adapter_dropout=0.1,
)

trainer = AdapterTrainer(model_name="google/medgemma-4b-it", config=config)
```

---

### 2.2 Alignment Algorithms

#### 2.2.1 DPO (Direct Preference Optimization)

DPO optimizes the model directly on preference data without a separate reward model.

**Implementation**: `medai_compass/training/algorithms/trainers.py::DPOTrainer`

```python
from medai_compass.training.algorithms import DPOConfig, DPOTrainer

config = DPOConfig(
    beta=0.1,                    # KL penalty coefficient
    loss_type="sigmoid",         # sigmoid, hinge, or ipo
    reference_free=False,        # Use reference model
    label_smoothing=0.0,

    # LoRA settings (combine with PEFT)
    use_peft=True,
    lora_r=64,
    lora_alpha=128,
)

# Prepare preference dataset
# Format: {"prompt": str, "chosen": str, "rejected": str}
preference_data = load_medical_preferences()

trainer = DPOTrainer(
    model_name="google/medgemma-27b-it",
    config=config,
)

trainer.train(preference_data)
```

**Use Cases**:
- Improving response quality without reward modeling
- Aligning to medical guidelines
- Reducing harmful outputs

---

#### 2.2.2 KTO (Kahneman-Tversky Optimization)

KTO uses prospect theory for human-aligned optimization without paired preferences.

**Implementation**: `medai_compass/training/algorithms/trainers.py::KTOTrainer`

```python
from medai_compass.training.algorithms import KTOConfig, KTOTrainer

config = KTOConfig(
    beta=0.1,
    desirable_weight=1.0,      # Weight for positive examples
    undesirable_weight=1.0,    # Weight for negative examples
)

# Dataset format: {"prompt": str, "completion": str, "label": bool}
# label=True for good responses, False for bad

trainer = KTOTrainer(model_name="google/medgemma-27b-it", config=config)
```

**Advantages**:
- Works with unpaired feedback (thumbs up/down)
- Simpler data collection than DPO
- Better handles asymmetric preferences

---

#### 2.2.3 GRPO (Group Relative Policy Optimization)

GRPO optimizes using group-level relative rankings.

**Implementation**: `medai_compass/training/algorithms/trainers.py::GRPOTrainer`

```python
from medai_compass.training.algorithms import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    group_size=4,              # Number of responses per prompt
    temperature=1.0,
    use_advantage_normalization=True,
)

trainer = GRPOTrainer(model_name="google/medgemma-27b-it", config=config)
```

---

#### 2.2.4 RLHF/PPO (Reinforcement Learning from Human Feedback)

Full RLHF pipeline with PPO optimization.

**Implementation**: `medai_compass/training/algorithms/trainers.py::RLHFTrainer`

```python
from medai_compass.training.algorithms import RLHFConfig, RLHFTrainer

config = RLHFConfig(
    # PPO settings
    ppo_epochs=4,
    mini_batch_size=4,
    learning_rate=1e-5,

    # KL control
    init_kl_coef=0.2,
    target_kl=6.0,

    # Reward model
    reward_model_name="path/to/reward_model",

    # Generation
    max_new_tokens=256,
    temperature=0.7,
)

trainer = RLHFTrainer(model_name="google/medgemma-27b-it", config=config)
trainer.train(prompt_dataset)
```

---

### 2.3 Distributed Training Strategies

#### 2.3.1 DeepSpeed ZeRO

ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients, and parameters across GPUs.

**Implementation**: `medai_compass/training/distributed/deepspeed_trainer.py`

```python
from medai_compass.training.distributed import DeepSpeedConfig, DeepSpeedTrainer

# ZeRO Stage 3 (full sharding)
config = DeepSpeedConfig(
    zero_stage=3,

    # Offloading
    offload_optimizer=True,      # CPU offload optimizer states
    offload_param=False,         # Keep params on GPU

    # Memory optimization
    overlap_comm=True,           # Overlap communication
    contiguous_gradients=True,
    reduce_bucket_size=5e8,

    # Mixed precision
    bf16=True,

    # Gradient settings
    gradient_accumulation_steps=16,
    gradient_clipping=1.0,
)

trainer = DeepSpeedTrainer(
    model_name="google/medgemma-27b-it",
    config=config,
    num_gpus=8,
)

trainer.train(dataset)
```

**ZeRO Stages**:
| Stage | Partitions | Memory Reduction |
|-------|------------|------------------|
| 1 | Optimizer states | 4x |
| 2 | + Gradients | 8x |
| 3 | + Parameters | Linear with #GPUs |

---

#### 2.3.2 FSDP (Fully Sharded Data Parallel)

PyTorch native sharding with automatic policy-based wrapping.

**Implementation**: `medai_compass/training/distributed/fsdp2_trainer.py`

```python
from medai_compass.training.distributed import FSDP2Config, FSDP2Trainer

config = FSDP2Config(
    sharding_strategy="FULL_SHARD",  # or SHARD_GRAD_OP, NO_SHARD

    # Mixed precision
    mixed_precision="bf16",

    # CPU offload
    cpu_offload=False,

    # Backward prefetch
    backward_prefetch="BACKWARD_PRE",

    # Activation checkpointing
    activation_checkpointing=True,
)

trainer = FSDP2Trainer(
    model_name="google/medgemma-27b-it",
    config=config,
)
```

---

#### 2.3.3 Megatron-LM (Tensor & Pipeline Parallelism)

For extreme scale training with tensor and pipeline parallelism.

**Implementation**: `medai_compass/training/distributed/megatron_parallelism.py`

```python
from medai_compass.training.distributed import MegatronConfig, MegatronTPPPTrainer

config = MegatronConfig(
    tensor_parallel_size=4,      # Split model across 4 GPUs
    pipeline_parallel_size=2,    # 2-stage pipeline
    micro_batch_size=1,
    global_batch_size=32,

    # Sequence parallelism
    sequence_parallel=True,

    # Activation recomputation
    recompute_granularity="selective",
)

trainer = MegatronTPPPTrainer(
    model_name="google/medgemma-27b-it",
    config=config,
)
```

---

#### 2.3.4 5D Parallelism

Combines all parallelism dimensions: Data, Tensor, Pipeline, Sequence, Expert.

**Implementation**: `medai_compass/training/distributed/parallelism_5d.py`

```python
from medai_compass.training.distributed import Parallelism5DConfig, HybridParallelTrainer

config = Parallelism5DConfig(
    data_parallel_size=2,
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    sequence_parallel_size=1,
    expert_parallel_size=1,  # For MoE models
)

trainer = HybridParallelTrainer(config=config)
```

---

## 3. Inference Optimization

### 3.1 Attention Optimizations

#### 3.1.1 Flash Attention 2

Memory-efficient attention with O(N) memory complexity instead of O(N²).

**Implementation**: `medai_compass/inference/optimized.py::OptimizedModelLoader`

```python
from medai_compass.inference.optimized import OptimizedModelLoader, H100InferenceConfig

config = H100InferenceConfig(
    use_flash_attention_2=True,
    use_sdpa_fallback=True,  # Fall back to SDPA if flash-attn unavailable
)

loader = OptimizedModelLoader(
    use_flash_attention_2=True,
    compute_dtype="bfloat16",
)

model, tokenizer = loader.load_model("google/medgemma-27b-it")
```

**Benefits**:
- 2-4x faster attention computation
- 5-10x memory reduction
- Enables 8K+ context length

**Requirements**:
- NVIDIA GPU with SM 8.0+ (A100, H100)
- `flash-attn` package or PyTorch 2.0+ SDPA

---

#### 3.1.2 CUDA Graphs

Captures and replays GPU kernel sequences for reduced launch overhead.

**Implementation**: `medai_compass/inference/optimized.py::CUDAGraphRunner`

```python
from medai_compass.inference.optimized import CUDAGraphRunner

# Initialize with supported batch sizes
cuda_runner = CUDAGraphRunner(
    batch_sizes=[1, 2, 4, 8, 16],
    warmup_iterations=3,
)

# Warm up (captures graphs)
cuda_runner.warmup(model, sample_input)

# Inference (replays captured graph)
output = cuda_runner.replay_graph(batch_size=4, inputs=batch_inputs)
```

**Benefits**:
- 20-40% latency reduction
- Eliminates kernel launch overhead
- Best for fixed batch sizes

---

### 3.2 Memory Optimizations

#### 3.2.1 FP8 KV Cache

H100-specific FP8 quantization for KV cache.

**Implementation**: `medai_compass/inference/optimized.py::KVCacheManager`

```python
from medai_compass.inference.optimized import KVCacheManager

cache_manager = KVCacheManager(
    max_length=8192,
    dtype="fp8",                 # FP8 for H100, bf16 otherwise
    use_paged_attention=True,
    page_size=16,
    num_layers=32,
    num_heads=32,
    head_dim=128,
)

# Get cache for request
cache = cache_manager.get_cache(request_id=123)

# Release when done
cache_manager.release_cache(request_id=123)
```

**Memory Savings**:
- FP8 vs FP16: 2x reduction
- Paged attention: Dynamic allocation, no pre-allocation waste

---

#### 3.2.2 Dynamic Batching

Groups requests for efficient batch processing.

**Implementation**: `medai_compass/inference/optimized.py::DynamicBatcher`

```python
from medai_compass.inference.optimized import DynamicBatcher

batcher = DynamicBatcher(
    max_batch_size=16,
    max_wait_ms=50,       # Maximum wait time to form batch
    pad_to_multiple=8,    # Pad batch size for tensor core efficiency
    continuous_batching=True,
)

# Add requests
batcher.add_request(request_id=1, inputs=input1, callback=on_complete)
batcher.add_request(request_id=2, inputs=input2)

# Get batch for processing
batch = batcher.get_batch(timeout_ms=50)
```

---

### 3.3 Serving Optimizations

#### 3.3.1 vLLM Backend

High-throughput serving with PagedAttention and continuous batching.

**Implementation**: `medai_compass/inference/optimized.py::VLLMInferenceEngine`

```python
from medai_compass.inference.optimized import VLLMInferenceEngine

engine = VLLMInferenceEngine(
    model_name="google/medgemma-27b-it",
    tensor_parallel_size=4,           # Split across 4 GPUs
    gpu_memory_utilization=0.9,       # Use 90% of GPU memory
    max_model_len=8192,
    enable_prefix_caching=True,       # Cache common prefixes
    use_speculative_decoding=False,
)

engine.initialize()

# Generate
responses = engine.generate(
    prompts=["What is diabetes?", "Explain hypertension"],
    max_tokens=256,
    temperature=0.7,
)
```

**Features**:
- Continuous batching (no batch waiting)
- PagedAttention (efficient KV cache)
- Prefix caching (faster repeated prompts)

---

#### 3.3.2 Ray Serve Backend

Autoscaling deployment with Ray ecosystem integration.

**Implementation**: `medai_compass/inference/optimized.py::RayServeEngine`

```python
from medai_compass.inference.optimized import RayServeEngine

engine = RayServeEngine(
    model_name="google/medgemma-27b-it",
    num_replicas=2,
    max_concurrent_queries=100,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
        "upscale_delay_s": 30,
        "downscale_delay_s": 300,
    },
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 4,
    },
)

engine.deploy()

# Async generation
responses = await engine.generate(
    prompts=["What are symptoms of pneumonia?"],
    max_tokens=256,
)
```

---

#### 3.3.3 Triton Inference Server

Enterprise-grade serving with NVIDIA Triton.

**Implementation**: `medai_compass/inference/optimized.py::TritonInferenceEngine`

```python
from medai_compass.inference.optimized import TritonInferenceEngine

engine = TritonInferenceEngine(
    model_name="medgemma-27b",
    model_repository="/models",
    grpc_url="localhost:8001",
    max_batch_size=32,
    preferred_batch_sizes=[1, 4, 8, 16, 32],
    max_queue_delay_microseconds=100000,  # 100ms
)

engine.initialize()

# Check server status
if engine.is_server_ready() and engine.is_model_ready():
    responses = engine.generate(prompts=["What is hypertension?"])
```

---

#### 3.3.4 Unified Factory

Use the factory to create any backend.

```python
from medai_compass.inference.optimized import ProductionServingFactory, H100InferenceConfig

config = H100InferenceConfig.for_model("medgemma-27b")

# Get recommended backend
backend = ProductionServingFactory.get_recommended_backend(config)
print(f"Recommended: {backend}")  # "vllm" for tensor parallel

# Create engine
engine = ProductionServingFactory.create(
    backend="vllm",
    config=config,
    model_name="google/medgemma-27b-it",
)
```

---

## 4. Production Deployment

### 4.1 Modal Cloud Deployment

For serverless H100 GPU access.

**Implementation**: `medai_compass/modal/training_app.py`

```python
# Deploy training
uv run modal deploy medai_compass/modal/training_app.py

# Run training job
uv run modal run medai_compass/modal/training_app.py::MedGemmaTrainer.train \
    --model-name "google/medgemma-27b-it" \
    --train-data-path "/data/medical/train.jsonl" \
    --max-steps 1000 \
    --batch-size 1 \
    --learning-rate 1e-4

# Deploy inference
uv run modal deploy medai_compass/modal/app.py
```

### 4.2 Ray Pipeline

For orchestrated training across distributed resources.

```python
# Full pipeline: download -> train -> evaluate -> save
uv run python scripts/ray_modal_train.py pipeline \
    --model medgemma-27b \
    --max-steps 1000 \
    --batch-size 1

# Just training
uv run python scripts/ray_modal_train.py train \
    --model medgemma-27b \
    --max-steps 1000
```

### 4.3 Docker Deployment

```bash
# Start all services
docker-compose up -d

# With GPU
docker-compose --profile gpu up -d

# With Modal (cloud GPU)
docker-compose --profile modal up -d
```

---

## 5. Benchmarking & Performance

### 5.1 Training Throughput

**Implementation**: `medai_compass/training/optimized.py::ThroughputTracker`

```python
from medai_compass.training.optimized import ThroughputTracker

tracker = ThroughputTracker()

# During training loop
for step, batch in enumerate(dataloader):
    start_time = time.time()

    # Training step
    loss = model(batch)
    loss.backward()

    step_time = time.time() - start_time
    tracker.log_step(
        step=step,
        num_samples=len(batch),
        num_tokens=batch_tokens,
        step_time=step_time,
    )

# Get summary
print(tracker.get_summary())
# {
#     "samples_per_second": 12.5,
#     "tokens_per_second": 25000,
#     "gpu_utilization": 95.2,
#     "gpu_memory_used_gb": 72.5,
# }
```

### 5.2 Inference Throughput

**Implementation**: `medai_compass/inference/optimized.py::ThroughputBenchmark`

```python
from medai_compass.inference.optimized import ThroughputBenchmark

benchmark = ThroughputBenchmark()

results = benchmark.run(
    inference_fn=model.generate,
    prompts=test_prompts,
    num_iterations=100,
    warmup_iterations=10,
)

print(f"Throughput: {results['tokens_per_second']:.0f} tok/s")
print(f"P50 Latency: {results['latency_p50_ms']:.1f} ms")
print(f"P99 Latency: {results['latency_p99_ms']:.1f} ms")
```

### 5.3 Expected Performance

#### Training Performance (8x H100 80GB)

| Model | Method | Throughput | Memory/GPU |
|-------|--------|------------|------------|
| 4B | LoRA | 50k tok/s | 20GB |
| 4B | QLoRA | 40k tok/s | 10GB |
| 27B | LoRA + ZeRO-3 | 15k tok/s | 60GB |
| 27B | QLoRA + ZeRO-3 | 12k tok/s | 25GB |

#### Inference Performance

| Model | Backend | P50 Latency | Throughput |
|-------|---------|-------------|------------|
| 4B | vLLM | 150ms | 150 req/s |
| 4B | Ray Serve | 200ms | 100 req/s |
| 27B | vLLM (4 GPU) | 400ms | 50 req/s |
| 27B | Modal H100 | 600ms | 30 req/s |

---

## 6. Quick Reference

### 6.1 Training Method Selection

```
Need extreme memory efficiency? → QLoRA
Need best quality? → LoRA with high rank (r=128)
Have unpaired feedback? → KTO
Have paired preferences? → DPO
Need full RLHF? → PPO with reward model
```

### 6.2 Distributed Strategy Selection

```
Single GPU (4B model)? → Standard training
Multi-GPU, large batch? → DDP
Multi-GPU, large model? → FSDP or DeepSpeed ZeRO-3
8+ GPUs, 27B+ model? → ZeRO-3 + LoRA
```

### 6.3 Inference Backend Selection

```
Maximum throughput? → vLLM
Autoscaling needs? → Ray Serve
Enterprise/Triton existing? → Triton
Serverless/no GPU? → Modal
```

### 6.4 Key Configuration Files

| Purpose | File |
|---------|------|
| Model configs | `config/hydra/model/*.yaml` |
| Training configs | `config/hydra/training/*.yaml` |
| Training algorithms | `medai_compass/training/algorithms/` |
| Distributed training | `medai_compass/training/distributed/` |
| Inference optimization | `medai_compass/inference/optimized.py` |
| Modal deployment | `medai_compass/modal/` |

---

## Additional Resources

- [INFERENCE_GUIDE.md](operations/INFERENCE_GUIDE.md) - Detailed inference deployment
- [PRODUCTION_DEPLOYMENT.md](operations/PRODUCTION_DEPLOYMENT.md) - Production setup
- [HIPAA_COMPLIANCE.md](compliance/HIPAA_COMPLIANCE.md) - Security requirements
- [PRINCIPAL_ARCHITECT_ANALYSIS.md](PRINCIPAL_ARCHITECT_ANALYSIS.md) - System architecture

---

*Last Updated: January 2026*
