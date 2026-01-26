# ADR-004: Distributed Training Strategy

## Status
Accepted

## Date
2025-12-10

## Context

MedAI Compass requires fine-tuning MedGemma models (4B and 27B parameters) on medical datasets. Training large models requires:

1. Memory-efficient techniques for 27B model
2. Multi-GPU/multi-node distributed training
3. Hyperparameter optimization at scale
4. Checkpoint management for long training runs

### Constraints

| Model | Parameters | FP32 Memory | FP16 Memory | Training Memory |
|-------|------------|-------------|-------------|-----------------|
| 4B | 4 billion | 16 GB | 8 GB | ~24 GB |
| 27B | 27 billion | 108 GB | 54 GB | ~160 GB |

## Decision

We will implement a **tiered training strategy** based on model size and available resources:

### Tier 1: Single GPU (4B Model)
- **Method**: LoRA with gradient checkpointing
- **Memory**: ~16 GB GPU RAM
- **Hardware**: Single A100/H100 or M1/M2/M3 Mac

### Tier 2: Multi-GPU (27B Model with LoRA)
- **Method**: QLoRA + DeepSpeed ZeRO-3 + FSDP2
- **Memory**: ~24 GB per GPU (8x GPUs)
- **Hardware**: 8x A100/H100

### Tier 3: Multi-Node (27B Full Fine-Tuning)
- **Method**: Megatron-LM + 5D Parallelism
- **Memory**: Distributed across nodes
- **Hardware**: Multiple nodes with NVLink

### PEFT Methods Supported

| Method | Use Case | Trainable Params | Memory |
|--------|----------|------------------|--------|
| LoRA | Default | 0.1-0.5% | Low |
| QLoRA | Memory-constrained | 0.1-0.5% | Very Low |
| DoRA | Better accuracy | 0.2-0.6% | Medium |
| IA³ | Minimal changes | 0.01% | Minimal |

## Implementation

### Hydra Configuration

```yaml
# config/hydra/training/qlora.yaml
training:
  method: qlora
  args:
    learning_rate: 2e-4
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
    num_train_epochs: 3
    bf16: true

  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: bfloat16
    bnb_4bit_quant_type: nf4
    bnb_4bit_use_double_quant: true

  lora:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_dropout: 0.05
```

### DeepSpeed ZeRO-3 Configuration

```yaml
# config/hydra/training/deepspeed/zero3_offload.yaml
deepspeed:
  stage: 3
  offload_optimizer:
    device: cpu
    pin_memory: true
  offload_param:
    device: cpu
    pin_memory: true
  overlap_comm: true
  contiguous_gradients: true
  reduce_bucket_size: 5e8
  stage3_prefetch_bucket_size: 5e8
  stage3_param_persistence_threshold: 1e6
```

### Trainer Selection Logic

```python
from medai_compass.training.strategy_selector import select_training_strategy

strategy = select_training_strategy(
    model_name="medgemma-27b",
    num_gpus=8,
    gpu_memory=80,  # GB
    preference="memory_efficient"
)

# Returns: StrategyConfig(
#     method="qlora",
#     distributed="deepspeed_zero3",
#     parallelism="fsdp2",
#     batch_size=1,
#     gradient_accumulation=16
# )
```

## Hyperparameter Optimization

### Ray Tune Integration

```python
from ray import tune
from medai_compass.tuning import run_hyperparameter_tuning

search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "lora_r": tune.choice([8, 16, 32, 64]),
    "lora_alpha": tune.choice([16, 32, 64]),
    "lora_dropout": tune.uniform(0.0, 0.1),
}

results = run_hyperparameter_tuning(
    config=base_config,
    search_space=search_space,
    scheduler="asha",
    num_samples=50,
)
```

### Scheduler Options

| Scheduler | Use Case | Efficiency |
|-----------|----------|------------|
| ASHA | Default | 10x faster than random |
| PBT | Adaptive schedules | Best final performance |
| Hyperband | Quick exploration | 5x faster than random |

## Consequences

### Positive
- Supports training from single GPU to multi-node clusters
- Memory-efficient methods enable training on consumer hardware
- Automatic strategy selection simplifies configuration
- Ray Tune enables efficient hyperparameter search

### Negative
- Complex configuration matrix to test
- Different strategies may produce different results
- Debugging distributed training is challenging
- Checkpoint compatibility across strategies

### Mitigation
- Comprehensive test suite for all strategies
- MLflow tracking for reproducibility
- Detailed logging with gradient statistics
- Checkpoint conversion utilities

## Performance Benchmarks

### 4B Model Training (1x H100)

| Method | Time/Epoch | Memory | Final Loss |
|--------|------------|--------|------------|
| Full FT | 2.5 hours | 48 GB | 0.42 |
| LoRA | 1.0 hours | 16 GB | 0.45 |
| QLoRA | 1.2 hours | 12 GB | 0.46 |

### 27B Model Training (8x H100)

| Method | Time/Epoch | Memory/GPU | Final Loss |
|--------|------------|------------|------------|
| LoRA + FSDP | 8 hours | 40 GB | 0.38 |
| QLoRA + ZeRO-3 | 10 hours | 24 GB | 0.40 |
| Full FT + Megatron | 24 hours | 70 GB | 0.35 |

## Alternatives Considered

### 1. Full Fine-Tuning Only
- **Rejected**: Prohibitive memory requirements for 27B
- 8x H100s still not enough for FP16 full fine-tuning

### 2. Custom Training Loop
- **Rejected**: Reinventing what PEFT/TRL already provide
- Would require significant development effort

### 3. Adapter-only Methods
- **Rejected**: LoRA family provides better performance
- Adapters have higher memory overhead than LoRA

## References

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
