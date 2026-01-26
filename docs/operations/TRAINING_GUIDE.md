# Training Guide

This guide covers fine-tuning MedGemma models using MedAI Compass's training infrastructure, from single GPU to multi-node distributed training.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Methods](#training-methods)
3. [Configuration](#configuration)
4. [Distributed Training](#distributed-training)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Monitoring Training](#monitoring-training)
7. [Best Practices](#best-practices)

---

## Quick Start

### Using the Pipeline CLI (Recommended)

The unified Pipeline CLI provides the easiest way to train models:

```bash
# Run full pipeline (data -> train -> evaluate)
uv run python -m medai_compass.pipelines run --model medgemma-4b

# Train with Hydra overrides
uv run python -m medai_compass.pipelines train \
    model=medgemma_27b \
    training.args.learning_rate=1e-4

# Train with Modal cloud GPUs
uv run python -m medai_compass.pipelines train --backend modal

# Dry run to verify configuration
uv run python -m medai_compass.pipelines train --dry-run
```

### Single GPU Training (LoRA)

```bash
# Using Pipeline CLI (recommended)
uv run python -m medai_compass.pipelines train model=medgemma_4b training=lora

# Or using direct module
python -m medai_compass.train \
    model=medgemma_4b \
    training=lora \
    compute=modal_h100
```

### Multi-GPU Training (QLoRA + ZeRO-3)

```bash
# 8x GPU training with DeepSpeed ZeRO-3
uv run python -m medai_compass.pipelines train \
    model=medgemma_27b \
    training=qlora \
    training/deepspeed=zero3_offload
```

### Quick Test

```bash
# Fast training for testing (100 steps)
uv run python -m medai_compass.pipelines train --max-steps 100 --dry-run
```

---

## Training Methods

### LoRA (Low-Rank Adaptation)

**Best for**: Fast training with good performance

```yaml
# config/hydra/training/lora.yaml
training:
  method: lora

  args:
    learning_rate: 2e-4
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    num_train_epochs: 3
    bf16: true

  lora:
    r: 16
    lora_alpha: 32
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
    lora_dropout: 0.05
    bias: none
```

**Usage**:
```bash
python -m medai_compass.train training=lora
```

**Trainable Parameters**: ~0.1-0.5% of model

### QLoRA (Quantized LoRA)

**Best for**: Memory-constrained environments

```yaml
# config/hydra/training/qlora.yaml
training:
  method: qlora

  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: bfloat16
    bnb_4bit_quant_type: nf4
    bnb_4bit_use_double_quant: true

  lora:
    r: 16
    lora_alpha: 32
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
```

**Usage**:
```bash
python -m medai_compass.train training=qlora
```

**Memory Reduction**: ~4x compared to LoRA

### DoRA (Weight-Decomposed LoRA)

**Best for**: Better accuracy on complex tasks

```yaml
# config/hydra/training/dora.yaml
training:
  method: dora

  lora:
    r: 16
    lora_alpha: 32
    use_dora: true  # Enable weight decomposition
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
```

**Usage**:
```bash
python -m medai_compass.train training=dora
```

**Accuracy Improvement**: ~5-10% on medical QA tasks

### IA³ (Infused Adapter by Inhibiting and Amplifying)

**Best for**: Minimal parameter overhead

```yaml
# config/hydra/training/ia3.yaml
training:
  method: ia3

  ia3:
    target_modules:
      - k_proj
      - v_proj
      - down_proj
    feedforward_modules:
      - down_proj
```

**Usage**:
```bash
python -m medai_compass.train training=ia3
```

**Trainable Parameters**: ~0.01% of model

### DPO (Direct Preference Optimization)

**Best for**: Alignment with human preferences

```yaml
# config/hydra/training/dpo.yaml
training:
  method: dpo

  dpo:
    beta: 0.1
    loss_type: sigmoid
    reference_free: false
    precompute_ref_log_probs: true

  args:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
    learning_rate: 5e-5
```

**Usage**:
```bash
python -m medai_compass.train training=dpo
```

**Data Format**:
```json
{
  "prompt": "What are the symptoms of pneumonia?",
  "chosen": "Pneumonia symptoms include fever, cough, and difficulty breathing...",
  "rejected": "Just Google it..."
}
```

### GRPO (Group Relative Policy Optimization)

**Best for**: More stable alignment training

```yaml
# config/hydra/training/grpo.yaml
training:
  method: grpo

  grpo:
    group_size: 4
    normalize_within_group: true
    baseline: mean

  args:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32
```

**Usage**:
```bash
python -m medai_compass.train training=grpo
```

---

## Configuration

### Model Selection

```bash
# 4B model (default)
python -m medai_compass.train model=medgemma_4b

# 27B model (requires more memory)
python -m medai_compass.train model=medgemma_27b
```

**Model Configurations**:

| Model | HF ID | Parameters | Min GPU Memory |
|-------|-------|------------|----------------|
| 4B | google/medgemma-4b-it | 4B | 16 GB |
| 27B | google/medgemma-27b-it | 27B | 40 GB (QLoRA) |

### Data Configuration

```yaml
# config/hydra/data/combined.yaml
data:
  train_file: data/train.jsonl
  eval_file: data/eval.jsonl
  max_seq_length: 8192
  preprocessing_num_workers: 4

  # Data format
  prompt_column: prompt
  response_column: response

  # Optional PHI filtering
  apply_phi_filter: true
```

**Usage**:
```bash
python -m medai_compass.train data=combined
```

### Compute Targets

```bash
# Modal H100 (cloud)
python -m medai_compass.train compute=modal_h100

# Modal A100 (cloud)
python -m medai_compass.train compute=modal_a100

# Local GPU
python -m medai_compass.train compute=local
```

### Parameter Overrides

```bash
# Override learning rate
python -m medai_compass.train training.args.learning_rate=1e-5

# Override batch size
python -m medai_compass.train training.args.per_device_train_batch_size=2

# Override LoRA rank
python -m medai_compass.train training.lora.r=32

# Multiple overrides
python -m medai_compass.train \
    training.args.learning_rate=1e-5 \
    training.args.num_train_epochs=5 \
    training.lora.r=32
```

---

## Distributed Training

### DeepSpeed ZeRO

#### ZeRO Stage 1 (Optimizer State Partitioning)

```yaml
# config/hydra/training/deepspeed/zero1.yaml
deepspeed:
  stage: 1
  allgather_partitions: true
  allgather_bucket_size: 5e8
  reduce_scatter: true
  reduce_bucket_size: 5e8
```

**Memory Reduction**: 4x

#### ZeRO Stage 2 (Gradient Partitioning)

```yaml
# config/hydra/training/deepspeed/zero2.yaml
deepspeed:
  stage: 2
  allgather_partitions: true
  allgather_bucket_size: 5e8
  reduce_scatter: true
  reduce_bucket_size: 5e8
  overlap_comm: true
  contiguous_gradients: true
```

**Memory Reduction**: 8x

#### ZeRO Stage 3 with CPU Offload

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

**Memory Reduction**: Linear with GPU count

**Usage**:
```bash
python -m medai_compass.train \
    model=medgemma_27b \
    training=qlora \
    training/deepspeed=zero3_offload
```

### FSDP2 (Fully Sharded Data Parallel)

```yaml
# config/hydra/training/fsdp2.yaml
fsdp:
  sharding_strategy: FULL_SHARD
  cpu_offload: false
  mixed_precision: bf16
  backward_prefetch: BACKWARD_PRE
  forward_prefetch: true
  use_orig_params: true
  limit_all_gathers: true
```

**Usage**:
```bash
python -m medai_compass.train \
    model=medgemma_27b \
    training=fsdp2
```

### Megatron-LM Parallelism

```yaml
# config/hydra/training/megatron.yaml
megatron:
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
  sequence_parallel: true
  micro_batch_size: 1
  global_batch_size: 32
```

**Usage**:
```bash
python -m medai_compass.train \
    model=medgemma_27b \
    training=megatron
```

### 5D Parallelism

```yaml
# config/hydra/training/parallelism_5d.yaml
parallelism:
  data_parallel_size: 2
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
  expert_parallel_size: 1
  optimizer_parallel_size: 1
```

---

## Hyperparameter Optimization

### ASHA Scheduler (Recommended)

```yaml
# config/hydra/tuning/asha.yaml
tuning:
  scheduler: asha
  num_samples: 50

  asha:
    time_attr: training_iteration
    max_t: 100
    grace_period: 10
    reduction_factor: 3
    brackets: 1

  search_space:
    learning_rate:
      type: loguniform
      min: 1e-5
      max: 1e-3
    lora_r:
      type: choice
      values: [8, 16, 32, 64]
    lora_alpha:
      type: choice
      values: [16, 32, 64]
```

**Usage**:
```bash
python -m medai_compass.tune tuning=asha
```

### Population-Based Training (PBT)

```yaml
# config/hydra/tuning/pbt.yaml
tuning:
  scheduler: pbt
  num_samples: 20

  pbt:
    time_attr: training_iteration
    perturbation_interval: 10
    hyperparam_mutations:
      learning_rate:
        type: loguniform
        min: 1e-5
        max: 1e-3
      weight_decay:
        type: uniform
        min: 0.0
        max: 0.1
```

**Usage**:
```bash
python -m medai_compass.tune tuning=pbt
```

### Hyperband

```yaml
# config/hydra/tuning/hyperband.yaml
tuning:
  scheduler: hyperband
  num_samples: 50

  hyperband:
    time_attr: training_iteration
    max_t: 100
    reduction_factor: 3
```

**Usage**:
```bash
python -m medai_compass.tune tuning=hyperband
```

### Multi-Run Sweeps

```bash
# Grid search over learning rates and LoRA ranks
python -m medai_compass.train --multirun \
    training.args.learning_rate=1e-5,1e-4,2e-4 \
    training.lora.r=8,16,32

# Random search with 50 trials
python -m medai_compass.train --multirun \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=50
```

---

## Monitoring Training

### MLflow Tracking

```bash
# Start MLflow server
uv run mlflow server \
    --backend-store-uri postgresql://mlflow:password@localhost/mlflow \
    --default-artifact-root s3://mlflow-artifacts \
    --host 0.0.0.0 --port 5000

# Open MLflow UI
open http://localhost:5000
```

**Tracked Metrics**:
- Training loss
- Evaluation loss
- Learning rate
- Gradient norm
- GPU memory usage

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=outputs/

# Open TensorBoard
open http://localhost:6006
```

### Ray Dashboard

```bash
# Start Ray dashboard (automatically started with training)
open http://localhost:8265
```

### Custom Callbacks

```python
from medai_compass.training.callbacks import (
    MedicalMetricsCallback,
    PHIFilterCallback,
    CheckpointCallback,
)

callbacks = [
    MedicalMetricsCallback(),  # Medical-specific metrics
    PHIFilterCallback(),        # PHI filtering in training
    CheckpointCallback(
        save_steps=500,
        save_total_limit=3,
    ),
]
```

---

## Best Practices

### Memory Optimization

```yaml
# Enable gradient checkpointing
training:
  args:
    gradient_checkpointing: true

# Use smaller batch with gradient accumulation
training:
  args:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
```

### Learning Rate Schedule

```yaml
training:
  args:
    warmup_ratio: 0.1
    lr_scheduler_type: cosine
```

### Regularization

```yaml
training:
  args:
    weight_decay: 0.01
    max_grad_norm: 1.0

  lora:
    lora_dropout: 0.05
```

### Early Stopping

```yaml
training:
  args:
    load_best_model_at_end: true
    metric_for_best_model: eval_loss
    greater_is_better: false
    early_stopping_patience: 3
```

### Checkpointing

```yaml
training:
  args:
    save_strategy: steps
    save_steps: 500
    save_total_limit: 3
    save_safetensors: true
```

### Data Quality

1. **PHI Filtering**: Always enable PHI filtering for medical data
2. **Deduplication**: Remove duplicate training examples
3. **Quality Validation**: Review sample training data
4. **Balanced Datasets**: Ensure class balance for classification

### Reproducibility

```yaml
project:
  seed: 42

training:
  args:
    seed: 42
    dataloader_num_workers: 4
    dataloader_pin_memory: true
```

---

## Training Recipes

### Recipe 1: Quick Fine-Tuning (4B)

```bash
python -m medai_compass.train \
    model=medgemma_4b \
    training=lora \
    training.args.num_train_epochs=1 \
    training.args.max_steps=1000
```

**Time**: ~1 hour on H100
**Best for**: Initial experiments

### Recipe 2: Production Fine-Tuning (4B)

```bash
python -m medai_compass.train \
    model=medgemma_4b \
    training=lora \
    training.args.num_train_epochs=3 \
    +experiment=production
```

**Time**: ~3 hours on H100
**Best for**: Production deployment

### Recipe 3: Memory-Efficient 27B

```bash
python -m medai_compass.train \
    model=medgemma_27b \
    training=qlora \
    training/deepspeed=zero3_offload \
    compute=modal_h100
```

**Time**: ~8 hours on 8x H100
**Best for**: Large model with limited memory

### Recipe 4: Maximum Accuracy 27B

```bash
python -m medai_compass.train \
    model=medgemma_27b \
    training=dora \
    training/deepspeed=zero3_offload \
    training.lora.r=64 \
    training.args.num_train_epochs=5
```

**Time**: ~20 hours on 8x H100
**Best for**: Best possible accuracy

### Recipe 5: Alignment Training

```bash
python -m medai_compass.train \
    model=medgemma_4b \
    training=dpo \
    data=preference_data \
    training.dpo.beta=0.1
```

**Time**: ~4 hours on H100
**Best for**: Aligning with clinical preferences

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
training.args.per_device_train_batch_size=1

# Enable gradient checkpointing
training.args.gradient_checkpointing=true

# Use QLoRA instead of LoRA
training=qlora

# Use DeepSpeed ZeRO-3
training/deepspeed=zero3_offload
```

### Slow Training

```bash
# Enable TF32 on H100
training.args.tf32=true

# Enable Flash Attention
training.args.use_flash_attention_2=true

# Increase workers
data.preprocessing_num_workers=8
```

### Poor Convergence

```bash
# Lower learning rate
training.args.learning_rate=1e-5

# Increase warmup
training.args.warmup_ratio=0.2

# Use cosine scheduler
training.args.lr_scheduler_type=cosine
```

### NaN Loss

```bash
# Enable gradient clipping
training.args.max_grad_norm=1.0

# Lower learning rate
training.args.learning_rate=1e-5

# Check data for issues
python scripts/validate_data.py
```

---

## Output Artifacts

After training, you'll find:

```
outputs/YYYY-MM-DD/HH-MM-SS/
├── adapter_config.json     # LoRA/PEFT configuration
├── adapter_model.safetensors  # Trained adapter weights
├── training_args.bin       # Training arguments
├── trainer_state.json      # Training state
├── config.yaml             # Hydra configuration
├── train_metrics.json      # Training metrics
├── eval_metrics.json       # Evaluation metrics
└── checkpoint-*/           # Intermediate checkpoints
```

### Loading Trained Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")

# Load adapter
model = PeftModel.from_pretrained(base_model, "outputs/2026-01-26/12-00-00")

# Merge for inference (optional)
model = model.merge_and_unload()
```
