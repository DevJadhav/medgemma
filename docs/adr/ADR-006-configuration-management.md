# ADR-006: Configuration Management with Hydra

## Status
Accepted

## Date
2026-01-10

## Context

MedAI Compass has complex configuration requirements:

1. Multiple model sizes (4B, 27B)
2. Multiple training methods (LoRA, QLoRA, DoRA, DPO, GRPO, etc.)
3. Multiple distributed strategies (DeepSpeed, FSDP, Megatron)
4. Multiple compute targets (local, Modal, cloud)
5. Hyperparameter tuning search spaces

We need a configuration system that:
- Supports hierarchical configuration composition
- Enables command-line overrides
- Supports multi-run sweeps for HPO
- Integrates with Ray Tune

## Decision

We will use **Hydra** as our configuration management framework, with:

### Hierarchical Configuration Structure

```
config/hydra/
├── config.yaml           # Main entry point with defaults
├── model/                # Model configurations
│   ├── medgemma_4b.yaml
│   └── medgemma_27b.yaml
├── training/             # Training method configurations
│   ├── lora.yaml
│   ├── qlora.yaml
│   ├── dora.yaml
│   ├── dpo.yaml
│   ├── grpo.yaml
│   └── deepspeed/
│       ├── zero1.yaml
│       ├── zero2.yaml
│       ├── zero3_offload.yaml
│       └── zero_infinity.yaml
├── compute/              # Compute target configurations
│   ├── modal_h100.yaml
│   ├── modal_a100.yaml
│   └── local.yaml
├── tuning/               # HPO scheduler configurations
│   ├── asha.yaml
│   ├── pbt.yaml
│   └── hyperband.yaml
├── data/                 # Dataset configurations
│   ├── combined.yaml
│   └── medqa.yaml
└── experiment/           # Pre-composed experiment profiles
    ├── production.yaml
    └── quick_test.yaml
```

### Default Composition

```yaml
# config/hydra/config.yaml
defaults:
  - model: medgemma_4b
  - training: lora
  - training/deepspeed: zero3_offload
  - compute: modal_h100
  - tuning: asha
  - data: combined
  - _self_
```

### Environment Variable Override

```python
# Support for UI configurability
model_name = os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-4b")
environment = os.environ.get("MEDAI_ENVIRONMENT", "development")
```

## Implementation

### Hydra Application Entry Point

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config/hydra", config_name="config")
def train(cfg: DictConfig) -> float:
    """Main training entry point with Hydra config."""

    # Load configuration
    model_config = cfg.model
    training_config = cfg.training

    # Initialize trainer
    trainer = create_trainer(model_config, training_config)

    # Train and return metrics
    metrics = trainer.train()
    return metrics["eval_loss"]

if __name__ == "__main__":
    train()
```

### Command-Line Overrides

```bash
# Default: 4B + LoRA + ZeRO-3 + Modal H100
python -m medai_compass.train

# Override model to 27B
python -m medai_compass.train model=medgemma_27b

# Override training method
python -m medai_compass.train training=qlora

# Override specific parameter
python -m medai_compass.train training.args.learning_rate=1e-5

# Multiple overrides
python -m medai_compass.train model=medgemma_27b training=qlora compute=local

# Multi-run sweep (hyperparameter search)
python -m medai_compass.train --multirun \
    training.args.learning_rate=1e-5,1e-4,2e-4 \
    training.lora.r=8,16,32
```

### Configuration Dataclasses

```python
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class ModelConfig:
    """Model configuration schema."""
    name: str = "medgemma-4b"
    hf_model_id: str = "google/medgemma-4b-it"
    max_seq_length: int = 8192
    torch_dtype: str = "bfloat16"

@dataclass
class TrainingConfig:
    """Training configuration schema."""
    method: str = "lora"
    args: TrainingArgs = field(default_factory=TrainingArgs)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: Optional[QuantConfig] = None

@dataclass
class Config:
    """Root configuration schema."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
```

### Ray Tune Integration

```python
from ray import tune
from hydra.utils import instantiate

def ray_tune_trainable(config: dict, hydra_cfg: DictConfig):
    """Trainable function for Ray Tune with Hydra config."""

    # Merge Ray Tune config with Hydra config
    merged_cfg = OmegaConf.merge(hydra_cfg, OmegaConf.create(config))

    # Create trainer from merged config
    trainer = instantiate(merged_cfg.trainer)

    # Train and report
    for epoch in range(merged_cfg.training.args.num_train_epochs):
        metrics = trainer.train_epoch()
        tune.report(loss=metrics["loss"], epoch=epoch)

# Search space from Hydra config
search_space = {
    "training.args.learning_rate": tune.loguniform(1e-5, 1e-3),
    "training.lora.r": tune.choice([8, 16, 32]),
}
```

## Consequences

### Positive
- Clean separation of configuration concerns
- Easy experimentation with different configurations
- Command-line overrides without code changes
- Type-safe configuration with dataclasses
- Native multi-run support for HPO
- Configuration versioning and reproducibility

### Negative
- Learning curve for Hydra concepts
- Complex configuration debugging
- OmegaConf interpolation can be confusing
- IDE support is limited

### Mitigation
- Comprehensive documentation with examples
- Pre-composed experiment profiles for common cases
- Validation with Pydantic/dataclass schemas
- Configuration printing in logs for debugging

## Configuration Examples

### Production 27B Training

```yaml
# config/hydra/experiment/production.yaml
defaults:
  - /model: medgemma_27b
  - /training: qlora
  - /training/deepspeed: zero3_offload
  - /compute: modal_h100
  - override /data: combined

training:
  args:
    num_train_epochs: 3
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
    learning_rate: 2e-4
    warmup_ratio: 0.1
    save_strategy: steps
    save_steps: 500
```

### Quick Test Configuration

```yaml
# config/hydra/experiment/quick_test.yaml
defaults:
  - /model: medgemma_4b
  - /training: lora
  - /compute: local

training:
  args:
    max_steps: 100
    per_device_train_batch_size: 1
    logging_steps: 10
```

## Alternatives Considered

### 1. Plain YAML Files
- **Rejected**: No composition, override, or multi-run support
- Would require custom parsing logic

### 2. Python Dictionaries
- **Rejected**: No type safety or validation
- Hard to manage at scale

### 3. Gin-Config
- **Rejected**: Less intuitive syntax than Hydra
- Smaller community and ecosystem

### 4. ML-Collections
- **Rejected**: Less powerful than Hydra for composition
- No native CLI override support

## References

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf](https://omegaconf.readthedocs.io/)
- [Hydra + Ray Tune Integration](https://hydra.cc/docs/plugins/ray_launcher/)
