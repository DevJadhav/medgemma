"""Hydra configuration management for MedGemma training.

This module provides type-safe configuration management using Hydra and dataclasses.
It supports hierarchical configuration composition, command-line overrides, and
conversion to HuggingFace training arguments.

Usage:
    # Load default configuration
    cfg = load_config()

    # Load with overrides
    cfg = load_config_with_overrides(["model=medgemma_27b", "training.args.learning_rate=1e-4"])

    # Convert to training arguments
    training_args = to_training_args(cfg)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

from omegaconf import DictConfig, OmegaConf, MISSING


# ============================================================================
# Model Configuration
# ============================================================================


@dataclass
class ModelArchitectureConfig:
    """Model architecture configuration (read-only reference)."""

    hidden_size: int = 3072
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    num_hidden_layers: int = 28
    intermediate_size: int = 8192
    vocab_size: int = 262144
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0


@dataclass
class QuantizationConfig:
    """Quantization configuration for BitsAndBytes."""

    enabled: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class MemoryConfig:
    """Memory optimization configuration."""

    gradient_checkpointing: bool = True
    use_reentrant: bool = False


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "google/medgemma-4b-it"
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    use_cache: bool = False
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""

    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: Optional[List[str]] = None
    fan_in_fan_out: bool = False
    init_lora_weights: bool = True


@dataclass
class TrainingArgsConfig:
    """HuggingFace TrainingArguments configuration."""

    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1

    # Batch size
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Gradient handling
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Optimizer
    optim: str = "adamw_torch_fused"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0

    # Precision
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True

    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True
    logging_nan_inf_filter: bool = True
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    save_safetensors: bool = True
    save_on_each_node: bool = False

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    eval_accumulation_steps: int = 1
    eval_on_start: bool = False

    # Performance
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True

    # Sequence handling
    group_by_length: bool = True
    length_column_name: str = "length"

    # Reproducibility
    seed: int = 42
    data_seed: int = 42

    # Output
    output_dir: str = "/checkpoints"
    logging_dir: str = "/logs"
    overwrite_output_dir: bool = True

    # Distributed training
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration."""

    enabled: bool = True
    config: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """Training configuration."""

    method: str = "lora"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    args: TrainingArgsConfig = field(default_factory=TrainingArgsConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    quantization: Optional[QuantizationConfig] = None


# ============================================================================
# Data Configuration
# ============================================================================


@dataclass
class DataPreprocessingConfig:
    """Data preprocessing configuration."""

    max_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"


@dataclass
class DataConfig:
    """Data configuration."""

    name: str = "combined_medical"
    path: str = "/data/combined_medical"
    source: Optional[str] = None
    split: str = "train"
    preprocessing: DataPreprocessingConfig = field(default_factory=DataPreprocessingConfig)
    template: Optional[str] = None
    streaming: bool = False
    num_proc: int = 4
    shuffle: bool = True
    shuffle_seed: int = 42
    train_split: float = 0.95
    eval_split: float = 0.05
    cache_dir: str = "/data/cache"
    use_cache: bool = True


# ============================================================================
# Compute Configuration
# ============================================================================


@dataclass
class GPUConfig:
    """GPU configuration."""

    type: str = "H100"
    count: int = 8
    memory_gb: Optional[int] = 80


@dataclass
class ContainerConfig:
    """Container configuration."""

    cpu: int = 32
    memory_gb: int = 256
    timeout_seconds: int = 86400


@dataclass
class VolumeConfig:
    """Volume mount configuration."""

    name: str = ""
    mount_path: str = ""


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    enabled: bool = True
    strategy: str = "deepspeed"
    world_size: int = 8
    backend: str = "nccl"
    nccl: Optional[Dict[str, str]] = None


@dataclass
class ComputeConfig:
    """Compute configuration."""

    backend: str = "modal"
    provider: str = "modal"
    gpu: GPUConfig = field(default_factory=GPUConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    volumes: Dict[str, VolumeConfig] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=lambda: ["huggingface-secret"])
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    env: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Tuning Configuration
# ============================================================================


@dataclass
class SearchSpaceParamConfig:
    """Search space parameter configuration."""

    type: str = "uniform"
    lower: Optional[float] = None
    upper: Optional[float] = None
    values: Optional[List[Any]] = None


@dataclass
class ASHAConfig:
    """ASHA scheduler configuration."""

    grace_period: int = 100
    max_t: int = 5000
    reduction_factor: int = 3
    brackets: int = 1
    stop_last_trials: bool = True


@dataclass
class PBTConfig:
    """PBT scheduler configuration."""

    perturbation_interval: int = 100
    quantile_fraction: float = 0.25
    resample_probability: float = 0.25
    perturbation_factors: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    synch: bool = False
    log_config: bool = True


@dataclass
class HyperbandConfig:
    """Hyperband scheduler configuration."""

    max_t: int = 5000
    grace_period: int = 50
    reduction_factor: int = 3
    stop_last_trials: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint configuration for tuning."""

    num_to_keep: int = 3
    checkpoint_score_attribute: str = "eval_loss"
    checkpoint_score_order: str = "min"


@dataclass
class FailureConfig:
    """Failure handling configuration."""

    max_failures: int = 3
    fail_fast: bool = False


@dataclass
class ResourcesPerTrialConfig:
    """Resources per trial configuration."""

    cpu: int = 4
    gpu: int = 1


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration."""

    enabled: bool = True
    scheduler: str = "asha"
    num_samples: int = 50
    max_concurrent_trials: int = 8
    metric: str = "eval_loss"
    mode: str = "min"
    search_algorithm: str = "optuna"
    resources_per_trial: ResourcesPerTrialConfig = field(
        default_factory=ResourcesPerTrialConfig
    )
    search_space: Dict[str, SearchSpaceParamConfig] = field(default_factory=dict)
    asha: ASHAConfig = field(default_factory=ASHAConfig)
    pbt: PBTConfig = field(default_factory=PBTConfig)
    hyperband: HyperbandConfig = field(default_factory=HyperbandConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)
    storage_path: str = "/checkpoints/tune"


# ============================================================================
# Project Configuration
# ============================================================================


@dataclass
class ExperimentTrackingConfig:
    """Experiment tracking configuration."""

    enabled: bool = True
    backend: str = "mlflow"
    project_name: str = "medgemma"


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str = "medgemma-training"
    version: str = "1.0.0"
    seed: int = 42
    output_dir: str = "/checkpoints"
    logging_dir: str = "/logs"
    experiment_tracking: ExperimentTrackingConfig = field(
        default_factory=ExperimentTrackingConfig
    )


# ============================================================================
# Root Configuration
# ============================================================================


@dataclass
class MedGemmaConfig:
    """Root configuration for MedGemma training."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


# ============================================================================
# Configuration Loading Functions
# ============================================================================


def get_config_path() -> Path:
    """Get the path to the Hydra config directory."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "config" / "hydra",
        Path.cwd() / "config" / "hydra",
        Path(os.environ.get("MEDGEMMA_CONFIG_PATH", "")) / "hydra",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Default to relative path from package
    return Path(__file__).parent.parent.parent / "config" / "hydra"


def load_config(
    config_path: Optional[str] = None,
    config_name: str = "config",
) -> DictConfig:
    """
    Load configuration using Hydra.

    Args:
        config_path: Path to config directory. If None, uses default.
        config_name: Name of the config file (without .yaml extension).

    Returns:
        OmegaConf DictConfig with resolved configuration.
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    if config_path is None:
        config_path = str(get_config_path())

    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name=config_name)

    return cfg


def load_config_with_overrides(
    overrides: List[str],
    config_path: Optional[str] = None,
    config_name: str = "config",
) -> DictConfig:
    """
    Load configuration with command-line style overrides.

    Args:
        overrides: List of overrides in Hydra format, e.g.,
                   ["model=medgemma_27b", "training.args.learning_rate=1e-4"]
        config_path: Path to config directory. If None, uses default.
        config_name: Name of the config file (without .yaml extension).

    Returns:
        OmegaConf DictConfig with resolved configuration.
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()

    if config_path is None:
        config_path = str(get_config_path())

    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


# ============================================================================
# Conversion Functions
# ============================================================================


def to_training_args(cfg: DictConfig) -> Dict[str, Any]:
    """
    Convert Hydra config to HuggingFace TrainingArguments dict.

    Args:
        cfg: OmegaConf DictConfig with training configuration.

    Returns:
        Dictionary suitable for TrainingArguments(**dict).
    """
    # Extract training args
    args = OmegaConf.to_container(cfg.training.args, resolve=True)

    # Ensure output directories are set
    args["output_dir"] = cfg.project.output_dir
    args["logging_dir"] = cfg.project.logging_dir
    args["seed"] = cfg.project.seed

    # Add DeepSpeed config if enabled
    if cfg.training.deepspeed.enabled:
        args["deepspeed"] = OmegaConf.to_container(
            cfg.training.deepspeed.config, resolve=True
        )

    # Handle gradient checkpointing kwargs
    if "gradient_checkpointing_kwargs" in args:
        args["gradient_checkpointing_kwargs"] = dict(args["gradient_checkpointing_kwargs"])

    return args


def to_lora_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Convert Hydra config to PEFT LoraConfig dict.

    Args:
        cfg: OmegaConf DictConfig with training configuration.

    Returns:
        Dictionary suitable for LoraConfig(**dict).
    """
    lora_cfg = OmegaConf.to_container(cfg.training.lora, resolve=True)

    # Ensure target_modules is a list
    if isinstance(lora_cfg.get("target_modules"), str):
        lora_cfg["target_modules"] = [lora_cfg["target_modules"]]

    return lora_cfg


def to_deepspeed_config(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """
    Convert Hydra config to DeepSpeed config dict.

    Args:
        cfg: OmegaConf DictConfig with training configuration.

    Returns:
        Dictionary suitable for DeepSpeed, or None if disabled.
    """
    if not cfg.training.deepspeed.enabled:
        return None

    return OmegaConf.to_container(cfg.training.deepspeed.config, resolve=True)


def to_quantization_config(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """
    Convert Hydra config to BitsAndBytes quantization config.

    Args:
        cfg: OmegaConf DictConfig with model configuration.

    Returns:
        Dictionary for BitsAndBytesConfig, or None if disabled.
    """
    quant_cfg = cfg.model.quantization
    if not quant_cfg.enabled:
        return None

    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    return {
        "load_in_4bit": quant_cfg.load_in_4bit,
        "load_in_8bit": quant_cfg.load_in_8bit,
        "bnb_4bit_compute_dtype": dtype_map.get(quant_cfg.bnb_4bit_compute_dtype, torch.bfloat16),
        "bnb_4bit_quant_type": quant_cfg.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": quant_cfg.bnb_4bit_use_double_quant,
    }


# ============================================================================
# Search Space Conversion
# ============================================================================


def search_space_to_ray_tune(cfg: DictConfig) -> Dict[str, Any]:
    """
    Convert Hydra search space config to Ray Tune search space.

    Args:
        cfg: OmegaConf DictConfig with tuning configuration.

    Returns:
        Dictionary of Ray Tune search space objects.
    """
    from ray import tune

    space = {}
    for name, param in cfg.tuning.search_space.items():
        param_type = param.type

        if param_type == "uniform":
            space[name] = tune.uniform(param.lower, param.upper)
        elif param_type == "loguniform":
            space[name] = tune.loguniform(param.lower, param.upper)
        elif param_type == "choice":
            space[name] = tune.choice(param.values)
        elif param_type == "randint":
            space[name] = tune.randint(param.lower, param.upper)
        elif param_type == "quniform":
            space[name] = tune.quniform(param.lower, param.upper, param.get("q", 1))
        else:
            raise ValueError(f"Unknown search space type: {param_type}")

    return space


def validate_config(cfg: DictConfig) -> List[str]:
    """
    Validate configuration for common issues.

    Args:
        cfg: OmegaConf DictConfig to validate.

    Returns:
        List of warning/error messages. Empty if valid.
    """
    issues = []

    # Check model exists
    if not cfg.model.name:
        issues.append("Model name is required")

    # Check batch size and gradient accumulation
    effective_batch = (
        cfg.training.args.per_device_train_batch_size
        * cfg.training.args.gradient_accumulation_steps
        * cfg.compute.gpu.count
    )
    if effective_batch > 512:
        issues.append(f"Effective batch size ({effective_batch}) is very large")

    # Check learning rate
    if cfg.training.args.learning_rate > 1e-3:
        issues.append(f"Learning rate ({cfg.training.args.learning_rate}) is very high")

    # Check LoRA configuration
    if cfg.training.lora.r > cfg.training.lora.lora_alpha:
        issues.append("LoRA alpha should typically be >= r")

    # Check compute resources
    if cfg.compute.backend == "modal":
        if cfg.compute.gpu.count > 8:
            issues.append("Modal supports max 8 GPUs per container")

    return issues
