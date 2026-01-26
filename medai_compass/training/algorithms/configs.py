"""
Configuration classes for all training algorithms.

Each config class provides model-specific defaults and validation.
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Model-specific base configurations
MODEL_CONFIGS = {
    "medgemma-4b": {
        "hf_model_id": "google/medgemma-4b-it",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_length": 8192,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "medgemma-27b": {
        "hf_model_id": "google/medgemma-27b-it",
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "max_seq_length": 4096,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}

MODEL_ALIASES = {
    "4b": "medgemma-4b",
    "medgemma-4b-it": "medgemma-4b",
    "google/medgemma-4b-it": "medgemma-4b",
    "27b": "medgemma-27b",
    "medgemma-27b-it": "medgemma-27b",
    "google/medgemma-27b-it": "medgemma-27b",
}


def _resolve_model_name(model_name: str) -> str:
    """Resolve model alias to canonical name."""
    return MODEL_ALIASES.get(model_name.lower(), model_name.lower())


def _get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration."""
    resolved = _resolve_model_name(model_name)
    return MODEL_CONFIGS.get(resolved, MODEL_CONFIGS["medgemma-4b"])


# =============================================================================
# Base Configuration
# =============================================================================

@dataclass
class BaseTrainingConfig:
    """Base configuration for all training algorithms."""

    model_name: str = "medgemma-4b"
    hf_model_id: str = ""

    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 10000
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 8192

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: int = 500

    def __post_init__(self):
        """Initialize model-specific defaults."""
        config = _get_model_config(self.model_name)
        if not self.hf_model_id:
            self.hf_model_id = config["hf_model_id"]

    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments dictionary."""
        return {
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
        }


# =============================================================================
# LoRA Configuration
# =============================================================================

@dataclass
class LoRAConfig(BaseTrainingConfig):
    """
    LoRA (Low-Rank Adaptation) configuration.

    LoRA adds trainable low-rank matrices to attention layers,
    enabling efficient fine-tuning with minimal parameters.
    """

    # LoRA parameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        super().__post_init__()
        config = _get_model_config(self.model_name)
        resolved = _resolve_model_name(self.model_name)

        if resolved == "medgemma-27b":
            self.r = config["lora_r"]
            self.lora_alpha = config["lora_alpha"]
            self.batch_size = config["batch_size"]
            self.gradient_accumulation_steps = config["gradient_accumulation_steps"]

        self.target_modules = config["target_modules"]

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "LoRAConfig":
        """Create LoRA config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT config dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


# =============================================================================
# QLoRA Configuration
# =============================================================================

@dataclass
class QLoRAConfig(LoRAConfig):
    """
    QLoRA (Quantized LoRA) configuration.

    QLoRA combines 4-bit quantization with LoRA for memory-efficient
    fine-tuning of large models.
    """

    # Quantization parameters
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "QLoRAConfig":
        """Create QLoRA config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization config dictionary."""
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }


# =============================================================================
# DoRA Configuration
# =============================================================================

@dataclass
class DoRAConfig(BaseTrainingConfig):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) configuration.

    DoRA decomposes weight updates into magnitude and direction components,
    providing better training stability and performance.
    """

    # LoRA base parameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # DoRA specific
    use_dora: bool = True

    def __post_init__(self):
        super().__post_init__()
        config = _get_model_config(self.model_name)
        resolved = _resolve_model_name(self.model_name)

        if resolved == "medgemma-27b":
            self.r = config["lora_r"]
            self.lora_alpha = config["lora_alpha"]
            self.batch_size = config["batch_size"]
            self.gradient_accumulation_steps = config["gradient_accumulation_steps"]

        self.target_modules = config["target_modules"]

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "DoRAConfig":
        """Create DoRA config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT config dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "use_dora": self.use_dora,
        }


# =============================================================================
# Adapter Configuration
# =============================================================================

@dataclass
class AdapterConfig(BaseTrainingConfig):
    """
    Adapter Module configuration (Houlsby/Pfeiffer).

    Adapters are bottleneck layers inserted into transformer blocks.
    - Houlsby: Adapters after both attention and FFN
    - Pfeiffer: Adapters only after FFN (more efficient)
    """

    adapter_type: str = "houlsby"  # "houlsby" or "pfeiffer"
    bottleneck_dim: int = 64
    reduction_factor: int = 16
    non_linearity: str = "relu"
    init_weights: str = "bert"
    scaling: float = 1.0

    # Which layers to add adapters
    leave_out: List[int] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        resolved = _resolve_model_name(self.model_name)

        if resolved == "medgemma-27b":
            self.bottleneck_dim = 128
            self.reduction_factor = 32

    @classmethod
    def for_model(cls, model_name: str, adapter_type: str = "houlsby", **kwargs) -> "AdapterConfig":
        """Create Adapter config for a specific model."""
        return cls(model_name=model_name, adapter_type=adapter_type, **kwargs)

    def get_adapter_config(self) -> Dict[str, Any]:
        """Get adapter configuration dictionary."""
        return {
            "adapter_type": self.adapter_type,
            "bottleneck_dim": self.bottleneck_dim,
            "reduction_factor": self.reduction_factor,
            "non_linearity": self.non_linearity,
            "init_weights": self.init_weights,
            "scaling": self.scaling,
            "leave_out": self.leave_out,
        }


# =============================================================================
# IA3 Configuration
# =============================================================================

@dataclass
class IA3Config(BaseTrainingConfig):
    """
    IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations).

    IA3 learns scaling vectors for key, value, and feedforward layers,
    requiring far fewer parameters than LoRA.
    """

    target_modules: List[str] = field(default_factory=lambda: [
        "k_proj", "v_proj", "down_proj"
    ])
    feedforward_modules: List[str] = field(default_factory=lambda: [
        "down_proj"
    ])
    task_type: str = "CAUSAL_LM"
    init_ia3_weights: bool = True

    def __post_init__(self):
        super().__post_init__()
        config = _get_model_config(self.model_name)
        # IA3 typically uses a subset of modules
        self.target_modules = ["k_proj", "v_proj", "down_proj"]
        self.feedforward_modules = ["down_proj"]

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "IA3Config":
        """Create IA3 config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT IA3 config dictionary."""
        return {
            "target_modules": self.target_modules,
            "feedforward_modules": self.feedforward_modules,
            "task_type": self.task_type,
            "init_ia3_weights": self.init_ia3_weights,
        }


# =============================================================================
# RLHF/PPO Configuration
# =============================================================================

@dataclass
class RLHFConfig(BaseTrainingConfig):
    """
    RLHF (Reinforcement Learning from Human Feedback) with PPO.

    Uses a reward model to provide feedback and PPO to optimize the policy.
    """

    # PPO parameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    value_coefficient: float = 0.1
    entropy_coefficient: float = 0.01
    gamma: float = 1.0
    lam: float = 0.95

    # Reward model
    reward_model_name: str = ""
    kl_penalty: str = "kl"  # "kl", "abs", "mse", "full"
    kl_coefficient: float = 0.1
    target_kl: float = 0.1

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self):
        super().__post_init__()
        # RLHF typically uses smaller learning rates
        self.learning_rate = 1e-5

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "RLHFConfig":
        """Create RLHF config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_ppo_config(self) -> Dict[str, Any]:
        """Get PPO configuration dictionary."""
        return {
            "ppo_epochs": self.ppo_epochs,
            "clip_range": self.clip_range,
            "clip_range_value": self.clip_range_value,
            "vf_coef": self.value_coefficient,
            "ent_coef": self.entropy_coefficient,
            "gamma": self.gamma,
            "lam": self.lam,
            "kl_penalty": self.kl_penalty,
            "init_kl_coef": self.kl_coefficient,
            "target_kl": self.target_kl,
        }


# =============================================================================
# DPO Configuration
# =============================================================================

@dataclass
class DPOConfig(BaseTrainingConfig):
    """
    DPO (Direct Preference Optimization) configuration.

    DPO directly optimizes for human preferences without a reward model,
    using a binary cross-entropy objective.
    """

    beta: float = 0.1  # Temperature parameter
    loss_type: str = "sigmoid"  # "sigmoid", "hinge", "ipo"
    label_smoothing: float = 0.0
    reference_free: bool = False
    generate_during_eval: bool = False

    # Margin for hinge loss
    margin: float = 0.0

    # For IPO loss
    ipo_tau: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        # DPO typically uses smaller learning rates
        self.learning_rate = 5e-7

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "DPOConfig":
        """Create DPO config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_dpo_config(self) -> Dict[str, Any]:
        """Get DPO configuration dictionary."""
        return {
            "beta": self.beta,
            "loss_type": self.loss_type,
            "label_smoothing": self.label_smoothing,
            "reference_free": self.reference_free,
            "generate_during_eval": self.generate_during_eval,
        }


# =============================================================================
# KTO Configuration
# =============================================================================

@dataclass
class KTOConfig(BaseTrainingConfig):
    """
    KTO (Kahneman-Tversky Optimization) configuration.

    KTO uses prospect theory to weight losses more heavily than gains,
    requiring only binary feedback (good/bad) instead of pairwise preferences.
    """

    beta: float = 0.1
    loss_aversion: float = 1.5  # Lambda in prospect theory (typically > 1)
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.learning_rate = 5e-7

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "KTOConfig":
        """Create KTO config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_kto_config(self) -> Dict[str, Any]:
        """Get KTO configuration dictionary."""
        return {
            "beta": self.beta,
            "desirable_weight": self.desirable_weight,
            "undesirable_weight": self.undesirable_weight,
        }


# =============================================================================
# GRPO Configuration
# =============================================================================

@dataclass
class GRPOConfig(BaseTrainingConfig):
    """
    GRPO (Group Relative Policy Optimization) configuration.

    GRPO generates multiple responses per prompt and uses group-relative
    rewards for more stable training without a separate reward model.
    """

    group_size: int = 4  # Number of responses per prompt
    num_generations: int = 4
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 256

    # Reward aggregation
    reward_aggregation: str = "mean"  # "mean", "max", "min"
    normalize_rewards: bool = True

    # KL penalty
    kl_coefficient: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        self.learning_rate = 1e-6

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "GRPOConfig":
        """Create GRPO config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_grpo_config(self) -> Dict[str, Any]:
        """Get GRPO configuration dictionary."""
        return {
            "num_generations": self.num_generations,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "reward_aggregation": self.reward_aggregation,
            "normalize_rewards": self.normalize_rewards,
        }


# =============================================================================
# mHC Configuration
# =============================================================================

@dataclass
class MHCConfig(BaseTrainingConfig):
    """
    mHC (Manifold-Constrained Hyper-Connections) configuration.

    mHC adds learnable connections between layers while constraining
    updates to lie on a low-dimensional manifold for stability.
    """

    manifold_dim: int = 64
    constraint_strength: float = 0.1
    hyper_connection_type: str = "residual"  # "residual", "dense", "skip"
    connection_rank: int = 8
    use_layer_norm: bool = True
    dropout: float = 0.1

    # Target layers for hyper-connections
    connection_layers: List[int] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        resolved = _resolve_model_name(self.model_name)

        if resolved == "medgemma-27b":
            self.manifold_dim = 128
            self.connection_rank = 16

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "MHCConfig":
        """Create mHC config for a specific model."""
        return cls(model_name=model_name, **kwargs)

    def get_mhc_config(self) -> Dict[str, Any]:
        """Get mHC configuration dictionary."""
        return {
            "manifold_dim": self.manifold_dim,
            "constraint_strength": self.constraint_strength,
            "hyper_connection_type": self.hyper_connection_type,
            "connection_rank": self.connection_rank,
            "use_layer_norm": self.use_layer_norm,
            "dropout": self.dropout,
            "connection_layers": self.connection_layers,
        }


# =============================================================================
# Unified Training Configuration
# =============================================================================

VALID_ALGORITHMS = [
    "lora", "qlora", "dora", "adapter_houlsby", "adapter_pfeiffer",
    "ia3", "rlhf_ppo", "dpo", "kto", "grpo", "mhc"
]


@dataclass
class UnifiedTrainingConfig:
    """
    Unified configuration that can represent any training algorithm.

    Provides a single interface for configuring all supported algorithms.
    """

    model_name: str = "medgemma-4b"
    algorithm: str = "lora"

    # Common parameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    warmup_ratio: float = 0.1

    # Algorithm-specific parameters (optional overrides)
    algorithm_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. "
                f"Valid options: {VALID_ALGORITHMS}"
            )

    def validate(self) -> bool:
        """Validate configuration."""
        return self.algorithm in VALID_ALGORITHMS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            "algorithm_params": self.algorithm_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UnifiedTrainingConfig":
        """Create from dictionary."""
        return cls(**config_dict)

    def get_algorithm_config(self):
        """Get the specific algorithm configuration."""
        config_map = {
            "lora": DoRAConfig,  # LoRA uses DoRA config with use_dora=False
            "qlora": DoRAConfig,
            "dora": DoRAConfig,
            "adapter_houlsby": AdapterConfig,
            "adapter_pfeiffer": AdapterConfig,
            "ia3": IA3Config,
            "rlhf_ppo": RLHFConfig,
            "dpo": DPOConfig,
            "kto": KTOConfig,
            "grpo": GRPOConfig,
            "mhc": MHCConfig,
        }

        config_cls = config_map.get(self.algorithm)
        if config_cls is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Build config with overrides
        kwargs = {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            **self.algorithm_params,
        }

        # Handle special cases
        if self.algorithm == "lora":
            kwargs["use_dora"] = False
        elif self.algorithm == "adapter_houlsby":
            kwargs["adapter_type"] = "houlsby"
        elif self.algorithm == "adapter_pfeiffer":
            kwargs["adapter_type"] = "pfeiffer"

        return config_cls(**kwargs)
