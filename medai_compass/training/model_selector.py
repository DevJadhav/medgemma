"""Model selector for MedGemma training.

Provides utilities to select between MedGemma 4B IT and MedGemma 27B IT models,
with appropriate configurations for training and inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ModelNotFoundError(Exception):
    """Raised when requested model is not found in configuration."""
    pass


# Model aliases for CLI convenience
MODEL_ALIASES: Dict[str, str] = {
    "medgemma-4b": "medgemma_4b_it",
    "medgemma_4b": "medgemma_4b_it",
    "4b": "medgemma_4b_it",
    "small": "medgemma_4b_it",
    "medgemma-27b": "medgemma_27b_it",
    "medgemma_27b": "medgemma_27b_it",
    "27b": "medgemma_27b_it",
    "large": "medgemma_27b_it",
}


@dataclass
class ModelConfig:
    """Configuration for a MedGemma model variant."""
    
    model_id: str
    hf_model_id: str
    display_name: str
    parameters: int
    vram_requirement_gb: float
    gpu_count: int
    training_gpu_count: int
    max_sequence_length: int
    supports_multimodal: bool
    
    # Quantization settings
    quantization_enabled: bool = True
    quantization_bits: int = 4
    quantization_type: str = "nf4"
    
    # Training defaults
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    distributed_strategy: Optional[str] = None
    deepspeed_config: Optional[Dict[str, Any]] = None


def _load_models_config() -> Dict[str, Any]:
    """Load models configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found at {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_model_name(model_name: str) -> str:
    """Resolve model alias to canonical name."""
    normalized = model_name.lower().strip().replace("-", "_")
    
    # Check if it's an alias
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    
    # Check if it's already a canonical name
    config = _load_models_config()
    if normalized in config.get("models", {}):
        return normalized
    
    # Check if the original name matches
    if model_name in config.get("models", {}):
        return model_name
    
    raise ModelNotFoundError(
        f"Model '{model_name}' not found. Available models: "
        f"{list(config.get('models', {}).keys())}"
    )


def select_model(model_name: str) -> Dict[str, Any]:
    """
    Select a MedGemma model by name and return its configuration.
    
    Args:
        model_name: Model name or alias (e.g., "medgemma-4b", "27b", "large")
        
    Returns:
        Dictionary with model configuration including:
        - model_id: Canonical model identifier
        - hf_model_id: HuggingFace model ID
        - gpu_count: Number of GPUs for inference
        - training_gpu_count: Number of GPUs for training
        - vram_requirement_gb: VRAM requirement in GB
        
    Raises:
        ModelNotFoundError: If model is not found
    """
    canonical_name = _resolve_model_name(model_name)
    config = _load_models_config()
    model_config = config["models"][canonical_name]
    
    return {
        "model_id": model_config["hf_model_id"],
        "hf_model_id": model_config["hf_model_id"],
        "canonical_name": canonical_name,
        "display_name": model_config["display_name"],
        "parameters": model_config["parameters"],
        "gpu_count": model_config["inference"]["gpu_count"],
        "training_gpu_count": model_config["training"]["gpu_count"],
        "vram_requirement_gb": model_config["vram_requirement_gb"],
        "max_sequence_length": model_config.get("context_length", 8192),
        "supports_multimodal": model_config["capabilities"].get("image_understanding", False),
        "quantization": model_config["inference"].get("quantization", {}),
    }


def get_training_config(model_name: str) -> Dict[str, Any]:
    """
    Get training-specific configuration for a model.
    
    Args:
        model_name: Model name or alias
        
    Returns:
        Dictionary with training configuration including:
        - batch_size: Training batch size
        - learning_rate: Learning rate
        - lora_r: LoRA rank
        - lora_alpha: LoRA alpha
        - gpu_count: Number of GPUs required
        - distributed_strategy: Strategy for distributed training
        - deepspeed_config: DeepSpeed configuration (for 27B)
    """
    canonical_name = _resolve_model_name(model_name)
    config = _load_models_config()
    model_config = config["models"][canonical_name]
    training_config = model_config["training"]
    finetune_config = model_config.get("finetune", {})
    
    result = {
        "model_id": model_config["hf_model_id"],
        "canonical_name": canonical_name,
        "batch_size": finetune_config.get("batch_size", training_config.get("batch_size", 1)),
        "learning_rate": finetune_config.get("learning_rate", training_config.get("learning_rate", 1e-4)),
        "lora_r": finetune_config.get("lora_r", 16),
        "lora_alpha": finetune_config.get("lora_alpha", 32),
        "lora_dropout": finetune_config.get("lora_dropout", 0.05),
        "target_modules": finetune_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "gpu_count": training_config["gpu_count"],
        "gradient_checkpointing": training_config.get("gradient_checkpointing", True),
        "gradient_accumulation_steps": finetune_config.get("gradient_accumulation_steps", 4),
        "warmup_ratio": finetune_config.get("warmup_ratio", 0.1),
        "max_steps": finetune_config.get("max_steps", 10000),
    }
    
    # Add distributed strategy based on model size
    if training_config["gpu_count"] > 1:
        result["distributed_strategy"] = training_config.get("distributed_strategy", "deepspeed_zero3")
        
        # Add DeepSpeed config for 27B model
        if "deepspeed_config" in training_config:
            result["deepspeed_config"] = training_config["deepspeed_config"]
    else:
        result["distributed_strategy"] = "single_gpu"
    
    return result


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of all available models with their configurations.
    
    Returns:
        List of model configurations
    """
    config = _load_models_config()
    models = []
    
    for name, model_config in config.get("models", {}).items():
        models.append({
            "name": name,
            "hf_model_id": model_config["hf_model_id"],
            "display_name": model_config["display_name"],
            "parameters": model_config["parameters"],
            "training_gpu_count": model_config["training"]["gpu_count"],
        })
    
    return models


def get_model_for_use_case(use_case: str) -> str:
    """
    Get recommended model for a specific use case.
    
    Args:
        use_case: One of "inference", "training", "evaluation"
        
    Returns:
        Recommended model canonical name
    """
    config = _load_models_config()
    defaults = config.get("selection", {}).get("defaults", {})
    
    return defaults.get(use_case, "medgemma_4b_it")


def validate_gpu_requirements(model_name: str, available_gpus: int) -> bool:
    """
    Validate if available GPUs meet model requirements for training.
    
    Args:
        model_name: Model name or alias
        available_gpus: Number of available GPUs
        
    Returns:
        True if requirements are met
        
    Raises:
        ValueError: If GPU requirements are not met
    """
    config = get_training_config(model_name)
    required_gpus = config["gpu_count"]
    
    if available_gpus < required_gpus:
        raise ValueError(
            f"Model {model_name} requires {required_gpus} GPUs for training, "
            f"but only {available_gpus} available"
        )
    
    return True
