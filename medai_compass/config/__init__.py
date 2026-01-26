"""
Configuration module for MedAI Compass.

Provides centralized configuration management with support for:
- Environment variables
- Config files (YAML/JSON)
- Dynamic configuration updates
- Hydra hierarchical configuration (for training)
"""

from medai_compass.config.settings import (
    ConfidenceThresholds,
    EscalationConfig,
    InferenceConfig,
    MedAIConfig,
    PHIDetectionConfig,
    RateLimitConfig,
    RetentionConfig,
    SecurityConfig,
    ConfigLoader,
    get_config,
    get_config_loader,
    get_confidence_threshold,
    get_escalation_keywords,
    get_model_name,
    is_debug,
    is_production,
    reload_config,
)

# Hydra configuration for training (imported lazily to avoid dependency on hydra)
try:
    from medai_compass.config.hydra_config import (
        MedGemmaConfig,
        ModelConfig,
        TrainingConfig,
        ComputeConfig,
        DataConfig,
        TuningConfig,
        load_config as load_hydra_config,
        load_config_with_overrides,
        to_training_args,
        to_lora_config,
        to_deepspeed_config,
        search_space_to_ray_tune,
        validate_config,
    )
    _HYDRA_AVAILABLE = True
except ImportError:
    _HYDRA_AVAILABLE = False

__all__ = [
    # Configuration classes
    "MedAIConfig",
    "ConfidenceThresholds",
    "EscalationConfig",
    "InferenceConfig",
    "PHIDetectionConfig",
    "RateLimitConfig",
    "RetentionConfig",
    "SecurityConfig",
    # Loader
    "ConfigLoader",
    # Functions
    "get_config",
    "get_config_loader",
    "get_confidence_threshold",
    "get_escalation_keywords",
    "get_model_name",
    "is_debug",
    "is_production",
    "reload_config",
    # Hydra configuration (for training)
    "MedGemmaConfig",
    "ModelConfig",
    "TrainingConfig",
    "ComputeConfig",
    "DataConfig",
    "TuningConfig",
    "load_hydra_config",
    "load_config_with_overrides",
    "to_training_args",
    "to_lora_config",
    "to_deepspeed_config",
    "search_space_to_ray_tune",
    "validate_config",
]
