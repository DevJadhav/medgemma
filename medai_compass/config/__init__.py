"""
Configuration module for MedAI Compass.

Provides centralized configuration management with support for:
- Environment variables
- Config files (YAML/JSON)
- Dynamic configuration updates
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
]
