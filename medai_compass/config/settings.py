"""
External configuration management for MedAI Compass.

Provides:
- Environment-based configuration
- Config file support (YAML/JSON)
- Database-backed configuration
- Dynamic configuration updates
- Type-safe configuration access

All hard-coded values should be moved here for centralized management.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for AI decision making."""

    high_confidence: float = 0.90
    medium_confidence: float = 0.85
    low_confidence: float = 0.80
    escalation_threshold: float = 0.70
    auto_approve_threshold: float = 0.95


@dataclass
class EscalationConfig:
    """Configuration for escalation triggers."""

    keywords: List[str] = field(default_factory=lambda: [
        "emergency", "urgent", "critical", "life-threatening",
        "chest pain", "difficulty breathing", "severe bleeding",
        "stroke symptoms", "loss of consciousness", "allergic reaction"
    ])

    confidence_threshold: float = 0.70
    auto_escalate_phi_breach: bool = True
    escalation_timeout_seconds: int = 300


@dataclass
class PHIDetectionConfig:
    """Configuration for PHI detection patterns."""

    ssn_pattern: str = r'\b\d{3}-\d{2}-\d{4}\b'
    mrn_pattern: str = r'\bMRN[:\s]*(\d{6,10})\b'
    phone_pattern: str = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    email_pattern: str = r'\b[\w.-]+@[\w.-]+\.\w+\b'
    dob_pattern: str = r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}\b'

    enable_ner: bool = True
    enable_context_aware: bool = True
    enable_extended_patterns: bool = True


@dataclass
class InferenceConfig:
    """Configuration for model inference."""

    default_model: str = "medgemma-27b"
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout_seconds: int = 90
    retry_attempts: int = 3

    # Backend selection
    preferred_backend: str = "modal"  # modal, local, ray
    fallback_backend: str = "local"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10

    # Per-user limits
    user_requests_per_minute: int = 30
    user_requests_per_hour: int = 500


@dataclass
class RetentionConfig:
    """Configuration for data retention."""

    audit_log_retention_days: int = 2190  # 6 years for HIPAA
    session_retention_hours: int = 24
    cache_ttl_seconds: int = 3600
    temp_file_retention_hours: int = 1


@dataclass
class SecurityConfig:
    """Security-related configuration."""

    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "https://localhost:3000"
    ])


@dataclass
class MedAIConfig:
    """
    Master configuration for MedAI Compass.

    All configurable values should be accessible through this class.
    """

    # Sub-configurations
    confidence: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    phi_detection: PHIDetectionConfig = field(default_factory=PHIDetectionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    log_level: str = "INFO"


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class ConfigLoader:
    """
    Load configuration from multiple sources.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file (YAML/JSON)
    3. Database
    4. Defaults
    """

    def __init__(
        self,
        config_file: Optional[Path] = None,
        env_prefix: str = "MEDAI_"
    ):
        """
        Initialize config loader.

        Args:
            config_file: Path to config file (YAML or JSON)
            env_prefix: Prefix for environment variables
        """
        self.config_file = config_file
        self.env_prefix = env_prefix
        self._config: Optional[MedAIConfig] = None
        self._callbacks: List[Callable[[MedAIConfig], None]] = []
        self._last_loaded: Optional[datetime] = None

    def load(self) -> MedAIConfig:
        """
        Load configuration from all sources.

        Returns:
            Merged MedAIConfig
        """
        # Start with defaults
        config = MedAIConfig()

        # Load from file if specified
        if self.config_file and self.config_file.exists():
            file_config = self._load_from_file(self.config_file)
            config = self._merge_config(config, file_config)

        # Override with environment variables
        config = self._apply_env_overrides(config)

        self._config = config
        self._last_loaded = datetime.now(timezone.utc)

        logger.info(f"Configuration loaded (env={config.environment})")
        return config

    def _load_from_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                return yaml.safe_load(content)
            except ImportError:
                logger.warning("PyYAML not installed, skipping YAML config")
                return {}
        elif path.suffix == ".json":
            return json.loads(content)
        else:
            logger.warning(f"Unknown config file format: {path.suffix}")
            return {}

    def _merge_config(self, base: MedAIConfig, overrides: Dict[str, Any]) -> MedAIConfig:
        """Merge file config into base config."""
        if not overrides:
            return base

        # Merge top-level simple fields
        if "environment" in overrides:
            base.environment = overrides["environment"]
        if "debug" in overrides:
            base.debug = overrides["debug"]
        if "log_level" in overrides:
            base.log_level = overrides["log_level"]

        # Merge confidence thresholds
        if "confidence" in overrides:
            conf = overrides["confidence"]
            if "high_confidence" in conf:
                base.confidence.high_confidence = conf["high_confidence"]
            if "medium_confidence" in conf:
                base.confidence.medium_confidence = conf["medium_confidence"]
            if "low_confidence" in conf:
                base.confidence.low_confidence = conf["low_confidence"]
            if "escalation_threshold" in conf:
                base.confidence.escalation_threshold = conf["escalation_threshold"]

        # Merge escalation config
        if "escalation" in overrides:
            esc = overrides["escalation"]
            if "keywords" in esc:
                base.escalation.keywords = esc["keywords"]
            if "confidence_threshold" in esc:
                base.escalation.confidence_threshold = esc["confidence_threshold"]

        # Merge inference config
        if "inference" in overrides:
            inf = overrides["inference"]
            if "default_model" in inf:
                base.inference.default_model = inf["default_model"]
            if "max_tokens" in inf:
                base.inference.max_tokens = inf["max_tokens"]
            if "temperature" in inf:
                base.inference.temperature = inf["temperature"]

        return base

    def _apply_env_overrides(self, config: MedAIConfig) -> MedAIConfig:
        """Apply environment variable overrides."""
        # Environment
        env_val = os.environ.get(f"{self.env_prefix}ENVIRONMENT")
        if env_val:
            config.environment = env_val

        # Debug
        debug_val = os.environ.get(f"{self.env_prefix}DEBUG")
        if debug_val:
            config.debug = debug_val.lower() in ("true", "1", "yes")

        # Log level
        log_level = os.environ.get(f"{self.env_prefix}LOG_LEVEL")
        if log_level:
            config.log_level = log_level.upper()

        # Model name (commonly used)
        model_name = os.environ.get("MEDGEMMA_MODEL_NAME")
        if model_name:
            config.inference.default_model = model_name

        # Confidence thresholds
        high_conf = os.environ.get(f"{self.env_prefix}CONFIDENCE_HIGH")
        if high_conf:
            config.confidence.high_confidence = float(high_conf)

        medium_conf = os.environ.get(f"{self.env_prefix}CONFIDENCE_MEDIUM")
        if medium_conf:
            config.confidence.medium_confidence = float(medium_conf)

        low_conf = os.environ.get(f"{self.env_prefix}CONFIDENCE_LOW")
        if low_conf:
            config.confidence.low_confidence = float(low_conf)

        # Inference timeout
        timeout = os.environ.get(f"{self.env_prefix}INFERENCE_TIMEOUT")
        if timeout:
            config.inference.timeout_seconds = int(timeout)

        # Rate limits
        rpm = os.environ.get(f"{self.env_prefix}RATE_LIMIT_RPM")
        if rpm:
            config.rate_limit.requests_per_minute = int(rpm)

        # CORS origins
        cors = os.environ.get(f"{self.env_prefix}CORS_ORIGINS")
        if cors:
            config.security.cors_origins = [o.strip() for o in cors.split(",")]

        return config

    def get(self) -> MedAIConfig:
        """
        Get current configuration.

        Loads if not already loaded.
        """
        if self._config is None:
            return self.load()
        return self._config

    def reload(self) -> MedAIConfig:
        """Force reload configuration."""
        config = self.load()

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(f"Config reload callback failed: {e}")

        return config

    def on_reload(self, callback: Callable[[MedAIConfig], None]) -> None:
        """Register a callback for configuration reloads."""
        self._callbacks.append(callback)


# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Default config file path
_default_config_path = Path(__file__).parent.parent.parent / "config" / "medai_config.yaml"

# Global config loader instance
_config_loader = ConfigLoader(
    config_file=_default_config_path if _default_config_path.exists() else None
)


def get_config() -> MedAIConfig:
    """Get the global configuration."""
    return _config_loader.get()


def reload_config() -> MedAIConfig:
    """Reload the global configuration."""
    return _config_loader.reload()


def get_config_loader() -> ConfigLoader:
    """Get the global config loader for advanced operations."""
    return _config_loader


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_confidence_threshold(level: str = "high") -> float:
    """
    Get confidence threshold by level.

    Args:
        level: "high", "medium", "low", or "escalation"

    Returns:
        Threshold value
    """
    config = get_config()
    thresholds = {
        "high": config.confidence.high_confidence,
        "medium": config.confidence.medium_confidence,
        "low": config.confidence.low_confidence,
        "escalation": config.confidence.escalation_threshold,
    }
    return thresholds.get(level, config.confidence.medium_confidence)


def get_model_name() -> str:
    """Get the configured model name."""
    return get_config().inference.default_model


def get_escalation_keywords() -> List[str]:
    """Get the list of escalation trigger keywords."""
    return get_config().escalation.keywords


def is_production() -> bool:
    """Check if running in production environment."""
    return get_config().environment == "production"


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return get_config().debug
