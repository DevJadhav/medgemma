"""
MedAI Compass Serving Module.

This module provides model serving infrastructure including:
- Ray Serve deployments
- Health checks
- Request/response models
- Metrics collection
- Canary deployments
- A/B testing
- Rollback mechanisms
"""

from medai_compass.serving.ray_serve_app import (
    DeploymentConfig,
    MedGemmaDeployment,
    create_deployment,
    serve_application,
    load_model_from_registry,
    load_model,
)
from medai_compass.serving.health import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    check_gpu_available,
    check_model_loaded,
    get_gpu_memory_usage,
)
from medai_compass.serving.models import (
    GenerateRequest,
    GenerateResponse,
)
from medai_compass.serving.metrics import (
    ServingMetricsCollector,
)
from medai_compass.serving.canary import (
    CanaryManager,
    CanaryStatus,
)
from medai_compass.serving.ab_testing import (
    ABTest,
    ABTestConfig,
    ABTestManager,
)
from medai_compass.serving.rollback import (
    RollbackManager,
    RollbackError,
    RollbackTrigger,
    DeploymentRecord,
)

__all__ = [
    # Ray Serve
    "DeploymentConfig",
    "MedGemmaDeployment",
    "create_deployment",
    "serve_application",
    "load_model_from_registry",
    "load_model",
    # Health
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "check_gpu_available",
    "check_model_loaded",
    "get_gpu_memory_usage",
    # Models
    "GenerateRequest",
    "GenerateResponse",
    # Metrics
    "ServingMetricsCollector",
    # Canary
    "CanaryManager",
    "CanaryStatus",
    # A/B Testing
    "ABTest",
    "ABTestConfig",
    "ABTestManager",
    # Rollback
    "RollbackManager",
    "RollbackError",
    "RollbackTrigger",
    "DeploymentRecord",
]
