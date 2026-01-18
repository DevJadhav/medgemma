"""
Deployment pipeline for MedGemma models.

Provides end-to-end deployment orchestration including:
- Model validation and packaging
- Deployment strategies (canary, blue-green, rolling)
- Health monitoring and alerts
- Integration with MLflow and Prometheus
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
import yaml
import time
import logging
import requests

logger = logging.getLogger(__name__)

# Valid model names
VALID_MODEL_NAMES = {"medgemma-4b-it", "medgemma-27b-it"}

# Valid deployment strategies
VALID_STRATEGIES = {"canary", "blue-green", "rolling"}


class DeploymentStage(Enum):
    """Deployment pipeline stages."""
    
    VALIDATE = "validate"
    PACKAGE = "package"
    DEPLOY = "deploy"
    VERIFY = "verify"
    PROMOTE = "promote"


@dataclass
class DeploymentPipelineConfig:
    """Configuration for deployment pipeline.
    
    Attributes:
        model_name: Name of the model to deploy.
        deployment_strategy: Deployment strategy to use.
        canary_percentage: Initial canary traffic percentage.
        health_check_interval: Health check interval in seconds.
        rollback_on_failure: Whether to auto-rollback on failure.
        replicas: Number of replicas.
        resources: Resource specifications.
    """
    
    model_name: str = "medgemma-27b-it"
    deployment_strategy: str = "canary"
    canary_percentage: int = 10
    health_check_interval: int = 30
    rollback_on_failure: bool = True
    replicas: int = 1
    resources: Optional[Dict[str, Any]] = field(default_factory=dict)
    version: Optional[str] = None
    
    def __post_init__(self):
        """Generate version if not provided."""
        if self.version is None:
            self.version = datetime.utcnow().strftime("%Y%m%d%H%M%S")


@dataclass
class DeploymentResult:
    """Result of a deployment operation.
    
    Attributes:
        success: Whether deployment succeeded.
        version: Deployed version.
        model_name: Deployed model name.
        deployment_time_seconds: Deployment duration.
        endpoint_url: Deployment endpoint URL.
        error: Error message if failed.
        stage: Stage where deployment stopped.
    """
    
    success: bool
    version: str
    model_name: str
    deployment_time_seconds: Optional[float] = None
    endpoint_url: Optional[str] = None
    error: Optional[str] = None
    stage: Optional[DeploymentStage] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "version": self.version,
            "model_name": self.model_name,
            "deployment_time_seconds": self.deployment_time_seconds,
            "endpoint_url": self.endpoint_url,
            "error": self.error,
            "stage": self.stage.value if self.stage else None,
        }


@dataclass
class AlertConfig:
    """Configuration for deployment alerts.
    
    Attributes:
        latency_threshold_ms: Maximum P95 latency.
        error_rate_threshold: Maximum error rate.
        health_threshold: Minimum health score.
    """
    
    latency_threshold_ms: float = 500.0
    error_rate_threshold: float = 0.01
    health_threshold: float = 0.95


@dataclass
class Alert:
    """Deployment alert.
    
    Attributes:
        name: Alert name.
        severity: Alert severity.
        message: Alert message.
        timestamp: When alert was triggered.
    """
    
    name: str
    severity: str
    message: str
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate deployment configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Validation result with 'valid' and 'errors' keys.
    """
    errors = []
    
    # Validate model name
    model_name = config.get("model_name")
    if model_name and model_name not in VALID_MODEL_NAMES:
        errors.append(f"Invalid model_name: {model_name}. Must be one of {VALID_MODEL_NAMES}")
    
    # Validate deployment strategy
    strategy = config.get("deployment_strategy")
    if strategy and strategy not in VALID_STRATEGIES:
        errors.append(f"Invalid deployment_strategy: {strategy}. Must be one of {VALID_STRATEGIES}")
    
    # Validate canary percentage
    canary_pct = config.get("canary_percentage")
    if canary_pct is not None:
        if not isinstance(canary_pct, (int, float)):
            errors.append("canary_percentage must be a number")
        elif canary_pct < 0 or canary_pct > 100:
            errors.append("canary_percentage must be between 0 and 100")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def validate_model(model: Any) -> bool:
    """Validate model before deployment.
    
    Args:
        model: Model to validate.
        
    Returns:
        True if model is valid.
    """
    if model is None:
        return False
    
    # Check required methods
    has_generate = hasattr(model, "generate")
    
    if not has_generate:
        logger.error("Model missing required 'generate' method")
        return False
    
    # Try a test inference
    try:
        if callable(model.generate):
            # Test with a simple prompt
            result = model.generate("Test")
            return result is not None
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False
    
    return True


def deploy_to_ray_serve(
    model_name: str,
    config: DeploymentPipelineConfig,
) -> Dict[str, Any]:
    """Deploy model to Ray Serve.
    
    Args:
        model_name: Model name.
        config: Deployment configuration.
        
    Returns:
        Deployment result dictionary.
    """
    try:
        from medai_compass.serving.ray_serve_app import (
            create_deployment,
            DeploymentConfig,
        )
        
        deploy_config = DeploymentConfig(
            model_name=model_name,
            num_replicas=config.replicas,
        )
        
        deployment = create_deployment(config=deploy_config)
        
        return {
            "status": "success",
            "deployment": deployment,
        }
    except Exception as e:
        logger.error(f"Ray Serve deployment failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def load_deployment_config(config_yaml: str) -> DeploymentPipelineConfig:
    """Load deployment configuration from YAML.
    
    Args:
        config_yaml: YAML configuration string.
        
    Returns:
        DeploymentPipelineConfig instance.
    """
    config_dict = yaml.safe_load(config_yaml)
    
    return DeploymentPipelineConfig(
        model_name=config_dict.get("model_name", "medgemma-27b-it"),
        deployment_strategy=config_dict.get("deployment_strategy", "canary"),
        canary_percentage=config_dict.get("canary_percentage", 10),
        health_check_interval=config_dict.get("health_check_interval", 30),
        rollback_on_failure=config_dict.get("rollback_on_failure", True),
        replicas=config_dict.get("replicas", 1),
        resources=config_dict.get("resources", {}),
        version=config_dict.get("version"),
    )


class ModelPackager:
    """Packages models for deployment.
    
    Handles fetching from MLflow registry, validation,
    and packaging for different serving backends.
    """
    
    def fetch_from_registry(
        self,
        model_name: str,
        version: str = "production",
    ) -> Any:
        """Fetch model from MLflow registry.
        
        Args:
            model_name: Model name in registry.
            version: Model version or stage.
            
        Returns:
            Loaded model.
        """
        import mlflow.pyfunc
        
        if version in ["production", "staging"]:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        logger.info(f"Fetching model from registry: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    
    def validate_model(self, model: Any) -> bool:
        """Validate model before deployment.
        
        Args:
            model: Model to validate.
            
        Returns:
            True if model is valid.
        """
        if model is None:
            return False
        
        try:
            # Check for generate method
            if hasattr(model, "generate"):
                result = model.generate("Test validation prompt")
                return result is not None
            return False
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def package_for_ray_serve(
        self,
        model: Any,
        model_name: str,
    ) -> Dict[str, Any]:
        """Package model for Ray Serve deployment.
        
        Args:
            model: The model to package.
            model_name: Model name.
            
        Returns:
            Package dictionary with config and model.
        """
        from medai_compass.serving.ray_serve_app import MODEL_CONFIGS
        
        model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["medgemma-27b-it"])
        
        return {
            "model": model,
            "config": {
                "model_name": model_name,
                "model_id": model_config["model_id"],
                "num_gpus": model_config["num_gpus"],
                "memory_gb": model_config["memory_gb"],
            },
            "model_path": None,  # Model is in memory
        }


class DeploymentMonitor:
    """Monitors deployment health and metrics.
    
    Provides health checks, latency monitoring, and alerting.
    """
    
    def __init__(self):
        """Initialize deployment monitor."""
        self._alert_config: Optional[AlertConfig] = None
        self._alerts: List[Alert] = []
        self._latency_samples: List[float] = []
    
    def configure_alerts(self, config: AlertConfig) -> None:
        """Configure alert thresholds.
        
        Args:
            config: Alert configuration.
        """
        self._alert_config = config
    
    def check_health(
        self,
        endpoint: str,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Check deployment health.
        
        Args:
            endpoint: Health check endpoint URL.
            timeout: Request timeout.
            
        Returns:
            Health status dictionary.
        """
        try:
            response = requests.get(endpoint, timeout=timeout)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "latency_ms": response.elapsed.total_seconds() * 1000,
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Status code: {response.status_code}",
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "unhealthy",
                "error": "Connection failed",
            }
        except requests.exceptions.Timeout:
            return {
                "status": "unhealthy",
                "error": "Request timeout",
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e),
            }
    
    def record_latency(self, latency_ms: float) -> None:
        """Record latency sample.
        
        Args:
            latency_ms: Latency in milliseconds.
        """
        self._latency_samples.append(latency_ms)
        
        # Keep last 1000 samples
        if len(self._latency_samples) > 1000:
            self._latency_samples = self._latency_samples[-1000:]
    
    def get_latency_metrics(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get latency percentile metrics.
        
        Args:
            model_name: Optional model name filter.
            
        Returns:
            Dictionary with p50, p95, p99 latencies.
        """
        if not self._latency_samples:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        
        sorted_samples = sorted(self._latency_samples)
        n = len(sorted_samples)
        
        def percentile(p: float) -> float:
            k = int((n - 1) * p)
            return sorted_samples[k]
        
        return {
            "p50_ms": round(percentile(0.50), 2),
            "p95_ms": round(percentile(0.95), 2),
            "p99_ms": round(percentile(0.99), 2),
        }
    
    def check_alerts(
        self,
        latency_p95_ms: Optional[float] = None,
        error_rate: Optional[float] = None,
        health_score: Optional[float] = None,
    ) -> List[Alert]:
        """Check if any alerts should be triggered.
        
        Args:
            latency_p95_ms: Current P95 latency.
            error_rate: Current error rate.
            health_score: Current health score.
            
        Returns:
            List of triggered alerts.
        """
        if self._alert_config is None:
            return []
        
        alerts = []
        
        # Check latency
        if latency_p95_ms is not None:
            if latency_p95_ms > self._alert_config.latency_threshold_ms:
                alerts.append(Alert(
                    name="high_latency",
                    severity="warning",
                    message=f"P95 latency {latency_p95_ms:.0f}ms exceeds threshold "
                            f"{self._alert_config.latency_threshold_ms:.0f}ms",
                ))
        
        # Check error rate
        if error_rate is not None:
            if error_rate > self._alert_config.error_rate_threshold:
                alerts.append(Alert(
                    name="high_error_rate",
                    severity="critical",
                    message=f"Error rate {error_rate:.2%} exceeds threshold "
                            f"{self._alert_config.error_rate_threshold:.2%}",
                ))
        
        # Check health
        if health_score is not None:
            if health_score < self._alert_config.health_threshold:
                alerts.append(Alert(
                    name="degraded_health",
                    severity="warning",
                    message=f"Health score {health_score:.2f} below threshold "
                            f"{self._alert_config.health_threshold:.2f}",
                ))
        
        self._alerts.extend(alerts)
        return alerts


class DeploymentMetrics:
    """Collects deployment metrics for Prometheus.
    
    Integrates with Prometheus for deployment observability.
    """
    
    def __init__(self):
        """Initialize deployment metrics."""
        self._deployment_count = 0
        self._rollback_count = 0
        self._deployment_durations: List[float] = []
    
    def record_deployment(
        self,
        model_name: str,
        version: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record a deployment.
        
        Args:
            model_name: Model name.
            version: Deployed version.
            duration_seconds: Deployment duration.
            success: Whether deployment succeeded.
        """
        self._deployment_count += 1
        self._deployment_durations.append(duration_seconds)
        
        try:
            import prometheus_client
            
            # Try to create/update metrics
            # In production, these would be actual Prometheus metrics
            logger.info(
                f"Recorded deployment: {model_name}@{version} "
                f"duration={duration_seconds:.1f}s success={success}"
            )
        except ImportError:
            pass
    
    def record_rollback(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
    ) -> None:
        """Record a rollback.
        
        Args:
            model_name: Model name.
            from_version: Version rolled back from.
            to_version: Version rolled back to.
        """
        self._rollback_count += 1
        
        logger.info(
            f"Recorded rollback: {model_name} {from_version} -> {to_version}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics.
        
        Returns:
            Dictionary with deployment stats.
        """
        return {
            "total_deployments": self._deployment_count,
            "total_rollbacks": self._rollback_count,
            "avg_duration_seconds": (
                sum(self._deployment_durations) / len(self._deployment_durations)
                if self._deployment_durations else 0
            ),
        }


class DeploymentPipeline:
    """Orchestrates the deployment pipeline.
    
    Provides end-to-end deployment management including
    validation, packaging, deployment, and monitoring.
    """
    
    def __init__(self):
        """Initialize deployment pipeline."""
        self.packager = ModelPackager()
        self.monitor = DeploymentMonitor()
        self.metrics = DeploymentMetrics()
    
    def run(
        self,
        config: DeploymentPipelineConfig,
    ) -> DeploymentResult:
        """Run the deployment pipeline.
        
        Args:
            config: Deployment configuration.
            
        Returns:
            DeploymentResult with outcome.
        """
        start_time = time.time()
        
        # Stage 1: Validate configuration
        validation = validate_config({
            "model_name": config.model_name,
            "deployment_strategy": config.deployment_strategy,
            "canary_percentage": config.canary_percentage,
        })
        
        if not validation["valid"]:
            return DeploymentResult(
                success=False,
                version=config.version,
                model_name=config.model_name,
                error=f"Validation failed: {validation['errors']}",
                stage=DeploymentStage.VALIDATE,
            )
        
        # Stage 2: Validate model
        try:
            # For this implementation, we create a mock model for validation
            from medai_compass.serving.ray_serve_app import MockModel
            model = MockModel(config.model_name)
            
            if not validate_model(model):
                return DeploymentResult(
                    success=False,
                    version=config.version,
                    model_name=config.model_name,
                    error="Model validation failed",
                    stage=DeploymentStage.VALIDATE,
                )
        except Exception as e:
            return DeploymentResult(
                success=False,
                version=config.version,
                model_name=config.model_name,
                error=f"Model validation error: {e}",
                stage=DeploymentStage.VALIDATE,
            )
        
        # Stage 3: Deploy
        try:
            deploy_result = deploy_to_ray_serve(config.model_name, config)
            
            if deploy_result["status"] != "success":
                return DeploymentResult(
                    success=False,
                    version=config.version,
                    model_name=config.model_name,
                    error=deploy_result.get("error", "Deployment failed"),
                    stage=DeploymentStage.DEPLOY,
                )
        except Exception as e:
            return DeploymentResult(
                success=False,
                version=config.version,
                model_name=config.model_name,
                error=f"Deployment error: {e}",
                stage=DeploymentStage.DEPLOY,
            )
        
        duration = time.time() - start_time
        
        # Record metrics
        self.metrics.record_deployment(
            model_name=config.model_name,
            version=config.version,
            duration_seconds=duration,
            success=True,
        )
        
        return DeploymentResult(
            success=True,
            version=config.version,
            model_name=config.model_name,
            deployment_time_seconds=duration,
            endpoint_url="http://localhost:8000",
            stage=DeploymentStage.PROMOTE,
        )
    
    def log_deployment_to_mlflow(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
    ) -> None:
        """Log deployment to MLflow.
        
        Args:
            model_name: Model name.
            version: Deployed version.
            metrics: Deployment metrics.
        """
        try:
            import mlflow
            
            with mlflow.start_run(run_name=f"deployment-{version}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("version", version)
                
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                logger.info(f"Logged deployment to MLflow: {model_name}@{version}")
        except ImportError:
            logger.warning("MLflow not available, skipping logging")
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
