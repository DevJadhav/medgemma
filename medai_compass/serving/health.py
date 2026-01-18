"""
Health check functionality for serving deployments.

Provides health monitoring for Ray Serve deployments including:
- Model loading status
- GPU availability
- Memory usage
- Latency monitoring
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check.
    
    Attributes:
        status: Overall health status.
        model_loaded: Whether the model is loaded.
        gpu_available: Whether GPU is available.
        memory_usage_percent: Memory usage percentage.
        latency_ms: Last inference latency in milliseconds.
        details: Additional details about the health check.
    """
    
    status: HealthStatus
    model_loaded: bool
    gpu_available: bool
    memory_usage_percent: float
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "model_loaded": self.model_loaded,
            "gpu_available": self.gpu_available,
            "memory_usage_percent": self.memory_usage_percent,
            "latency_ms": self.latency_ms,
            "details": self.details or {},
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if status is healthy."""
        return self.status == HealthStatus.HEALTHY


def check_gpu_available() -> bool:
    """Check if GPU is available.
    
    Returns:
        True if CUDA GPU is available, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not installed, GPU check unavailable")
        return False
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return False


def check_model_loaded(model: Any = None) -> bool:
    """Check if model is loaded and functional.
    
    Args:
        model: The model instance to check.
        
    Returns:
        True if model is loaded, False otherwise.
    """
    if model is None:
        return False
    
    # Check if model has required methods
    has_generate = hasattr(model, "generate")
    has_forward = hasattr(model, "forward") or hasattr(model, "__call__")
    
    return has_generate or has_forward


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "max_gb": 0.0,
                "utilization_percent": 0.0,
            }
        
        allocated = torch.cuda.memory_allocated()
        max_allocated = torch.cuda.max_memory_allocated()
        
        # Get total memory
        total = torch.cuda.get_device_properties(0).total_memory
        
        allocated_gb = allocated / (1024 ** 3)
        max_gb = max_allocated / (1024 ** 3)
        utilization = (allocated / total) * 100 if total > 0 else 0.0
        
        return {
            "allocated_gb": round(allocated_gb, 2),
            "max_gb": round(max_gb, 2),
            "utilization_percent": round(utilization, 2),
        }
    except ImportError:
        return {
            "allocated_gb": 0.0,
            "max_gb": 0.0,
            "utilization_percent": 0.0,
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory usage: {e}")
        return {
            "allocated_gb": 0.0,
            "max_gb": 0.0,
            "utilization_percent": 0.0,
        }


def get_system_memory_usage() -> Dict[str, float]:
    """Get system memory usage statistics.
    
    Returns:
        Dictionary with system memory statistics.
    """
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "used_gb": round(memory.used / (1024 ** 3), 2),
            "percent": memory.percent,
        }
    except ImportError:
        logger.warning("psutil not installed, system memory check unavailable")
        return {
            "total_gb": 0.0,
            "available_gb": 0.0,
            "used_gb": 0.0,
            "percent": 0.0,
        }


class HealthChecker:
    """Health checker for serving deployments.
    
    Monitors deployment health including model status, GPU availability,
    memory usage, and inference latency.
    """
    
    def __init__(
        self,
        model: Any = None,
        latency_threshold_ms: float = 500.0,
        memory_threshold_percent: float = 90.0,
    ):
        """Initialize health checker.
        
        Args:
            model: The model instance to monitor.
            latency_threshold_ms: Latency threshold for degraded status.
            memory_threshold_percent: Memory threshold for degraded status.
        """
        self.model = model
        self.latency_threshold_ms = latency_threshold_ms
        self.memory_threshold_percent = memory_threshold_percent
        self._last_latency_ms: Optional[float] = None
    
    def set_model(self, model: Any) -> None:
        """Set the model to monitor.
        
        Args:
            model: The model instance.
        """
        self.model = model
    
    def record_latency(self, latency_ms: float) -> None:
        """Record inference latency.
        
        Args:
            latency_ms: Latency in milliseconds.
        """
        self._last_latency_ms = latency_ms
    
    def check(self) -> HealthCheckResult:
        """Perform health check.
        
        Returns:
            HealthCheckResult with current health status.
        """
        model_loaded = check_model_loaded(self.model)
        gpu_available = check_gpu_available()
        gpu_memory = get_gpu_memory_usage()
        system_memory = get_system_memory_usage()
        
        memory_usage = max(
            gpu_memory.get("utilization_percent", 0),
            system_memory.get("percent", 0)
        )
        
        # Determine health status
        if not model_loaded:
            status = HealthStatus.UNHEALTHY
        elif not gpu_available:
            status = HealthStatus.DEGRADED
        elif memory_usage > self.memory_threshold_percent:
            status = HealthStatus.DEGRADED
        elif self._last_latency_ms and self._last_latency_ms > self.latency_threshold_ms:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return HealthCheckResult(
            status=status,
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage_percent=memory_usage,
            latency_ms=self._last_latency_ms,
            details={
                "gpu_memory": gpu_memory,
                "system_memory": system_memory,
            }
        )
    
    async def check_async(self) -> HealthCheckResult:
        """Perform health check asynchronously.
        
        Returns:
            HealthCheckResult with current health status.
        """
        return self.check()


class LivenessProbe:
    """Liveness probe for deployment health.
    
    Simple check that the service is running.
    """
    
    def __init__(self):
        """Initialize liveness probe."""
        self._started_at = time.time()
    
    def check(self) -> Dict[str, Any]:
        """Perform liveness check.
        
        Returns:
            Dictionary with liveness status.
        """
        return {
            "alive": True,
            "uptime_seconds": time.time() - self._started_at,
        }


class ReadinessProbe:
    """Readiness probe for deployment health.
    
    Checks that the service is ready to receive traffic.
    """
    
    def __init__(self, health_checker: HealthChecker):
        """Initialize readiness probe.
        
        Args:
            health_checker: The health checker instance.
        """
        self.health_checker = health_checker
    
    def check(self) -> Dict[str, Any]:
        """Perform readiness check.
        
        Returns:
            Dictionary with readiness status.
        """
        health = self.health_checker.check()
        
        return {
            "ready": health.model_loaded,
            "status": health.status.value,
            "model_loaded": health.model_loaded,
        }
