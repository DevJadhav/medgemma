"""
Canary deployment management.

Provides canary deployment functionality for gradual rollout
of new model versions with traffic splitting.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)


class CanaryState(Enum):
    """Canary deployment state."""
    
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CanaryStatus:
    """Status of a canary deployment.
    
    Attributes:
        current_version: The stable/current version.
        canary_version: The canary/new version.
        canary_percentage: Percentage of traffic to canary.
        is_active: Whether canary is active.
        state: Current canary state.
        started_at: When canary was started.
        metrics: Collected metrics during canary.
    """
    
    current_version: str
    canary_version: str
    canary_percentage: int
    is_active: bool
    state: CanaryState = CanaryState.INACTIVE
    started_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_version": self.current_version,
            "canary_version": self.canary_version,
            "canary_percentage": self.canary_percentage,
            "is_active": self.is_active,
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "metrics": self.metrics or {},
        }


@dataclass
class CanaryConfig:
    """Configuration for canary deployment.
    
    Attributes:
        initial_percentage: Initial canary traffic percentage.
        increment_percentage: Traffic increment per step.
        target_percentage: Final target percentage before promotion.
        health_threshold: Minimum health score for promotion.
        latency_threshold_ms: Maximum latency for promotion.
        min_requests: Minimum requests before evaluation.
        auto_promote: Whether to auto-promote on success.
    """
    
    initial_percentage: int = 10
    increment_percentage: int = 20
    target_percentage: int = 100
    health_threshold: float = 0.95
    latency_threshold_ms: float = 500.0
    min_requests: int = 100
    auto_promote: bool = False


class CanaryManager:
    """Manager for canary deployments.
    
    Handles gradual rollout of new model versions with
    traffic splitting and automatic rollback on failure.
    """
    
    def __init__(
        self,
        current_version: str,
        canary_version: str,
        config: Optional[CanaryConfig] = None,
    ):
        """Initialize canary manager.
        
        Args:
            current_version: The stable/current version.
            canary_version: The canary/new version to test.
            config: Optional canary configuration.
        """
        self.current_version = current_version
        self.canary_version = canary_version
        self.config = config or CanaryConfig()
        
        self._canary_percentage: int = 0
        self._is_active: bool = False
        self._state: CanaryState = CanaryState.INACTIVE
        self._started_at: Optional[datetime] = None
        
        # Metrics tracking
        self._current_metrics: Dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "latency_sum": 0.0,
        }
        self._canary_metrics: Dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "latency_sum": 0.0,
        }
    
    @property
    def canary_percentage(self) -> int:
        """Get current canary traffic percentage."""
        return self._canary_percentage
    
    @property
    def is_active(self) -> bool:
        """Check if canary is active."""
        return self._is_active
    
    def start_rollout(self, percentage: Optional[int] = None) -> None:
        """Start canary rollout.
        
        Args:
            percentage: Initial traffic percentage (default from config).
        """
        if percentage is None:
            percentage = self.config.initial_percentage
        
        self._canary_percentage = percentage
        self._is_active = True
        self._state = CanaryState.ACTIVE
        self._started_at = datetime.utcnow()
        
        logger.info(
            f"Started canary rollout: {self.canary_version} at {percentage}% traffic"
        )
    
    def increase_traffic(self, percentage: int) -> None:
        """Increase canary traffic percentage.
        
        Args:
            percentage: New traffic percentage.
        """
        if not self._is_active:
            raise ValueError("Cannot increase traffic: canary not active")
        
        if percentage < 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100")
        
        old_percentage = self._canary_percentage
        self._canary_percentage = percentage
        
        logger.info(
            f"Canary traffic increased: {old_percentage}% -> {percentage}%"
        )
    
    def increment_traffic(self) -> int:
        """Increment canary traffic by configured amount.
        
        Returns:
            New traffic percentage.
        """
        new_percentage = min(
            self._canary_percentage + self.config.increment_percentage,
            self.config.target_percentage,
        )
        self.increase_traffic(new_percentage)
        return new_percentage
    
    def complete_rollout(self) -> None:
        """Complete canary rollout and promote canary to current."""
        if not self._is_active:
            raise ValueError("Cannot complete: canary not active")
        
        logger.info(
            f"Completing canary rollout: {self.canary_version} promoted to current"
        )
        
        self.current_version = self.canary_version
        self._canary_percentage = 100
        self._is_active = False
        self._state = CanaryState.COMPLETED
    
    def rollback(self, reason: Optional[str] = None) -> None:
        """Rollback canary deployment.
        
        Args:
            reason: Reason for rollback.
        """
        logger.warning(
            f"Rolling back canary {self.canary_version}. "
            f"Reason: {reason or 'Not specified'}"
        )
        
        self._canary_percentage = 0
        self._is_active = False
        self._state = CanaryState.ROLLED_BACK
    
    def should_route_to_canary(self) -> bool:
        """Determine if request should route to canary.
        
        Returns:
            True if request should go to canary.
        """
        if not self._is_active or self._canary_percentage == 0:
            return False
        
        return random.randint(1, 100) <= self._canary_percentage
    
    def route_request(self) -> str:
        """Route request to appropriate version.
        
        Returns:
            Version to route to.
        """
        if self.should_route_to_canary():
            return self.canary_version
        return self.current_version
    
    def record_request(
        self,
        version: str,
        latency_ms: float,
        is_error: bool = False,
    ) -> None:
        """Record request metrics.
        
        Args:
            version: Version that handled request.
            latency_ms: Request latency.
            is_error: Whether request resulted in error.
        """
        if version == self.canary_version:
            metrics = self._canary_metrics
        else:
            metrics = self._current_metrics
        
        metrics["requests"] += 1
        metrics["latency_sum"] += latency_ms
        if is_error:
            metrics["errors"] += 1
    
    def get_status(self) -> CanaryStatus:
        """Get current canary status.
        
        Returns:
            CanaryStatus object.
        """
        return CanaryStatus(
            current_version=self.current_version,
            canary_version=self.canary_version,
            canary_percentage=self._canary_percentage,
            is_active=self._is_active,
            state=self._state,
            started_at=self._started_at,
            metrics={
                "current": self._current_metrics.copy(),
                "canary": self._canary_metrics.copy(),
            },
        )
    
    def get_canary_health(self) -> Dict[str, Any]:
        """Get canary health metrics.
        
        Returns:
            Dictionary with health metrics.
        """
        canary_requests = self._canary_metrics["requests"]
        canary_errors = self._canary_metrics["errors"]
        canary_latency_sum = self._canary_metrics["latency_sum"]
        
        if canary_requests == 0:
            return {
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "total_requests": 0,
            }
        
        return {
            "error_rate": canary_errors / canary_requests,
            "avg_latency_ms": canary_latency_sum / canary_requests,
            "total_requests": canary_requests,
        }
    
    def should_promote(self) -> bool:
        """Check if canary should be promoted.
        
        Returns:
            True if canary should be promoted.
        """
        health = self.get_canary_health()
        
        # Check minimum requests
        if health["total_requests"] < self.config.min_requests:
            return False
        
        # Check error rate
        error_rate = health["error_rate"]
        if error_rate > (1 - self.config.health_threshold):
            return False
        
        # Check latency
        avg_latency = health["avg_latency_ms"]
        if avg_latency > self.config.latency_threshold_ms:
            return False
        
        return True
    
    def should_rollback(self) -> bool:
        """Check if canary should be rolled back.
        
        Returns:
            True if canary should be rolled back.
        """
        health = self.get_canary_health()
        
        # Only evaluate after minimum requests
        if health["total_requests"] < self.config.min_requests:
            return False
        
        # Check error rate (more than 2x threshold triggers rollback)
        error_rate = health["error_rate"]
        if error_rate > (1 - self.config.health_threshold) * 2:
            return True
        
        # Check latency (more than 2x threshold triggers rollback)
        avg_latency = health["avg_latency_ms"]
        if avg_latency > self.config.latency_threshold_ms * 2:
            return True
        
        return False
    
    def auto_manage(self) -> Optional[str]:
        """Automatically manage canary based on metrics.
        
        Returns:
            Action taken ("promote", "rollback", "increment", or None).
        """
        if not self._is_active:
            return None
        
        # Check for rollback
        if self.should_rollback():
            self.rollback(reason="Health threshold breached")
            return "rollback"
        
        # Check for promotion
        if self._canary_percentage >= self.config.target_percentage:
            if self.should_promote():
                self.complete_rollout()
                return "promote"
            return None
        
        # Check for traffic increment
        if self.should_promote() and self.config.auto_promote:
            self.increment_traffic()
            return "increment"
        
        return None


class BlueGreenManager:
    """Manager for blue-green deployments.
    
    Provides instant traffic switching between two environments.
    """
    
    def __init__(
        self,
        blue_version: str,
        green_version: Optional[str] = None,
    ):
        """Initialize blue-green manager.
        
        Args:
            blue_version: The blue (current) version.
            green_version: The green (new) version.
        """
        self.blue_version = blue_version
        self.green_version = green_version
        self._active_environment = "blue"
    
    @property
    def active_version(self) -> str:
        """Get the currently active version."""
        if self._active_environment == "blue":
            return self.blue_version
        return self.green_version or self.blue_version
    
    def deploy_to_green(self, version: str) -> None:
        """Deploy a new version to green environment.
        
        Args:
            version: Version to deploy.
        """
        self.green_version = version
        logger.info(f"Deployed {version} to green environment")
    
    def switch_to_green(self) -> None:
        """Switch traffic to green environment."""
        if self.green_version is None:
            raise ValueError("No green version deployed")
        
        self._active_environment = "green"
        logger.info(f"Switched traffic to green: {self.green_version}")
    
    def switch_to_blue(self) -> None:
        """Switch traffic to blue environment."""
        self._active_environment = "blue"
        logger.info(f"Switched traffic to blue: {self.blue_version}")
    
    def promote_green_to_blue(self) -> None:
        """Promote green to become the new blue."""
        if self.green_version is None:
            raise ValueError("No green version to promote")
        
        self.blue_version = self.green_version
        self.green_version = None
        self._active_environment = "blue"
        
        logger.info(f"Promoted green to blue: {self.blue_version}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get blue-green deployment status.
        
        Returns:
            Dictionary with status information.
        """
        return {
            "blue_version": self.blue_version,
            "green_version": self.green_version,
            "active_environment": self._active_environment,
            "active_version": self.active_version,
        }
