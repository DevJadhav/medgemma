"""
Rollback mechanism for model deployments.

Provides rollback functionality for safe deployment management,
including version tracking and automatic rollback triggers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RollbackError(Exception):
    """Error during rollback operation."""
    pass


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class DeploymentRecord:
    """Record of a deployment.
    
    Attributes:
        version: Deployment version.
        model_name: Name of the deployed model.
        config: Deployment configuration.
        status: Current deployment status.
        deployed_at: Timestamp of deployment.
        metrics: Deployment metrics.
    """
    
    version: str
    model_name: str
    config: Optional[Dict[str, Any]] = field(default_factory=dict)
    status: DeploymentStatus = DeploymentStatus.ACTIVE
    deployed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.deployed_at is None:
            self.deployed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "model_name": self.model_name,
            "config": self.config,
            "status": self.status.value,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "metrics": self.metrics,
        }


@dataclass
class RollbackResult:
    """Result of a rollback operation.
    
    Attributes:
        success: Whether rollback succeeded.
        rolled_back_to: Version rolled back to.
        rolled_back_from: Version rolled back from.
        error: Error message if failed.
        duration_seconds: Rollback duration.
    """
    
    success: bool
    rolled_back_to: str
    rolled_back_from: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "rolled_back_to": self.rolled_back_to,
            "rolled_back_from": self.rolled_back_from,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


def deploy_version(
    version: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Deploy a specific version.
    
    This is a placeholder that should be replaced with actual deployment logic.
    
    Args:
        version: Version to deploy.
        model_name: Model name.
        config: Deployment configuration.
        
    Returns:
        True if deployment succeeded.
    """
    logger.info(f"Deploying version {version} of {model_name}")
    # In production, this would trigger actual deployment
    return True


class RollbackManager:
    """Manager for deployment rollback operations.
    
    Tracks deployment history and provides rollback functionality
    for safe deployment management.
    """
    
    def __init__(self, max_history: int = 10):
        """Initialize rollback manager.
        
        Args:
            max_history: Maximum deployment history to retain.
        """
        self.deployment_history: List[DeploymentRecord] = []
        self.max_history = max_history
        self._current_version: Optional[str] = None
    
    def record_deployment(
        self,
        version: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> DeploymentRecord:
        """Record a new deployment.
        
        Args:
            version: Deployment version.
            model_name: Model name.
            config: Deployment configuration.
            metrics: Initial deployment metrics.
            
        Returns:
            The created DeploymentRecord.
        """
        # Mark previous deployment as not active
        if self.deployment_history:
            self.deployment_history[-1].status = DeploymentStatus.ROLLED_BACK
        
        record = DeploymentRecord(
            version=version,
            model_name=model_name,
            config=config or {},
            metrics=metrics or {},
        )
        
        self.deployment_history.append(record)
        self._current_version = version
        
        # Trim history if needed
        if len(self.deployment_history) > self.max_history:
            self.deployment_history = self.deployment_history[-self.max_history:]
        
        logger.info(f"Recorded deployment: {version}")
        return record
    
    def get_current_version(self) -> Optional[str]:
        """Get currently deployed version.
        
        Returns:
            Current version or None if no deployments.
        """
        if not self.deployment_history:
            return None
        return self.deployment_history[-1].version
    
    def get_rollback_target(self, steps_back: int = 1) -> DeploymentRecord:
        """Get the deployment record to rollback to.
        
        Args:
            steps_back: Number of versions to go back.
            
        Returns:
            DeploymentRecord for rollback target.
            
        Raises:
            RollbackError: If not enough history for rollback.
        """
        if len(self.deployment_history) <= steps_back:
            raise RollbackError(
                f"Not enough deployment history for rollback. "
                f"Have {len(self.deployment_history)} records, need {steps_back + 1}"
            )
        
        target_index = -(steps_back + 1)
        return self.deployment_history[target_index]
    
    def execute_rollback(
        self,
        steps_back: int = 1,
        reason: Optional[str] = None,
    ) -> RollbackResult:
        """Execute rollback to previous version.
        
        Args:
            steps_back: Number of versions to go back.
            reason: Reason for rollback.
            
        Returns:
            RollbackResult with outcome.
            
        Raises:
            RollbackError: If rollback cannot be executed.
        """
        import time
        start_time = time.time()
        
        if len(self.deployment_history) <= 1:
            raise RollbackError("No deployment history available for rollback")
        
        try:
            target = self.get_rollback_target(steps_back)
            current_version = self.get_current_version()
            
            logger.info(
                f"Rolling back from {current_version} to {target.version}. "
                f"Reason: {reason or 'Not specified'}"
            )
            
            # Execute the rollback
            success = deploy_version(
                version=target.version,
                model_name=target.model_name,
                config=target.config,
            )
            
            if success:
                # Update history
                self.deployment_history[-1].status = DeploymentStatus.ROLLED_BACK
                
                # Record the rollback as a new deployment
                self.record_deployment(
                    version=target.version,
                    model_name=target.model_name,
                    config=target.config,
                )
            
            duration = time.time() - start_time
            
            return RollbackResult(
                success=success,
                rolled_back_to=target.version,
                rolled_back_from=current_version,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Rollback failed: {e}")
            
            return RollbackResult(
                success=False,
                rolled_back_to="",
                rolled_back_from=self.get_current_version(),
                error=str(e),
                duration_seconds=duration,
            )
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history.
        
        Returns:
            List of deployment records as dictionaries.
        """
        return [record.to_dict() for record in self.deployment_history]
    
    def can_rollback(self) -> bool:
        """Check if rollback is possible.
        
        Returns:
            True if rollback is possible.
        """
        return len(self.deployment_history) > 1


class RollbackTrigger:
    """Automatic rollback trigger based on health metrics.
    
    Monitors deployment health and triggers rollback when
    thresholds are exceeded.
    """
    
    def __init__(
        self,
        manager: RollbackManager,
        health_threshold: float = 0.95,
        latency_threshold_ms: float = 500.0,
        error_rate_threshold: float = 0.05,
    ):
        """Initialize rollback trigger.
        
        Args:
            manager: RollbackManager instance.
            health_threshold: Minimum health score (0-1).
            latency_threshold_ms: Maximum P95 latency.
            error_rate_threshold: Maximum error rate (0-1).
        """
        self.manager = manager
        self.health_threshold = health_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self._consecutive_failures = 0
        self._failure_threshold = 3  # Consecutive failures before trigger
    
    def evaluate(
        self,
        health_score: Optional[float] = None,
        latency_p95_ms: Optional[float] = None,
        error_rate: Optional[float] = None,
    ) -> bool:
        """Evaluate if rollback should be triggered.
        
        Args:
            health_score: Current health score (0-1).
            latency_p95_ms: Current P95 latency.
            error_rate: Current error rate (0-1).
            
        Returns:
            True if rollback should be triggered.
        """
        should_rollback = False
        reasons = []
        
        # Check health score
        if health_score is not None and health_score < self.health_threshold:
            reasons.append(f"Health score {health_score:.2f} below threshold {self.health_threshold}")
            should_rollback = True
        
        # Check latency
        if latency_p95_ms is not None and latency_p95_ms > self.latency_threshold_ms:
            reasons.append(f"Latency P95 {latency_p95_ms:.0f}ms exceeds threshold {self.latency_threshold_ms}ms")
            should_rollback = True
        
        # Check error rate
        if error_rate is not None and error_rate > self.error_rate_threshold:
            reasons.append(f"Error rate {error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}")
            should_rollback = True
        
        if should_rollback:
            self._consecutive_failures += 1
            logger.warning(f"Rollback trigger evaluation: {', '.join(reasons)}")
        else:
            self._consecutive_failures = 0
        
        # Only trigger after consecutive failures
        return self._consecutive_failures >= self._failure_threshold
    
    def execute_if_needed(
        self,
        health_score: Optional[float] = None,
        latency_p95_ms: Optional[float] = None,
        error_rate: Optional[float] = None,
    ) -> Optional[RollbackResult]:
        """Evaluate and execute rollback if needed.
        
        Args:
            health_score: Current health score.
            latency_p95_ms: Current P95 latency.
            error_rate: Current error rate.
            
        Returns:
            RollbackResult if rollback was executed, None otherwise.
        """
        if not self.manager.can_rollback():
            return None
        
        should_rollback = self.evaluate(
            health_score=health_score,
            latency_p95_ms=latency_p95_ms,
            error_rate=error_rate,
        )
        
        if should_rollback:
            logger.warning("Automatic rollback triggered due to health degradation")
            return self.manager.execute_rollback(
                reason="Automatic rollback due to health threshold breach"
            )
        
        return None
    
    def reset(self) -> None:
        """Reset failure counter."""
        self._consecutive_failures = 0
