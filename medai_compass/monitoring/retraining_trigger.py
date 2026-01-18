"""
Auto-Retraining Triggers Module (Phase 8: Monitoring & Observability).

Provides automatic retraining triggers based on drift detection,
accuracy degradation, and scheduling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Types
# ============================================================================

class TriggerType(Enum):
    """Types of retraining triggers."""
    
    DRIFT = "drift"
    ACCURACY = "accuracy"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DATA_VOLUME = "data_volume"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RetrainingConfig:
    """Configuration for retraining triggers.
    
    Attributes:
        model_name: Name of the model
        min_samples_for_retrain: Minimum new samples before retraining
        drift_threshold: Drift score threshold to trigger retraining
        accuracy_threshold: Accuracy threshold (retrain if below)
        max_retrains_per_day: Rate limit for automatic retraining
    """
    
    model_name: str = "medgemma-27b-it"
    min_samples_for_retrain: int = 1000
    drift_threshold: float = 0.2
    accuracy_threshold: float = 0.75
    max_retrains_per_day: int = 2


def get_model_retrain_config(model_name: str) -> RetrainingConfig:
    """Get retraining configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        RetrainingConfig for the model
    """
    return RetrainingConfig(
        model_name=model_name,
        drift_threshold=0.2,
        accuracy_threshold=0.75,  # MedQA threshold
    )


# ============================================================================
# Trigger Reason
# ============================================================================

@dataclass
class TriggerReason:
    """Reason for triggering retraining.
    
    Attributes:
        trigger_type: Type of trigger
        description: Human-readable description
        metric_name: Name of the metric that triggered
        metric_value: Current metric value
        threshold: Threshold that was exceeded
        timestamp: When the trigger occurred
    """
    
    trigger_type: TriggerType
    description: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "trigger_type": self.trigger_type.value,
            "description": self.description,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Trigger Result
# ============================================================================

@dataclass
class TriggerResult:
    """Result of a trigger evaluation.
    
    Attributes:
        should_retrain: Whether retraining should be triggered
        reason: Reason for triggering (if should_retrain)
    """
    
    should_retrain: bool
    reason: Optional[TriggerReason] = None


# ============================================================================
# Trigger Base Class
# ============================================================================

class BaseTrigger:
    """Base class for retraining triggers."""
    
    def __init__(self, config: RetrainingConfig):
        """Initialize trigger.
        
        Args:
            config: Retraining configuration
        """
        self.config = config
    
    def evaluate(self, **kwargs) -> TriggerResult:
        """Evaluate the trigger.
        
        Returns:
            TriggerResult
        """
        raise NotImplementedError


# ============================================================================
# Drift-Based Trigger
# ============================================================================

class DriftBasedTrigger(BaseTrigger):
    """Triggers retraining based on drift detection."""
    
    def __init__(self, config: RetrainingConfig):
        """Initialize drift trigger.
        
        Args:
            config: Retraining configuration
        """
        super().__init__(config)
        self.threshold = config.drift_threshold
    
    def evaluate(self, drift_score: float) -> TriggerResult:
        """Evaluate drift score against threshold.
        
        Args:
            drift_score: Drift score from detector
            
        Returns:
            TriggerResult
        """
        if drift_score >= self.threshold:
            return TriggerResult(
                should_retrain=True,
                reason=TriggerReason(
                    trigger_type=TriggerType.DRIFT,
                    description=f"Drift score {drift_score:.3f} exceeds threshold {self.threshold}",
                    metric_name="drift_score",
                    metric_value=drift_score,
                    threshold=self.threshold,
                ),
            )
        
        return TriggerResult(should_retrain=False)
    
    def evaluate_multiple(self, drift_metrics: dict[str, float]) -> TriggerResult:
        """Evaluate multiple drift metrics.
        
        Args:
            drift_metrics: Dictionary of drift scores by type
            
        Returns:
            TriggerResult
        """
        for name, score in drift_metrics.items():
            if score >= self.threshold:
                return TriggerResult(
                    should_retrain=True,
                    reason=TriggerReason(
                        trigger_type=TriggerType.DRIFT,
                        description=f"{name} drift score {score:.3f} exceeds threshold",
                        metric_name=name,
                        metric_value=score,
                        threshold=self.threshold,
                    ),
                )
        
        return TriggerResult(should_retrain=False)


# ============================================================================
# Accuracy-Based Trigger
# ============================================================================

class AccuracyBasedTrigger(BaseTrigger):
    """Triggers retraining based on accuracy degradation."""
    
    def __init__(self, config: RetrainingConfig, consider_trend: bool = False):
        """Initialize accuracy trigger.
        
        Args:
            config: Retraining configuration
            consider_trend: Whether to consider accuracy trend
        """
        super().__init__(config)
        self.threshold = config.accuracy_threshold
        self.consider_trend = consider_trend
    
    def evaluate(self, accuracy: float) -> TriggerResult:
        """Evaluate accuracy against threshold.
        
        Args:
            accuracy: Current accuracy score
            
        Returns:
            TriggerResult
        """
        if accuracy < self.threshold:
            return TriggerResult(
                should_retrain=True,
                reason=TriggerReason(
                    trigger_type=TriggerType.ACCURACY,
                    description=f"Accuracy {accuracy:.3f} below threshold {self.threshold}",
                    metric_name="accuracy",
                    metric_value=accuracy,
                    threshold=self.threshold,
                ),
            )
        
        return TriggerResult(should_retrain=False)
    
    def evaluate_with_history(self, accuracy_history: list[float]) -> TriggerResult:
        """Evaluate accuracy trend.
        
        Args:
            accuracy_history: List of recent accuracy values (oldest to newest)
            
        Returns:
            TriggerResult
        """
        if not accuracy_history:
            return TriggerResult(should_retrain=False)
        
        current = accuracy_history[-1]
        
        # Check current accuracy
        if current < self.threshold:
            return TriggerResult(
                should_retrain=True,
                reason=TriggerReason(
                    trigger_type=TriggerType.ACCURACY,
                    description=f"Accuracy {current:.3f} below threshold",
                    metric_name="accuracy",
                    metric_value=current,
                    threshold=self.threshold,
                ),
            )
        
        # Check declining trend
        if self.consider_trend and len(accuracy_history) >= 3:
            import numpy as np
            
            trend = np.polyfit(range(len(accuracy_history)), accuracy_history, 1)[0]
            
            if trend < -0.01:  # Declining trend
                predicted = current + trend * 5  # Project 5 steps ahead
                if predicted < self.threshold:
                    return TriggerResult(
                        should_retrain=True,
                        reason=TriggerReason(
                            trigger_type=TriggerType.ACCURACY,
                            description="Declining accuracy trend detected",
                            metric_name="accuracy_trend",
                            metric_value=float(trend),
                            threshold=-0.01,
                        ),
                    )
        
        return TriggerResult(should_retrain=False)


# ============================================================================
# Scheduled Trigger
# ============================================================================

class ScheduledTrigger(BaseTrigger):
    """Triggers retraining on a schedule."""
    
    def __init__(self, config: RetrainingConfig, schedule_days: int = 7):
        """Initialize scheduled trigger.
        
        Args:
            config: Retraining configuration
            schedule_days: Days between scheduled retraining
        """
        super().__init__(config)
        self.schedule_days = schedule_days
        self.last_retrain: Optional[datetime] = None
    
    def evaluate(self) -> TriggerResult:
        """Check if scheduled retraining is due.
        
        Returns:
            TriggerResult
        """
        if self.last_retrain is None:
            return TriggerResult(should_retrain=False)
        
        days_since = (datetime.now() - self.last_retrain).days
        
        if days_since >= self.schedule_days:
            return TriggerResult(
                should_retrain=True,
                reason=TriggerReason(
                    trigger_type=TriggerType.SCHEDULED,
                    description=f"Scheduled retraining: {days_since} days since last",
                    metric_name="days_since_retrain",
                    metric_value=float(days_since),
                    threshold=float(self.schedule_days),
                ),
            )
        
        return TriggerResult(should_retrain=False)


# ============================================================================
# Data Volume Trigger
# ============================================================================

class DataVolumeTrigger(BaseTrigger):
    """Triggers retraining based on new data volume."""
    
    def __init__(self, config: RetrainingConfig):
        """Initialize data volume trigger.
        
        Args:
            config: Retraining configuration
        """
        super().__init__(config)
        self.min_samples = config.min_samples_for_retrain
    
    def evaluate(self, new_samples: int) -> TriggerResult:
        """Check if enough new data for retraining.
        
        Args:
            new_samples: Number of new samples collected
            
        Returns:
            TriggerResult
        """
        if new_samples >= self.min_samples:
            return TriggerResult(
                should_retrain=True,
                reason=TriggerReason(
                    trigger_type=TriggerType.DATA_VOLUME,
                    description=f"Collected {new_samples} new samples",
                    metric_name="new_samples",
                    metric_value=float(new_samples),
                    threshold=float(self.min_samples),
                ),
            )
        
        return TriggerResult(should_retrain=False)


# ============================================================================
# Composite Trigger
# ============================================================================

class CompositeTrigger:
    """Combines multiple triggers with AND/OR logic."""
    
    def __init__(
        self,
        triggers: list[BaseTrigger],
        mode: str = "any",  # "any" or "all"
    ):
        """Initialize composite trigger.
        
        Args:
            triggers: List of triggers to combine
            mode: Combination mode ("any" or "all")
        """
        self.triggers = triggers
        self.mode = mode
    
    def evaluate(self, metrics: dict[str, Any]) -> TriggerResult:
        """Evaluate all triggers.
        
        Args:
            metrics: Dictionary of metrics for evaluation
            
        Returns:
            TriggerResult
        """
        results = []
        
        for trigger in self.triggers:
            if isinstance(trigger, DriftBasedTrigger) and "drift_score" in metrics:
                results.append(trigger.evaluate(metrics["drift_score"]))
            elif isinstance(trigger, AccuracyBasedTrigger) and "accuracy" in metrics:
                results.append(trigger.evaluate(metrics["accuracy"]))
            elif isinstance(trigger, ScheduledTrigger):
                results.append(trigger.evaluate())
            elif isinstance(trigger, DataVolumeTrigger) and "new_samples" in metrics:
                results.append(trigger.evaluate(metrics["new_samples"]))
        
        if self.mode == "any":
            for result in results:
                if result.should_retrain:
                    return result
            return TriggerResult(should_retrain=False)
        else:  # "all"
            if all(r.should_retrain for r in results) and results:
                return results[0]
            return TriggerResult(should_retrain=False)


# ============================================================================
# Retraining History
# ============================================================================

class RetrainingHistory:
    """Tracks history of retraining events."""
    
    def __init__(self, max_events: int = 100):
        """Initialize retraining history.
        
        Args:
            max_events: Maximum events to retain
        """
        self.events: list[dict] = []
        self.max_events = max_events
    
    def record(
        self,
        model_name: str,
        trigger_type: str,
        success: bool,
        details: Optional[dict] = None,
    ) -> None:
        """Record a retraining event.
        
        Args:
            model_name: Name of the model
            trigger_type: Type of trigger
            success: Whether retraining succeeded
            details: Additional details
        """
        self.events.append({
            "model_name": model_name,
            "trigger_type": trigger_type,
            "success": success,
            "timestamp": datetime.now(),
            "details": details or {},
        })
        
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_stats(self) -> dict[str, Any]:
        """Get retraining statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.events:
            return {
                "total_retrains": 0,
                "success_rate": 0.0,
            }
        
        successes = sum(1 for e in self.events if e["success"])
        
        return {
            "total_retrains": len(self.events),
            "success_rate": successes / len(self.events),
            "trigger_types": self._count_by_trigger_type(),
        }
    
    def _count_by_trigger_type(self) -> dict[str, int]:
        """Count events by trigger type."""
        counts: dict[str, int] = {}
        for event in self.events:
            trigger_type = event["trigger_type"]
            counts[trigger_type] = counts.get(trigger_type, 0) + 1
        return counts
    
    def get_count_today(self) -> int:
        """Get count of retrains today.
        
        Returns:
            Number of retrains today
        """
        today = datetime.now().date()
        return sum(1 for e in self.events if e["timestamp"].date() == today)


# ============================================================================
# Retraining Manager
# ============================================================================

class RetrainingManager:
    """Manages retraining triggers and orchestration."""
    
    def __init__(self, config: RetrainingConfig):
        """Initialize retraining manager.
        
        Args:
            config: Retraining configuration
        """
        self.config = config
        self.drift_trigger = DriftBasedTrigger(config)
        self.accuracy_trigger = AccuracyBasedTrigger(config)
        self.triggers: list[BaseTrigger] = [
            self.drift_trigger,
            self.accuracy_trigger,
        ]
        self.history = RetrainingHistory()
        self._retrain_count_today = 0
    
    def add_trigger(self, trigger: BaseTrigger) -> None:
        """Add a custom trigger.
        
        Args:
            trigger: Trigger to add
        """
        self.triggers.append(trigger)
    
    def evaluate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Evaluate all triggers.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Evaluation result
        """
        # Check rate limiting
        if self._retrain_count_today >= self.config.max_retrains_per_day:
            return {
                "should_retrain": False,
                "rate_limited": True,
                "reasons": [],
            }
        
        reasons = []
        should_retrain = False
        
        if "drift_score" in metrics:
            result = self.drift_trigger.evaluate(metrics["drift_score"])
            if result.should_retrain:
                should_retrain = True
                reasons.append(result.reason)
        
        if "accuracy" in metrics:
            result = self.accuracy_trigger.evaluate(metrics["accuracy"])
            if result.should_retrain:
                should_retrain = True
                reasons.append(result.reason)
        
        return {
            "should_retrain": should_retrain,
            "rate_limited": False,
            "reasons": [r.to_dict() if r else {} for r in reasons],
        }
    
    def record_retrain(self) -> None:
        """Record a retraining event for rate limiting."""
        self._retrain_count_today += 1
    
    async def trigger_retrain(self) -> dict[str, Any]:
        """Trigger the retraining process.
        
        Returns:
            Job status
        """
        return await self._start_retraining_job()
    
    async def _start_retraining_job(self) -> dict[str, Any]:
        """Start a retraining job (stub for integration).
        
        Returns:
            Job information
        """
        import uuid
        
        job_id = f"retrain-{uuid.uuid4().hex[:8]}"
        
        return {
            "job_id": job_id,
            "status": "started",
            "model_name": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
        }
    
    def generate_retrain_config(self, reason: TriggerReason) -> dict[str, Any]:
        """Generate retraining configuration.
        
        Args:
            reason: Trigger reason
            
        Returns:
            Retraining configuration
        """
        return {
            "model_name": self.config.model_name,
            "trigger_reason": reason.to_dict(),
            "min_samples": self.config.min_samples_for_retrain,
            "config": {
                "learning_rate": 1e-5,
                "epochs": 3,
                "batch_size": 8,
            },
        }
    
    async def retrain_and_deploy(self) -> dict[str, Any]:
        """Retrain and deploy new model.
        
        Returns:
            Deployment result
        """
        retrain_result = await self.trigger_retrain()
        
        if retrain_result["status"] == "started":
            deploy_result = await self._deploy_new_model()
            return {
                "retrain": retrain_result,
                "deploy": deploy_result,
            }
        
        return retrain_result
    
    async def _deploy_new_model(self) -> dict[str, Any]:
        """Deploy newly trained model (stub).
        
        Returns:
            Deployment result
        """
        return {
            "status": "deployed",
            "version": "1.0.1",
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_rollback_config(self) -> dict[str, Any]:
        """Get rollback configuration.
        
        Returns:
            Rollback configuration
        """
        return {
            "previous_version": "1.0.0",
            "auto_rollback": True,
            "rollback_on_accuracy_drop": True,
            "accuracy_threshold": self.config.accuracy_threshold,
        }
