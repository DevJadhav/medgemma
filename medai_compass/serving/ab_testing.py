"""
A/B testing framework for model deployments.

Provides A/B testing functionality for comparing model variants
with statistical analysis and outcome tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import hashlib
import logging
import math

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """A/B test status."""
    
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test.
    
    Attributes:
        name: Unique test name.
        variant_a: First variant (control).
        variant_b: Second variant (treatment).
        traffic_split: Fraction of traffic to variant B (0-1).
        description: Optional test description.
        min_samples: Minimum samples before evaluation.
        confidence_level: Statistical confidence level (0-1).
    """
    
    name: str
    variant_a: str
    variant_b: str
    traffic_split: float = 0.5
    description: Optional[str] = None
    min_samples: int = 100
    confidence_level: float = 0.95
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.traffic_split <= 1:
            raise ValueError("traffic_split must be between 0 and 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")


@dataclass
class ABTestOutcome:
    """Recorded outcome for A/B test.
    
    Attributes:
        user_id: User identifier.
        variant: Assigned variant.
        metric_name: Name of the metric.
        value: Metric value.
        timestamp: When outcome was recorded.
        metadata: Additional metadata.
    """
    
    user_id: str
    variant: str
    metric_name: str
    value: float
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "variant": self.variant,
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {},
        }


@dataclass
class ABTestResults:
    """Results of an A/B test.
    
    Attributes:
        variant_a: Statistics for variant A.
        variant_b: Statistics for variant B.
        statistical_significance: Whether difference is significant.
        p_value: Statistical p-value.
        effect_size: Measured effect size.
        confidence_interval: Confidence interval for effect.
        recommendation: Recommended action.
    """
    
    variant_a: Dict[str, Any]
    variant_b: Dict[str, Any]
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: tuple
    recommendation: str


class ABTest:
    """A/B test for comparing model variants.
    
    Provides consistent user assignment, outcome tracking,
    and statistical analysis.
    """
    
    def __init__(self, config: ABTestConfig):
        """Initialize A/B test.
        
        Args:
            config: Test configuration.
        """
        self.config = config
        self._status = ABTestStatus.RUNNING
        self._outcomes: List[ABTestOutcome] = []
        self._assignments: Dict[str, str] = {}
        self._started_at = datetime.utcnow()
    
    @property
    def name(self) -> str:
        """Get test name."""
        return self.config.name
    
    @property
    def variant_a(self) -> str:
        """Get variant A."""
        return self.config.variant_a
    
    @property
    def variant_b(self) -> str:
        """Get variant B."""
        return self.config.variant_b
    
    @property
    def traffic_split(self) -> float:
        """Get traffic split."""
        return self.config.traffic_split
    
    @property
    def status(self) -> ABTestStatus:
        """Get test status."""
        return self._status
    
    def _hash_user_id(self, user_id: str) -> float:
        """Hash user ID to deterministic value between 0-1.
        
        Args:
            user_id: User identifier.
            
        Returns:
            Deterministic float between 0 and 1.
        """
        hash_input = f"{self.name}:{user_id}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        return hash_int / (2 ** 64)
    
    def assign_variant(self, user_id: str) -> str:
        """Assign user to a variant.
        
        Uses consistent hashing for deterministic assignment.
        
        Args:
            user_id: User identifier.
            
        Returns:
            Assigned variant name.
        """
        # Check cache first
        if user_id in self._assignments:
            return self._assignments[user_id]
        
        # Use consistent hashing
        hash_value = self._hash_user_id(user_id)
        
        if hash_value < self.config.traffic_split:
            variant = self.config.variant_b
        else:
            variant = self.config.variant_a
        
        self._assignments[user_id] = variant
        return variant
    
    def record_outcome(
        self,
        user_id: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an outcome for a user.
        
        Args:
            user_id: User identifier.
            metric_name: Name of the metric.
            value: Metric value.
            metadata: Optional additional metadata.
        """
        variant = self._assignments.get(user_id)
        
        if variant is None:
            # Assign if not already assigned
            variant = self.assign_variant(user_id)
        
        outcome = ABTestOutcome(
            user_id=user_id,
            variant=variant,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {},
        )
        
        self._outcomes.append(outcome)
    
    def get_outcomes(self) -> List[ABTestOutcome]:
        """Get all recorded outcomes.
        
        Returns:
            List of outcomes.
        """
        return self._outcomes.copy()
    
    def _calculate_statistics(
        self,
        values: List[float]
    ) -> Dict[str, float]:
        """Calculate statistics for a list of values.
        
        Args:
            values: List of metric values.
            
        Returns:
            Dictionary with statistics.
        """
        n = len(values)
        if n == 0:
            return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
        
        mean = sum(values) / n
        
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0
        
        return {
            "n": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    
    def _calculate_p_value(
        self,
        mean_a: float,
        std_a: float,
        n_a: int,
        mean_b: float,
        std_b: float,
        n_b: int,
    ) -> float:
        """Calculate p-value using Welch's t-test.
        
        Args:
            mean_a: Mean of variant A.
            std_a: Standard deviation of variant A.
            n_a: Sample size of variant A.
            mean_b: Mean of variant B.
            std_b: Standard deviation of variant B.
            n_b: Sample size of variant B.
            
        Returns:
            Two-tailed p-value.
        """
        if n_a < 2 or n_b < 2:
            return 1.0
        
        if std_a == 0 and std_b == 0:
            return 1.0 if mean_a == mean_b else 0.0
        
        # Welch's t-test
        se_a = (std_a ** 2) / n_a
        se_b = (std_b ** 2) / n_b
        
        se_diff = math.sqrt(se_a + se_b)
        if se_diff == 0:
            return 1.0
        
        t_stat = abs(mean_a - mean_b) / se_diff
        
        # Approximate p-value using normal distribution for large samples
        # For small samples, this is an approximation
        p_value = 2 * (1 - self._normal_cdf(t_stat))
        
        return round(p_value, 6)
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function.
        
        Args:
            x: Input value.
            
        Returns:
            CDF value.
        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def compute_results(
        self,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute A/B test results.
        
        Args:
            metric_name: Optional specific metric to analyze.
            
        Returns:
            Dictionary with test results.
        """
        # Filter outcomes by metric if specified
        outcomes = self._outcomes
        if metric_name:
            outcomes = [o for o in outcomes if o.metric_name == metric_name]
        
        # Separate by variant
        values_a = [o.value for o in outcomes if o.variant == self.config.variant_a]
        values_b = [o.value for o in outcomes if o.variant == self.config.variant_b]
        
        # Calculate statistics
        stats_a = self._calculate_statistics(values_a)
        stats_b = self._calculate_statistics(values_b)
        
        # Calculate p-value
        p_value = self._calculate_p_value(
            stats_a["mean"], stats_a["std"], stats_a["n"],
            stats_b["mean"], stats_b["std"], stats_b["n"],
        )
        
        # Statistical significance
        alpha = 1 - self.config.confidence_level
        is_significant = p_value < alpha
        
        # Effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((stats_a["n"] - 1) * stats_a["std"] ** 2 + 
             (stats_b["n"] - 1) * stats_b["std"] ** 2) /
            (stats_a["n"] + stats_b["n"] - 2)
        ) if (stats_a["n"] + stats_b["n"] > 2) else 1
        
        effect_size = (stats_b["mean"] - stats_a["mean"]) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference
        diff = stats_b["mean"] - stats_a["mean"]
        se_diff = math.sqrt(
            (stats_a["std"] ** 2 / stats_a["n"] if stats_a["n"] > 0 else 0) +
            (stats_b["std"] ** 2 / stats_b["n"] if stats_b["n"] > 0 else 0)
        )
        z = 1.96  # 95% confidence
        ci_lower = diff - z * se_diff
        ci_upper = diff + z * se_diff
        
        # Recommendation
        if not is_significant:
            recommendation = "No significant difference detected. Continue testing."
        elif effect_size > 0.2:  # Small effect size threshold
            if stats_b["mean"] > stats_a["mean"]:
                recommendation = f"Variant B ({self.config.variant_b}) shows improvement. Consider promotion."
            else:
                recommendation = f"Variant A ({self.config.variant_a}) performs better. Consider keeping current."
        else:
            recommendation = "Effect size is small. More data recommended."
        
        return {
            "variant_a": {
                "name": self.config.variant_a,
                "statistics": stats_a,
            },
            "variant_b": {
                "name": self.config.variant_b,
                "statistics": stats_b,
            },
            "statistical_significance": is_significant,
            "p_value": p_value,
            "effect_size": round(effect_size, 4),
            "confidence_interval": (round(ci_lower, 4), round(ci_upper, 4)),
            "recommendation": recommendation,
            "total_samples": stats_a["n"] + stats_b["n"],
        }
    
    def pause(self) -> None:
        """Pause the A/B test."""
        self._status = ABTestStatus.PAUSED
        logger.info(f"A/B test '{self.name}' paused")
    
    def resume(self) -> None:
        """Resume the A/B test."""
        self._status = ABTestStatus.RUNNING
        logger.info(f"A/B test '{self.name}' resumed")
    
    def complete(self) -> None:
        """Complete the A/B test."""
        self._status = ABTestStatus.COMPLETED
        logger.info(f"A/B test '{self.name}' completed")
    
    def cancel(self) -> None:
        """Cancel the A/B test."""
        self._status = ABTestStatus.CANCELLED
        logger.info(f"A/B test '{self.name}' cancelled")


class ABTestManager:
    """Manager for multiple A/B tests.
    
    Handles creation, tracking, and lifecycle of A/B tests.
    """
    
    def __init__(self):
        """Initialize A/B test manager."""
        self._tests: Dict[str, ABTest] = {}
    
    def create_test(self, config: ABTestConfig) -> ABTest:
        """Create a new A/B test.
        
        Args:
            config: Test configuration.
            
        Returns:
            Created ABTest instance.
        """
        if config.name in self._tests:
            raise ValueError(f"Test '{config.name}' already exists")
        
        test = ABTest(config)
        self._tests[config.name] = test
        
        logger.info(f"Created A/B test: {config.name}")
        return test
    
    def get_test(self, name: str) -> Optional[ABTest]:
        """Get A/B test by name.
        
        Args:
            name: Test name.
            
        Returns:
            ABTest or None if not found.
        """
        return self._tests.get(name)
    
    def list_tests(self) -> List[ABTest]:
        """List all A/B tests.
        
        Returns:
            List of all tests.
        """
        return list(self._tests.values())
    
    def delete_test(self, name: str) -> bool:
        """Delete an A/B test.
        
        Args:
            name: Test name.
            
        Returns:
            True if deleted, False if not found.
        """
        if name in self._tests:
            del self._tests[name]
            logger.info(f"Deleted A/B test: {name}")
            return True
        return False
    
    def get_active_tests(self) -> List[ABTest]:
        """Get all active A/B tests.
        
        Returns:
            List of running tests.
        """
        return [t for t in self._tests.values() if t.status == ABTestStatus.RUNNING]
    
    def assign_variant_for_all(self, user_id: str) -> Dict[str, str]:
        """Assign user to variants for all active tests.
        
        Args:
            user_id: User identifier.
            
        Returns:
            Dictionary mapping test names to assigned variants.
        """
        assignments = {}
        for test in self.get_active_tests():
            assignments[test.name] = test.assign_variant(user_id)
        return assignments
