"""
Drift Detection Module (Phase 8: Monitoring & Observability).

Provides comprehensive drift detection for medical AI models using alibi-detect.
Supports input drift, output drift, and concept drift detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from scipy import stats
from scipy.special import kl_div

# alibi-detect imports (fully integrated, not optional)
from alibi_detect.cd import KSDrift, MMDDrift, TabularDrift
from alibi_detect.cd.base import BaseUnivariateDrift

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Types
# ============================================================================

class DriftType(Enum):
    """Types of drift that can be detected."""
    
    INPUT = "input"
    OUTPUT = "output"
    CONCEPT = "concept"
    FEATURE = "feature"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DriftConfig:
    """Configuration for drift detection.
    
    Attributes:
        model_name: Name of the model (medgemma-4b-it or medgemma-27b-it)
        kl_divergence_threshold: Threshold for KL divergence
        psi_threshold: Threshold for Population Stability Index
        p_value_threshold: Statistical significance threshold
        window_size: Number of samples in sliding window
    """
    
    model_name: str = "medgemma-27b-it"
    kl_divergence_threshold: float = 0.1
    psi_threshold: float = 0.2
    p_value_threshold: float = 0.05
    window_size: int = 1000


# ============================================================================
# Drift Result
# ============================================================================

@dataclass
class DriftResult:
    """Result of a drift detection check.
    
    Attributes:
        drift_detected: Whether drift was detected
        drift_type: Type of drift detected
        score: Drift score/statistic
        threshold: Threshold used for detection
        p_value: Statistical p-value (if applicable)
        details: Additional details about the drift
        timestamp: When the check was performed
    """
    
    drift_detected: bool
    drift_type: DriftType
    score: float
    threshold: float
    p_value: Optional[float] = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type.value,
            "score": self.score,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Statistical Utilities
# ============================================================================

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL divergence between two distributions.
    
    Args:
        p: Reference distribution (probabilities)
        q: Current distribution (probabilities)
        
    Returns:
        KL divergence value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # Normalize to ensure valid probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return float(np.sum(kl_div(p, q)))


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI).
    
    PSI measures how much a distribution has shifted.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change
    
    Args:
        expected: Expected/baseline distribution
        actual: Current distribution
        bins: Number of bins for histogram
        
    Returns:
        PSI value
    """
    epsilon = 1e-10
    
    # Create bins based on expected distribution
    _, bin_edges = np.histogram(expected, bins=bins)
    
    # Calculate proportions in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    expected_props = expected_counts / len(expected) + epsilon
    actual_props = actual_counts / len(actual) + epsilon
    
    # Calculate PSI
    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    
    return float(psi)


# ============================================================================
# Model-Specific Thresholds
# ============================================================================

def get_model_thresholds(model_name: str) -> dict:
    """Get drift detection thresholds for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of thresholds
    """
    thresholds = {
        "medgemma-27b-it": {
            "kl_divergence": 0.1,
            "psi": 0.2,
            "p_value": 0.05,
            "latency_p95_ms": 500,
            "accuracy_min": 0.75,
        },
        "medgemma-4b-it": {
            "kl_divergence": 0.1,
            "psi": 0.2,
            "p_value": 0.05,
            "latency_p95_ms": 250,  # Faster model
            "accuracy_min": 0.75,
        },
    }
    
    return thresholds.get(model_name, thresholds["medgemma-27b-it"])


# ============================================================================
# Input Drift Detector
# ============================================================================

class InputDriftDetector:
    """Detects drift in input data distribution.
    
    Uses statistical tests to detect changes in the input feature
    distribution compared to a baseline.
    """
    
    def __init__(self, config: DriftConfig):
        """Initialize input drift detector.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.baseline: Optional[np.ndarray] = None
        self._baseline_stats: Optional[dict] = None
    
    def set_baseline(self, data: np.ndarray) -> None:
        """Set baseline distribution for comparison.
        
        Args:
            data: Baseline input data (n_samples, n_features)
        """
        self.baseline = data.astype(np.float32)
        self._baseline_stats = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "n_samples": len(data),
        }
        logger.info(f"Baseline set with {len(data)} samples")
    
    def detect(self, data: np.ndarray) -> DriftResult:
        """Detect drift in new data compared to baseline.
        
        Args:
            data: Current input data (n_samples, n_features)
            
        Returns:
            DriftResult with detection results
            
        Raises:
            ValueError: If baseline not set
        """
        if self.baseline is None:
            raise ValueError("Baseline must be set before drift detection")
        
        data = data.astype(np.float32)
        
        # Use Kolmogorov-Smirnov test for each feature
        p_values = []
        for i in range(data.shape[1]):
            _, p_value = stats.ks_2samp(self.baseline[:, i], data[:, i])
            p_values.append(p_value)
        
        # Drift detected if any feature has significant drift
        min_p_value = min(p_values)
        drift_detected = bool(min_p_value < self.config.p_value_threshold)
        
        # Calculate overall drift score (1 - min p-value for interpretability)
        drift_score = 1 - min_p_value
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.INPUT,
            score=drift_score,
            threshold=1 - self.config.p_value_threshold,
            p_value=min_p_value,
            details={
                "feature_p_values": p_values,
                "n_features_drifted": sum(1 for p in p_values if p < self.config.p_value_threshold),
            },
        )


# ============================================================================
# Output Drift Detector
# ============================================================================

class OutputDriftDetector:
    """Detects drift in model output distribution.
    
    Monitors changes in prediction confidence, class distribution,
    or output feature distributions.
    """
    
    def __init__(self, config: DriftConfig):
        """Initialize output drift detector.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.baseline: Optional[np.ndarray] = None
    
    def set_baseline(self, data: np.ndarray) -> None:
        """Set baseline output distribution.
        
        Args:
            data: Baseline output data (confidence scores, predictions, etc.)
        """
        self.baseline = np.asarray(data).flatten().astype(np.float32)
        logger.info(f"Output baseline set with {len(self.baseline)} samples")
    
    def detect(self, data: np.ndarray) -> DriftResult:
        """Detect drift in output distribution.
        
        Args:
            data: Current output data
            
        Returns:
            DriftResult with detection results
        """
        if self.baseline is None:
            raise ValueError("Baseline must be set before drift detection")
        
        data = np.asarray(data).flatten().astype(np.float32)
        
        # Use KS test for output distribution
        statistic, p_value = stats.ks_2samp(self.baseline, data)
        
        # Also calculate PSI for additional insight
        psi_score = calculate_psi(self.baseline, data)
        
        drift_detected = bool(
            p_value < self.config.p_value_threshold or 
            psi_score > self.config.psi_threshold
        )
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.OUTPUT,
            score=psi_score,
            threshold=self.config.psi_threshold,
            p_value=p_value,
            details={
                "ks_statistic": float(statistic),
                "psi_score": psi_score,
                "baseline_mean": float(np.mean(self.baseline)),
                "current_mean": float(np.mean(data)),
            },
        )


# ============================================================================
# Concept Drift Detector
# ============================================================================

class ConceptDriftDetector:
    """Detects concept drift (change in input-output relationship).
    
    Monitors whether the mapping from inputs to outputs has changed,
    indicating that the model's understanding may be outdated.
    """
    
    def __init__(self, config: DriftConfig):
        """Initialize concept drift detector.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.baseline_X: Optional[np.ndarray] = None
        self.baseline_y: Optional[np.ndarray] = None
        self._baseline_correlation: Optional[np.ndarray] = None
    
    def set_baseline(self, X: np.ndarray, y: np.ndarray) -> None:
        """Set baseline input-output relationship.
        
        Args:
            X: Baseline input features
            y: Baseline outputs
        """
        self.baseline_X = X.astype(np.float32)
        self.baseline_y = y.astype(np.float32).flatten()
        
        # Calculate baseline correlations between features and output
        self._baseline_correlation = np.array([
            np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])
        ])
        
        logger.info(f"Concept baseline set with {len(X)} samples")
    
    def detect(self, X: np.ndarray, y: np.ndarray) -> DriftResult:
        """Detect concept drift in new data.
        
        Args:
            X: Current input features
            y: Current outputs
            
        Returns:
            DriftResult with detection results
        """
        if self.baseline_X is None:
            raise ValueError("Baseline must be set before concept drift detection")
        
        X = X.astype(np.float32)
        y = y.astype(np.float32).flatten()
        
        # Calculate current correlations
        current_correlation = np.array([
            np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])
        ])
        
        # Handle NaN correlations
        current_correlation = np.nan_to_num(current_correlation, nan=0.0)
        baseline_corr = np.nan_to_num(self._baseline_correlation, nan=0.0)
        
        # Calculate correlation change (concept drift indicator)
        correlation_change = np.abs(current_correlation - baseline_corr)
        max_change = np.max(correlation_change)
        
        # Significant change indicates concept drift
        drift_detected = max_change > self.config.kl_divergence_threshold
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.CONCEPT,
            score=float(max_change),
            threshold=self.config.kl_divergence_threshold,
            details={
                "correlation_changes": correlation_change.tolist(),
                "max_correlation_change": float(max_change),
                "feature_with_max_change": int(np.argmax(correlation_change)),
            },
        )


# ============================================================================
# Alibi-Detect Integration
# ============================================================================

class AlibiTabularDriftDetector:
    """Drift detector using alibi-detect library.
    
    Provides advanced drift detection methods including:
    - MMD (Maximum Mean Discrepancy)
    - KS (Kolmogorov-Smirnov)
    - Tabular drift detection
    """
    
    def __init__(self, config: DriftConfig, method: str = "mmd"):
        """Initialize alibi-detect based detector.
        
        Args:
            config: Drift detection configuration
            method: Detection method ('mmd', 'ks', 'tabular')
        """
        self.config = config
        self.method = method
        self.detector: Optional[BaseUnivariateDrift] = None
    
    def set_baseline(self, data: np.ndarray) -> None:
        """Set baseline and initialize detector.
        
        Args:
            data: Baseline data for reference
        """
        data = data.astype(np.float32)
        
        if self.method == "mmd":
            self.detector = MMDDrift(
                data,
                p_val=self.config.p_value_threshold,
                backend="pytorch" if self._has_torch() else "tensorflow",
            )
        elif self.method == "ks":
            self.detector = KSDrift(
                data,
                p_val=self.config.p_value_threshold,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Alibi-detect {self.method} detector initialized with {len(data)} samples")
    
    def detect(self, data: np.ndarray) -> DriftResult:
        """Detect drift using alibi-detect.
        
        Args:
            data: Current data to check for drift
            
        Returns:
            DriftResult with detection results
        """
        if self.detector is None:
            raise ValueError("Baseline must be set before drift detection")
        
        data = data.astype(np.float32)
        
        # Run prediction
        prediction = self.detector.predict(data)
        
        drift_detected = bool(prediction["data"]["is_drift"])
        
        # Handle p_val which can be scalar or array
        p_val = prediction["data"]["p_val"]
        if hasattr(p_val, '__iter__') and not isinstance(p_val, str):
            p_value = float(np.min(p_val))  # Use minimum p-value
        else:
            p_value = float(p_val)
        
        # Get distance/statistic
        if "distance" in prediction["data"]:
            dist = prediction["data"]["distance"]
            if hasattr(dist, '__iter__') and not isinstance(dist, str):
                score = float(np.mean(dist))
            else:
                score = float(dist)
        else:
            score = 1 - p_value  # Fallback
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.INPUT,
            score=score,
            threshold=self.config.p_value_threshold,
            p_value=p_value,
            details={
                "method": self.method,
                "alibi_prediction": {
                    k: v.tolist() if hasattr(v, 'tolist') else v 
                    for k, v in prediction["data"].items()
                },
            },
        )
    
    @staticmethod
    def _has_torch() -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False


# ============================================================================
# Drift History
# ============================================================================

class DriftHistory:
    """Tracks history of drift detection events."""
    
    def __init__(self, max_events: int = 1000):
        """Initialize drift history.
        
        Args:
            max_events: Maximum events to keep in history
        """
        self.events: list[DriftResult] = []
        self.max_events = max_events
    
    def record(self, result: DriftResult) -> None:
        """Record a drift detection result.
        
        Args:
            result: Drift detection result
        """
        self.events.append(result)
        
        # Trim if exceeding max
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_events(self, limit: Optional[int] = None) -> list[DriftResult]:
        """Get recent drift events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of drift results
        """
        if limit:
            return self.events[-limit:]
        return self.events
    
    def get_drift_rate(self) -> float:
        """Calculate drift detection rate.
        
        Returns:
            Proportion of checks that detected drift
        """
        if not self.events:
            return 0.0
        
        drift_count = sum(1 for e in self.events if e.drift_detected)
        return drift_count / len(self.events)


# ============================================================================
# Unified Drift Manager
# ============================================================================

class DriftManager:
    """Unified interface for all drift detection.
    
    Manages input, output, and concept drift detection with
    support for both MedGemma 4B and 27B models.
    """
    
    def __init__(self, config: DriftConfig):
        """Initialize drift manager.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.input_detector = InputDriftDetector(config)
        self.output_detector = OutputDriftDetector(config)
        self.concept_detector = ConceptDriftDetector(config)
        self.history = DriftHistory()
        self._last_results: dict[str, DriftResult] = {}
    
    def set_baselines(
        self,
        input_data: Optional[np.ndarray] = None,
        output_data: Optional[np.ndarray] = None,
        concept_X: Optional[np.ndarray] = None,
        concept_y: Optional[np.ndarray] = None,
    ) -> None:
        """Set baselines for all drift detectors.
        
        Args:
            input_data: Baseline input features
            output_data: Baseline outputs/predictions
            concept_X: Baseline features for concept drift
            concept_y: Baseline labels for concept drift
        """
        if input_data is not None:
            self.input_detector.set_baseline(input_data)
        
        if output_data is not None:
            self.output_detector.set_baseline(output_data)
        
        if concept_X is not None and concept_y is not None:
            self.concept_detector.set_baseline(concept_X, concept_y)
        
        logger.info(f"Baselines set for model: {self.config.model_name}")
    
    def check_all_drift(
        self,
        input_data: Optional[np.ndarray] = None,
        output_data: Optional[np.ndarray] = None,
        concept_X: Optional[np.ndarray] = None,
        concept_y: Optional[np.ndarray] = None,
    ) -> dict[str, DriftResult]:
        """Check all types of drift.
        
        Args:
            input_data: Current input features
            output_data: Current outputs
            concept_X: Current features for concept drift
            concept_y: Current labels for concept drift
            
        Returns:
            Dictionary of drift results by type
        """
        results = {}
        
        if input_data is not None and self.input_detector.baseline is not None:
            result = self.input_detector.detect(input_data)
            results["input"] = result
            self.history.record(result)
        
        if output_data is not None and self.output_detector.baseline is not None:
            result = self.output_detector.detect(output_data)
            results["output"] = result
            self.history.record(result)
        
        if concept_X is not None and concept_y is not None:
            if self.concept_detector.baseline_X is not None:
                result = self.concept_detector.detect(concept_X, concept_y)
                results["concept"] = result
                self.history.record(result)
        
        self._last_results = results
        return results
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of drift detection status.
        
        Returns:
            Summary dictionary
        """
        any_drift = any(r.drift_detected for r in self._last_results.values())
        
        return {
            "model_name": self.config.model_name,
            "drift_detected": any_drift,
            "checks_performed": list(self._last_results.keys()),
            "results": {k: v.to_dict() for k, v in self._last_results.items()},
            "drift_rate": self.history.get_drift_rate(),
            "timestamp": datetime.now().isoformat(),
        }
