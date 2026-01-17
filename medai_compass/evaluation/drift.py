"""
Model drift detection for MedAI Compass.

Provides:
- Population Stability Index (PSI) calculation
- Kolmogorov-Smirnov test for distribution comparison
- Comprehensive drift detection with severity assessment
"""

import numpy as np
from scipy import stats


class DriftDetector:
    """
    Detect distribution shift in model inputs and outputs.

    Uses multiple statistical methods to identify when the
    production data distribution differs from the reference
    distribution used during training.
    """

    def __init__(self, reference_data: np.ndarray, threshold: float = 0.1):
        """
        Initialize drift detector with reference distribution.

        Args:
            reference_data: Reference distribution data
            threshold: PSI threshold for drift detection
        """
        self.reference = reference_data.flatten()
        self.threshold = threshold

    def calculate_psi(self, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index.

        PSI measures the shift between two distributions:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change

        Args:
            current: Current distribution data
            bins: Number of bins for histogram

        Returns:
            PSI score
        """
        current = current.flatten()

        # Create bins from reference distribution
        _, bin_edges = np.histogram(self.reference, bins=bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(self.reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        ref_props = ref_counts / len(self.reference)
        cur_props = cur_counts / len(current)

        # Avoid division by zero and log(0)
        ref_props = np.clip(ref_props, 1e-10, 1)
        cur_props = np.clip(cur_props, 1e-10, 1)

        # PSI calculation
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    def ks_test(self, current: np.ndarray) -> tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.

        The KS test measures the maximum distance between two
        cumulative distribution functions.

        Args:
            current: Current distribution data

        Returns:
            Tuple of (KS statistic, p-value)
        """
        current = current.flatten()
        statistic, p_value = stats.ks_2samp(self.reference, current)
        return float(statistic), float(p_value)

    def check_drift(self, current: np.ndarray) -> dict:
        """
        Comprehensive drift detection.

        Combines PSI and KS test for robust drift detection.

        Args:
            current: Current distribution data

        Returns:
            Dictionary with drift metrics and assessment
        """
        psi = self.calculate_psi(current)
        ks_stat, ks_pvalue = self.ks_test(current)

        # Determine if drift detected
        drift_detected = psi > self.threshold or ks_pvalue < 0.05

        # Assess severity
        if psi >= 0.25:
            severity = "high"
        elif psi >= 0.1:
            severity = "medium"
        else:
            severity = "low"

        return {
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "drift_detected": drift_detected,
            "severity": severity
        }


class FeatureDriftMonitor:
    """
    Monitor drift across multiple features.

    Tracks drift for each input feature and provides
    aggregate drift assessment.
    """

    def __init__(self, reference_features: np.ndarray, feature_names: list[str] = None):
        """
        Initialize feature drift monitor.

        Args:
            reference_features: Reference data (N, num_features)
            feature_names: Optional names for features
        """
        self.reference = reference_features
        self.num_features = reference_features.shape[1] if reference_features.ndim > 1 else 1
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.num_features)]

        # Create detectors for each feature
        if reference_features.ndim > 1:
            self.detectors = {
                name: DriftDetector(reference_features[:, i])
                for i, name in enumerate(self.feature_names)
            }
        else:
            self.detectors = {self.feature_names[0]: DriftDetector(reference_features)}

    def check_all_features(self, current_features: np.ndarray) -> dict:
        """
        Check drift for all features.

        Args:
            current_features: Current data (N, num_features)

        Returns:
            Dictionary with per-feature and aggregate results
        """
        feature_results = {}
        drifted_features = []

        if current_features.ndim == 1:
            current_features = current_features.reshape(-1, 1)

        for i, (name, detector) in enumerate(self.detectors.items()):
            result = detector.check_drift(current_features[:, i])
            feature_results[name] = result
            if result["drift_detected"]:
                drifted_features.append(name)

        return {
            "feature_results": feature_results,
            "drifted_features": drifted_features,
            "num_drifted": len(drifted_features),
            "drift_ratio": len(drifted_features) / len(self.feature_names),
            "overall_drift": len(drifted_features) > 0
        }


class PredictionDriftMonitor:
    """
    Monitor drift in model predictions.

    Detects when prediction distributions change significantly,
    which may indicate concept drift or data distribution shift.
    """

    def __init__(self, reference_predictions: np.ndarray):
        """
        Initialize prediction drift monitor.

        Args:
            reference_predictions: Reference prediction distribution
        """
        self.detector = DriftDetector(reference_predictions)
        self.history = []

    def record_batch(self, predictions: np.ndarray) -> dict:
        """
        Record a batch of predictions and check for drift.

        Args:
            predictions: Batch of model predictions

        Returns:
            Drift detection results
        """
        result = self.detector.check_drift(predictions)
        self.history.append({
            "batch_size": len(predictions),
            "mean_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
            **result
        })
        return result

    def get_history(self) -> list[dict]:
        """Get history of drift checks."""
        return self.history.copy()
