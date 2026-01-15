"""Uncertainty quantification for medical AI safety.

Provides:
- Monte Carlo Dropout uncertainty estimation
- Ensemble-based uncertainty
- Threshold-based escalation
"""

from typing import Any, Optional
import numpy as np


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimator.
    
    Performs multiple forward passes with dropout enabled
    to estimate model uncertainty.
    """
    
    def __init__(self, model: Any, n_samples: int = 30):
        """
        Initialize MC Dropout estimator.
        
        Args:
            model: PyTorch model with dropout layers
            n_samples: Number of forward passes for uncertainty
        """
        self.model = model
        self.n_samples = n_samples
    
    def enable_dropout(self) -> None:
        """Enable dropout during inference for MC sampling."""
        # For PyTorch models, we'd iterate through modules
        # and call train() on Dropout layers
        try:
            for module in self.model.modules():
                if module.__class__.__name__.startswith('Dropout'):
                    module.train()
        except AttributeError:
            pass  # Mock model doesn't have modules()
    
    def estimate_uncertainty(
        self, 
        inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty using MC Dropout.
        
        Args:
            inputs: Input array
            
        Returns:
            Tuple of (mean_prediction, uncertainty_std)
        """
        self.enable_dropout()
        
        predictions = []
        for _ in range(self.n_samples):
            output = self.model(inputs)
            if isinstance(output, np.ndarray):
                predictions.append(output)
            else:
                # Handle PyTorch tensors
                predictions.append(np.array(output))
        
        predictions = np.stack(predictions, axis=0)
        return calculate_uncertainty(predictions)


class EnsembleEstimator:
    """
    Ensemble-based uncertainty estimator.
    
    Uses multiple models to estimate epistemic uncertainty
    through prediction disagreement.
    """
    
    def __init__(self, models: list[Any]):
        """
        Initialize ensemble estimator.
        
        Args:
            models: List of models to ensemble
        """
        self.models = models
    
    def predict_with_uncertainty(
        self, 
        inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Get ensemble predictions with uncertainty metrics.
        
        Args:
            inputs: Input array
            
        Returns:
            Tuple of (mean_prediction, uncertainty, metrics_dict)
        """
        predictions = []
        
        for model in self.models:
            output = model(inputs)
            if isinstance(output, np.ndarray):
                predictions.append(output)
            else:
                predictions.append(np.array(output))
        
        predictions = np.stack(predictions, axis=0)
        mean_pred, uncertainty = calculate_uncertainty(predictions)
        
        # Calculate additional metrics
        metrics = {
            "epistemic_uncertainty": float(np.mean(uncertainty)),
            "max_uncertainty": float(np.max(uncertainty)),
            "model_agreement": self._calculate_agreement(predictions),
        }
        
        return mean_pred, uncertainty, metrics
    
    def _calculate_agreement(self, predictions: np.ndarray) -> float:
        """Calculate inter-model agreement."""
        std = np.std(predictions, axis=0)
        # Agreement is inverse of disagreement
        return float(1.0 / (1.0 + np.mean(std)))


def calculate_uncertainty(
    samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and uncertainty from samples.
    
    Args:
        samples: Array of shape (n_samples, ...) with predictions
        
    Returns:
        Tuple of (mean, std) arrays
    """
    mean_prediction = np.mean(samples, axis=0)
    uncertainty = np.std(samples, axis=0)
    
    return mean_prediction, uncertainty


def should_escalate_uncertainty(
    uncertainty: float,
    threshold: float = 0.20
) -> bool:
    """
    Check if uncertainty exceeds escalation threshold.
    
    Args:
        uncertainty: Uncertainty score (0-1)
        threshold: Escalation threshold
        
    Returns:
        True if should escalate to human review
    """
    return uncertainty > threshold
