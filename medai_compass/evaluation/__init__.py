"""MedAI Compass Evaluation Framework.

Provides comprehensive evaluation metrics for:
- Diagnostic imaging (AUC-ROC, sensitivity, Dice, IoU)
- Clinical NLP (ROUGE-L, entity F1, readability)
- Patient communication (appropriateness, harm potential, empathy)
- Model drift detection (PSI, KS test)
"""

from medai_compass.evaluation.metrics import (
    DiagnosticEvaluator,
    NLPEvaluator,
    CommunicationEvaluator,
    MetricsCollector,
    DiagnosticEvaluationResult,
    NLPEvaluationResult,
    CommunicationEvaluationResult,
)
from medai_compass.evaluation.drift import (
    DriftDetector,
    FeatureDriftMonitor,
    PredictionDriftMonitor,
)

__all__ = [
    # Evaluators
    "DiagnosticEvaluator",
    "NLPEvaluator",
    "CommunicationEvaluator",
    "MetricsCollector",
    # Result types
    "DiagnosticEvaluationResult",
    "NLPEvaluationResult",
    "CommunicationEvaluationResult",
    # Drift detection
    "DriftDetector",
    "FeatureDriftMonitor",
    "PredictionDriftMonitor",
]
