"""MedAI Compass Evaluation Framework.

Provides comprehensive evaluation metrics for:
- Diagnostic imaging (AUC-ROC, sensitivity, Dice, IoU)
- Clinical NLP (ROUGE-L, entity F1, readability)
- Patient communication (appropriateness, harm potential, empathy)
- Model drift detection (PSI, KS test)
- Medical benchmarks (MedQA, PubMedQA, MedMCQA)
- Safety evaluation (jailbreak, injection, PHI, misinformation)
- Fairness evaluation (demographic parity, equalized odds)

Quality Thresholds:
- MedQA Accuracy: >= 75%
- PubMedQA F1: >= 80%
- Hallucination Rate: <= 5%
- Safety Pass Rate: >= 99%
- Fairness Gap: <= 5%
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
from medai_compass.evaluation.benchmarks import (
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkType,
    QualityGateResult,
    QUALITY_THRESHOLDS,
)
from medai_compass.evaluation.safety_eval import (
    SafetyEvaluator,
    SafetyTestResult,
    SafetyCategory,
    SAFETY_PASS_RATE_THRESHOLD,
    create_mock_safe_model,
    create_mock_unsafe_model,
)
from medai_compass.evaluation.fairness import (
    FairnessEvaluator,
    BiasEvaluator,  # Alias for backward compatibility
    BiasTestResult,
    BiasCategory,
    FAIRNESS_GAP_THRESHOLD,
)
# Also export from ai_evaluation for backward compatibility
from medai_compass.evaluation.ai_evaluation import (
    ClinicalAccuracyEvaluator,
    ComprehensiveEvaluator,
    EvaluationReport,
)

__all__ = [
    # Domain-specific Evaluators
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
    # Benchmarks (Phase 6)
    "BenchmarkSuite",
    "BenchmarkResult",
    "BenchmarkType",
    "QualityGateResult",
    "QUALITY_THRESHOLDS",
    # Safety evaluation (Phase 6)
    "SafetyEvaluator",
    "SafetyTestResult",
    "SafetyCategory",
    "SAFETY_PASS_RATE_THRESHOLD",
    "create_mock_safe_model",
    "create_mock_unsafe_model",
    # Fairness evaluation (Phase 6)
    "FairnessEvaluator",
    "BiasEvaluator",
    "BiasTestResult",
    "BiasCategory",
    "FAIRNESS_GAP_THRESHOLD",
    # Clinical and comprehensive (backward compatibility)
    "ClinicalAccuracyEvaluator",
    "ComprehensiveEvaluator",
    "EvaluationReport",
]
