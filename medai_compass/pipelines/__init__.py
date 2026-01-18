"""Data pipelines for MedGemma training.

This module provides distributed data processing pipelines using Ray Data,
with validation, tokenization, PHI detection, versioning, and quality monitoring.

Modules:
- ray_pipeline: Ray Data distributed pipeline for medical data
- validation: Schema validation and data quality checks
- tokenization: Model-aware tokenization for MedGemma 4B/27B
- phi_detection: PHI detection and filtering (strict mode)
- versioning: DVC-based data versioning
- quality: Data quality monitoring with Great Expectations

Example:
    >>> from medai_compass.pipelines import MedicalDataPipeline
    >>> pipeline = MedicalDataPipeline(model_name="medgemma-4b")
    >>> dataset = pipeline.load_dataset("data/synthea")
    >>> train_ds, val_ds = pipeline.create_splits(dataset)
"""

from medai_compass.pipelines.ray_pipeline import (
    MedicalDataPipeline,
    PipelineConfig,
)
from medai_compass.pipelines.validation import (
    DataValidator,
    ValidationResult,
    MedicalRecordSchema,
)
from medai_compass.pipelines.tokenization import (
    MedicalTokenizer,
    TokenizationConfig,
)
from medai_compass.pipelines.phi_detection import (
    PHIPipelineFilter,
    PHIFilterConfig,
    PHIFilterResult,
)
from medai_compass.pipelines.versioning import (
    DataVersionManager,
    DatasetVersion,
)
from medai_compass.pipelines.quality import (
    DataQualityMonitor,
    QualityReport,
)
from medai_compass.pipelines.lora_trainer import (
    LoRAConfig,
    LoRATrainer,
    QLoRATrainer,
)
from medai_compass.pipelines.training_pipeline import (
    TrainingPipelineConfig,
    TrainingPipelineOrchestrator,
    prepare_data_task,
    train_model_task,
    evaluate_model_task,
    register_model_task,
)
from medai_compass.pipelines.mlflow_integration import (
    ExperimentConfig,
    MLflowTracker,
)

__all__ = [
    # Ray pipeline
    "MedicalDataPipeline",
    "PipelineConfig",
    # Validation
    "DataValidator",
    "ValidationResult",
    "MedicalRecordSchema",
    # Tokenization
    "MedicalTokenizer",
    "TokenizationConfig",
    # PHI Detection
    "PHIPipelineFilter",
    "PHIFilterConfig",
    "PHIFilterResult",
    # Versioning
    "DataVersionManager",
    "DatasetVersion",
    # Quality
    "DataQualityMonitor",
    "QualityReport",
    # Training - LoRA/QLoRA
    "LoRAConfig",
    "LoRATrainer",
    "QLoRATrainer",
    # Training Pipeline
    "TrainingPipelineConfig",
    "TrainingPipelineOrchestrator",
    "prepare_data_task",
    "train_model_task",
    "evaluate_model_task",
    "register_model_task",
    # MLflow Integration
    "ExperimentConfig",
    "MLflowTracker",
]
