"""MedGemma Training Module.

Provides distributed training infrastructure for MedGemma 4B IT and 27B IT models
using Ray Train, PEFT (LoRA/QLoRA), and MLflow for experiment tracking.

Modules:
- ray_trainer: Ray Train distributed training orchestration
- model_selector: Model selection and configuration for 4B/27B variants
- data_pipeline: Ray Data integration for training data
- checkpoint_manager: MinIO/S3 checkpoint management
- metrics: Training metrics and MLflow integration
- pipeline: High-level training pipeline API
- callbacks: Custom training callbacks

Example:
    >>> from medai_compass.training import MedGemmaTrainer, TrainingConfig
    >>> config = TrainingConfig(model_name="medgemma-4b", max_steps=1000)
    >>> trainer = MedGemmaTrainer(config)
    >>> result = trainer.train()
"""

from medai_compass.training.ray_trainer import (
    MedGemmaTrainer,
    TrainingConfig,
    TrainingResult,
)
from medai_compass.training.model_selector import (
    select_model,
    get_training_config,
    validate_gpu_requirements,
    get_available_models,
    MODEL_ALIASES,
)
from medai_compass.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
)
from medai_compass.training.metrics import (
    MetricsTracker,
    MetricSnapshot,
)
from medai_compass.training.pipeline import (
    run_training_pipeline,
)
from medai_compass.training.data_pipeline import (
    MedicalDataPipeline,
    DataConfig,
)
from medai_compass.training.callbacks import (
    CallbackBase,
    MLflowCallback,
    EarlyStoppingCallback,
    GradientAccumulationCallback,
    CheckpointCallback,
    CompositeCallback,
    create_default_callbacks,
)

# H100 Optimized Training Components
from medai_compass.training.optimized import (
    # Configuration
    H100TrainingConfig,
    # Ray Data Integration
    RayDataDICOMLoader,
    RayDICOMDataset,
    # FSDP Training
    FSDPTrainer,
    # Optimized Trainer
    OptimizedTrainer,
    # Throughput Tracking
    ThroughputTracker,
    # DICOM Training Pipeline
    DICOMTrainingPipeline,
    # H100 Optimizations
    H100Optimizer,
    # End-to-End Pipeline
    OptimizedTrainingPipeline,
)

__all__ = [
    # Core trainer
    "MedGemmaTrainer",
    "TrainingConfig",
    "TrainingResult",
    # Model selection
    "select_model",
    "get_training_config",
    "validate_gpu_requirements",
    "get_available_models",
    "MODEL_ALIASES",
    # Checkpointing
    "CheckpointManager",
    "CheckpointMetadata",
    # Metrics
    "MetricsTracker",
    "MetricSnapshot",
    # Pipeline
    "run_training_pipeline",
    # Data
    "MedicalDataPipeline",
    "DataConfig",
    # Callbacks
    "CallbackBase",
    "MLflowCallback",
    "EarlyStoppingCallback",
    "GradientAccumulationCallback",
    "CheckpointCallback",
    "CompositeCallback",
    "create_default_callbacks",
    # H100 Optimized Training
    "H100TrainingConfig",
    "RayDataDICOMLoader",
    "RayDICOMDataset",
    "FSDPTrainer",
    "OptimizedTrainer",
    "ThroughputTracker",
    "DICOMTrainingPipeline",
    "H100Optimizer",
    "OptimizedTrainingPipeline",
]

# Model profile constants for easy access
MODEL_PROFILES = {
    "medgemma-4b": {
        "hf_model_id": "google/medgemma-4b-it",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_length": 8192,
        "distributed_strategy": "single_gpu",
        "gpu_count": 1,
    },
    "medgemma-27b": {
        "hf_model_id": "google/medgemma-27b-it",
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "lora_r": 64,
        "lora_alpha": 128,
        "max_seq_length": 4096,
        "distributed_strategy": "deepspeed_zero3",
        "gpu_count": 8,
    },
}
