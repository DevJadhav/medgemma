"""
MedGemma ML Workflows Package.

Provides Ray Workflow-based ML pipelines for:
- End-to-end training pipelines
- Evaluation workflows
- Deployment workflows
- Long-running Ray actors for background processing
"""

from medai_compass.workflows.training_workflow import (
    TrainingWorkflow,
    WorkflowStepResult,
    create_training_workflow,
    run_training_pipeline,
)
from medai_compass.workflows.actors import (
    EvaluationActor,
    MetricsAggregator,
    CheckpointManager,
)
from medai_compass.workflows.ray_actors import (
    EvaluationResult,
    MetricEntry,
    CheckpointInfo,
    RayActorManager,
    create_ray_evaluation_actor,
    create_ray_metrics_aggregator,
    create_ray_checkpoint_manager,
    create_ray_health_monitor,
)

__all__ = [
    # Training Workflow
    "TrainingWorkflow",
    "WorkflowStepResult",
    "create_training_workflow",
    "run_training_pipeline",
    # Legacy Actors (non-Ray)
    "EvaluationActor",
    "MetricsAggregator",
    "CheckpointManager",
    # Ray Actors (production)
    "EvaluationResult",
    "MetricEntry",
    "CheckpointInfo",
    "RayActorManager",
    "create_ray_evaluation_actor",
    "create_ray_metrics_aggregator",
    "create_ray_checkpoint_manager",
    "create_ray_health_monitor",
]
