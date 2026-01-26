"""
Training Pipeline Management with Ray Integration.

Provides orchestration for distributed training with algorithm
selection, checkpointing, and monitoring.
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Status of the training pipeline."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CheckpointInfo:
    """Information about a training checkpoint."""
    path: str
    step: int
    epoch: float
    metrics: Dict[str, float]
    timestamp: str
    algorithm: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "step": self.step,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "algorithm": self.algorithm,
        }


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    algorithm: str = "dora"
    model_name: str = "google/medgemma-4b-it"
    output_dir: str = "./training_output"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Ray configuration
    use_ray: bool = True
    num_workers: int = 1
    gpus_per_worker: float = 1.0
    cpus_per_worker: int = 4

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Evaluation
    eval_steps: int = 100
    eval_strategy: str = "steps"

    # Callbacks
    enable_safety_callback: bool = True
    enable_evaluation_callback: bool = True

    # Algorithm-specific config (optional override)
    algorithm_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "use_ray": self.use_ray,
            "num_workers": self.num_workers,
            "gpus_per_worker": self.gpus_per_worker,
            "cpus_per_worker": self.cpus_per_worker,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "eval_steps": self.eval_steps,
            "eval_strategy": self.eval_strategy,
            "enable_safety_callback": self.enable_safety_callback,
            "enable_evaluation_callback": self.enable_evaluation_callback,
            "algorithm_config": self.algorithm_config,
        }


class TrainingPipelineManager:
    """
    Manages the training pipeline lifecycle.

    Handles algorithm selection, trainer creation, checkpointing,
    and monitoring across training runs.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_name: Optional[str] = None,
    ):
        if config is None:
            config = PipelineConfig()
        if model_name:
            config.model_name = model_name

        self.config = config
        self.status = PipelineStatus.PENDING
        self.current_trainer = None
        self.checkpoints: List[CheckpointInfo] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self._callbacks: List[Any] = []
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._current_algorithm: str = config.algorithm

    @property
    def current_algorithm(self) -> str:
        """Get the currently selected algorithm."""
        return self._current_algorithm

    def set_algorithm(self, algorithm_name: str, algorithm_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the training algorithm.

        Args:
            algorithm_name: Name of the algorithm (e.g., 'dora', 'dpo', 'rlhf')
            algorithm_config: Optional algorithm-specific configuration
        """
        from .registry import get_algorithm

        info = get_algorithm(algorithm_name)
        if info is None:
            from .registry import ALGORITHM_REGISTRY
            ALGORITHM_REGISTRY._ensure_initialized()
            available = ALGORITHM_REGISTRY.list_names()
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {available}")

        self._current_algorithm = algorithm_name
        self.config.algorithm = algorithm_name
        if algorithm_config:
            self.config.algorithm_config = algorithm_config

        logger.info(f"Algorithm set to: {algorithm_name}")

    def check_compatibility(self, available_vram_gb: float, num_gpus: int = 1) -> Dict[str, Any]:
        """Check if the current configuration is compatible with hardware."""
        from .registry import check_algorithm_compatibility

        return check_algorithm_compatibility(
            algorithm_name=self.config.algorithm,
            model_name=self.config.model_name,
            available_vram_gb=available_vram_gb,
            num_gpus=num_gpus,
        )

    def add_callback(self, callback: Any) -> None:
        """Add a training callback."""
        self._callbacks.append(callback)

    def initialize(self, model: Any, train_dataset: Any, eval_dataset: Optional[Any] = None) -> None:
        """
        Initialize the training pipeline.

        Args:
            model: The model to train
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        self.status = PipelineStatus.INITIALIZING

        try:
            from .registry import get_algorithm
            from .callbacks import MedicalSafetyCallback, MedicalEvaluationCallback

            # Get algorithm info
            info = get_algorithm(self.config.algorithm)
            if info is None:
                raise ValueError(f"Algorithm not found: {self.config.algorithm}")

            # Create algorithm config
            if self.config.algorithm_config:
                algo_config = info.config_class(**self.config.algorithm_config)
            else:
                algo_config = info.config_class()

            # Create trainer
            self.current_trainer = info.trainer_class(
                model=model,
                config=algo_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            # Add default callbacks
            if self.config.enable_safety_callback:
                self._callbacks.append(MedicalSafetyCallback())

            if self.config.enable_evaluation_callback and eval_dataset is not None:
                self._callbacks.append(MedicalEvaluationCallback())

            # Set callbacks on trainer
            for callback in self._callbacks:
                self.current_trainer.add_callback(callback)

            self.status = PipelineStatus.PENDING
            logger.info(f"Pipeline initialized with algorithm: {self.config.algorithm}")

        except Exception as e:
            self.status = PipelineStatus.FAILED
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the training pipeline.

        Args:
            resume_from_checkpoint: Optional checkpoint path to resume from

        Returns:
            Training results dictionary
        """
        if self.current_trainer is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        self.status = PipelineStatus.RUNNING
        self._start_time = datetime.now()

        try:
            # Run training
            checkpoint = resume_from_checkpoint or self.config.resume_from_checkpoint
            result = self.current_trainer.train(resume_from_checkpoint=checkpoint)

            self.status = PipelineStatus.COMPLETED
            self._end_time = datetime.now()

            # Record final metrics
            if hasattr(result, "metrics"):
                self.metrics_history.append({
                    "type": "final",
                    "metrics": result.metrics,
                    "timestamp": datetime.now().isoformat(),
                })

            return {
                "status": "completed",
                "algorithm": self.config.algorithm,
                "duration": str(self._end_time - self._start_time),
                "metrics": getattr(result, "metrics", {}),
            }

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self._end_time = datetime.now()
            logger.error(f"Training failed: {e}")
            raise

    def save_checkpoint(self, step: int, metrics: Optional[Dict[str, float]] = None) -> str:
        """Save a training checkpoint."""
        if self.current_trainer is None:
            raise RuntimeError("No active trainer")

        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save trainer state
        self.current_trainer.save_model(str(checkpoint_path))

        # Record checkpoint info
        checkpoint_info = CheckpointInfo(
            path=str(checkpoint_path),
            step=step,
            epoch=step / 1000,  # Approximate
            metrics=metrics or {},
            timestamp=datetime.now().isoformat(),
            algorithm=self.config.algorithm,
        )
        self.checkpoints.append(checkpoint_info)

        # Enforce save limit
        if len(self.checkpoints) > self.config.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            # Could delete old checkpoint files here

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "status": self.status.value,
            "algorithm": self.config.algorithm,
            "model": self.config.model_name,
            "checkpoints": len(self.checkpoints),
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
        }

    def export_config(self, path: str) -> None:
        """Export pipeline configuration to JSON."""
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_config_file(cls, path: str) -> "TrainingPipelineManager":
        """Load pipeline from configuration file."""
        with open(path, "r") as f:
            config_dict = json.load(f)

        config = PipelineConfig(**config_dict)
        return cls(config=config)


class RayTrainingOrchestrator:
    """
    Orchestrates distributed training using Ray.

    Manages worker allocation, data distribution, and
    fault tolerance for large-scale training.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_name: Optional[str] = None,
    ):
        if config is None:
            config = PipelineConfig()
        if model_name:
            config.model_name = model_name

        self.config = config
        self._ray_initialized = False
        self._trainer = None
        self._scaling_config = None
        self._algorithm = config.algorithm

    @property
    def algorithm(self) -> str:
        """Get the currently configured algorithm."""
        return self._algorithm

    @property
    def scaling_config(self) -> Optional[Dict[str, Any]]:
        """Get the scaling configuration as a dict."""
        if self._scaling_config is None:
            return None
        # Return as dict for easier access
        return {
            "num_workers": self.config.num_workers,
            "use_gpu": True,
            "gpus_per_worker": self.config.gpus_per_worker,
            "cpus_per_worker": self.config.cpus_per_worker,
        }

    def configure_algorithm(self, algorithm_name: str) -> None:
        """Configure the training algorithm."""
        from .registry import get_algorithm

        info = get_algorithm(algorithm_name)
        if info is None:
            from .registry import ALGORITHM_REGISTRY
            ALGORITHM_REGISTRY._ensure_initialized()
            available = ALGORITHM_REGISTRY.list_names()
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {available}")

        self._algorithm = algorithm_name
        self.config.algorithm = algorithm_name
        logger.info(f"Algorithm configured: {algorithm_name}")

    def initialize_ray(self, **ray_init_kwargs) -> None:
        """Initialize Ray cluster connection."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(**ray_init_kwargs)
            self._ray_initialized = True
            logger.info("Ray initialized successfully")
        except ImportError:
            logger.warning("Ray not installed. Falling back to local training.")
            self._ray_initialized = False

    def create_scaling_config(
        self,
        num_workers: Optional[int] = None,
        use_gpu: bool = True,
        gpus_per_worker: Optional[float] = None,
        cpus_per_worker: Optional[int] = None,
    ) -> Any:
        """Create Ray scaling configuration."""
        try:
            from ray.train import ScalingConfig

            self._scaling_config = ScalingConfig(
                num_workers=num_workers or self.config.num_workers,
                use_gpu=use_gpu,
                resources_per_worker={
                    "GPU": gpus_per_worker or self.config.gpus_per_worker,
                    "CPU": cpus_per_worker or self.config.cpus_per_worker,
                },
            )
            return self._scaling_config

        except ImportError:
            logger.warning("Ray Train not available")
            return None

    def create_trainer(
        self,
        train_func: Callable,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        algorithm_name: Optional[str] = None,
    ) -> Any:
        """
        Create a Ray trainer for distributed training.

        Args:
            train_func: Training function to distribute
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            algorithm_name: Algorithm to use (defaults to config)

        Returns:
            Ray TorchTrainer instance
        """
        if not self._ray_initialized:
            self.initialize_ray()

        try:
            from ray.train.torch import TorchTrainer
            from ray.train import RunConfig, CheckpointConfig

            if self._scaling_config is None:
                self.create_scaling_config()

            # Create checkpoint config
            checkpoint_config = CheckpointConfig(
                num_to_keep=self.config.save_total_limit,
            )

            # Create run config
            run_config = RunConfig(
                name=f"medgemma-{algorithm_name or self.config.algorithm}",
                storage_path=self.config.output_dir,
                checkpoint_config=checkpoint_config,
            )

            # Create trainer
            self._trainer = TorchTrainer(
                train_loop_per_worker=train_func,
                scaling_config=self._scaling_config,
                run_config=run_config,
            )

            logger.info(f"Ray trainer created for {algorithm_name or self.config.algorithm}")
            return self._trainer

        except ImportError as e:
            logger.error(f"Failed to create Ray trainer: {e}")
            raise RuntimeError("Ray Train not installed. Install with: pip install ray[train]")

    def fit(self) -> Any:
        """Run distributed training (alias for run_training)."""
        return self.run_training()

    def run_training(self) -> Any:
        """Run distributed training."""
        if self._trainer is None:
            raise RuntimeError("Trainer not created. Call create_trainer() first.")

        try:
            result = self._trainer.fit()
            logger.info("Distributed training completed")
            return result
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            raise

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get Ray cluster information."""
        if not self._ray_initialized:
            return {"status": "not_initialized"}

        try:
            import ray
            return {
                "status": "initialized",
                "nodes": len(ray.nodes()),
                "resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def shutdown(self) -> None:
        """Shutdown Ray connection."""
        if self._ray_initialized:
            try:
                import ray
                ray.shutdown()
                self._ray_initialized = False
                logger.info("Ray shutdown complete")
            except Exception as e:
                logger.warning(f"Error during Ray shutdown: {e}")


def create_ray_trainer(
    algorithm: str,
    model_name: str = "google/medgemma-4b-it",
    train_dataset: Any = None,
    eval_dataset: Any = None,
    num_workers: int = 1,
    gpus_per_worker: float = 1.0,
    use_gpu: bool = True,
    output_dir: str = "./output",
    **training_kwargs,
) -> RayTrainingOrchestrator:
    """
    Convenience function to create a Ray-distributed trainer.

    Args:
        algorithm: Training algorithm to use
        model_name: Model identifier
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        num_workers: Number of Ray workers
        gpus_per_worker: GPUs per worker
        use_gpu: Whether to use GPU
        output_dir: Output directory
        **training_kwargs: Additional training arguments

    Returns:
        Configured RayTrainingOrchestrator
    """
    from .registry import get_algorithm

    # Validate algorithm
    info = get_algorithm(algorithm)
    if info is None:
        from .registry import ALGORITHM_REGISTRY
        ALGORITHM_REGISTRY._ensure_initialized()
        available = ALGORITHM_REGISTRY.list_names()
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")

    # Create config
    config = PipelineConfig(
        algorithm=algorithm,
        model_name=model_name,
        output_dir=output_dir,
        num_workers=num_workers,
        gpus_per_worker=gpus_per_worker,
        **{k: v for k, v in training_kwargs.items() if hasattr(PipelineConfig, k)},
    )

    # Create orchestrator
    orchestrator = RayTrainingOrchestrator(config=config)

    # Initialize scaling config
    orchestrator.create_scaling_config(
        num_workers=num_workers,
        use_gpu=use_gpu,
        gpus_per_worker=gpus_per_worker,
    )

    logger.info(f"Created Ray trainer for {algorithm} with {num_workers} workers")
    return orchestrator
