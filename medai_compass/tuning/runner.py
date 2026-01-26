"""
Unified Tuning Runner.

Provides a single interface for running hyperparameter tuning
with any scheduler (ASHA, PBT, Hyperband).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TuningRunner:
    """
    Unified tuning runner for MedGemma.

    Supports:
    - ASHA (Asynchronous Successive Halving)
    - PBT (Population Based Training)
    - Hyperband (Multi-fidelity optimization)
    """

    def __init__(self, cfg: "DictConfig"):
        """
        Initialize tuning runner from Hydra config.

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.scheduler_type = cfg.tuning.scheduler
        self.tuner = self._create_tuner()

    def _create_tuner(self):
        """Create appropriate tuner based on scheduler type."""
        if self.scheduler_type == "asha":
            from medai_compass.tuning.asha_tuner import ASHATuner
            return ASHATuner(self.cfg)

        elif self.scheduler_type == "pbt":
            from medai_compass.tuning.pbt_tuner import PBTTuner
            return PBTTuner(self.cfg)

        elif self.scheduler_type == "hyperband":
            from medai_compass.tuning.hyperband_tuner import HyperbandTuner
            return HyperbandTuner(self.cfg)

        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def get_search_space(self) -> Dict[str, Any]:
        """Get search space from config."""
        from medai_compass.tuning.utils import config_to_search_space

        if hasattr(self.cfg.tuning, "search_space"):
            return config_to_search_space(self.cfg.tuning.search_space)

        # Default search space
        try:
            from ray import tune

            return {
                "learning_rate": tune.loguniform(1e-5, 5e-4),
                "lora_r": tune.choice([8, 16, 32, 64, 128]),
                "per_device_train_batch_size": tune.choice([1, 2, 4]),
                "warmup_ratio": tune.uniform(0.01, 0.1),
                "weight_decay": tune.loguniform(1e-4, 0.1),
            }
        except ImportError:
            return {}

    def get_ray_config(self) -> Dict[str, Any]:
        """Get Ray Tune configuration."""
        from ray.train import RunConfig, CheckpointConfig

        search_space = self.get_search_space()

        # Add fixed config values
        search_space["model_name"] = self.cfg.model.name
        search_space["output_dir"] = self.cfg.project.output_dir

        tune_config = {
            "scheduler": self.tuner.scheduler,
            "num_samples": self.cfg.tuning.num_samples,
            "max_concurrent_trials": self.cfg.tuning.max_concurrent_trials,
        }

        run_config = RunConfig(
            name=f"{self.cfg.project.name}-{self.scheduler_type}",
            storage_path=self.cfg.project.output_dir,
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute=self.cfg.tuning.metric,
                checkpoint_score_order="min" if self.cfg.tuning.mode == "min" else "max",
            ),
        )

        return {
            "param_space": search_space,
            "tune_config": tune_config,
            "run_config": run_config,
        }

    def run(self, trainable=None) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            trainable: Optional custom trainable. If not provided,
                      uses MedGemmaTrainable.

        Returns:
            Best configuration and metrics
        """
        if trainable is None:
            from medai_compass.tuning.trainable import MedGemmaTrainable
            trainable = MedGemmaTrainable

        search_space = self.get_search_space()

        # Add fixed values
        search_space["model_name"] = self.cfg.model.name
        search_space["output_dir"] = self.cfg.project.output_dir

        results = self.tuner.run(trainable, search_space)
        return self.tuner.get_best_result(results)

    @staticmethod
    def suggest_scheduler(
        search_space_size: int,
        max_training_hours: float,
        gpu_budget: int,
        is_long_running: bool = False,
    ) -> str:
        """
        Suggest best scheduler for given constraints.

        Args:
            search_space_size: Number of configurations to explore
            max_training_hours: Maximum hours for tuning
            gpu_budget: Number of GPUs available
            is_long_running: Whether trials are long (>1 hour)

        Returns:
            Recommended scheduler name
        """
        from medai_compass.tuning.utils import suggest_scheduler

        return suggest_scheduler(
            search_space_size,
            max_training_hours,
            gpu_budget,
            is_long_running,
        )


def run_hyperparameter_tuning(cfg: "DictConfig") -> Dict[str, Any]:
    """
    Run hyperparameter tuning from Hydra config.

    This is the main entry point for tuning.

    Args:
        cfg: Hydra configuration

    Returns:
        Best configuration and metrics
    """
    import ray

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    runner = TuningRunner(cfg)
    return runner.run()
