"""
ASHA (Asynchronous Successive Halving Algorithm) Tuner.

Provides efficient early stopping for hyperparameter search.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def create_asha_scheduler(
    max_t: int = 5000,
    grace_period: int = 100,
    reduction_factor: int = 3,
    brackets: int = 1,
    time_attr: str = "training_iteration",
    metric: str = "eval_loss",
    mode: str = "min",
    stop_last_trials: bool = True,
) -> "ASHAScheduler":
    """
    Create ASHA scheduler with specified configuration.

    Args:
        max_t: Maximum training iterations
        grace_period: Minimum iterations before stopping
        reduction_factor: Halving rate
        brackets: Number of brackets
        time_attr: Attribute to use for time
        metric: Metric to optimize
        mode: 'min' or 'max'
        stop_last_trials: Whether to stop underperforming trials

    Returns:
        ASHAScheduler instance
    """
    try:
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        raise ImportError("ray[tune] is required for ASHA tuning")

    return ASHAScheduler(
        time_attr=time_attr,
        metric=metric,
        mode=mode,
        max_t=max_t,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        brackets=brackets,
        stop_last_trials=stop_last_trials,
    )


class ASHATuner:
    """
    ASHA-based hyperparameter tuner for MedGemma.

    Features:
    - Asynchronous early stopping
    - Efficient resource utilization
    - Integration with Optuna for smart sampling
    """

    def __init__(self, cfg: "DictConfig"):
        """
        Initialize ASHA tuner from Hydra config.

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.scheduler = self._create_scheduler()
        self.search_alg = self._create_search_algorithm()

    def _create_scheduler(self) -> "ASHAScheduler":
        """Create ASHA scheduler from config."""
        asha_cfg = self.cfg.tuning.asha if hasattr(self.cfg.tuning, "asha") else {}

        return create_asha_scheduler(
            max_t=getattr(asha_cfg, "max_t", 5000),
            grace_period=getattr(asha_cfg, "grace_period", 100),
            reduction_factor=getattr(asha_cfg, "reduction_factor", 3),
            brackets=getattr(asha_cfg, "brackets", 1),
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
            stop_last_trials=getattr(asha_cfg, "stop_last_trials", True),
        )

    def _create_search_algorithm(self) -> Optional[Any]:
        """Create search algorithm for smart sampling."""
        try:
            from ray.tune.search.optuna import OptunaSearch

            return OptunaSearch(
                metric=self.cfg.tuning.metric,
                mode=self.cfg.tuning.mode,
            )
        except ImportError:
            logger.warning("Optuna not available, using random search")
            return None

    def run(self, trainable, param_space: Dict[str, Any]) -> "ResultGrid":
        """
        Run ASHA hyperparameter tuning.

        Args:
            trainable: Ray Tune trainable class or function
            param_space: Parameter search space

        Returns:
            Ray Tune ResultGrid
        """
        import ray
        from ray import tune
        from ray.train import RunConfig, CheckpointConfig

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                scheduler=self.scheduler,
                search_alg=self.search_alg,
                num_samples=self.cfg.tuning.num_samples,
                max_concurrent_trials=self.cfg.tuning.max_concurrent_trials,
                reuse_actors=True,
            ),
            run_config=RunConfig(
                name=f"{self.cfg.project.name}-asha",
                storage_path=self.cfg.project.output_dir,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_score_attribute=self.cfg.tuning.metric,
                    checkpoint_score_order="min" if self.cfg.tuning.mode == "min" else "max",
                ),
            ),
        )

        return tuner.fit()

    def get_best_result(self, results: "ResultGrid") -> Dict[str, Any]:
        """
        Get best result from tuning run.

        Args:
            results: Ray Tune ResultGrid

        Returns:
            Best configuration and metrics
        """
        best_result = results.get_best_result(
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
        )

        return {
            "best_config": best_result.config,
            "best_metrics": best_result.metrics,
            "best_checkpoint": best_result.checkpoint,
            "num_trials": len(results),
            "num_errors": len([r for r in results if r.error]),
        }
