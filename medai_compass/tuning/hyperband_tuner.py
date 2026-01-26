"""
Hyperband Tuner.

Multi-fidelity hyperparameter optimization with bracket-based resource allocation.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def create_hyperband_scheduler(
    max_t: int = 5000,
    reduction_factor: int = 3,
    time_attr: str = "training_iteration",
    metric: str = "eval_loss",
    mode: str = "min",
    stop_last_trials: bool = True,
) -> "HyperBandScheduler":
    """
    Create Hyperband scheduler with specified configuration.

    Args:
        max_t: Maximum training iterations
        reduction_factor: Halving rate
        time_attr: Attribute to use for time
        metric: Metric to optimize
        mode: 'min' or 'max'
        stop_last_trials: Whether to stop last trials

    Returns:
        HyperBandScheduler instance
    """
    try:
        from ray.tune.schedulers import HyperBandScheduler
    except ImportError:
        raise ImportError("ray[tune] is required for Hyperband tuning")

    return HyperBandScheduler(
        time_attr=time_attr,
        metric=metric,
        mode=mode,
        max_t=max_t,
        reduction_factor=reduction_factor,
        stop_last_trials=stop_last_trials,
    )


class HyperbandTuner:
    """
    Hyperband-based hyperparameter tuner for MedGemma.

    Features:
    - Multi-fidelity optimization
    - Bracket-based resource allocation
    - Early stopping of poor performers
    """

    def __init__(self, cfg: "DictConfig"):
        """
        Initialize Hyperband tuner from Hydra config.

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.scheduler = self._create_scheduler()
        self.search_alg = self._create_search_algorithm()

    def _create_scheduler(self) -> "HyperBandScheduler":
        """Create Hyperband scheduler from config."""
        hb_cfg = self.cfg.tuning.hyperband if hasattr(self.cfg.tuning, "hyperband") else {}

        return create_hyperband_scheduler(
            max_t=getattr(hb_cfg, "max_t", 5000),
            reduction_factor=getattr(hb_cfg, "reduction_factor", 3),
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
            stop_last_trials=getattr(hb_cfg, "stop_last_trials", True),
        )

    def _create_search_algorithm(self) -> Optional[Any]:
        """Create search algorithm for smart sampling."""
        try:
            import hyperopt  # Check if hyperopt is installed
            from ray.tune.search.hyperopt import HyperOptSearch

            return HyperOptSearch(
                metric=self.cfg.tuning.metric,
                mode=self.cfg.tuning.mode,
            )
        except (ImportError, AssertionError):
            logger.warning("HyperOpt not available, using random search")
            return None

    def run(self, trainable, param_space: Dict[str, Any]) -> "ResultGrid":
        """
        Run Hyperband hyperparameter tuning.

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
                name=f"{self.cfg.project.name}-hyperband",
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
        """Get best result from tuning run."""
        best_result = results.get_best_result(
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
        )

        return {
            "best_config": best_result.config,
            "best_metrics": best_result.metrics,
            "best_checkpoint": best_result.checkpoint,
            "num_trials": len(results),
        }

    def get_bracket_info(self) -> Dict[str, Any]:
        """Get information about Hyperband brackets."""
        from medai_compass.tuning.utils import calculate_hyperband_brackets

        hb_cfg = self.cfg.tuning.hyperband if hasattr(self.cfg.tuning, "hyperband") else {}

        return calculate_hyperband_brackets(
            max_t=getattr(hb_cfg, "max_t", 5000),
            reduction_factor=getattr(hb_cfg, "reduction_factor", 3),
        )
