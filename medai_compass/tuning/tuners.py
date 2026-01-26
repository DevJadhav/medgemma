"""Ray Tune hyperparameter tuners for MedGemma training."""

import os
from typing import Any, Dict, Optional
import logging

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ASHATuner:
    """
    ASHA (Asynchronous Successive Halving Algorithm) hyperparameter tuner.

    ASHA is ideal for:
    - Large search spaces where you want quick initial results
    - When you can evaluate trials at intermediate points
    - When trials are independent and can be stopped early
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize ASHA tuner.

        Args:
            cfg: Configuration with tuning settings.
        """
        self.cfg = cfg
        self.scheduler = None
        self.search_alg = None

    def _create_scheduler(self):
        """Create ASHA scheduler."""
        from ray.tune.schedulers import ASHAScheduler

        asha_cfg = self.cfg.tuning.asha
        return ASHAScheduler(
            time_attr="training_iteration",
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
            max_t=asha_cfg.max_t,
            grace_period=asha_cfg.grace_period,
            reduction_factor=asha_cfg.reduction_factor,
            brackets=asha_cfg.brackets,
            stop_last_trials=asha_cfg.stop_last_trials,
        )

    def _create_search_algorithm(self):
        """Create search algorithm based on configuration."""
        search_alg_name = self.cfg.tuning.get("search_algorithm", "optuna")

        if search_alg_name == "optuna":
            from ray.tune.search.optuna import OptunaSearch

            return OptunaSearch(
                metric=self.cfg.tuning.metric,
                mode=self.cfg.tuning.mode,
            )
        elif search_alg_name == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch

            return HyperOptSearch(
                metric=self.cfg.tuning.metric,
                mode=self.cfg.tuning.mode,
            )
        else:
            # Random search (no search algorithm)
            return None

    def run(self, trainable=None) -> Any:
        """
        Run ASHA hyperparameter tuning.

        Args:
            trainable: Optional trainable class. If None, uses default.

        Returns:
            Ray Tune ResultGrid.
        """
        import ray
        from ray import tune, train
        from medai_compass.tuning.trainable import create_trainable_class
        from medai_compass.tuning.utils import search_space_to_ray

        if trainable is None:
            base_config = OmegaConf.to_container(self.cfg, resolve=True)
            trainable = create_trainable_class(base_config)

        # Create search space
        search_space = search_space_to_ray(
            OmegaConf.to_container(self.cfg.tuning.search_space, resolve=True)
        )

        # Add fixed config values
        search_space["model_name"] = self.cfg.model.name
        search_space["output_dir"] = self.cfg.project.output_dir

        # Create scheduler and search algorithm
        scheduler = self._create_scheduler()
        search_alg = self._create_search_algorithm()

        # Create tuner
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=self.cfg.tuning.num_samples,
                max_concurrent_trials=self.cfg.tuning.max_concurrent_trials,
                reuse_actors=True,
            ),
            run_config=train.RunConfig(
                name=f"{self.cfg.project.name}-asha",
                storage_path=self.cfg.tuning.storage_path,
                stop={"training_iteration": self.cfg.tuning.asha.max_t},
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=self.cfg.tuning.checkpoint.num_to_keep,
                    checkpoint_score_attribute=self.cfg.tuning.metric,
                    checkpoint_score_order=self.cfg.tuning.mode,
                ),
                failure_config=train.FailureConfig(
                    max_failures=self.cfg.tuning.failure.max_failures,
                ),
            ),
        )

        logger.info(f"Starting ASHA tuning with {self.cfg.tuning.num_samples} samples")
        return tuner.fit()


class PBTTuner:
    """
    Population Based Training (PBT) hyperparameter tuner.

    PBT is ideal for:
    - Long training runs where hyperparameters should adapt
    - When you want to exploit good configurations early
    - When checkpointing is available
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize PBT tuner.

        Args:
            cfg: Configuration with tuning settings.
        """
        self.cfg = cfg

    def _create_scheduler(self):
        """Create PBT scheduler."""
        from ray.tune.schedulers import PopulationBasedTraining
        from medai_compass.tuning.utils import mutations_to_ray

        pbt_cfg = self.cfg.tuning.pbt

        # Convert mutation config to Ray format
        hyperparam_mutations = mutations_to_ray(
            OmegaConf.to_container(
                self.cfg.tuning.get("hyperparam_mutations", {}),
                resolve=True,
            )
        )

        return PopulationBasedTraining(
            time_attr="training_iteration",
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
            perturbation_interval=pbt_cfg.perturbation_interval,
            quantile_fraction=pbt_cfg.quantile_fraction,
            resample_probability=pbt_cfg.resample_probability,
            hyperparam_mutations=hyperparam_mutations,
            log_config=pbt_cfg.log_config,
            synch=pbt_cfg.synch,
        )

    def run(self, trainable=None) -> Any:
        """
        Run PBT hyperparameter tuning.

        Args:
            trainable: Optional trainable class.

        Returns:
            Ray Tune ResultGrid.
        """
        import ray
        from ray import tune, train
        from medai_compass.tuning.trainable import create_trainable_class
        from medai_compass.tuning.utils import search_space_to_ray

        if trainable is None:
            base_config = OmegaConf.to_container(self.cfg, resolve=True)
            trainable = create_trainable_class(base_config)

        # Create search space
        search_space = search_space_to_ray(
            OmegaConf.to_container(self.cfg.tuning.search_space, resolve=True)
        )
        search_space["model_name"] = self.cfg.model.name
        search_space["output_dir"] = self.cfg.project.output_dir

        scheduler = self._create_scheduler()

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=self.cfg.tuning.num_samples,
                max_concurrent_trials=self.cfg.tuning.max_concurrent_trials,
                reuse_actors=True,
            ),
            run_config=train.RunConfig(
                name=f"{self.cfg.project.name}-pbt",
                storage_path=self.cfg.tuning.storage_path,
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=self.cfg.tuning.checkpoint.num_to_keep,
                    checkpoint_score_attribute=self.cfg.tuning.metric,
                    checkpoint_score_order=self.cfg.tuning.mode,
                ),
                failure_config=train.FailureConfig(
                    max_failures=self.cfg.tuning.failure.max_failures,
                ),
            ),
        )

        logger.info(
            f"Starting PBT tuning with population size {self.cfg.tuning.num_samples}"
        )
        return tuner.fit()


class HyperbandTuner:
    """
    Hyperband hyperparameter tuner.

    Hyperband is ideal for:
    - Resource-efficient exploration
    - When you need structured bracket-based scheduling
    - Multi-fidelity optimization
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize Hyperband tuner.

        Args:
            cfg: Configuration with tuning settings.
        """
        self.cfg = cfg

    def _create_scheduler(self):
        """Create Hyperband scheduler."""
        from ray.tune.schedulers import HyperBandScheduler

        hb_cfg = self.cfg.tuning.hyperband
        return HyperBandScheduler(
            time_attr="training_iteration",
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
            max_t=hb_cfg.max_t,
            reduction_factor=hb_cfg.reduction_factor,
            stop_last_trials=hb_cfg.stop_last_trials,
        )

    def _create_search_algorithm(self):
        """Create search algorithm for Hyperband."""
        search_alg_name = self.cfg.tuning.get("search_algorithm", "hyperopt")

        if search_alg_name == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch

            return HyperOptSearch(
                metric=self.cfg.tuning.metric,
                mode=self.cfg.tuning.mode,
            )
        elif search_alg_name == "optuna":
            from ray.tune.search.optuna import OptunaSearch

            return OptunaSearch(
                metric=self.cfg.tuning.metric,
                mode=self.cfg.tuning.mode,
            )
        else:
            return None

    def run(self, trainable=None) -> Any:
        """
        Run Hyperband hyperparameter tuning.

        Args:
            trainable: Optional trainable class.

        Returns:
            Ray Tune ResultGrid.
        """
        import ray
        from ray import tune, train
        from medai_compass.tuning.trainable import create_trainable_class
        from medai_compass.tuning.utils import search_space_to_ray

        if trainable is None:
            base_config = OmegaConf.to_container(self.cfg, resolve=True)
            trainable = create_trainable_class(base_config)

        search_space = search_space_to_ray(
            OmegaConf.to_container(self.cfg.tuning.search_space, resolve=True)
        )
        search_space["model_name"] = self.cfg.model.name
        search_space["output_dir"] = self.cfg.project.output_dir

        scheduler = self._create_scheduler()
        search_alg = self._create_search_algorithm()

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=self.cfg.tuning.num_samples,
                max_concurrent_trials=self.cfg.tuning.max_concurrent_trials,
                reuse_actors=True,
            ),
            run_config=train.RunConfig(
                name=f"{self.cfg.project.name}-hyperband",
                storage_path=self.cfg.tuning.storage_path,
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=self.cfg.tuning.checkpoint.num_to_keep,
                    checkpoint_score_attribute=self.cfg.tuning.metric,
                    checkpoint_score_order=self.cfg.tuning.mode,
                ),
            ),
        )

        logger.info(
            f"Starting Hyperband tuning with {self.cfg.tuning.num_samples} samples"
        )
        return tuner.fit()


def run_hyperparameter_tuning(cfg: DictConfig, trainable=None) -> Dict[str, Any]:
    """
    Run hyperparameter tuning based on configuration.

    Args:
        cfg: Configuration with tuning settings.
        trainable: Optional trainable class.

    Returns:
        Dictionary with best configuration and metrics.

    Example:
        >>> cfg = load_config_with_overrides(["tuning=asha"])
        >>> result = run_hyperparameter_tuning(cfg)
        >>> print(result["best_config"])
    """
    from medai_compass.tuning.utils import get_best_trial_config

    scheduler = cfg.tuning.scheduler

    if scheduler == "asha":
        tuner = ASHATuner(cfg)
    elif scheduler == "pbt":
        tuner = PBTTuner(cfg)
    elif scheduler == "hyperband":
        tuner = HyperbandTuner(cfg)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    results = tuner.run(trainable)

    # Extract best result
    best = get_best_trial_config(
        results,
        metric=cfg.tuning.metric,
        mode=cfg.tuning.mode,
    )

    return {
        "best_config": best.get("config", {}),
        "best_metrics": best.get("metrics", {}),
        "best_checkpoint": best.get("checkpoint"),
        "num_trials": len(results),
        "num_errors": len([r for r in results if r.error]),
        "results": results,
    }


def suggest_scheduler(
    search_space_size: int,
    max_training_hours: float,
    gpu_budget: int,
    checkpointing_supported: bool = True,
) -> str:
    """
    Suggest the best scheduler based on constraints.

    Args:
        search_space_size: Approximate number of configurations to try.
        max_training_hours: Maximum hours for tuning.
        gpu_budget: Number of GPUs available.
        checkpointing_supported: Whether checkpointing is available.

    Returns:
        Recommended scheduler name ("asha", "pbt", or "hyperband").

    Example:
        >>> scheduler = suggest_scheduler(
        ...     search_space_size=100,
        ...     max_training_hours=12,
        ...     gpu_budget=8,
        ... )
        >>> print(scheduler)
        "asha"
    """
    # Large search space with limited time -> ASHA for quick pruning
    if search_space_size > 100 and max_training_hours < 12:
        return "asha"

    # Long training with good resources and checkpointing -> PBT
    if (
        max_training_hours > 24
        and gpu_budget >= 8
        and checkpointing_supported
    ):
        return "pbt"

    # Limited resources -> Hyperband for efficiency
    if gpu_budget < 4:
        return "hyperband"

    # Medium-sized search, moderate resources -> ASHA
    if search_space_size <= 50:
        return "asha"

    # Default to ASHA
    return "asha"
