"""
PBT (Population Based Training) Tuner.

Adapts hyperparameters during training for continuous optimization.
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def create_pbt_scheduler(
    perturbation_interval: int = 100,
    hyperparam_mutations: Optional[Dict[str, Any]] = None,
    quantile_fraction: float = 0.25,
    resample_probability: float = 0.25,
    time_attr: str = "training_iteration",
    metric: str = "eval_loss",
    mode: str = "min",
    synch: bool = False,
) -> "PopulationBasedTraining":
    """
    Create PBT scheduler with specified configuration.

    Args:
        perturbation_interval: Steps between perturbations
        hyperparam_mutations: Dict of hyperparams to mutate
        quantile_fraction: Bottom % to replace
        resample_probability: Probability of resampling vs perturbing
        time_attr: Attribute to use for time
        metric: Metric to optimize
        mode: 'min' or 'max'
        synch: Whether to synchronize workers

    Returns:
        PopulationBasedTraining scheduler
    """
    try:
        from ray.tune.schedulers import PopulationBasedTraining
    except ImportError:
        raise ImportError("ray[tune] is required for PBT tuning")

    if hyperparam_mutations is None:
        hyperparam_mutations = {}

    return PopulationBasedTraining(
        time_attr=time_attr,
        metric=metric,
        mode=mode,
        perturbation_interval=perturbation_interval,
        quantile_fraction=quantile_fraction,
        resample_probability=resample_probability,
        hyperparam_mutations=hyperparam_mutations,
        log_config=True,
        require_attrs=True,
        synch=synch,
    )


def create_pbt_explore_fn(
    learning_rate_range: Tuple[float, float] = (1e-6, 1e-3),
    weight_decay_range: Tuple[float, float] = (1e-5, 0.1),
    perturbation_factors: Optional[List[float]] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create custom exploration function for PBT.

    Args:
        learning_rate_range: Min and max learning rate
        weight_decay_range: Min and max weight decay
        perturbation_factors: List of multiplication factors

    Returns:
        Exploration function
    """
    if perturbation_factors is None:
        perturbation_factors = [0.5, 0.8, 1.0, 1.25, 2.0]

    def explore_fn(config: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb configuration for exploration."""
        new_config = config.copy()

        # Perturb learning rate (log scale)
        if "learning_rate" in config:
            factor = random.choice(perturbation_factors)
            new_lr = config["learning_rate"] * factor
            new_lr = np.clip(new_lr, learning_rate_range[0], learning_rate_range[1])
            new_config["learning_rate"] = float(new_lr)

        # Perturb weight decay (log scale)
        if "weight_decay" in config:
            factor = random.choice(perturbation_factors)
            new_wd = config["weight_decay"] * factor
            new_wd = np.clip(new_wd, weight_decay_range[0], weight_decay_range[1])
            new_config["weight_decay"] = float(new_wd)

        # Perturb LoRA rank (discrete)
        if "lora_r" in config:
            ranks = [8, 16, 32, 64, 128]
            if random.random() < 0.3:
                current_idx = ranks.index(config["lora_r"]) if config["lora_r"] in ranks else 2
                delta = random.choice([-1, 0, 1])
                new_idx = max(0, min(len(ranks) - 1, current_idx + delta))
                new_config["lora_r"] = ranks[new_idx]
                new_config["lora_alpha"] = ranks[new_idx] * 2

        return new_config

    return explore_fn


class PBTTuner:
    """
    PBT-based hyperparameter tuner for MedGemma.

    Features:
    - Adaptive hyperparameter optimization
    - Population-based exploration
    - Continuous training with perturbation
    """

    def __init__(self, cfg: "DictConfig"):
        """
        Initialize PBT tuner from Hydra config.

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self) -> "PopulationBasedTraining":
        """Create PBT scheduler from config."""
        from medai_compass.tuning.utils import config_to_mutation_space

        pbt_cfg = self.cfg.tuning.pbt if hasattr(self.cfg.tuning, "pbt") else {}

        # Get mutation space
        mutations = {}
        if hasattr(pbt_cfg, "hyperparam_mutations") and pbt_cfg.hyperparam_mutations is not None:
            mutations = config_to_mutation_space(pbt_cfg.hyperparam_mutations)
        else:
            # Default mutations
            try:
                from ray import tune
                mutations = {
                    "learning_rate": tune.loguniform(1e-6, 1e-3),
                    "weight_decay": tune.loguniform(1e-5, 0.1),
                }
            except ImportError:
                pass

        return create_pbt_scheduler(
            perturbation_interval=getattr(pbt_cfg, "perturbation_interval", 100),
            hyperparam_mutations=mutations,
            quantile_fraction=getattr(pbt_cfg, "quantile_fraction", 0.25),
            resample_probability=getattr(pbt_cfg, "resample_probability", 0.25),
            metric=self.cfg.tuning.metric,
            mode=self.cfg.tuning.mode,
        )

    def run(self, trainable, param_space: Dict[str, Any]) -> "ResultGrid":
        """
        Run PBT hyperparameter tuning.

        Args:
            trainable: Ray Tune trainable class or function
            param_space: Parameter search space

        Returns:
            Ray Tune ResultGrid
        """
        import ray
        from ray import tune
        from ray.train import RunConfig, CheckpointConfig, FailureConfig

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                scheduler=self.scheduler,
                num_samples=self.cfg.tuning.num_samples,
                max_concurrent_trials=self.cfg.tuning.max_concurrent_trials,
                reuse_actors=True,
            ),
            run_config=RunConfig(
                name=f"{self.cfg.project.name}-pbt",
                storage_path=self.cfg.project.output_dir,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute=self.cfg.tuning.metric,
                    checkpoint_score_order="min" if self.cfg.tuning.mode == "min" else "max",
                ),
                failure_config=FailureConfig(
                    max_failures=5,
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

        # Collect population history
        population_history = []
        for result in results:
            if not result.error:
                population_history.append({
                    "trial_id": result.metrics.get("trial_id"),
                    "final_loss": result.metrics.get("eval_loss"),
                    "config": result.config,
                })

        return {
            "best_config": best_result.config,
            "best_metrics": best_result.metrics,
            "best_checkpoint": best_result.checkpoint,
            "population_history": population_history,
            "num_perturbations": results.num_terminated if hasattr(results, "num_terminated") else 0,
        }
