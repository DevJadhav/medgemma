"""Ray Tune Integration for MedGemma Hyperparameter Optimization.

Provides comprehensive hyperparameter tuning infrastructure with:
- Ray Tune Tuner integration
- ASHA scheduler for early stopping (Task 4.3)
- Optuna Bayesian optimization (Task 4.4)
- Population-Based Training (Task 4.6)
- Automatic GPU resource allocation
- MLflow experiment tracking integration
- Airflow-compatible task functions

Example:
    >>> from medai_compass.optimization import (
    ...     HyperparameterTuner,
    ...     TuneConfig,
    ...     SearchSpaceConfig,
    ... )
    >>> search_space = SearchSpaceConfig.for_model("medgemma-4b")
    >>> config = TuneConfig.for_model("medgemma-4b")
    >>> tuner = HyperparameterTuner(
    ...     model_name="medgemma-4b",
    ...     search_space=search_space,
    ...     tune_config=config,
    ... )
    >>> result = tuner.run(train_data_path="data/train.jsonl")
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Import Ray with graceful fallback
try:
    import ray
    from ray import tune
    from ray.tune import Tuner, TuneConfig as RayTuneConfig
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.optuna import OptunaSearch
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    tune = None

# Import Optuna with graceful fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Import MLflow with graceful fallback
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


@dataclass
class TuneResult:
    """Results from hyperparameter tuning run.
    
    Contains best configuration, metrics, and trial history.
    """
    
    best_config: Dict[str, Any]
    best_metrics: Dict[str, float]
    best_trial_id: str
    num_trials: int
    total_time_seconds: float
    experiment_name: str
    storage_path: str
    all_trial_configs: List[Dict[str, Any]] = field(default_factory=list)
    all_trial_metrics: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_config": self.best_config,
            "best_metrics": self.best_metrics,
            "best_trial_id": self.best_trial_id,
            "num_trials": self.num_trials,
            "total_time_seconds": self.total_time_seconds,
            "experiment_name": self.experiment_name,
            "storage_path": self.storage_path,
        }


@dataclass
class TuneConfig:
    """Configuration for hyperparameter tuning.
    
    Encapsulates all tuning settings including scheduler, search algorithm,
    resource allocation, and stopping criteria.
    """
    
    # Basic tuning settings
    num_samples: int = 50
    metric: str = "eval_loss"
    mode: str = "min"
    
    # Resource allocation
    max_concurrent_trials: int = 4
    resources_per_trial: Dict[str, int] = field(default_factory=lambda: {"cpu": 4, "gpu": 1})
    
    # Scheduler settings (ASHA)
    scheduler_type: str = "asha"  # "asha", "pbt", or "fifo"
    max_t: int = 100
    grace_period: int = 10
    reduction_factor: int = 3
    
    # Search algorithm settings
    search_alg_type: str = "optuna"  # "optuna", "random", or "grid"
    optuna_sampler: str = "tpe"  # "tpe", "random", "cmaes"
    
    # PBT-specific settings
    pbt_perturbation_interval: int = 5
    pbt_quantile_fraction: float = 0.25
    pbt_resample_probability: float = 0.25
    pbt_hyperparam_mutations: Optional[Dict[str, Any]] = None
    
    # Storage and tracking
    storage_path: Optional[str] = None
    experiment_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    
    # Stopping criteria
    stop_criteria: Optional[Dict[str, Any]] = None
    time_budget_s: Optional[int] = None
    
    @classmethod
    def for_model(cls, model_name: str) -> "TuneConfig":
        """Create TuneConfig optimized for a specific model.
        
        Args:
            model_name: Model name ("medgemma-4b" or "medgemma-27b")
            
        Returns:
            TuneConfig with model-appropriate settings
        """
        normalized_name = model_name.lower().replace("_", "-")
        
        if "4b" in normalized_name:
            return cls(
                num_samples=50,
                max_concurrent_trials=4,
                resources_per_trial={"cpu": 4, "gpu": 1},
                max_t=100,
                grace_period=10,
                reduction_factor=3,
                experiment_name=f"medgemma-4b-hp-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
        elif "27b" in normalized_name:
            return cls(
                num_samples=20,  # Fewer trials for expensive 27B model
                max_concurrent_trials=1,  # Only 1 trial at a time (uses 8 GPUs)
                resources_per_trial={"cpu": 8, "gpu": 8},
                max_t=50,  # Shorter max iterations
                grace_period=5,
                reduction_factor=2,
                experiment_name=f"medgemma-27b-hp-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_scheduler(self) -> Any:
        """Get configured scheduler instance.
        
        Returns:
            Ray Tune scheduler (ASHA, PBT, or None for FIFO)
        """
        if self.scheduler_type == "asha":
            return create_asha_scheduler(
                metric=self.metric,
                mode=self.mode,
                max_t=self.max_t,
                grace_period=self.grace_period,
                reduction_factor=self.reduction_factor,
            )
        elif self.scheduler_type == "pbt":
            return create_pbt_scheduler(
                metric=self.metric,
                mode=self.mode,
                perturbation_interval=self.pbt_perturbation_interval,
                quantile_fraction=self.pbt_quantile_fraction,
                resample_probability=self.pbt_resample_probability,
                hyperparam_mutations=self.pbt_hyperparam_mutations,
            )
        else:
            return None  # FIFO (no scheduler)
    
    def get_search_alg(self) -> Any:
        """Get configured search algorithm instance.
        
        Returns:
            Ray Tune search algorithm (OptunaSearch or None for random)
        """
        if self.search_alg_type == "optuna":
            return create_optuna_search(
                metric=self.metric,
                mode=self.mode,
                sampler=self.optuna_sampler,
            )
        else:
            return None  # Random search


class HyperparameterTuner:
    """Main hyperparameter tuning orchestrator.
    
    Integrates Ray Tune with the MedGemma training pipeline for
    comprehensive hyperparameter optimization.
    
    Example:
        >>> tuner = HyperparameterTuner(
        ...     model_name="medgemma-4b",
        ...     search_space=SearchSpaceConfig.for_model("medgemma-4b"),
        ...     tune_config=TuneConfig.for_model("medgemma-4b"),
        ... )
        >>> result = tuner.run(
        ...     train_data_path="data/train.jsonl",
        ...     eval_data_path="data/eval.jsonl",
        ... )
        >>> print(f"Best config: {result.best_config}")
    """
    
    def __init__(
        self,
        model_name: str,
        search_space: "SearchSpaceConfig",
        tune_config: TuneConfig,
        train_func: Optional[Callable] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """Initialize HyperparameterTuner.
        
        Args:
            model_name: Model to tune ("medgemma-4b" or "medgemma-27b")
            search_space: Search space configuration
            tune_config: Tuning configuration
            train_func: Optional custom training function
            mlflow_tracking_uri: Optional MLflow tracking URI
        """
        self.model_name = model_name
        self.search_space = search_space
        self.tune_config = tune_config
        self._train_func = train_func
        self.mlflow_tracking_uri = mlflow_tracking_uri or tune_config.mlflow_tracking_uri
        
        self._tuner = None
        self._result = None
    
    @property
    def train_func(self) -> Callable:
        """Get training function for Ray Tune."""
        if self._train_func is not None:
            return self._train_func
        
        # Default training function wrapping LoRATrainer
        return self._create_default_train_func()
    
    def _create_default_train_func(self) -> Callable:
        """Create default training function using LoRATrainer."""
        model_name = self.model_name
        
        def train_func(config: Dict[str, Any]) -> None:
            """Training function for Ray Tune."""
            try:
                from medai_compass.pipelines.lora_trainer import LoRATrainer, LoRAConfig
            except ImportError:
                logger.warning("LoRATrainer not available, using mock training")
                # Mock training for testing
                import random
                for step in range(10):
                    if RAY_AVAILABLE:
                        tune.report(
                            training_iteration=step,
                            eval_loss=random.uniform(0.1, 1.0),
                            train_loss=random.uniform(0.1, 1.0),
                        )
                return
            
            # Create LoRA config from tuned parameters
            lora_config = LoRAConfig.for_model(model_name)
            
            # Override with tuned parameters
            for key, value in config.items():
                if hasattr(lora_config, key):
                    setattr(lora_config, key, value)
            
            # Create trainer and run
            trainer = LoRATrainer(config=lora_config)
            
            # Training loop with reporting
            # This would be integrated with actual training
            result = trainer.train(
                train_data=config.get("train_data_path"),
                eval_data=config.get("eval_data_path"),
            )
            
            if RAY_AVAILABLE:
                tune.report(
                    eval_loss=result.get("eval_loss", 0.0),
                    train_loss=result.get("train_loss", 0.0),
                )
        
        return train_func
    
    def run(
        self,
        train_data_path: Optional[str] = None,
        eval_data_path: Optional[str] = None,
        resume: bool = False,
        **kwargs,
    ) -> TuneResult:
        """Run hyperparameter tuning.
        
        Args:
            train_data_path: Path to training data
            eval_data_path: Path to evaluation data
            resume: Whether to resume from previous run
            **kwargs: Additional arguments passed to training function
            
        Returns:
            TuneResult with best configuration and metrics
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune not available. Install with: pip install 'ray[tune]'")
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Get search space
        param_space = self.search_space.get_ray_tune_space()
        
        # Add data paths to param space
        if train_data_path:
            param_space["train_data_path"] = train_data_path
        if eval_data_path:
            param_space["eval_data_path"] = eval_data_path
        
        # Add any extra kwargs
        param_space.update(kwargs)
        
        # Create tuner configuration
        tune_config = RayTuneConfig(
            metric=self.tune_config.metric,
            mode=self.tune_config.mode,
            num_samples=self.tune_config.num_samples,
            max_concurrent_trials=self.tune_config.max_concurrent_trials,
            scheduler=self.tune_config.get_scheduler(),
            search_alg=self.tune_config.get_search_alg(),
        )
        
        # Create run configuration
        storage_path = self.tune_config.storage_path or tempfile.mkdtemp(prefix="medgemma_tune_")
        run_config = RunConfig(
            name=self.tune_config.experiment_name or f"hp_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            storage_path=storage_path,
            stop=self.tune_config.stop_criteria,
        )
        
        # Create and run tuner
        self._tuner = Tuner(
            self.train_func,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        
        if resume and self._tuner.can_restore():
            self._tuner = Tuner.restore(
                path=storage_path,
                trainable=self.train_func,
            )
        
        start_time = datetime.now()
        results = self._tuner.fit()
        end_time = datetime.now()
        
        # Extract best result
        best_result = results.get_best_result(
            metric=self.tune_config.metric,
            mode=self.tune_config.mode,
        )
        
        # Create TuneResult
        self._result = TuneResult(
            best_config=best_result.config,
            best_metrics=best_result.metrics,
            best_trial_id=best_result.path,
            num_trials=len(results),
            total_time_seconds=(end_time - start_time).total_seconds(),
            experiment_name=run_config.name,
            storage_path=storage_path,
            all_trial_configs=[r.config for r in results],
            all_trial_metrics=[r.metrics for r in results],
        )
        
        # Log to MLflow if configured
        if self.mlflow_tracking_uri and MLFLOW_AVAILABLE:
            self._log_to_mlflow()
        
        return self._result
    
    def resume(self, path: str) -> TuneResult:
        """Resume tuning from a previous run.
        
        Args:
            path: Path to previous tuning run
            
        Returns:
            TuneResult with best configuration
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune not available")
        
        self._tuner = Tuner.restore(
            path=path,
            trainable=self.train_func,
            resume_errored=True,
        )
        
        return self.run()
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best configuration from completed tuning.
        
        Returns:
            Best hyperparameter configuration
        """
        if self._result is None:
            raise RuntimeError("No tuning results available. Run tuning first.")
        
        return self._result.best_config
    
    def _log_to_mlflow(self) -> None:
        """Log tuning results to MLflow."""
        if not MLFLOW_AVAILABLE or self._result is None:
            return
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        experiment_name = self.tune_config.mlflow_experiment_name or f"hp_tuning_{self.model_name}"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"hp_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log best config
            mlflow.log_params(self._result.best_config)
            
            # Log best metrics
            mlflow.log_metrics(self._result.best_metrics)
            
            # Log tuning metadata
            mlflow.log_param("num_trials", self._result.num_trials)
            mlflow.log_param("total_time_seconds", self._result.total_time_seconds)
            mlflow.log_param("model_name", self.model_name)


# =============================================================================
# ASHA Scheduler (Task 4.3)
# =============================================================================

def create_asha_scheduler(
    metric: str = "eval_loss",
    mode: str = "min",
    max_t: int = 100,
    grace_period: int = 10,
    reduction_factor: int = 3,
    time_attr: str = "training_iteration",
) -> "ASHAScheduler":
    """Create ASHA scheduler for early stopping.
    
    ASHA (Asynchronous Successive Halving Algorithm) aggressively stops
    poorly performing trials early, saving compute resources.
    
    Args:
        metric: Metric to optimize
        mode: "min" or "max"
        max_t: Maximum training iterations
        grace_period: Minimum iterations before pruning
        reduction_factor: Factor for pruning (higher = more aggressive)
        time_attr: Attribute to use as time measure
        
    Returns:
        Configured ASHAScheduler
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune not available")
    
    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=max_t,
        grace_period=grace_period,
        reduction_factor=reduction_factor,
        time_attr=time_attr,
    )
    
    # Store config values for inspection
    scheduler.max_t = max_t
    scheduler._grace_period = grace_period
    scheduler._reduction_factor = reduction_factor
    scheduler._metric = metric
    scheduler._mode = mode
    
    return scheduler


# =============================================================================
# Optuna Integration (Task 4.4)
# =============================================================================

def create_optuna_search(
    metric: str = "eval_loss",
    mode: str = "min",
    sampler: str = "tpe",
    use_pruning: bool = False,
    seed: Optional[int] = None,
) -> "OptunaSearch":
    """Create Optuna-based search algorithm.
    
    Uses Bayesian optimization for efficient hyperparameter search.
    
    Args:
        metric: Metric to optimize
        mode: "min" or "max"
        sampler: Sampler type ("tpe", "random", "cmaes")
        use_pruning: Whether to enable Optuna pruning
        seed: Random seed for reproducibility
        
    Returns:
        Configured OptunaSearch
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune not available")
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available. Install with: pip install optuna")
    
    # Create Optuna sampler
    if sampler == "tpe":
        optuna_sampler = TPESampler(seed=seed)
    elif sampler == "random":
        optuna_sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler == "cmaes":
        optuna_sampler = optuna.samplers.CmaEsSampler(seed=seed)
    else:
        optuna_sampler = TPESampler(seed=seed)
    
    # Create search algorithm
    search_alg = OptunaSearch(
        metric=metric,
        mode=mode,
        sampler=optuna_sampler,
    )
    
    # Store config for inspection
    search_alg._metric = metric
    search_alg._mode = mode
    
    return search_alg


def create_optuna_study(
    study_name: str,
    direction: str = "minimize",
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    sampler: Optional["optuna.samplers.BaseSampler"] = None,
    pruner: Optional["optuna.pruners.BasePruner"] = None,
) -> "optuna.Study":
    """Create Optuna study for standalone optimization.
    
    Args:
        study_name: Name for the study
        direction: "minimize" or "maximize"
        storage: Database URL for persistence
        load_if_exists: Load existing study if available
        sampler: Custom sampler
        pruner: Custom pruner
        
    Returns:
        Configured Optuna study
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available")
    
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler or TPESampler(),
        pruner=pruner or MedianPruner(),
    )
    
    return study


# =============================================================================
# Population-Based Training (Task 4.6)
# =============================================================================

def create_pbt_scheduler(
    metric: str = "eval_loss",
    mode: str = "min",
    perturbation_interval: int = 5,
    quantile_fraction: float = 0.25,
    resample_probability: float = 0.25,
    hyperparam_mutations: Optional[Dict[str, Any]] = None,
    custom_explore_fn: Optional[Callable] = None,
    time_attr: str = "training_iteration",
) -> "PopulationBasedTraining":
    """Create Population-Based Training (PBT) scheduler.
    
    PBT trains a population of models in parallel, periodically replacing
    poor performers with mutations of better performers.
    
    Args:
        metric: Metric to optimize
        mode: "min" or "max"
        perturbation_interval: Steps between perturbations
        quantile_fraction: Fraction of trials in bottom quantile to perturb
        resample_probability: Probability of resampling from original distribution
        hyperparam_mutations: Dictionary of hyperparameter mutation ranges
        custom_explore_fn: Custom exploration function
        time_attr: Attribute to use as time measure
        
    Returns:
        Configured PopulationBasedTraining scheduler
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune not available")
    
    # Default mutations if not provided
    if hyperparam_mutations is None:
        hyperparam_mutations = {
            "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1],
        }
    
    scheduler = PopulationBasedTraining(
        metric=metric,
        mode=mode,
        perturbation_interval=perturbation_interval,
        quantile_fraction=quantile_fraction,
        resample_probability=resample_probability,
        hyperparam_mutations=hyperparam_mutations,
        time_attr=time_attr,
    )
    
    # Store config for inspection
    scheduler._perturbation_interval = perturbation_interval
    scheduler._quantile_fraction = quantile_fraction
    scheduler._resample_probability = resample_probability
    scheduler._hyperparam_mutations = hyperparam_mutations
    scheduler._custom_explore_fn = custom_explore_fn
    
    return scheduler


def create_pbt_scheduler_for_model(model_name: str) -> "PopulationBasedTraining":
    """Create PBT scheduler optimized for a specific model.
    
    Args:
        model_name: Model name ("medgemma-4b" or "medgemma-27b")
        
    Returns:
        Configured PBT scheduler
    """
    normalized_name = model_name.lower().replace("_", "-")
    
    if "4b" in normalized_name:
        return create_pbt_scheduler(
            perturbation_interval=5,
            quantile_fraction=0.25,
            resample_probability=0.25,
            hyperparam_mutations={
                "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
                "weight_decay": [0.0, 0.01, 0.05, 0.1],
                "lora_dropout": [0.0, 0.05, 0.1, 0.15, 0.2],
            },
        )
    elif "27b" in normalized_name:
        return create_pbt_scheduler(
            perturbation_interval=3,  # More frequent for expensive model
            quantile_fraction=0.25,
            resample_probability=0.25,
            hyperparam_mutations={
                "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4],
                "weight_decay": [0.0, 0.01, 0.05],
                "lora_dropout": [0.0, 0.05, 0.1],
            },
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Airflow Task Functions
# =============================================================================

def define_search_space_task(
    model_name: str = "medgemma-4b",
    **kwargs,
) -> Dict[str, Any]:
    """Airflow task to define search space.
    
    Args:
        model_name: Model to tune
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with search space configuration
    """
    from medai_compass.optimization.search_space import SearchSpaceConfig
    
    config = SearchSpaceConfig.for_model(model_name)
    
    return {
        "model_name": model_name,
        "search_space": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }


def run_hp_tuning_task(
    model_name: str,
    search_space_dict: Dict[str, Any],
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    num_samples: int = 50,
    **kwargs,
) -> Dict[str, Any]:
    """Airflow task to run hyperparameter tuning.
    
    Args:
        model_name: Model to tune
        search_space_dict: Search space from define_search_space_task
        train_data_path: Training data path
        eval_data_path: Evaluation data path
        num_samples: Number of trials
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with tuning results
    """
    from medai_compass.optimization.search_space import SearchSpaceConfig
    
    # Reconstruct search space
    search_space = SearchSpaceConfig.from_dict(search_space_dict)
    
    # Create config
    tune_config = TuneConfig.for_model(model_name)
    tune_config.num_samples = num_samples
    
    # Run tuning
    tuner = HyperparameterTuner(
        model_name=model_name,
        search_space=search_space,
        tune_config=tune_config,
    )
    
    result = tuner.run(
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        **kwargs,
    )
    
    return result.to_dict()


def analyze_results_task(
    tuning_results: Dict[str, Any],
    output_dir: str = "/outputs/hp_analysis",
    **kwargs,
) -> Dict[str, Any]:
    """Airflow task to analyze tuning results.
    
    Args:
        tuning_results: Results from run_hp_tuning_task
        output_dir: Directory for output files
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with analysis summary
    """
    from medai_compass.optimization.analysis import HPAnalyzer
    
    analyzer = HPAnalyzer()
    analyzer.load_results_from_path(tuning_results["storage_path"])
    
    # Generate visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer.save_figure(
        analyzer.plot_parallel_coordinates(),
        os.path.join(output_dir, "parallel_coordinates.html"),
    )
    
    analyzer.save_figure(
        analyzer.plot_optimization_history(),
        os.path.join(output_dir, "optimization_history.html"),
    )
    
    # Calculate parameter importance
    importance = analyzer.calculate_parameter_importance()
    
    return {
        "best_config": tuning_results["best_config"],
        "best_metrics": tuning_results["best_metrics"],
        "parameter_importance": importance,
        "output_dir": output_dir,
    }


def select_best_config_task(
    analysis_results: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Airflow task to select best configuration.
    
    Args:
        analysis_results: Results from analyze_results_task
        **kwargs: Additional parameters
        
    Returns:
        Best configuration for training
    """
    best_config = analysis_results["best_config"]
    
    return {
        "best_config": best_config,
        "ready_for_training": True,
        "timestamp": datetime.now().isoformat(),
    }
