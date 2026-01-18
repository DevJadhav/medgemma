"""MedGemma Hyperparameter Optimization Module.

Provides comprehensive hyperparameter optimization infrastructure for
MedGemma 4B IT and 27B IT LoRA fine-tuning using Ray Tune and Optuna.

Modules:
- search_space: Search space definitions for LoRA parameters
- ray_tune_integration: Ray Tune integration with ASHA, Optuna, PBT
- analysis: HP search analysis tools and visualization

Features:
- Model-specific search spaces (4B vs 27B)
- ASHA early stopping scheduler
- Optuna Bayesian optimization
- Population-Based Training (PBT)
- Automatic GPU resource allocation
- MLflow experiment tracking integration
- Comprehensive analysis and visualization

Example:
    >>> from medai_compass.optimization import (
    ...     HyperparameterTuner,
    ...     SearchSpaceConfig,
    ...     TuneConfig,
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

from medai_compass.optimization.search_space import (
    SearchSpaceConfig,
    ParameterSpace,
    get_default_search_space,
)
from medai_compass.optimization.ray_tune_integration import (
    HyperparameterTuner,
    TuneConfig,
    TuneResult,
    create_asha_scheduler,
    create_optuna_search,
    create_optuna_study,
    create_pbt_scheduler,
    create_pbt_scheduler_for_model,
    # Airflow task functions
    define_search_space_task,
    run_hp_tuning_task,
    analyze_results_task,
    select_best_config_task,
)
from medai_compass.optimization.analysis import (
    HPAnalyzer,
    TrialResult,
    AnalysisReport,
)

__all__ = [
    # Search Space
    "SearchSpaceConfig",
    "ParameterSpace",
    "get_default_search_space",
    # Ray Tune Integration
    "HyperparameterTuner",
    "TuneConfig",
    "TuneResult",
    "create_asha_scheduler",
    "create_optuna_search",
    "create_optuna_study",
    "create_pbt_scheduler",
    "create_pbt_scheduler_for_model",
    # Analysis
    "HPAnalyzer",
    "TrialResult",
    "AnalysisReport",
    # Airflow Tasks
    "define_search_space_task",
    "run_hp_tuning_task",
    "analyze_results_task",
    "select_best_config_task",
    # Constants
    "OPTIMIZATION_PROFILES",
]

# Model-specific optimization profiles
OPTIMIZATION_PROFILES = {
    "medgemma-4b": {
        "max_concurrent_trials": 4,
        "resources_per_trial": {"cpu": 4, "gpu": 1},
        "num_samples": 50,
        "asha_config": {
            "max_t": 100,
            "grace_period": 10,
            "reduction_factor": 3,
        },
        "pbt_config": {
            "perturbation_interval": 5,
            "quantile_fraction": 0.25,
            "resample_probability": 0.25,
        },
        "search_space_defaults": {
            "lora_r": [8, 16, 32],
            "lora_alpha": [16, 32, 64],
            "batch_size": [2, 4, 8],
            "learning_rate_range": (1e-5, 1e-3),
        },
    },
    "medgemma-27b": {
        "max_concurrent_trials": 1,
        "resources_per_trial": {"cpu": 8, "gpu": 8},
        "num_samples": 20,
        "asha_config": {
            "max_t": 50,
            "grace_period": 5,
            "reduction_factor": 2,
        },
        "pbt_config": {
            "perturbation_interval": 3,
            "quantile_fraction": 0.25,
            "resample_probability": 0.25,
        },
        "search_space_defaults": {
            "lora_r": [32, 64, 128],
            "lora_alpha": [64, 128, 256],
            "batch_size": [1],
            "learning_rate_range": (1e-5, 5e-4),
        },
    },
}
