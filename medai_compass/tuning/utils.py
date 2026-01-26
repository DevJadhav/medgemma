"""Utility functions for Ray Tune hyperparameter optimization."""

from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def search_space_to_ray(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a search space configuration to Ray Tune search space objects.

    Args:
        search_space: Dictionary with parameter names and specifications.
                     Each spec should have 'type' and relevant bounds/values.

    Returns:
        Dictionary of Ray Tune search space objects.

    Example:
        >>> space = search_space_to_ray({
        ...     "learning_rate": {"type": "loguniform", "lower": 1e-5, "upper": 1e-3},
        ...     "batch_size": {"type": "choice", "values": [8, 16, 32]},
        ... })
    """
    from ray import tune

    ray_space = {}

    for name, spec in search_space.items():
        if isinstance(spec, dict):
            param_type = spec.get("type", "uniform")

            if param_type == "uniform":
                ray_space[name] = tune.uniform(
                    spec.get("lower", 0),
                    spec.get("upper", 1),
                )
            elif param_type == "loguniform":
                ray_space[name] = tune.loguniform(
                    spec.get("lower", 1e-5),
                    spec.get("upper", 1e-1),
                )
            elif param_type == "choice":
                ray_space[name] = tune.choice(
                    spec.get("values", spec.get("categories", []))
                )
            elif param_type == "randint":
                ray_space[name] = tune.randint(
                    spec.get("lower", 0),
                    spec.get("upper", 10),
                )
            elif param_type == "quniform":
                ray_space[name] = tune.quniform(
                    spec.get("lower", 0),
                    spec.get("upper", 1),
                    spec.get("q", 0.1),
                )
            elif param_type == "qloguniform":
                ray_space[name] = tune.qloguniform(
                    spec.get("lower", 1e-5),
                    spec.get("upper", 1e-1),
                    spec.get("q", 1e-5),
                )
            elif param_type == "grid":
                ray_space[name] = tune.grid_search(
                    spec.get("values", spec.get("categories", []))
                )
            else:
                logger.warning(f"Unknown search space type '{param_type}' for {name}")
                ray_space[name] = spec
        else:
            # Pass through as constant value
            ray_space[name] = spec

    return ray_space


def mutations_to_ray(mutations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert mutation specifications to Ray Tune PBT mutation functions.

    Args:
        mutations: Dictionary with parameter names and mutation specs.

    Returns:
        Dictionary of mutation functions for PBT.
    """
    from ray import tune

    ray_mutations = {}

    for name, spec in mutations.items():
        if isinstance(spec, dict):
            param_type = spec.get("type", "uniform")

            if param_type == "uniform":
                ray_mutations[name] = tune.uniform(
                    spec.get("lower", 0),
                    spec.get("upper", 1),
                )
            elif param_type == "loguniform":
                ray_mutations[name] = tune.loguniform(
                    spec.get("lower", 1e-5),
                    spec.get("upper", 1e-1),
                )
            elif param_type == "choice":
                ray_mutations[name] = spec.get("values", spec.get("categories", []))
            elif param_type == "perturbation":
                # Custom perturbation factors
                factors = spec.get("factors", [0.8, 1.0, 1.2])
                ray_mutations[name] = lambda x, f=factors: x * tune.choice(f).sample()
            else:
                logger.warning(f"Unknown mutation type '{param_type}' for {name}")
        else:
            ray_mutations[name] = spec

    return ray_mutations


def get_best_trial_config(
    results,
    metric: str = "eval_loss",
    mode: str = "min",
) -> Dict[str, Any]:
    """
    Extract the best trial configuration from Ray Tune results.

    Args:
        results: Ray Tune ResultGrid or ExperimentAnalysis.
        metric: Metric to optimize.
        mode: "min" or "max".

    Returns:
        Dictionary with best configuration and metrics.
    """
    try:
        best_result = results.get_best_result(metric=metric, mode=mode)

        return {
            "config": best_result.config,
            "metrics": best_result.metrics,
            "checkpoint": best_result.checkpoint,
            "log_dir": best_result.path,
        }
    except Exception as e:
        logger.error(f"Error getting best trial config: {e}")
        return {}


def calculate_hyperband_brackets(
    max_t: int,
    reduction_factor: int = 3,
) -> Dict[str, Any]:
    """
    Calculate Hyperband bracket configuration.

    Args:
        max_t: Maximum training iterations.
        reduction_factor: Halving rate.

    Returns:
        Dictionary with bracket information for planning.
    """
    import math

    s_max = int(math.log(max_t) / math.log(reduction_factor))
    brackets = []

    for s in range(s_max + 1):
        n_i = int(math.ceil(
            (s_max + 1) * (reduction_factor ** s) / (s + 1)
        ))
        r_i = max_t / (reduction_factor ** s)

        brackets.append({
            "bracket": s,
            "num_configs": n_i,
            "initial_budget": int(r_i),
            "total_budget": int(n_i * r_i * (s + 1)),
        })

    return {
        "brackets": brackets,
        "total_configs": sum(b["num_configs"] for b in brackets),
        "max_bracket_budget": brackets[0]["total_budget"] if brackets else 0,
        "s_max": s_max,
    }


def estimate_tuning_cost(
    num_samples: int,
    max_iterations: int,
    cost_per_gpu_hour: float,
    seconds_per_iteration: float = 1.0,
    scheduler: str = "asha",
    reduction_factor: int = 3,
) -> Dict[str, float]:
    """
    Estimate the cost of a hyperparameter tuning run.

    Args:
        num_samples: Number of trials to run.
        max_iterations: Maximum iterations per trial.
        cost_per_gpu_hour: Cost per GPU hour in dollars.
        seconds_per_iteration: Average seconds per training iteration.
        scheduler: Scheduler type ("asha", "pbt", "hyperband").
        reduction_factor: Halving rate for ASHA/Hyperband.

    Returns:
        Dictionary with cost estimates.
    """
    import math

    # Calculate expected iterations based on scheduler
    if scheduler == "asha":
        # ASHA stops trials early, so average is less than max
        # Rough estimate: sum of geometric series
        avg_iterations = max_iterations / reduction_factor
        total_iterations = num_samples * avg_iterations
    elif scheduler == "pbt":
        # PBT runs all trials for full duration
        total_iterations = num_samples * max_iterations
    elif scheduler == "hyperband":
        # Use bracket calculation
        brackets = calculate_hyperband_brackets(max_iterations, reduction_factor)
        total_iterations = sum(
            b["num_configs"] * b["initial_budget"]
            for b in brackets["brackets"]
        )
    else:
        total_iterations = num_samples * max_iterations

    # Calculate time and cost
    total_seconds = total_iterations * seconds_per_iteration
    total_hours = total_seconds / 3600
    total_cost = total_hours * cost_per_gpu_hour

    return {
        "total_iterations": total_iterations,
        "total_hours": total_hours,
        "total_cost_usd": total_cost,
        "cost_per_trial": total_cost / num_samples if num_samples > 0 else 0,
    }
