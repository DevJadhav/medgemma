"""Search Space Definitions for MedGemma Hyperparameter Optimization.

Provides model-specific search space configurations for LoRA fine-tuning
hyperparameters with support for Ray Tune and Optuna.

Features:
- Pre-defined search spaces for MedGemma 4B and 27B
- Support for categorical, uniform, log-uniform distributions
- Custom parameter constraints
- Export to Ray Tune and Optuna formats
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import Ray Tune - graceful fallback if not available
try:
    from ray import tune
    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False
    tune = None

# Try to import Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


@dataclass
class ParameterSpace:
    """Definition of a single hyperparameter search space.
    
    Supports multiple distribution types:
    - "choice": Categorical choices from a list
    - "uniform": Uniform distribution between lower and upper
    - "loguniform": Log-uniform distribution (for learning rates)
    - "quniform": Quantized uniform (for integers)
    - "randint": Random integer in range
    
    Example:
        >>> lr_space = ParameterSpace(
        ...     type="loguniform",
        ...     lower=1e-5,
        ...     upper=1e-3,
        ... )
        >>> lora_r_space = ParameterSpace(
        ...     type="choice",
        ...     choices=[8, 16, 32, 64],
        ... )
    """
    
    type: str  # "choice", "uniform", "loguniform", "quniform", "randint"
    lower: Optional[float] = None
    upper: Optional[float] = None
    choices: Optional[List[Any]] = None
    q: Optional[float] = None  # For quantized distributions
    log: bool = False  # For log-scale distributions
    
    def __post_init__(self):
        """Validate parameter space configuration."""
        if self.type == "choice" and not self.choices:
            raise ValueError("'choice' type requires 'choices' list")
        if self.type in ("uniform", "loguniform", "quniform", "randint"):
            if self.lower is None or self.upper is None:
                raise ValueError(f"'{self.type}' type requires 'lower' and 'upper'")
    
    def to_ray_tune(self) -> Any:
        """Convert to Ray Tune search space format.
        
        Returns:
            Ray Tune distribution object
        """
        if not RAY_TUNE_AVAILABLE:
            raise ImportError("Ray Tune not available. Install with: pip install 'ray[tune]'")
        
        if self.type == "choice":
            return tune.choice(self.choices)
        elif self.type == "uniform":
            return tune.uniform(self.lower, self.upper)
        elif self.type == "loguniform":
            return tune.loguniform(self.lower, self.upper)
        elif self.type == "quniform":
            return tune.quniform(self.lower, self.upper, self.q or 1)
        elif self.type == "randint":
            return tune.randint(int(self.lower), int(self.upper))
        else:
            raise ValueError(f"Unknown distribution type: {self.type}")
    
    def to_optuna(self, trial: "optuna.Trial", name: str) -> Any:
        """Convert to Optuna trial suggestion.
        
        Args:
            trial: Optuna trial object
            name: Parameter name
            
        Returns:
            Suggested parameter value
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        if self.type == "choice":
            return trial.suggest_categorical(name, self.choices)
        elif self.type == "uniform":
            return trial.suggest_float(name, self.lower, self.upper)
        elif self.type == "loguniform":
            return trial.suggest_float(name, self.lower, self.upper, log=True)
        elif self.type == "quniform":
            return trial.suggest_float(
                name, self.lower, self.upper, step=self.q or 1.0
            )
        elif self.type == "randint":
            return trial.suggest_int(name, int(self.lower), int(self.upper))
        else:
            raise ValueError(f"Unknown distribution type: {self.type}")


@dataclass
class SearchSpaceConfig:
    """Complete search space configuration for hyperparameter tuning.
    
    Contains all hyperparameters to tune with their search spaces,
    optimized for the specific model variant (4B or 27B).
    
    Example:
        >>> config = SearchSpaceConfig.for_model("medgemma-4b")
        >>> ray_space = config.get_ray_tune_space()
        >>> print(ray_space.keys())
    """
    
    model_name: str
    parameters: Dict[str, ParameterSpace] = field(default_factory=dict)
    constraints: List[Callable[[Dict], bool]] = field(default_factory=list)
    
    @classmethod
    def for_model(cls, model_name: str) -> "SearchSpaceConfig":
        """Create search space configuration for a specific model.
        
        Args:
            model_name: Model name ("medgemma-4b" or "medgemma-27b")
            
        Returns:
            SearchSpaceConfig with model-appropriate search spaces
        """
        normalized_name = model_name.lower().replace("_", "-")
        
        if "4b" in normalized_name:
            return cls._create_4b_config()
        elif "27b" in normalized_name:
            return cls._create_27b_config()
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'medgemma-4b' or 'medgemma-27b'")
    
    @classmethod
    def _create_4b_config(cls) -> "SearchSpaceConfig":
        """Create search space for MedGemma 4B model."""
        return cls(
            model_name="medgemma-4b",
            parameters={
                # LoRA parameters
                "lora_r": ParameterSpace(
                    type="choice",
                    choices=[8, 16, 32],
                ),
                "lora_alpha": ParameterSpace(
                    type="choice",
                    choices=[16, 32, 64],
                ),
                "lora_dropout": ParameterSpace(
                    type="uniform",
                    lower=0.0,
                    upper=0.2,
                ),
                # Training parameters
                "learning_rate": ParameterSpace(
                    type="loguniform",
                    lower=1e-5,
                    upper=1e-3,
                ),
                "batch_size": ParameterSpace(
                    type="choice",
                    choices=[2, 4, 8],
                ),
                "warmup_ratio": ParameterSpace(
                    type="uniform",
                    lower=0.0,
                    upper=0.1,
                ),
                "weight_decay": ParameterSpace(
                    type="loguniform",
                    lower=1e-4,
                    upper=1e-1,
                ),
                "gradient_accumulation_steps": ParameterSpace(
                    type="choice",
                    choices=[2, 4, 8],
                ),
            },
            constraints=[
                # Constraint: lora_alpha should be >= lora_r
                lambda params: params.get("lora_alpha", 32) >= params.get("lora_r", 16),
            ],
        )
    
    @classmethod
    def _create_27b_config(cls) -> "SearchSpaceConfig":
        """Create search space for MedGemma 27B model."""
        return cls(
            model_name="medgemma-27b",
            parameters={
                # LoRA parameters - larger values for 27B
                "lora_r": ParameterSpace(
                    type="choice",
                    choices=[32, 64, 128],
                ),
                "lora_alpha": ParameterSpace(
                    type="choice",
                    choices=[64, 128, 256],
                ),
                "lora_dropout": ParameterSpace(
                    type="uniform",
                    lower=0.0,
                    upper=0.15,
                ),
                # Training parameters - more conservative for 27B
                "learning_rate": ParameterSpace(
                    type="loguniform",
                    lower=1e-5,
                    upper=5e-4,
                ),
                "batch_size": ParameterSpace(
                    type="choice",
                    choices=[1],  # Only batch_size=1 for 27B
                ),
                "warmup_ratio": ParameterSpace(
                    type="uniform",
                    lower=0.0,
                    upper=0.1,
                ),
                "weight_decay": ParameterSpace(
                    type="loguniform",
                    lower=1e-4,
                    upper=1e-1,
                ),
                "gradient_accumulation_steps": ParameterSpace(
                    type="choice",
                    choices=[8, 16, 32],
                ),
            },
            constraints=[
                lambda params: params.get("lora_alpha", 128) >= params.get("lora_r", 64),
            ],
        )
    
    def get_ray_tune_space(self) -> Dict[str, Any]:
        """Get search space in Ray Tune format.
        
        Returns:
            Dictionary mapping parameter names to Ray Tune distributions
        """
        return {
            name: space.to_ray_tune()
            for name, space in self.parameters.items()
        }
    
    def get_parameter_choices(self, param_name: str) -> List[Any]:
        """Get choices for a categorical parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            List of choices
        """
        if param_name not in self.parameters:
            raise KeyError(f"Parameter '{param_name}' not in search space")
        
        space = self.parameters[param_name]
        if space.type != "choice":
            raise ValueError(f"Parameter '{param_name}' is not categorical")
        
        return space.choices
    
    def get_parameter_type(self, param_name: str) -> str:
        """Get distribution type for a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Distribution type string
        """
        if param_name not in self.parameters:
            raise KeyError(f"Parameter '{param_name}' not in search space")
        
        return self.parameters[param_name].type
    
    def get_parameter_bounds(self, param_name: str) -> Dict[str, float]:
        """Get bounds for a continuous parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Dictionary with 'lower' and 'upper' bounds
        """
        if param_name not in self.parameters:
            raise KeyError(f"Parameter '{param_name}' not in search space")
        
        space = self.parameters[param_name]
        return {
            "lower": space.lower,
            "upper": space.upper,
        }
    
    def add_parameter(
        self,
        name: str,
        space: ParameterSpace,
    ) -> None:
        """Add a parameter to the search space.
        
        Args:
            name: Parameter name
            space: ParameterSpace definition
        """
        self.parameters[name] = space
        logger.info(f"Added parameter '{name}' to search space")
    
    def remove_parameter(self, name: str) -> None:
        """Remove a parameter from the search space.
        
        Args:
            name: Parameter name to remove
        """
        if name in self.parameters:
            del self.parameters[name]
            logger.info(f"Removed parameter '{name}' from search space")
    
    def update_parameter(
        self,
        name: str,
        **kwargs,
    ) -> None:
        """Update parameter bounds or choices.
        
        Args:
            name: Parameter name
            **kwargs: Fields to update (lower, upper, choices, etc.)
        """
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not in search space")
        
        space = self.parameters[name]
        for key, value in kwargs.items():
            if hasattr(space, key):
                setattr(space, key, value)
        
        logger.info(f"Updated parameter '{name}': {kwargs}")
    
    def add_constraint(
        self,
        constraint_fn: Callable[[Dict], bool],
    ) -> None:
        """Add a constraint function.
        
        Constraint functions take a parameter dict and return True if valid.
        
        Args:
            constraint_fn: Function that returns True for valid configs
        """
        self.constraints.append(constraint_fn)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate a configuration against constraints.
        
        Args:
            config: Parameter configuration to validate
            
        Returns:
            True if configuration satisfies all constraints
        """
        return all(constraint(config) for constraint in self.constraints)
    
    def sample_config(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space.
        
        Returns:
            Dictionary with sampled parameter values
        """
        import random
        
        config = {}
        for name, space in self.parameters.items():
            if space.type == "choice":
                config[name] = random.choice(space.choices)
            elif space.type == "uniform":
                config[name] = random.uniform(space.lower, space.upper)
            elif space.type == "loguniform":
                import math
                log_low = math.log(space.lower)
                log_high = math.log(space.upper)
                config[name] = math.exp(random.uniform(log_low, log_high))
            elif space.type == "randint":
                config[name] = random.randint(int(space.lower), int(space.upper))
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search space to dictionary representation.
        
        Returns:
            Dictionary representation for serialization
        """
        return {
            "model_name": self.model_name,
            "parameters": {
                name: {
                    "type": space.type,
                    "lower": space.lower,
                    "upper": space.upper,
                    "choices": space.choices,
                    "q": space.q,
                }
                for name, space in self.parameters.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSpaceConfig":
        """Create SearchSpaceConfig from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            SearchSpaceConfig instance
        """
        parameters = {
            name: ParameterSpace(
                type=space["type"],
                lower=space.get("lower"),
                upper=space.get("upper"),
                choices=space.get("choices"),
                q=space.get("q"),
            )
            for name, space in data.get("parameters", {}).items()
        }
        
        return cls(
            model_name=data["model_name"],
            parameters=parameters,
        )


def get_default_search_space(model_name: str) -> Dict[str, Any]:
    """Get default Ray Tune search space for a model.
    
    Convenience function to quickly get a search space dictionary.
    
    Args:
        model_name: Model name ("medgemma-4b" or "medgemma-27b")
        
    Returns:
        Dictionary with Ray Tune search space
        
    Example:
        >>> space = get_default_search_space("medgemma-4b")
        >>> # Use directly with Ray Tune
        >>> tuner = tune.Tuner(trainable, param_space=space)
    """
    config = SearchSpaceConfig.for_model(model_name)
    return config.get_ray_tune_space()


# Pre-defined search space templates
SEARCH_SPACE_TEMPLATES = {
    "minimal": {
        "learning_rate": ParameterSpace(type="loguniform", lower=1e-5, upper=1e-3),
        "lora_r": ParameterSpace(type="choice", choices=[8, 16]),
    },
    "standard": {
        "learning_rate": ParameterSpace(type="loguniform", lower=1e-5, upper=1e-3),
        "lora_r": ParameterSpace(type="choice", choices=[8, 16, 32]),
        "lora_alpha": ParameterSpace(type="choice", choices=[16, 32, 64]),
        "lora_dropout": ParameterSpace(type="uniform", lower=0.0, upper=0.2),
    },
    "comprehensive": {
        "learning_rate": ParameterSpace(type="loguniform", lower=1e-5, upper=1e-3),
        "lora_r": ParameterSpace(type="choice", choices=[8, 16, 32, 64]),
        "lora_alpha": ParameterSpace(type="choice", choices=[16, 32, 64, 128]),
        "lora_dropout": ParameterSpace(type="uniform", lower=0.0, upper=0.2),
        "batch_size": ParameterSpace(type="choice", choices=[2, 4, 8]),
        "warmup_ratio": ParameterSpace(type="uniform", lower=0.0, upper=0.1),
        "weight_decay": ParameterSpace(type="loguniform", lower=1e-4, upper=1e-1),
        "gradient_accumulation_steps": ParameterSpace(type="choice", choices=[2, 4, 8]),
    },
}
