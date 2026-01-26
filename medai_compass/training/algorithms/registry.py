"""
Algorithm Registry for Training Pipeline.

Provides a centralized registry for all training algorithms with
dynamic selection, compatibility checking, and trainer creation.
"""

from typing import Dict, Type, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class AlgorithmCategory(Enum):
    """Categories of training algorithms."""
    PEFT = "parameter_efficient_fine_tuning"
    ALIGNMENT = "alignment"
    HYBRID = "hybrid"


class ResourceRequirement(Enum):
    """Resource requirement levels."""
    LOW = "low"  # < 16GB VRAM
    MEDIUM = "medium"  # 16-40GB VRAM
    HIGH = "high"  # 40-80GB VRAM
    VERY_HIGH = "very_high"  # 80GB+ VRAM


@dataclass
class AlgorithmInfo:
    """Information about a registered algorithm."""
    name: str
    category: AlgorithmCategory
    description: str
    config_class: Type
    trainer_class: Type
    resource_requirement: ResourceRequirement
    supports_quantization: bool
    supports_multi_gpu: bool
    requires_reference_model: bool
    requires_reward_model: bool
    compatible_models: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "resource_requirement": self.resource_requirement.value,
            "supports_quantization": self.supports_quantization,
            "supports_multi_gpu": self.supports_multi_gpu,
            "requires_reference_model": self.requires_reference_model,
            "requires_reward_model": self.requires_reward_model,
            "compatible_models": self.compatible_models,
        }

    def get_config(self, **kwargs) -> Any:
        """Get config instance for this algorithm."""
        return self.config_class(**kwargs)

    def get_trainer(self, **kwargs) -> Any:
        """Get trainer instance for this algorithm."""
        return self.trainer_class(**kwargs)


class AlgorithmRegistry:
    """
    Central registry for training algorithms.

    Provides registration, lookup, and compatibility checking
    for all supported training algorithms.
    """

    def __init__(self):
        self._algorithms: Dict[str, AlgorithmInfo] = {}
        self._aliases: Dict[str, str] = {}
        self._initialized = False

    def __contains__(self, name: str) -> bool:
        """Check if algorithm is in registry."""
        self._ensure_initialized()
        key = name.lower()
        if key in self._aliases:
            key = self._aliases[key]
        return key in self._algorithms

    def __iter__(self):
        """Iterate over algorithm names."""
        self._ensure_initialized()
        return iter(self._algorithms.keys())

    def register(
        self,
        name: str,
        category: AlgorithmCategory,
        description: str,
        config_class: Type,
        trainer_class: Type,
        resource_requirement: ResourceRequirement = ResourceRequirement.MEDIUM,
        supports_quantization: bool = False,
        supports_multi_gpu: bool = True,
        requires_reference_model: bool = False,
        requires_reward_model: bool = False,
        compatible_models: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a new algorithm."""
        if compatible_models is None:
            compatible_models = ["medgemma-4b", "medgemma-27b"]

        info = AlgorithmInfo(
            name=name,
            category=category,
            description=description,
            config_class=config_class,
            trainer_class=trainer_class,
            resource_requirement=resource_requirement,
            supports_quantization=supports_quantization,
            supports_multi_gpu=supports_multi_gpu,
            requires_reference_model=requires_reference_model,
            requires_reward_model=requires_reward_model,
            compatible_models=compatible_models,
        )

        self._algorithms[name.lower()] = info

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name.lower()

    def get(self, name: str) -> Optional[AlgorithmInfo]:
        """Get algorithm info by name or alias."""
        self._ensure_initialized()
        key = name.lower()

        # Check aliases first
        if key in self._aliases:
            key = self._aliases[key]

        return self._algorithms.get(key)

    def list_all(self) -> List[AlgorithmInfo]:
        """List all registered algorithms."""
        self._ensure_initialized()
        return list(self._algorithms.values())

    def list_by_category(self, category: AlgorithmCategory) -> List[AlgorithmInfo]:
        """List algorithms by category."""
        self._ensure_initialized()
        return [
            info for info in self._algorithms.values()
            if info.category == category
        ]

    def list_names(self) -> List[str]:
        """List all algorithm names including aliases."""
        self._ensure_initialized()
        names = list(self._algorithms.keys())
        names.extend(self._aliases.keys())
        return names

    def is_registered(self, name: str) -> bool:
        """Check if an algorithm is registered."""
        return self.get(name) is not None

    def check_compatibility(
        self,
        algorithm_name: str,
        model_name: str,
        available_vram_gb: Optional[float] = None,
        num_gpus: int = 1,
    ) -> Dict[str, Any]:
        """
        Check if an algorithm is compatible with the given setup.

        Returns a dict with:
        - compatible: bool
        - warnings: list of warning messages
        - errors: list of error messages
        - recommendations: list of recommendations
        - requires_reward_model: bool (if algorithm requires reward model)
        """
        result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        info = self.get(algorithm_name)
        if info is None:
            result["compatible"] = False
            result["errors"].append(f"Algorithm '{algorithm_name}' not found in registry")
            return result

        # Add requires_reward_model to result
        if info.requires_reward_model:
            result["requires_reward_model"] = True

        # Check model compatibility
        model_lower = model_name.lower()
        compatible_lower = [m.lower() for m in info.compatible_models]

        if not any(m in model_lower for m in compatible_lower):
            result["warnings"].append(
                f"Model '{model_name}' not in tested models: {info.compatible_models}"
            )

        # If VRAM not provided, assume sufficient
        if available_vram_gb is None:
            return result

        # Check VRAM requirements
        vram_thresholds = {
            ResourceRequirement.LOW: 16,
            ResourceRequirement.MEDIUM: 40,
            ResourceRequirement.HIGH: 80,
            ResourceRequirement.VERY_HIGH: 160,
        }

        required_vram = vram_thresholds.get(info.resource_requirement, 40)
        effective_vram = available_vram_gb * num_gpus if info.supports_multi_gpu else available_vram_gb

        if effective_vram < required_vram:
            if info.supports_quantization:
                result["warnings"].append(
                    f"VRAM ({effective_vram}GB) below recommended ({required_vram}GB). "
                    "Consider using quantization."
                )
                result["recommendations"].append("Enable 4-bit or 8-bit quantization")
            else:
                result["compatible"] = False
                result["errors"].append(
                    f"Insufficient VRAM ({effective_vram}GB) for {algorithm_name} "
                    f"(requires ~{required_vram}GB)"
                )

        # Multi-GPU check
        if num_gpus > 1 and not info.supports_multi_gpu:
            result["warnings"].append(
                f"{algorithm_name} doesn't support multi-GPU. Only 1 GPU will be used."
            )

        # Special requirements
        if info.requires_reward_model:
            result["recommendations"].append(
                "Ensure a reward model is configured for RLHF training"
            )

        if info.requires_reference_model:
            result["recommendations"].append(
                "A reference model will be loaded for KL divergence computation"
            )

        return result

    def _ensure_initialized(self) -> None:
        """Lazy initialization of default algorithms."""
        if self._initialized:
            return

        # Import here to avoid circular imports
        from .configs import (
            DoRAConfig, AdapterConfig, IA3Config,
            RLHFConfig, DPOConfig, KTOConfig, GRPOConfig, MHCConfig,
            LoRAConfig, QLoRAConfig
        )
        from .trainers import (
            DoRATrainer, AdapterTrainer, IA3Trainer,
            RLHFTrainer, DPOTrainer, KTOTrainer, GRPOTrainer, MHCTrainer,
            LoRATrainer, QLoRATrainer
        )

        # Register LoRA (base)
        self.register(
            name="lora",
            category=AlgorithmCategory.PEFT,
            description="Low-Rank Adaptation for efficient fine-tuning",
            config_class=LoRAConfig,
            trainer_class=LoRATrainer,
            resource_requirement=ResourceRequirement.LOW,
            supports_quantization=True,
        )

        # Register QLoRA
        self.register(
            name="qlora",
            category=AlgorithmCategory.PEFT,
            description="Quantized LoRA - 4-bit quantized fine-tuning",
            config_class=QLoRAConfig,
            trainer_class=QLoRATrainer,
            resource_requirement=ResourceRequirement.LOW,
            supports_quantization=True,
        )

        # Register DoRA
        self.register(
            name="dora",
            category=AlgorithmCategory.PEFT,
            description="Weight-Decomposed Low-Rank Adaptation - improves upon LoRA by decomposing weights",
            config_class=DoRAConfig,
            trainer_class=DoRATrainer,
            resource_requirement=ResourceRequirement.LOW,
            supports_quantization=True,
            aliases=["weight_decomposed_lora"],
        )

        # Register Adapter variants separately
        self.register(
            name="adapter_houlsby",
            category=AlgorithmCategory.PEFT,
            description="Houlsby Adapter - bottleneck adapters after both attention and FFN",
            config_class=AdapterConfig,
            trainer_class=AdapterTrainer,
            resource_requirement=ResourceRequirement.LOW,
            supports_quantization=True,
        )

        self.register(
            name="adapter_pfeiffer",
            category=AlgorithmCategory.PEFT,
            description="Pfeiffer Adapter - efficient bottleneck adapters after FFN only",
            config_class=AdapterConfig,
            trainer_class=AdapterTrainer,
            resource_requirement=ResourceRequirement.LOW,
            supports_quantization=True,
        )

        # Keep generic adapter for backward compatibility
        self.register(
            name="adapter",
            category=AlgorithmCategory.PEFT,
            description="Adapter modules (Houlsby/Pfeiffer) - adds trainable bottleneck layers",
            config_class=AdapterConfig,
            trainer_class=AdapterTrainer,
            resource_requirement=ResourceRequirement.LOW,
            supports_quantization=True,
            aliases=["adapters", "houlsby", "pfeiffer"],
        )

        self.register(
            name="ia3",
            category=AlgorithmCategory.PEFT,
            description="Infused Adapter by Inhibiting and Amplifying Inner Activations",
            config_class=IA3Config,
            trainer_class=IA3Trainer,
            resource_requirement=ResourceRequirement.LOW,
            supports_quantization=True,
            aliases=["ia3_adapter"],
        )

        # Register alignment methods
        self.register(
            name="rlhf_ppo",
            category=AlgorithmCategory.ALIGNMENT,
            description="Reinforcement Learning from Human Feedback with PPO",
            config_class=RLHFConfig,
            trainer_class=RLHFTrainer,
            resource_requirement=ResourceRequirement.HIGH,
            supports_quantization=False,
            requires_reward_model=True,
            requires_reference_model=True,
            aliases=["rlhf", "ppo"],
        )

        self.register(
            name="dpo",
            category=AlgorithmCategory.ALIGNMENT,
            description="Direct Preference Optimization - simpler alternative to RLHF",
            config_class=DPOConfig,
            trainer_class=DPOTrainer,
            resource_requirement=ResourceRequirement.MEDIUM,
            supports_quantization=True,
            requires_reference_model=True,
            aliases=["direct_preference"],
        )

        self.register(
            name="kto",
            category=AlgorithmCategory.ALIGNMENT,
            description="Kahneman-Tversky Optimization - uses prospect theory for alignment",
            config_class=KTOConfig,
            trainer_class=KTOTrainer,
            resource_requirement=ResourceRequirement.MEDIUM,
            supports_quantization=True,
            requires_reference_model=True,
            aliases=["kahneman_tversky"],
        )

        self.register(
            name="grpo",
            category=AlgorithmCategory.ALIGNMENT,
            description="Group Relative Policy Optimization - groups responses for optimization",
            config_class=GRPOConfig,
            trainer_class=GRPOTrainer,
            resource_requirement=ResourceRequirement.HIGH,
            supports_quantization=True,
            requires_reference_model=True,
            aliases=["group_relative"],
        )

        # Register hybrid methods
        self.register(
            name="mhc",
            category=AlgorithmCategory.HYBRID,
            description="Manifold-Constrained Hyper-Connections - advanced architecture modification",
            config_class=MHCConfig,
            trainer_class=MHCTrainer,
            resource_requirement=ResourceRequirement.VERY_HIGH,
            supports_quantization=False,
            aliases=["manifold_hyper", "hyper_connections"],
        )

        self._initialized = True


# Global registry instance
ALGORITHM_REGISTRY = AlgorithmRegistry()


def get_algorithm(name: str) -> Optional[AlgorithmInfo]:
    """Get algorithm info by name."""
    ALGORITHM_REGISTRY._ensure_initialized()
    return ALGORITHM_REGISTRY.get(name)


def list_algorithms(category: Optional[AlgorithmCategory] = None) -> List[str]:
    """List all registered algorithm names, optionally filtered by category."""
    ALGORITHM_REGISTRY._ensure_initialized()

    if category is not None:
        return [info.name for info in ALGORITHM_REGISTRY.list_by_category(category)]
    return ALGORITHM_REGISTRY.list_names()


def create_trainer(
    algorithm_name: str,
    model_name: Optional[str] = None,
    model: Any = None,
    config: Optional[Any] = None,
    apply_phi_filter: bool = False,
    **kwargs,
) -> Any:
    """
    Create a trainer for the specified algorithm.

    Args:
        algorithm_name: Name of the algorithm
        model_name: Name/path of the model (for lazy loading)
        model: The model to train (if already loaded)
        config: Optional algorithm-specific config (uses defaults if None)
        apply_phi_filter: Whether to apply PHI filtering during training
        **kwargs: Additional arguments passed to trainer

    Returns:
        Configured trainer instance
    """
    ALGORITHM_REGISTRY._ensure_initialized()

    info = ALGORITHM_REGISTRY.get(algorithm_name)
    if info is None:
        available = ALGORITHM_REGISTRY.list_names()
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. Available: {available}"
        )

    # Create trainer with model_name or model
    trainer_kwargs = {**kwargs}
    if model_name:
        trainer_kwargs["model_name"] = model_name
    if model is not None:
        trainer_kwargs["model"] = model
    if config is not None:
        trainer_kwargs["config"] = config

    trainer = info.trainer_class(**trainer_kwargs)

    # Set PHI filter attribute
    trainer.apply_phi_filter = apply_phi_filter

    return trainer


def check_algorithm_compatibility(
    algorithm_name: str,
    model_name: str,
    available_vram_gb: Optional[float] = None,
    num_gpus: int = 1,
) -> Dict[str, Any]:
    """
    Check algorithm compatibility with given hardware setup.

    Returns compatibility report with warnings and recommendations.
    """
    ALGORITHM_REGISTRY._ensure_initialized()
    return ALGORITHM_REGISTRY.check_compatibility(
        algorithm_name=algorithm_name,
        model_name=model_name,
        available_vram_gb=available_vram_gb,
        num_gpus=num_gpus,
    )
