"""
Advanced Training Algorithms for MedGemma Models.

Provides implementations for:
- LoRA/QLoRA (Low-Rank Adaptation)
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- Adapter Modules (Houlsby/Pfeiffer)
- IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- RLHF/PPO (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- KTO (Kahneman-Tversky Optimization)
- GRPO (Group Relative Policy Optimization)
- mHC (Manifold-Constrained Hyper-Connections)
"""

from .registry import (
    ALGORITHM_REGISTRY,
    get_algorithm,
    list_algorithms,
    create_trainer,
    check_algorithm_compatibility,
)

from .configs import (
    LoRAConfig,
    QLoRAConfig,
    DoRAConfig,
    AdapterConfig,
    IA3Config,
    RLHFConfig,
    DPOConfig,
    KTOConfig,
    GRPOConfig,
    MHCConfig,
    UnifiedTrainingConfig,
)

from .trainers import (
    LoRATrainer,
    QLoRATrainer,
    DoRATrainer,
    AdapterTrainer,
    IA3Trainer,
    RLHFTrainer,
    DPOTrainer,
    KTOTrainer,
    GRPOTrainer,
    MHCTrainer,
)

from .pipeline import (
    TrainingPipelineManager,
    RayTrainingOrchestrator,
    create_ray_trainer,
)

from .callbacks import (
    MedicalSafetyCallback,
    MedicalEvaluationCallback,
)

__all__ = [
    # Registry
    "ALGORITHM_REGISTRY",
    "get_algorithm",
    "list_algorithms",
    "create_trainer",
    "check_algorithm_compatibility",
    # Configs
    "LoRAConfig",
    "QLoRAConfig",
    "DoRAConfig",
    "AdapterConfig",
    "IA3Config",
    "RLHFConfig",
    "DPOConfig",
    "KTOConfig",
    "GRPOConfig",
    "MHCConfig",
    "UnifiedTrainingConfig",
    # Trainers
    "LoRATrainer",
    "QLoRATrainer",
    "DoRATrainer",
    "AdapterTrainer",
    "IA3Trainer",
    "RLHFTrainer",
    "DPOTrainer",
    "KTOTrainer",
    "GRPOTrainer",
    "MHCTrainer",
    # Pipeline
    "TrainingPipelineManager",
    "RayTrainingOrchestrator",
    "create_ray_trainer",
    # Callbacks
    "MedicalSafetyCallback",
    "MedicalEvaluationCallback",
]
