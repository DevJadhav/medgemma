"""
Distributed Training Module for MedGemma.

Provides advanced distributed training strategies:
- DeepSpeed ZeRO (stages 1, 2, 3, Infinity)
- Megatron-LM (Tensor & Pipeline Parallelism)
- FSDP2 (Per-Parameter Sharding)
- 5D Parallelism (DP + TP + PP + SP + EP)

Example:
    >>> from medai_compass.training.distributed import DeepSpeedTrainer, DeepSpeedConfig
    >>> config = DeepSpeedConfig(zero_stage=3, offload_optimizer=True)
    >>> trainer = DeepSpeedTrainer(config)
    >>> trainer.train(dataset)
"""

from .configs import (
    DeepSpeedConfig,
    MegatronConfig,
    FSDP2Config,
    Parallelism5DConfig,
    ParallelismStrategy,
    ParallelismType,
)
from .deepspeed_trainer import (
    DeepSpeedTrainer,
    ZeROOptimizer,
    check_deepspeed_available,
)
from .megatron_parallelism import (
    TensorParallelTrainer,
    PipelineParallelTrainer,
    MegatronOptimizer,
    check_megatron_available,
)
from .fsdp2_trainer import (
    FSDP2Trainer,
    DTensorSharding,
    check_fsdp2_available,
)
from .parallelism_5d import (
    HybridParallelTrainer,
    Parallelism5DStrategy,
)

__all__ = [
    # Configurations
    "DeepSpeedConfig",
    "MegatronConfig",
    "FSDP2Config",
    "Parallelism5DConfig",
    "ParallelismStrategy",
    "ParallelismType",
    # DeepSpeed
    "DeepSpeedTrainer",
    "ZeROOptimizer",
    "check_deepspeed_available",
    # Megatron-LM
    "TensorParallelTrainer",
    "PipelineParallelTrainer",
    "MegatronOptimizer",
    "check_megatron_available",
    # FSDP2
    "FSDP2Trainer",
    "DTensorSharding",
    "check_fsdp2_available",
    # 5D Parallelism
    "HybridParallelTrainer",
    "Parallelism5DStrategy",
]
