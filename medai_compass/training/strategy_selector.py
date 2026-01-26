"""
Training Strategy Selector for MedGemma.

Provides a unified interface for selecting and configuring
different distributed training strategies:
- Single GPU
- DeepSpeed ZeRO (stages 1, 2, 3, Infinity)
- Megatron-LM (Tensor & Pipeline Parallelism)
- FSDP / FSDP2
- DualPipe
- 5D Parallelism
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available training strategy types."""
    SINGLE_GPU = "single_gpu"
    DEEPSPEED_ZERO1 = "deepspeed_zero1"
    DEEPSPEED_ZERO2 = "deepspeed_zero2"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"
    DEEPSPEED_INFINITY = "deepspeed_infinity"
    FSDP = "fsdp"
    FSDP2 = "fsdp2"
    MEGATRON_TP = "megatron_tp"
    MEGATRON_PP = "megatron_pp"
    MEGATRON_TP_PP = "megatron_tp_pp"
    DUALPIPE = "dualpipe"
    PARALLELISM_5D = "parallelism_5d"


@dataclass
class TrainingStrategy:
    """
    Base training strategy configuration.

    Represents a selected training strategy with its configuration
    and provides methods to create trainers and export configs.
    """
    name: str
    num_gpus: int = 1
    config: Any = None
    _trainer_class: Optional[Type] = None

    def is_valid(self) -> bool:
        """Check if strategy configuration is valid."""
        return self.name is not None and self.num_gpus >= 1

    def get_trainer(self) -> Optional[Type]:
        """Get the trainer class for this strategy."""
        return self._trainer_class

    def create_trainer(self, **kwargs) -> Any:
        """Create a trainer instance for this strategy."""
        if self._trainer_class is None:
            # Return a mock trainer for strategies without specific trainer
            return MockTrainer(self)

        return self._trainer_class(self.config, **kwargs)

    def to_deepspeed_config(self) -> Dict[str, Any]:
        """Export as DeepSpeed configuration dictionary."""
        if hasattr(self.config, "to_deepspeed_config"):
            return self.config.to_deepspeed_config()

        # Default DeepSpeed config structure
        return {
            "zero_optimization": {
                "stage": getattr(self.config, "zero_stage", 0),
            },
            "bf16": {"enabled": True},
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
        }

    def to_megatron_config(self) -> Dict[str, Any]:
        """Export as Megatron configuration dictionary."""
        config = {}

        if hasattr(self.config, "tensor_parallel_size"):
            config["tensor_model_parallel_size"] = self.config.tensor_parallel_size

        if hasattr(self.config, "pipeline_parallel_size"):
            config["pipeline_model_parallel_size"] = self.config.pipeline_parallel_size

        if hasattr(self.config, "sequence_parallel"):
            config["sequence_parallel"] = self.config.sequence_parallel

        return config


class MockTrainer:
    """Mock trainer for testing and strategies without specific implementations."""

    def __init__(self, strategy: TrainingStrategy):
        self.strategy = strategy

    def train(self, *args, **kwargs):
        logger.info(f"MockTrainer.train called for strategy: {self.strategy.name}")
        return {"status": "mock_completed"}


class TrainingStrategySelector:
    """
    Unified selector for training strategies.

    Provides a single interface to select and configure any
    distributed training strategy supported by MedAI Compass.

    Example:
        >>> selector = TrainingStrategySelector()
        >>> strategy = selector.select("deepspeed_zero3", num_gpus=8)
        >>> trainer = strategy.create_trainer()

        >>> # Auto-select based on model size
        >>> strategy = selector.auto_select(model_params=27e9, num_gpus=8)
    """

    # Memory per parameter in bytes (including gradients, optimizer states)
    MEMORY_PER_PARAM_BYTES = 20

    def __init__(self):
        """Initialize the strategy selector."""
        self._strategies = self._build_strategy_registry()

    def _build_strategy_registry(self) -> Dict[str, StrategyType]:
        """Build registry of available strategies."""
        return {
            "single_gpu": StrategyType.SINGLE_GPU,
            "deepspeed_zero1": StrategyType.DEEPSPEED_ZERO1,
            "deepspeed_zero2": StrategyType.DEEPSPEED_ZERO2,
            "deepspeed_zero3": StrategyType.DEEPSPEED_ZERO3,
            "deepspeed_infinity": StrategyType.DEEPSPEED_INFINITY,
            "fsdp": StrategyType.FSDP,
            "fsdp2": StrategyType.FSDP2,
            "megatron_tp": StrategyType.MEGATRON_TP,
            "megatron_pp": StrategyType.MEGATRON_PP,
            "megatron_tp_pp": StrategyType.MEGATRON_TP_PP,
            "dualpipe": StrategyType.DUALPIPE,
            "parallelism_5d": StrategyType.PARALLELISM_5D,
        }

    def list_strategies(self) -> List[str]:
        """
        List all available training strategies.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def select(
        self,
        strategy_name: str,
        num_gpus: int = 1,
        **kwargs
    ) -> TrainingStrategy:
        """
        Select a training strategy by name.

        Args:
            strategy_name: Name of the strategy to select
            num_gpus: Number of GPUs to use
            **kwargs: Strategy-specific configuration options

        Returns:
            TrainingStrategy with configuration

        Raises:
            ValueError: If strategy name is not recognized
        """
        if strategy_name not in self._strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {self.list_strategies()}"
            )

        strategy_type = self._strategies[strategy_name]

        # Create strategy based on type
        if strategy_type == StrategyType.SINGLE_GPU:
            return self._create_single_gpu_strategy()

        elif strategy_type in (
            StrategyType.DEEPSPEED_ZERO1,
            StrategyType.DEEPSPEED_ZERO2,
            StrategyType.DEEPSPEED_ZERO3,
            StrategyType.DEEPSPEED_INFINITY,
        ):
            return self._create_deepspeed_strategy(strategy_type, num_gpus, **kwargs)

        elif strategy_type in (
            StrategyType.MEGATRON_TP,
            StrategyType.MEGATRON_PP,
            StrategyType.MEGATRON_TP_PP,
        ):
            return self._create_megatron_strategy(strategy_type, num_gpus, **kwargs)

        elif strategy_type == StrategyType.FSDP:
            return self._create_fsdp_strategy(num_gpus, **kwargs)

        elif strategy_type == StrategyType.FSDP2:
            return self._create_fsdp2_strategy(num_gpus, **kwargs)

        elif strategy_type == StrategyType.DUALPIPE:
            return self._create_dualpipe_strategy(num_gpus, **kwargs)

        elif strategy_type == StrategyType.PARALLELISM_5D:
            return self._create_5d_parallelism_strategy(num_gpus, **kwargs)

        raise ValueError(f"Strategy not implemented: {strategy_name}")

    def auto_select(
        self,
        model_params: float,
        num_gpus: int,
        gpu_memory_gb: float = 80,
    ) -> TrainingStrategy:
        """
        Automatically select the best strategy for given model and hardware.

        Args:
            model_params: Number of model parameters
            num_gpus: Number of available GPUs
            gpu_memory_gb: GPU memory in GB (default: 80 for H100)

        Returns:
            Recommended TrainingStrategy
        """
        # Calculate memory requirements
        memory_needed_gb = (model_params * self.MEMORY_PER_PARAM_BYTES) / 1e9

        # Single GPU if model fits
        if memory_needed_gb < gpu_memory_gb * 0.7:
            if num_gpus == 1:
                return self.select("single_gpu")
            else:
                # Use FSDP for data parallelism with some sharding
                return self.select("fsdp", num_gpus=num_gpus)

        # Model needs sharding
        total_memory = num_gpus * gpu_memory_gb

        if memory_needed_gb < total_memory * 0.5:
            # ZeRO-2 is sufficient
            return self.select("deepspeed_zero2", num_gpus=num_gpus)

        elif memory_needed_gb < total_memory * 0.8:
            # Need ZeRO-3
            return self.select("deepspeed_zero3", num_gpus=num_gpus)

        else:
            # Very large model - use 5D parallelism or ZeRO-Infinity
            if num_gpus >= 8:
                # Calculate optimal parallelism dimensions
                tp_size = min(4, num_gpus)
                pp_size = max(1, num_gpus // tp_size // 2)
                dp_size = num_gpus // (tp_size * pp_size)

                return self.select(
                    "parallelism_5d",
                    num_gpus=num_gpus,
                    data_parallel_size=dp_size,
                    tensor_parallel_size=tp_size,
                    pipeline_parallel_size=pp_size,
                )
            else:
                return self.select("deepspeed_infinity", num_gpus=num_gpus)

    def _create_single_gpu_strategy(self) -> TrainingStrategy:
        """Create single GPU training strategy."""
        return TrainingStrategy(
            name="single_gpu",
            num_gpus=1,
            config=None,
        )

    def _create_deepspeed_strategy(
        self,
        strategy_type: StrategyType,
        num_gpus: int,
        **kwargs
    ) -> TrainingStrategy:
        """Create DeepSpeed ZeRO strategy."""
        from medai_compass.training.distributed import DeepSpeedConfig, DeepSpeedTrainer

        # Map strategy type to ZeRO stage
        zero_stages = {
            StrategyType.DEEPSPEED_ZERO1: 1,
            StrategyType.DEEPSPEED_ZERO2: 2,
            StrategyType.DEEPSPEED_ZERO3: 3,
            StrategyType.DEEPSPEED_INFINITY: 3,
        }

        zero_stage = zero_stages[strategy_type]
        offload = strategy_type == StrategyType.DEEPSPEED_INFINITY

        config = DeepSpeedConfig(
            zero_stage=zero_stage,
            offload_optimizer=offload,
            offload_param=offload,
            offload_optimizer_device="nvme" if offload else "cpu",
            **{k: v for k, v in kwargs.items() if hasattr(DeepSpeedConfig, k)}
        )

        return TrainingStrategy(
            name=strategy_type.value,
            num_gpus=num_gpus,
            config=config,
            _trainer_class=DeepSpeedTrainer,
        )

    def _create_megatron_strategy(
        self,
        strategy_type: StrategyType,
        num_gpus: int,
        **kwargs
    ) -> TrainingStrategy:
        """Create Megatron parallelism strategy."""
        from medai_compass.training.distributed import (
            MegatronConfig,
            TensorParallelTrainer,
            PipelineParallelTrainer,
            MegatronTPPPTrainer,
        )

        tp_size = kwargs.get("tensor_parallel_size", 1)
        pp_size = kwargs.get("pipeline_parallel_size", 1)

        if strategy_type == StrategyType.MEGATRON_TP:
            tp_size = kwargs.get("tensor_parallel_size", num_gpus)
            trainer_class = TensorParallelTrainer
        elif strategy_type == StrategyType.MEGATRON_PP:
            pp_size = kwargs.get("pipeline_parallel_size", num_gpus)
            trainer_class = PipelineParallelTrainer
        else:  # MEGATRON_TP_PP
            trainer_class = MegatronTPPPTrainer

        config = MegatronConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            **{k: v for k, v in kwargs.items()
               if hasattr(MegatronConfig, k) and k not in ["tensor_parallel_size", "pipeline_parallel_size"]}
        )

        return TrainingStrategy(
            name=strategy_type.value,
            num_gpus=num_gpus,
            config=config,
            _trainer_class=trainer_class,
        )

    def _create_fsdp_strategy(
        self,
        num_gpus: int,
        **kwargs
    ) -> TrainingStrategy:
        """Create FSDP strategy."""
        from medai_compass.training.distributed import FSDP2Config, FSDP2Trainer

        config = FSDP2Config(
            per_param_sharding=False,
            **{k: v for k, v in kwargs.items() if hasattr(FSDP2Config, k)}
        )

        return TrainingStrategy(
            name="fsdp",
            num_gpus=num_gpus,
            config=config,
            _trainer_class=FSDP2Trainer,
        )

    def _create_fsdp2_strategy(
        self,
        num_gpus: int,
        **kwargs
    ) -> TrainingStrategy:
        """Create FSDP2 with per-parameter sharding strategy."""
        from medai_compass.training.distributed import FSDP2Config, FSDP2Trainer

        config = FSDP2Config(
            per_param_sharding=True,
            use_dtensor=True,
            **{k: v for k, v in kwargs.items() if hasattr(FSDP2Config, k)}
        )

        return TrainingStrategy(
            name="fsdp2",
            num_gpus=num_gpus,
            config=config,
            _trainer_class=FSDP2Trainer,
        )

    def _create_dualpipe_strategy(
        self,
        num_gpus: int,
        **kwargs
    ) -> TrainingStrategy:
        """Create DualPipe strategy."""
        from medai_compass.training.distributed import DualPipeConfig, DualPipeTrainer

        num_stages = kwargs.get("num_stages", num_gpus)
        num_micro_batches = kwargs.get("num_micro_batches", num_stages * 2)

        config = DualPipeConfig(
            num_stages=num_stages,
            num_micro_batches=num_micro_batches,
            **{k: v for k, v in kwargs.items()
               if hasattr(DualPipeConfig, k) and k not in ["num_stages", "num_micro_batches"]}
        )

        return TrainingStrategy(
            name="dualpipe",
            num_gpus=num_gpus,
            config=config,
            _trainer_class=DualPipeTrainer,
        )

    def _create_5d_parallelism_strategy(
        self,
        num_gpus: int,
        **kwargs
    ) -> TrainingStrategy:
        """Create 5D Parallelism strategy."""
        from medai_compass.training.distributed import Parallelism5DConfig, HybridParallelTrainer

        config = Parallelism5DConfig(
            data_parallel_size=kwargs.get("data_parallel_size", 1),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            pipeline_parallel_size=kwargs.get("pipeline_parallel_size", 1),
            sequence_parallel_size=kwargs.get("sequence_parallel_size", 1),
            expert_parallel_size=kwargs.get("expert_parallel_size", 1),
        )

        return TrainingStrategy(
            name="parallelism_5d",
            num_gpus=num_gpus,
            config=config,
            _trainer_class=HybridParallelTrainer,
        )


# Convenience function for quick strategy selection
def select_training_strategy(
    strategy_name: str,
    **kwargs
) -> TrainingStrategy:
    """
    Convenience function to select a training strategy.

    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy-specific options

    Returns:
        Configured TrainingStrategy
    """
    selector = TrainingStrategySelector()
    return selector.select(strategy_name, **kwargs)


def auto_select_training_strategy(
    model_params: float,
    num_gpus: int,
    gpu_memory_gb: float = 80,
) -> TrainingStrategy:
    """
    Convenience function to auto-select training strategy.

    Args:
        model_params: Number of model parameters
        num_gpus: Number of GPUs
        gpu_memory_gb: GPU memory in GB

    Returns:
        Recommended TrainingStrategy
    """
    selector = TrainingStrategySelector()
    return selector.auto_select(model_params, num_gpus, gpu_memory_gb)
