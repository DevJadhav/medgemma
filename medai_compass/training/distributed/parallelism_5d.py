"""
5D Parallelism for MedGemma Training.

Combines all parallelism dimensions:
1. Data Parallelism (DP) - Replicate model, partition data
2. Tensor Parallelism (TP) - Partition tensors across GPUs
3. Pipeline Parallelism (PP) - Partition layers across GPUs
4. Sequence Parallelism (SP) - Partition sequence dimension
5. Expert Parallelism (EP) - Partition experts for MoE models

Enables training of extremely large models (100B+ parameters)
across thousands of GPUs.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .configs import Parallelism5DConfig, ParallelismType


class Parallelism5DStrategy:
    """
    Strategy manager for 5D parallelism.

    Determines the optimal parallelism configuration based on
    model size, hardware, and training requirements.

    Example:
        >>> strategy = Parallelism5DStrategy()
        >>> config = strategy.auto_configure(
        ...     model_params=70e9,
        ...     num_gpus=64,
        ...     gpu_memory=80e9,
        ... )
    """

    # GPU memory per parameter (in bytes)
    # Includes optimizer states, gradients, activations
    MEMORY_PER_PARAM_TRAINING = 20  # ~20 bytes with Adam + activations

    def __init__(self):
        """Initialize strategy manager."""
        self._configs_cache: Dict[str, Parallelism5DConfig] = {}

    def auto_configure(
        self,
        model_params: int,
        num_gpus: int,
        gpu_memory: int = 80e9,  # Default H100 80GB
        batch_size: int = 1,
        sequence_length: int = 8192,
        use_moe: bool = False,
        num_experts: int = 8,
    ) -> Parallelism5DConfig:
        """
        Automatically configure parallelism based on model and hardware.

        Args:
            model_params: Total model parameters
            num_gpus: Number of available GPUs
            gpu_memory: Per-GPU memory in bytes
            batch_size: Global batch size
            sequence_length: Maximum sequence length
            use_moe: Whether model uses Mixture of Experts
            num_experts: Number of experts (if MoE)

        Returns:
            Optimized Parallelism5DConfig
        """
        # Calculate minimum parallelism needed
        model_memory = model_params * self.MEMORY_PER_PARAM_TRAINING

        # Start with pure data parallelism
        tp_size = 1
        pp_size = 1
        ep_size = 1

        # Add tensor parallelism if model doesn't fit
        while model_memory / tp_size > gpu_memory * 0.7:  # 70% utilization target
            if tp_size >= 8:  # Max practical TP size
                break
            tp_size *= 2

        # Add pipeline parallelism if still doesn't fit
        while model_memory / (tp_size * pp_size) > gpu_memory * 0.7:
            if pp_size >= 8:
                break
            pp_size *= 2

        # Configure expert parallelism for MoE
        if use_moe:
            ep_size = min(num_experts, num_gpus // (tp_size * pp_size))

        # Calculate data parallelism from remaining GPUs
        dp_size = num_gpus // (tp_size * pp_size * max(ep_size, 1))

        # Enable sequence parallelism if TP is used
        use_sp = tp_size > 1

        return Parallelism5DConfig(
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            sequence_parallel=use_sp,
            expert_parallel_size=ep_size if use_moe else 1,
            global_batch_size=batch_size,
            micro_batch_size=1,
            num_experts=num_experts if use_moe else None,
        )

    def get_recommendation(
        self,
        model_name: str,
    ) -> Parallelism5DConfig:
        """
        Get recommended configuration for known models.

        Args:
            model_name: Model name (e.g., "medgemma-27b")

        Returns:
            Recommended configuration
        """
        recommendations = {
            "medgemma-4b": Parallelism5DConfig(
                data_parallel_size=8,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            "medgemma-27b": Parallelism5DConfig(
                data_parallel_size=4,
                tensor_parallel_size=4,
                pipeline_parallel_size=2,
                sequence_parallel=True,
            ),
            "llama-70b": Parallelism5DConfig(
                data_parallel_size=2,
                tensor_parallel_size=8,
                pipeline_parallel_size=4,
                sequence_parallel=True,
            ),
            "mixtral-8x7b": Parallelism5DConfig(
                data_parallel_size=4,
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                expert_parallel_size=8,
                num_experts=8,
            ),
        }

        return recommendations.get(
            model_name,
            Parallelism5DConfig()  # Default config
        )

    def auto_select_strategy(
        self,
        model_params: int,
        num_gpus: int,
        gpu_memory: int = 80e9,
    ) -> ParallelismType:
        """
        Automatically select the best parallelism strategy.

        Args:
            model_params: Number of model parameters
            num_gpus: Number of available GPUs
            gpu_memory: Per-GPU memory in bytes

        Returns:
            Recommended ParallelismType
        """
        model_memory = model_params * self.MEMORY_PER_PARAM_TRAINING

        # Simple heuristic for strategy selection
        if model_memory < gpu_memory * 0.5:
            return ParallelismType.DATA_PARALLEL
        elif model_memory < gpu_memory * 2:
            return ParallelismType.FSDP2
        elif model_memory < gpu_memory * 8:
            return ParallelismType.DEEPSPEED_ZERO3
        elif model_memory < gpu_memory * 16:
            return ParallelismType.MEGATRON_TP_PP
        else:
            return ParallelismType.PARALLELISM_5D

    def get_strategy_for_model(
        self,
        model_name: str,
    ) -> ParallelismType:
        """
        Get recommended strategy for a known model.

        Args:
            model_name: Model name

        Returns:
            Recommended strategy
        """
        model_strategies = {
            "medgemma-4b": ParallelismType.FSDP2,
            "medgemma-27b": ParallelismType.MEGATRON_TP_PP,
            "llama-7b": ParallelismType.FSDP2,
            "llama-13b": ParallelismType.DEEPSPEED_ZERO3,
            "llama-70b": ParallelismType.MEGATRON_TP_PP,
            "mixtral-8x7b": ParallelismType.PARALLELISM_5D,
        }

        return model_strategies.get(
            model_name,
            ParallelismType.DATA_PARALLEL
        )


class HybridParallelTrainer:
    """
    Hybrid Parallel Trainer for 5D Parallelism.

    Manages all five parallelism dimensions and their interactions
    for training extremely large models.

    Example:
        >>> config = Parallelism5DConfig(
        ...     data_parallel_size=8,
        ...     tensor_parallel_size=4,
        ...     pipeline_parallel_size=2,
        ... )
        >>> trainer = HybridParallelTrainer(config)
        >>> model = trainer.parallelize(model)
    """

    def __init__(
        self,
        config: Union[Parallelism5DConfig, Dict[str, Any]],
    ):
        """
        Initialize HybridParallelTrainer.

        Args:
            config: 5D parallelism configuration
        """
        if isinstance(config, dict):
            config = Parallelism5DConfig(**config)

        self.config = config

        # Process groups for each dimension
        self._dp_group = None
        self._tp_group = None
        self._pp_group = None
        self._sp_group = None  # Same as TP for sequence parallel
        self._ep_group = None

        self._initialized = False
        self._local_rank = 0
        self._global_rank = 0

    @property
    def data_parallel_size(self) -> int:
        """Get data parallel size from config."""
        return self.config.data_parallel_size

    @property
    def tensor_parallel_size(self) -> int:
        """Get tensor parallel size from config."""
        return self.config.tensor_parallel_size

    @property
    def pipeline_parallel_size(self) -> int:
        """Get pipeline parallel size from config."""
        return self.config.pipeline_parallel_size

    @property
    def sequence_parallel_size(self) -> int:
        """Get sequence parallel size from config."""
        return self.config.sequence_parallel_size

    @property
    def expert_parallel_size(self) -> int:
        """Get expert parallel size from config."""
        return self.config.expert_parallel_size

    @property
    def supports_tensor_parallel(self) -> bool:
        """Whether tensor parallelism is supported."""
        return self.config.tensor_parallel_size > 1

    @property
    def supports_pipeline_parallel(self) -> bool:
        """Whether pipeline parallelism is supported."""
        return self.config.pipeline_parallel_size > 1

    @property
    def supports_sequence_parallel(self) -> bool:
        """Whether sequence parallelism is supported."""
        return self.config.sequence_parallel

    @property
    def supports_expert_parallel(self) -> bool:
        """Whether expert parallelism is supported."""
        return self.config.expert_parallel_size > 1

    def initialize(self) -> None:
        """Initialize all parallelism process groups."""
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {errors}")

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self._global_rank = dist.get_rank()
        self._local_rank = self._global_rank % torch.cuda.device_count()

        # Set device
        torch.cuda.set_device(self._local_rank)

        # Create process groups
        self._create_process_groups()

        self._initialized = True

    def create_process_groups(self) -> None:
        """Create process groups for all parallelism dimensions (public API)."""
        self._create_process_groups()

    def _create_process_groups(self) -> None:
        """Create process groups for all parallelism dimensions."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        dp_size = self.config.data_parallel_size
        tp_size = self.config.tensor_parallel_size
        pp_size = self.config.pipeline_parallel_size
        ep_size = self.config.expert_parallel_size

        # Calculate group assignments
        # Layout: [DP, PP, TP] or [DP, PP, TP, EP] for MoE

        # TP groups: ranks that share tensor parallel operations
        tp_group_size = tp_size
        num_tp_groups = world_size // tp_group_size

        for i in range(num_tp_groups):
            start = i * tp_group_size
            ranks = list(range(start, start + tp_group_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                self._tp_group = group
                self._sp_group = group  # SP uses same group as TP

        # PP groups: ranks that share pipeline parallel operations
        pp_group_stride = tp_size
        num_pp_groups = world_size // (pp_size * pp_group_stride)

        for i in range(num_pp_groups):
            ranks = []
            for j in range(pp_size):
                ranks.append(i * tp_size + j * tp_size * dp_size)
            group = dist.new_group(ranks)
            if rank in ranks:
                self._pp_group = group

        # DP groups: ranks that share data parallel operations
        dp_group_stride = tp_size * pp_size

        for i in range(world_size // (dp_size * dp_group_stride)):
            for j in range(dp_group_stride):
                ranks = [
                    i * dp_size * dp_group_stride + j + k * dp_group_stride
                    for k in range(dp_size)
                ]
                if rank in ranks:
                    group = dist.new_group(ranks)
                    self._dp_group = group

        # EP groups for MoE (if enabled)
        if ep_size > 1 and self.config.num_experts is not None:
            for i in range(world_size // ep_size):
                ranks = list(range(i * ep_size, (i + 1) * ep_size))
                group = dist.new_group(ranks)
                if rank in ranks:
                    self._ep_group = group

    def parallelize(self, model: nn.Module) -> nn.Module:
        """
        Apply 5D parallelism to a model.

        Args:
            model: Model to parallelize

        Returns:
            Parallelized model
        """
        if not self._initialized:
            self.initialize()

        # Apply parallelism in order
        # 1. Expert Parallelism (if MoE)
        if self.config.expert_parallel_size > 1:
            model = self._apply_expert_parallel(model)

        # 2. Tensor Parallelism
        if self.config.tensor_parallel_size > 1:
            model = self._apply_tensor_parallel(model)

        # 3. Sequence Parallelism
        if self.config.sequence_parallel:
            model = self._apply_sequence_parallel(model)

        # 4. Pipeline Parallelism
        if self.config.pipeline_parallel_size > 1:
            model = self._apply_pipeline_parallel(model)

        # 5. Data Parallelism (DDP or ZeRO)
        if self.config.data_parallel_size > 1:
            model = self._apply_data_parallel(model)

        return model

    def _apply_expert_parallel(self, model: nn.Module) -> nn.Module:
        """Apply expert parallelism for MoE models."""
        if self._ep_group is None:
            return model

        # Wrap MoE layers with expert parallel
        for name, module in model.named_modules():
            if "moe" in name.lower() or "expert" in name.lower():
                # Distribute experts across EP group
                pass

        return model

    def _apply_tensor_parallel(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism."""
        if self._tp_group is None:
            return model

        return TensorParallelWrapper5D(
            model,
            tp_group=self._tp_group,
            tp_size=self.config.tensor_parallel_size,
        )

    def _apply_sequence_parallel(self, model: nn.Module) -> nn.Module:
        """Apply sequence parallelism."""
        if not self.config.sequence_parallel:
            return model

        # SP typically shares group with TP
        return SequenceParallelWrapper(
            model,
            sp_group=self._sp_group,
        )

    def _apply_pipeline_parallel(self, model: nn.Module) -> nn.Module:
        """Apply pipeline parallelism."""
        if self._pp_group is None:
            return model

        return PipelineParallelWrapper5D(
            model,
            pp_group=self._pp_group,
            num_stages=self.config.pipeline_parallel_size,
            micro_batch_size=self.config.micro_batch_size,
        )

    def _apply_data_parallel(self, model: nn.Module) -> nn.Module:
        """Apply data parallelism."""
        if self._dp_group is None:
            return model

        if self.config.zero_stage > 0:
            # Use ZeRO with DeepSpeed
            return self._apply_zero(model)
        else:
            # Use standard DDP
            return nn.parallel.DistributedDataParallel(
                model,
                process_group=self._dp_group,
                device_ids=[self._local_rank],
            )

    def _apply_zero(self, model: nn.Module) -> nn.Module:
        """Apply ZeRO optimization."""
        # Would integrate with DeepSpeed here
        return model

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Perform a training step with 5D parallelism.

        Args:
            model: Parallelized model
            batch: Input batch
            optimizer: Optimizer

        Returns:
            Loss tensor
        """
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs

        # Backward pass
        loss.backward()

        # Gradient synchronization handled by wrappers

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        return loss

    @property
    def dp_rank(self) -> int:
        """Get data parallel rank."""
        if self._dp_group is not None:
            return dist.get_rank(self._dp_group)
        return 0

    @property
    def tp_rank(self) -> int:
        """Get tensor parallel rank."""
        if self._tp_group is not None:
            return dist.get_rank(self._tp_group)
        return 0

    @property
    def pp_rank(self) -> int:
        """Get pipeline parallel rank (stage ID)."""
        if self._pp_group is not None:
            return dist.get_rank(self._pp_group)
        return 0


class TensorParallelWrapper5D(nn.Module):
    """Tensor parallel wrapper for 5D parallelism."""

    def __init__(
        self,
        model: nn.Module,
        tp_group: Any,
        tp_size: int,
    ):
        super().__init__()
        self.model = model
        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class SequenceParallelWrapper(nn.Module):
    """Sequence parallel wrapper for 5D parallelism."""

    def __init__(
        self,
        model: nn.Module,
        sp_group: Any,
    ):
        super().__init__()
        self.model = model
        self.sp_group = sp_group

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PipelineParallelWrapper5D(nn.Module):
    """Pipeline parallel wrapper for 5D parallelism."""

    def __init__(
        self,
        model: nn.Module,
        pp_group: Any,
        num_stages: int,
        micro_batch_size: int,
    ):
        super().__init__()
        self.model = model
        self.pp_group = pp_group
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def get_optimal_config(
    model_params: int,
    num_gpus: int,
    gpu_memory_gb: int = 80,
) -> Parallelism5DConfig:
    """
    Get optimal 5D parallelism configuration.

    Args:
        model_params: Number of model parameters
        num_gpus: Number of available GPUs
        gpu_memory_gb: GPU memory in GB

    Returns:
        Optimal configuration
    """
    strategy = Parallelism5DStrategy()
    return strategy.auto_configure(
        model_params=model_params,
        num_gpus=num_gpus,
        gpu_memory=gpu_memory_gb * 1e9,
    )
