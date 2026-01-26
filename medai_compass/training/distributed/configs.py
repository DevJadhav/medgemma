"""
Configuration Classes for Distributed Training.

Provides configuration dataclasses for:
- DeepSpeed ZeRO (stages 1, 2, 3, Infinity)
- Megatron-LM (Tensor & Pipeline Parallelism)
- FSDP2 (Per-Parameter Sharding)
- 5D Parallelism (DP + TP + PP + SP + EP)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class ParallelismType(Enum):
    """Enumeration of distributed training strategy types."""
    DATA_PARALLEL = "data_parallel"
    DEEPSPEED_ZERO1 = "deepspeed_zero1"
    DEEPSPEED_ZERO2 = "deepspeed_zero2"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"
    DEEPSPEED_INFINITY = "deepspeed_infinity"
    FSDP = "fsdp"
    FSDP2 = "fsdp2"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    MEGATRON_TP_PP = "megatron_tp_pp"
    PARALLELISM_5D = "parallelism_5d"


class ParallelismStrategy:
    """
    Strategy selector for distributed training.

    Provides automatic strategy selection based on model size,
    hardware, and training requirements.

    Example:
        >>> strategy = ParallelismStrategy()
        >>> config = strategy.select_for_model(
        ...     model_params=27e9,
        ...     num_gpus=8,
        ...     gpu_memory_gb=80,
        ... )
    """

    # Memory per parameter during training (bytes)
    MEMORY_PER_PARAM = 20

    def __init__(self):
        """Initialize ParallelismStrategy."""
        pass

    def select_for_model(
        self,
        model_params: int,
        num_gpus: int,
        gpu_memory_gb: int = 80,
    ) -> "Parallelism5DConfig":
        """
        Select optimal parallelism configuration for a model.

        Args:
            model_params: Number of model parameters
            num_gpus: Number of available GPUs
            gpu_memory_gb: Per-GPU memory in GB

        Returns:
            Optimal Parallelism5DConfig
        """
        gpu_memory = gpu_memory_gb * 1e9
        model_memory = model_params * self.MEMORY_PER_PARAM

        # Start with data parallelism
        tp_size = 1
        pp_size = 1

        # Add tensor parallelism if model doesn't fit
        while model_memory / tp_size > gpu_memory * 0.7:
            if tp_size >= 8:
                break
            tp_size *= 2

        # Add pipeline parallelism if still doesn't fit
        while model_memory / (tp_size * pp_size) > gpu_memory * 0.7:
            if pp_size >= 8:
                break
            pp_size *= 2

        # Calculate data parallel size
        dp_size = num_gpus // (tp_size * pp_size)
        dp_size = max(1, dp_size)

        # Enable sequence parallelism with TP
        use_sp = tp_size > 1

        return Parallelism5DConfig(
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            sequence_parallel=use_sp,
        )

    def get_strategy_type(
        self,
        model_params: int,
        num_gpus: int,
        gpu_memory_gb: int = 80,
    ) -> ParallelismType:
        """
        Get recommended strategy type for a model.

        Args:
            model_params: Number of model parameters
            num_gpus: Number of available GPUs
            gpu_memory_gb: Per-GPU memory in GB

        Returns:
            Recommended ParallelismType
        """
        gpu_memory = gpu_memory_gb * 1e9
        model_memory = model_params * self.MEMORY_PER_PARAM

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


@dataclass
class DeepSpeedConfig:
    """
    Configuration for DeepSpeed ZeRO training.

    Supports ZeRO stages 1, 2, 3, and Infinity with various
    memory optimization techniques.

    Example:
        >>> config = DeepSpeedConfig(
        ...     zero_stage=3,
        ...     offload_optimizer=True,
        ...     offload_param=True,
        ... )
    """

    # ZeRO Stage Configuration
    zero_stage: int = 3
    """ZeRO optimization stage (1, 2, or 3)"""

    @property
    def partition_optimizer_states(self) -> bool:
        """Whether optimizer states are partitioned (ZeRO-1+)."""
        return self.zero_stage >= 1

    @property
    def partition_gradients(self) -> bool:
        """Whether gradients are partitioned (ZeRO-2+)."""
        return self.zero_stage >= 2

    @property
    def partition_parameters(self) -> bool:
        """Whether parameters are partitioned (ZeRO-3)."""
        return self.zero_stage >= 3

    # Offloading Options (ZeRO-Infinity)
    offload_optimizer: bool = False
    """Offload optimizer states to CPU"""

    offload_param: bool = False
    """Offload parameters to CPU (ZeRO-3 only)"""

    offload_optimizer_device: str = "cpu"
    """Device for optimizer offloading ('cpu' or 'nvme')"""

    offload_param_device: str = "cpu"
    """Device for parameter offloading ('cpu' or 'nvme')"""

    nvme_path: Optional[str] = None
    """Path for NVMe offloading (ZeRO-Infinity)"""

    # Memory Optimization
    contiguous_gradients: bool = True
    """Use contiguous gradient buffers"""

    overlap_comm: bool = True
    """Overlap gradient communication with backward pass"""

    reduce_bucket_size: int = 500_000_000
    """Size of allreduce buckets in elements"""

    allgather_bucket_size: int = 500_000_000
    """Size of allgather buckets for ZeRO-3"""

    stage3_prefetch_bucket_size: int = 50_000_000
    """Prefetch bucket size for ZeRO-3"""

    stage3_param_persistence_threshold: int = 100_000
    """Parameter persistence threshold for ZeRO-3"""

    stage3_max_live_parameters: int = 1_000_000_000
    """Max live parameters for ZeRO-3"""

    stage3_max_reuse_distance: int = 1_000_000_000
    """Max reuse distance for ZeRO-3"""

    # Mixed Precision
    fp16_enabled: bool = False
    """Enable FP16 mixed precision"""

    bf16_enabled: bool = True
    """Enable BF16 mixed precision"""

    fp16_opt_level: str = "O2"
    """APEX FP16 optimization level"""

    # Gradient Handling
    gradient_clipping: float = 1.0
    """Gradient clipping value"""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps"""

    # Activation Checkpointing
    activation_checkpointing: bool = False
    """Enable activation checkpointing"""

    cpu_checkpointing: bool = False
    """Checkpoint activations to CPU"""

    partition_activations: bool = False
    """Partition activations across data parallel ranks"""

    # Communication
    allreduce_always_fp32: bool = False
    """Always use FP32 for allreduce"""

    communication_data_type: Optional[str] = None
    """Override communication data type"""

    def to_deepspeed_config(self) -> Dict[str, Any]:
        """Convert to DeepSpeed JSON config format."""
        config = {
            "zero_optimization": {
                "stage": self.zero_stage,
                "contiguous_gradients": self.contiguous_gradients,
                "overlap_comm": self.overlap_comm,
                "reduce_bucket_size": self.reduce_bucket_size,
                "allgather_bucket_size": self.allgather_bucket_size,
            },
            "gradient_clipping": self.gradient_clipping,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

        # ZeRO-3 specific settings
        if self.zero_stage == 3:
            config["zero_optimization"].update({
                "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.stage3_param_persistence_threshold,
                "stage3_max_live_parameters": self.stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.stage3_max_reuse_distance,
            })

        # Offloading
        if self.offload_optimizer:
            config["zero_optimization"]["offload_optimizer"] = {
                "device": self.offload_optimizer_device,
                "pin_memory": True,
            }
            if self.offload_optimizer_device == "nvme" and self.nvme_path:
                config["zero_optimization"]["offload_optimizer"]["nvme_path"] = self.nvme_path

        if self.offload_param:
            config["zero_optimization"]["offload_param"] = {
                "device": self.offload_param_device,
                "pin_memory": True,
            }
            if self.offload_param_device == "nvme" and self.nvme_path:
                config["zero_optimization"]["offload_param"]["nvme_path"] = self.nvme_path

        # Mixed precision
        if self.bf16_enabled:
            config["bf16"] = {"enabled": True}
        elif self.fp16_enabled:
            config["fp16"] = {
                "enabled": True,
                "opt_level": self.fp16_opt_level,
            }

        # Activation checkpointing
        if self.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": self.partition_activations,
                "cpu_checkpointing": self.cpu_checkpointing,
            }

        return config


@dataclass
class MegatronConfig:
    """
    Configuration for Megatron-LM parallelism.

    Supports Tensor Parallelism (TP), Pipeline Parallelism (PP),
    and Sequence Parallelism (SP).

    Example:
        >>> config = MegatronConfig(
        ...     tensor_parallel_size=4,
        ...     pipeline_parallel_size=2,
        ...     sequence_parallel=True,
        ... )
    """

    # Tensor Parallelism
    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism"""

    tensor_model_parallel_size: int = 1
    """Alias for tensor_parallel_size"""

    # Pipeline Parallelism
    pipeline_parallel_size: int = 1
    """Number of pipeline stages"""

    pipeline_model_parallel_size: int = 1
    """Alias for pipeline_parallel_size"""

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """Virtual pipeline stages for interleaved scheduling"""

    # Sequence Parallelism
    sequence_parallel: bool = False
    """Enable sequence parallelism"""

    # Micro-batching
    micro_batch_size: int = 1
    """Micro-batch size for pipeline parallelism"""

    num_micro_batches: int = 1
    """Number of micro-batches for pipeline parallelism"""

    global_batch_size: int = 8
    """Global batch size across all GPUs"""

    # Interleaved Pipeline
    interleaved_pipeline: bool = False
    """Enable interleaved pipeline scheduling"""

    # Communication
    async_tensor_model_parallel_allreduce: bool = True
    """Async allreduce for tensor parallelism"""

    pipeline_dtype: str = "bf16"
    """Data type for pipeline communication"""

    # Layer Distribution
    num_layers: Optional[int] = None
    """Total number of transformer layers"""

    encoder_num_layers: Optional[int] = None
    """Number of encoder layers (for encoder-decoder)"""

    decoder_num_layers: Optional[int] = None
    """Number of decoder layers (for encoder-decoder)"""

    # Memory Optimization
    recompute_granularity: str = "selective"
    """Activation recomputation granularity ('full', 'selective', 'none')"""

    recompute_method: str = "uniform"
    """Recomputation method ('uniform', 'block')"""

    recompute_num_layers: int = 1
    """Number of layers to recompute"""

    distribute_saved_activations: bool = True
    """Distribute saved activations across TP ranks"""

    # Expert Parallelism (MoE)
    expert_parallel_size: int = 1
    """Number of GPUs for expert parallelism"""

    num_experts: Optional[int] = None
    """Number of experts for MoE models"""

    def get_data_parallel_size(self, world_size: int) -> int:
        """Calculate data parallel size given world size."""
        return world_size // (self.tensor_parallel_size * self.pipeline_parallel_size)


@dataclass
class FSDP2Config:
    """
    Configuration for FSDP2 (Fully Sharded Data Parallel v2).

    FSDP2 provides per-parameter sharding using DTensor,
    offering finer granularity and better composability.

    Example:
        >>> config = FSDP2Config(
        ...     sharding_strategy="FULL_SHARD",
        ...     use_dtensor=True,
        ... )
    """

    # Sharding Strategy
    sharding_strategy: str = "FULL_SHARD"
    """Sharding strategy ('FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD')"""

    # DTensor Integration
    use_dtensor: bool = True
    """Use DTensor for per-parameter sharding"""

    per_param_sharding: bool = True
    """Enable per-parameter sharding"""

    compose_with_tp: bool = False
    """Compose FSDP2 with tensor parallelism"""

    tp_size: int = 1
    """Tensor parallel size when composing with TP"""

    @property
    def per_parameter_sharding(self) -> bool:
        """Whether per-parameter sharding is enabled."""
        return self.use_dtensor and self.per_param_sharding

    # CPU Offloading
    cpu_offload: bool = False
    """Offload parameters to CPU"""

    offload_params: bool = False
    """Alias for cpu_offload"""

    # Mixed Precision
    mixed_precision: bool = True
    """Enable mixed precision training"""

    param_dtype: str = "bf16"
    """Parameter data type"""

    reduce_dtype: str = "fp32"
    """Gradient reduction data type"""

    buffer_dtype: str = "bf16"
    """Buffer data type"""

    # Communication
    backward_prefetch: str = "BACKWARD_PRE"
    """Prefetch strategy ('BACKWARD_PRE', 'BACKWARD_POST', 'NONE')"""

    forward_prefetch: bool = True
    """Enable forward prefetching"""

    limit_all_gathers: bool = True
    """Limit concurrent all-gathers for memory efficiency"""

    # Checkpointing
    use_orig_params: bool = True
    """Use original parameter shapes (required for optimizer state dict)"""

    # Wrapping Policy
    auto_wrap_policy: str = "transformer_auto_wrap_policy"
    """Auto-wrap policy for FSDP wrapping"""

    min_num_params: int = 100_000
    """Minimum parameters for automatic wrapping"""

    # State Dict
    state_dict_type: str = "FULL_STATE_DICT"
    """State dict type ('FULL_STATE_DICT', 'SHARDED_STATE_DICT', 'LOCAL_STATE_DICT')"""


@dataclass
class Parallelism5DConfig:
    """
    Configuration for 5D Parallelism.

    Combines all parallelism dimensions:
    1. Data Parallelism (DP)
    2. Tensor Parallelism (TP)
    3. Pipeline Parallelism (PP)
    4. Sequence Parallelism (SP)
    5. Expert Parallelism (EP)

    Example:
        >>> config = Parallelism5DConfig(
        ...     data_parallel_size=8,
        ...     tensor_parallel_size=4,
        ...     pipeline_parallel_size=2,
        ...     sequence_parallel=True,
        ...     expert_parallel_size=2,
        ... )
    """

    # Parallelism Dimensions
    data_parallel_size: int = 1
    """Number of data parallel replicas"""

    tensor_parallel_size: int = 1
    """Number of tensor parallel partitions"""

    pipeline_parallel_size: int = 1
    """Number of pipeline stages"""

    sequence_parallel: bool = False
    """Enable sequence parallelism"""

    sequence_parallel_size: int = 1
    """Sequence parallel size (typically same as TP size)"""

    @property
    def sequence_parallel_enabled(self) -> bool:
        """Whether sequence parallelism is enabled."""
        return self.sequence_parallel or self.sequence_parallel_size > 1

    expert_parallel_size: int = 1
    """Number of expert parallel partitions (for MoE)"""

    # World Size Validation
    @property
    def world_size_requirement(self) -> int:
        """Calculate required world size for this configuration."""
        base_size = (
            self.data_parallel_size *
            self.tensor_parallel_size *
            self.pipeline_parallel_size
        )
        if self.expert_parallel_size > 1:
            base_size *= self.expert_parallel_size
        return base_size

    def total_gpus_required(self) -> int:
        """Calculate total GPUs required (alias for world_size_requirement)."""
        return self.world_size_requirement

    @property
    def total_gpus(self) -> int:
        """Total GPUs required for this configuration."""
        return self.world_size_requirement

    # Batch Sizing
    global_batch_size: int = 8
    """Global batch size across all data parallel replicas"""

    micro_batch_size: int = 1
    """Micro-batch size for pipeline parallelism"""

    # Communication Groups
    dp_comm_type: str = "nccl"
    """Communication backend for data parallelism"""

    tp_comm_type: str = "nccl"
    """Communication backend for tensor parallelism"""

    pp_comm_type: str = "nccl"
    """Communication backend for pipeline parallelism"""

    # Memory Optimization
    activation_checkpointing: bool = True
    """Enable activation checkpointing"""

    checkpoint_num_layers: int = 1
    """Number of layers per checkpoint"""

    # ZeRO Integration
    zero_stage: int = 0
    """ZeRO stage for data parallelism (0, 1, 2, or 3)"""

    # Async Communication
    overlap_grad_reduce: bool = True
    """Overlap gradient reduction with backward pass"""

    overlap_param_gather: bool = True
    """Overlap parameter gathering with forward pass"""

    # Expert Configuration (MoE)
    num_experts: Optional[int] = None
    """Number of experts in MoE layers"""

    experts_per_rank: Optional[int] = None
    """Number of experts per rank"""

    def validate(self) -> List[str]:
        """
        Validate configuration and return any errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check parallel sizes are positive
        if self.data_parallel_size < 1:
            errors.append("data_parallel_size must be >= 1")
        if self.tensor_parallel_size < 1:
            errors.append("tensor_parallel_size must be >= 1")
        if self.pipeline_parallel_size < 1:
            errors.append("pipeline_parallel_size must be >= 1")
        if self.expert_parallel_size < 1:
            errors.append("expert_parallel_size must be >= 1")

        # Check batch sizes
        if self.global_batch_size < self.data_parallel_size:
            errors.append("global_batch_size must be >= data_parallel_size")

        # Check ZeRO stage
        if self.zero_stage not in [0, 1, 2, 3]:
            errors.append("zero_stage must be 0, 1, 2, or 3")

        # Check expert configuration
        if self.expert_parallel_size > 1 and self.num_experts is None:
            errors.append("num_experts required when expert_parallel_size > 1")

        return errors

    def get_process_group_config(self) -> Dict[str, Any]:
        """Get configuration for creating process groups."""
        return {
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
        }


@dataclass
class HybridParallelismConfig:
    """
    Configuration for hybrid parallelism combining multiple strategies.

    This allows mixing different strategies like FSDP + TP or
    DeepSpeed + PP for optimal resource utilization.
    """

    # Primary Strategy
    primary_strategy: ParallelismType = ParallelismType.FSDP2
    """Primary parallelism strategy"""

    # Secondary Strategies
    use_tensor_parallel: bool = False
    """Add tensor parallelism"""

    use_pipeline_parallel: bool = False
    """Add pipeline parallelism"""

    use_sequence_parallel: bool = False
    """Add sequence parallelism"""

    # Strategy-specific configs
    deepspeed_config: Optional[DeepSpeedConfig] = None
    """DeepSpeed configuration (if using DeepSpeed)"""

    megatron_config: Optional[MegatronConfig] = None
    """Megatron configuration (if using TP/PP)"""

    fsdp2_config: Optional[FSDP2Config] = None
    """FSDP2 configuration (if using FSDP2)"""

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.deepspeed_config is None:
            self.deepspeed_config = DeepSpeedConfig()
        if self.megatron_config is None:
            self.megatron_config = MegatronConfig()
        if self.fsdp2_config is None:
            self.fsdp2_config = FSDP2Config()
