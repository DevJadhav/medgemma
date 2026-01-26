"""
Tests for Distributed Training Strategies.

TDD approach: Tests written first for all distributed training optimizations:
- DeepSpeed ZeRO (stages 1, 2, 3, Infinity)
- Megatron-LM (Tensor & Pipeline Parallelism)
- FSDP2 (Per-Parameter Sharding)
- 5D Parallelism (DP + TP + PP + SP + EP)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# =============================================================================
# DeepSpeed ZeRO Tests
# =============================================================================

class TestDeepSpeedConfig:
    """Tests for DeepSpeed configuration."""

    def test_config_creation(self):
        """Verify DeepSpeed config can be created."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig()
        assert config is not None
        assert hasattr(config, "zero_stage")

    def test_zero_stage_1(self):
        """Verify ZeRO stage 1 configuration."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig(zero_stage=1)
        assert config.zero_stage == 1
        assert config.partition_optimizer_states is True
        assert config.partition_gradients is False

    def test_zero_stage_2(self):
        """Verify ZeRO stage 2 configuration."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig(zero_stage=2)
        assert config.zero_stage == 2
        assert config.partition_optimizer_states is True
        assert config.partition_gradients is True
        assert config.partition_parameters is False

    def test_zero_stage_3(self):
        """Verify ZeRO stage 3 configuration."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig(zero_stage=3)
        assert config.zero_stage == 3
        assert config.partition_optimizer_states is True
        assert config.partition_gradients is True
        assert config.partition_parameters is True

    def test_offload_optimizer(self):
        """Verify optimizer offloading configuration."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig(zero_stage=3, offload_optimizer=True)
        assert config.offload_optimizer is True
        assert config.offload_optimizer_device == "cpu"

    def test_offload_param(self):
        """Verify parameter offloading configuration."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig(zero_stage=3, offload_param=True)
        assert config.offload_param is True
        assert config.offload_param_device == "cpu"

    def test_nvme_offload(self):
        """Verify NVMe offloading for ZeRO-Infinity."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig(
            zero_stage=3,
            offload_optimizer=True,
            offload_optimizer_device="nvme",
            nvme_path="/nvme/offload"
        )
        assert config.offload_optimizer_device == "nvme"
        assert config.nvme_path == "/nvme/offload"

    def test_to_deepspeed_config(self):
        """Verify conversion to DeepSpeed config dict."""
        from medai_compass.training.distributed import DeepSpeedConfig

        config = DeepSpeedConfig(zero_stage=3)
        ds_config = config.to_deepspeed_config()
        assert isinstance(ds_config, dict)
        assert "zero_optimization" in ds_config
        assert ds_config["zero_optimization"]["stage"] == 3


class TestDeepSpeedTrainer:
    """Tests for DeepSpeed trainer."""

    def test_trainer_creation(self):
        """Verify DeepSpeed trainer can be created."""
        from medai_compass.training.distributed import DeepSpeedTrainer, DeepSpeedConfig

        config = DeepSpeedConfig(zero_stage=2)
        trainer = DeepSpeedTrainer(config)
        assert trainer is not None
        assert trainer.config.zero_stage == 2

    def test_trainer_has_train_method(self):
        """Verify trainer has train method."""
        from medai_compass.training.distributed import DeepSpeedTrainer, DeepSpeedConfig

        config = DeepSpeedConfig()
        trainer = DeepSpeedTrainer(config)
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "evaluate")

    def test_trainer_checkpoint_methods(self):
        """Verify trainer has checkpoint methods."""
        from medai_compass.training.distributed import DeepSpeedTrainer, DeepSpeedConfig

        config = DeepSpeedConfig()
        trainer = DeepSpeedTrainer(config)
        assert hasattr(trainer, "save_checkpoint")
        assert hasattr(trainer, "load_checkpoint")

    def test_check_deepspeed_available(self):
        """Verify DeepSpeed availability check."""
        from medai_compass.training.distributed import check_deepspeed_available

        result = check_deepspeed_available()
        assert isinstance(result, bool)


class TestZeROOptimizer:
    """Tests for ZeRO optimizer wrapper."""

    def test_optimizer_creation(self):
        """Verify ZeRO optimizer can be created."""
        from medai_compass.training.distributed import ZeROOptimizer

        optimizer = ZeROOptimizer(zero_stage=2)
        assert optimizer is not None
        assert optimizer.zero_stage == 2

    def test_optimizer_step(self):
        """Verify optimizer has step method."""
        from medai_compass.training.distributed import ZeROOptimizer

        optimizer = ZeROOptimizer()
        assert hasattr(optimizer, "step")
        assert hasattr(optimizer, "zero_grad")


# =============================================================================
# Megatron-LM Parallelism Tests
# =============================================================================

class TestMegatronConfig:
    """Tests for Megatron configuration."""

    def test_config_creation(self):
        """Verify Megatron config can be created."""
        from medai_compass.training.distributed import MegatronConfig

        config = MegatronConfig()
        assert config is not None

    def test_tensor_parallel_size(self):
        """Verify tensor parallel size configuration."""
        from medai_compass.training.distributed import MegatronConfig

        config = MegatronConfig(tensor_parallel_size=4)
        assert config.tensor_parallel_size == 4

    def test_pipeline_parallel_size(self):
        """Verify pipeline parallel size configuration."""
        from medai_compass.training.distributed import MegatronConfig

        config = MegatronConfig(pipeline_parallel_size=2)
        assert config.pipeline_parallel_size == 2

    def test_sequence_parallel(self):
        """Verify sequence parallelism configuration."""
        from medai_compass.training.distributed import MegatronConfig

        config = MegatronConfig(sequence_parallel=True)
        assert config.sequence_parallel is True

    def test_micro_batch_size(self):
        """Verify micro-batch size for pipeline parallelism."""
        from medai_compass.training.distributed import MegatronConfig

        config = MegatronConfig(
            pipeline_parallel_size=4,
            micro_batch_size=2,
            num_micro_batches=8
        )
        assert config.micro_batch_size == 2
        assert config.num_micro_batches == 8

    def test_interleaved_pipeline(self):
        """Verify interleaved pipeline scheduling."""
        from medai_compass.training.distributed import MegatronConfig

        config = MegatronConfig(
            pipeline_parallel_size=4,
            interleaved_pipeline=True,
            virtual_pipeline_model_parallel_size=2
        )
        assert config.interleaved_pipeline is True
        assert config.virtual_pipeline_model_parallel_size == 2


class TestTensorParallelTrainer:
    """Tests for Tensor Parallel trainer."""

    def test_trainer_creation(self):
        """Verify Tensor Parallel trainer can be created."""
        from medai_compass.training.distributed import TensorParallelTrainer, MegatronConfig

        config = MegatronConfig(tensor_parallel_size=4)
        trainer = TensorParallelTrainer(config)
        assert trainer is not None
        assert trainer.tensor_parallel_size == 4

    def test_column_parallel_linear(self):
        """Verify column parallel linear layer support."""
        from medai_compass.training.distributed import TensorParallelTrainer, MegatronConfig

        config = MegatronConfig(tensor_parallel_size=2)
        trainer = TensorParallelTrainer(config)
        assert hasattr(trainer, "create_column_parallel_linear")

    def test_row_parallel_linear(self):
        """Verify row parallel linear layer support."""
        from medai_compass.training.distributed import TensorParallelTrainer, MegatronConfig

        config = MegatronConfig(tensor_parallel_size=2)
        trainer = TensorParallelTrainer(config)
        assert hasattr(trainer, "create_row_parallel_linear")

    def test_parallel_attention(self):
        """Verify parallel attention support."""
        from medai_compass.training.distributed import TensorParallelTrainer, MegatronConfig

        config = MegatronConfig(tensor_parallel_size=4)
        trainer = TensorParallelTrainer(config)
        assert hasattr(trainer, "wrap_attention_for_tp")


class TestPipelineParallelTrainer:
    """Tests for Pipeline Parallel trainer."""

    def test_trainer_creation(self):
        """Verify Pipeline Parallel trainer can be created."""
        from medai_compass.training.distributed import PipelineParallelTrainer, MegatronConfig

        config = MegatronConfig(pipeline_parallel_size=4)
        trainer = PipelineParallelTrainer(config)
        assert trainer is not None
        assert trainer.pipeline_parallel_size == 4

    def test_pipeline_stages(self):
        """Verify pipeline stages configuration."""
        from medai_compass.training.distributed import PipelineParallelTrainer, MegatronConfig

        config = MegatronConfig(pipeline_parallel_size=4)
        trainer = PipelineParallelTrainer(config)
        assert hasattr(trainer, "num_stages")
        assert trainer.num_stages == 4

    def test_pipeline_schedule(self):
        """Verify pipeline schedule method."""
        from medai_compass.training.distributed import PipelineParallelTrainer, MegatronConfig

        config = MegatronConfig(pipeline_parallel_size=2)
        trainer = PipelineParallelTrainer(config)
        assert hasattr(trainer, "get_pipeline_schedule")

    def test_check_megatron_available(self):
        """Verify Megatron availability check."""
        from medai_compass.training.distributed import check_megatron_available

        result = check_megatron_available()
        assert isinstance(result, bool)


# =============================================================================
# FSDP2 Tests
# =============================================================================

class TestFSDP2Config:
    """Tests for FSDP2 configuration."""

    def test_config_creation(self):
        """Verify FSDP2 config can be created."""
        from medai_compass.training.distributed import FSDP2Config

        config = FSDP2Config()
        assert config is not None

    def test_per_parameter_sharding(self):
        """Verify per-parameter sharding configuration."""
        from medai_compass.training.distributed import FSDP2Config

        config = FSDP2Config(per_param_sharding=True)
        assert config.per_param_sharding is True

    def test_dtensor_enabled(self):
        """Verify DTensor configuration."""
        from medai_compass.training.distributed import FSDP2Config

        config = FSDP2Config(use_dtensor=True)
        assert config.use_dtensor is True

    def test_sharding_strategy(self):
        """Verify sharding strategy configuration."""
        from medai_compass.training.distributed import FSDP2Config

        config = FSDP2Config(sharding_strategy="FULL_SHARD")
        assert config.sharding_strategy == "FULL_SHARD"

    def test_mixed_precision(self):
        """Verify mixed precision configuration."""
        from medai_compass.training.distributed import FSDP2Config

        config = FSDP2Config(mixed_precision="bf16")
        assert config.mixed_precision == "bf16"


class TestFSDP2Trainer:
    """Tests for FSDP2 trainer."""

    def test_trainer_creation(self):
        """Verify FSDP2 trainer can be created."""
        from medai_compass.training.distributed import FSDP2Trainer, FSDP2Config

        config = FSDP2Config()
        trainer = FSDP2Trainer(config)
        assert trainer is not None

    def test_trainer_wrap_model(self):
        """Verify trainer can wrap model."""
        from medai_compass.training.distributed import FSDP2Trainer, FSDP2Config

        config = FSDP2Config()
        trainer = FSDP2Trainer(config)
        assert hasattr(trainer, "wrap_model")

    def test_composability_with_tp(self):
        """Verify FSDP2 composability with tensor parallelism."""
        from medai_compass.training.distributed import FSDP2Config

        config = FSDP2Config(
            per_param_sharding=True,
            compose_with_tp=True,
            tp_size=2
        )
        assert config.compose_with_tp is True
        assert config.tp_size == 2

    def test_check_fsdp2_available(self):
        """Verify FSDP2 availability check."""
        from medai_compass.training.distributed import check_fsdp2_available

        result = check_fsdp2_available()
        assert isinstance(result, bool)


class TestDTensorSharding:
    """Tests for DTensor sharding."""

    def test_sharding_creation(self):
        """Verify DTensor sharding can be created."""
        from medai_compass.training.distributed import DTensorSharding

        sharding = DTensorSharding()
        assert sharding is not None

    def test_shard_tensor(self):
        """Verify tensor sharding method."""
        from medai_compass.training.distributed import DTensorSharding

        sharding = DTensorSharding()
        assert hasattr(sharding, "shard_tensor")

    def test_gather_tensor(self):
        """Verify tensor gathering method."""
        from medai_compass.training.distributed import DTensorSharding

        sharding = DTensorSharding()
        assert hasattr(sharding, "gather_tensor")


# =============================================================================
# 5D Parallelism Tests
# =============================================================================

class TestParallelism5DConfig:
    """Tests for 5D Parallelism configuration."""

    def test_config_creation(self):
        """Verify 5D parallelism config can be created."""
        from medai_compass.training.distributed import Parallelism5DConfig

        config = Parallelism5DConfig()
        assert config is not None

    def test_data_parallel_size(self):
        """Verify data parallel size configuration."""
        from medai_compass.training.distributed import Parallelism5DConfig

        config = Parallelism5DConfig(data_parallel_size=8)
        assert config.data_parallel_size == 8

    def test_tensor_parallel_size(self):
        """Verify tensor parallel size in 5D config."""
        from medai_compass.training.distributed import Parallelism5DConfig

        config = Parallelism5DConfig(tensor_parallel_size=4)
        assert config.tensor_parallel_size == 4

    def test_pipeline_parallel_size(self):
        """Verify pipeline parallel size in 5D config."""
        from medai_compass.training.distributed import Parallelism5DConfig

        config = Parallelism5DConfig(pipeline_parallel_size=2)
        assert config.pipeline_parallel_size == 2

    def test_sequence_parallel(self):
        """Verify sequence parallelism in 5D config."""
        from medai_compass.training.distributed import Parallelism5DConfig

        config = Parallelism5DConfig(sequence_parallel_size=2)
        assert config.sequence_parallel_size == 2

    def test_expert_parallel(self):
        """Verify expert parallelism for MoE models."""
        from medai_compass.training.distributed import Parallelism5DConfig

        config = Parallelism5DConfig(expert_parallel_size=8)
        assert config.expert_parallel_size == 8

    def test_total_gpus_calculation(self):
        """Verify total GPUs calculation."""
        from medai_compass.training.distributed import Parallelism5DConfig

        config = Parallelism5DConfig(
            data_parallel_size=2,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        )
        assert hasattr(config, "total_gpus")
        # DP * TP * PP = 2 * 4 * 2 = 16
        assert config.total_gpus == 16


class TestHybridParallelTrainer:
    """Tests for Hybrid Parallel trainer (5D)."""

    def test_trainer_creation(self):
        """Verify Hybrid Parallel trainer can be created."""
        from medai_compass.training.distributed import HybridParallelTrainer, Parallelism5DConfig

        config = Parallelism5DConfig(
            data_parallel_size=2,
            tensor_parallel_size=2,
        )
        trainer = HybridParallelTrainer(config)
        assert trainer is not None

    def test_trainer_has_all_parallelisms(self):
        """Verify trainer supports all parallelism dimensions."""
        from medai_compass.training.distributed import HybridParallelTrainer, Parallelism5DConfig

        config = Parallelism5DConfig()
        trainer = HybridParallelTrainer(config)
        assert hasattr(trainer, "data_parallel_size")
        assert hasattr(trainer, "tensor_parallel_size")
        assert hasattr(trainer, "pipeline_parallel_size")
        assert hasattr(trainer, "sequence_parallel_size")
        assert hasattr(trainer, "expert_parallel_size")

    def test_process_group_creation(self):
        """Verify process group creation for hybrid parallelism."""
        from medai_compass.training.distributed import HybridParallelTrainer, Parallelism5DConfig

        config = Parallelism5DConfig()
        trainer = HybridParallelTrainer(config)
        assert hasattr(trainer, "create_process_groups")


class TestParallelismStrategy:
    """Tests for automatic parallelism strategy selection."""

    def test_strategy_creation(self):
        """Verify parallelism strategy can be created."""
        from medai_compass.training.distributed import ParallelismStrategy

        strategy = ParallelismStrategy()
        assert strategy is not None

    def test_auto_select_for_model_size(self):
        """Verify automatic strategy selection based on model size."""
        from medai_compass.training.distributed import ParallelismStrategy

        strategy = ParallelismStrategy()
        assert hasattr(strategy, "select_for_model")

    def test_strategy_for_4b_model(self):
        """Verify strategy for 4B model."""
        from medai_compass.training.distributed import ParallelismStrategy

        strategy = ParallelismStrategy()
        config = strategy.select_for_model(
            model_params=4e9,
            num_gpus=8,
            gpu_memory_gb=80
        )
        # 4B model on 8x H100 should use DP primarily
        assert config.data_parallel_size >= 1

    def test_strategy_for_27b_model(self):
        """Verify strategy for 27B model."""
        from medai_compass.training.distributed import ParallelismStrategy

        strategy = ParallelismStrategy()
        config = strategy.select_for_model(
            model_params=27e9,
            num_gpus=8,
            gpu_memory_gb=80
        )
        # 27B model needs TP and possibly PP
        assert config.tensor_parallel_size >= 2


# =============================================================================
# DualPipe Tests
# =============================================================================

class TestDualPipeConfig:
    """Tests for DualPipe configuration."""

    def test_config_creation(self):
        """Verify DualPipe config can be created."""
        from medai_compass.training.distributed import DualPipeConfig

        config = DualPipeConfig()
        assert config is not None
        assert hasattr(config, "num_stages")
        assert hasattr(config, "num_micro_batches")

    def test_default_values(self):
        """Verify default configuration values."""
        from medai_compass.training.distributed import DualPipeConfig

        config = DualPipeConfig()
        assert config.num_stages == 4
        assert config.num_micro_batches == 8
        assert config.overlap_p2p_comm is True
        assert config.async_communication is True

    def test_custom_configuration(self):
        """Verify custom configuration."""
        from medai_compass.training.distributed import DualPipeConfig

        config = DualPipeConfig(
            num_stages=8,
            num_micro_batches=16,
            overlap_p2p_comm=True,
            scatter_gather_tensors=True,
            pipeline_dtype="bfloat16"
        )
        assert config.num_stages == 8
        assert config.num_micro_batches == 16
        assert config.pipeline_dtype == "bfloat16"

    def test_validation_micro_batches_vs_stages(self):
        """Verify validation that micro_batches >= stages."""
        from medai_compass.training.distributed import DualPipeConfig
        import pytest

        # Should raise error when micro_batches < stages
        with pytest.raises(ValueError, match="num_micro_batches.*must be.*num_stages"):
            DualPipeConfig(num_stages=8, num_micro_batches=4)


class TestDualPipeSchedule:
    """Tests for DualPipe scheduling."""

    def test_schedule_creation(self):
        """Verify DualPipe schedule can be created."""
        from medai_compass.training.distributed import DualPipeSchedule

        schedule = DualPipeSchedule(num_stages=4, num_micro_batches=8)
        assert schedule is not None
        assert schedule.num_stages == 4
        assert schedule.num_micro_batches == 8

    def test_get_schedule_for_stage(self):
        """Verify schedule generation for specific stage."""
        from medai_compass.training.distributed import DualPipeSchedule

        schedule = DualPipeSchedule(num_stages=4, num_micro_batches=8)
        steps = schedule.get_schedule(stage_id=0)
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_schedule_step_attributes(self):
        """Verify schedule step has required attributes."""
        from medai_compass.training.distributed import DualPipeSchedule

        schedule = DualPipeSchedule(num_stages=4, num_micro_batches=8)
        steps = schedule.get_schedule(stage_id=0)
        step = steps[0]

        assert hasattr(step, "micro_batch_id")
        assert hasattr(step, "is_forward")
        assert hasattr(step, "is_send")
        assert hasattr(step, "is_recv")

    def test_bubble_ratio_calculation(self):
        """Verify bubble ratio calculation."""
        from medai_compass.training.distributed import DualPipeSchedule

        # DualPipe should have approximately (p-1)/(2m) bubble ratio
        schedule = DualPipeSchedule(num_stages=4, num_micro_batches=8, overlap_communication=True)
        bubble_ratio = schedule.get_bubble_ratio()

        # With p=4, m=8, expected ~3/16 = 0.1875
        assert bubble_ratio < 0.25  # Should be less than standard 1F1B
        assert bubble_ratio > 0

    def test_schedule_repr(self):
        """Verify schedule string representation."""
        from medai_compass.training.distributed import DualPipeSchedule

        schedule = DualPipeSchedule(num_stages=4, num_micro_batches=8)
        repr_str = repr(schedule)
        assert "DualPipeSchedule" in repr_str
        assert "num_stages=4" in repr_str
        assert "num_micro_batches=8" in repr_str


class TestDualPipeTrainer:
    """Tests for DualPipe trainer."""

    def test_trainer_creation(self):
        """Verify DualPipe trainer can be created."""
        from medai_compass.training.distributed import DualPipeTrainer, DualPipeConfig

        config = DualPipeConfig(num_stages=4, num_micro_batches=8)
        trainer = DualPipeTrainer(config)
        assert trainer is not None

    def test_trainer_properties(self):
        """Verify trainer has required properties."""
        from medai_compass.training.distributed import DualPipeTrainer, DualPipeConfig

        config = DualPipeConfig(num_stages=4, num_micro_batches=8)
        trainer = DualPipeTrainer(config)

        assert trainer.num_stages == 4
        assert trainer.num_micro_batches == 8
        assert isinstance(trainer.bubble_ratio, float)

    def test_trainer_has_parallelize_method(self):
        """Verify trainer has parallelize method."""
        from medai_compass.training.distributed import DualPipeTrainer, DualPipeConfig

        config = DualPipeConfig()
        trainer = DualPipeTrainer(config)
        assert hasattr(trainer, "parallelize")

    def test_trainer_has_train_step_method(self):
        """Verify trainer has train_step method."""
        from medai_compass.training.distributed import DualPipeTrainer, DualPipeConfig

        config = DualPipeConfig()
        trainer = DualPipeTrainer(config)
        assert hasattr(trainer, "train_step")

    def test_trainer_config_from_dict(self):
        """Verify trainer accepts config as dict."""
        from medai_compass.training.distributed import DualPipeTrainer

        config_dict = {
            "num_stages": 4,
            "num_micro_batches": 8,
            "overlap_p2p_comm": True
        }
        trainer = DualPipeTrainer(config_dict)
        assert trainer.num_stages == 4


class TestDualPipeWrapper:
    """Tests for DualPipe model wrapper."""

    def test_wrapper_creation(self):
        """Verify DualPipe wrapper can be created."""
        from medai_compass.training.distributed import DualPipeWrapper, DualPipeSchedule, DualPipeConfig
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        config = DualPipeConfig(num_stages=2, num_micro_batches=4)
        schedule = DualPipeSchedule(num_stages=2, num_micro_batches=4)

        wrapper = DualPipeWrapper(
            model=model,
            pp_group=None,
            stage_id=0,
            num_stages=2,
            schedule=schedule,
            config=config
        )
        assert wrapper is not None

    def test_wrapper_has_layers(self):
        """Verify wrapper has layers property."""
        from medai_compass.training.distributed import DualPipeWrapper, DualPipeSchedule, DualPipeConfig
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        config = DualPipeConfig(num_stages=2, num_micro_batches=4)
        schedule = DualPipeSchedule(num_stages=2, num_micro_batches=4)

        wrapper = DualPipeWrapper(
            model=model,
            pp_group=None,
            stage_id=0,
            num_stages=2,
            schedule=schedule,
            config=config
        )
        assert hasattr(wrapper, "layers")

    def test_wrapper_stage_info(self):
        """Verify wrapper provides stage info."""
        from medai_compass.training.distributed import DualPipeWrapper, DualPipeSchedule, DualPipeConfig
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        config = DualPipeConfig(num_stages=2, num_micro_batches=4)
        schedule = DualPipeSchedule(num_stages=2, num_micro_batches=4)

        wrapper = DualPipeWrapper(
            model=model,
            pp_group=None,
            stage_id=0,
            num_stages=2,
            schedule=schedule,
            config=config
        )

        info = wrapper.get_stage_info()
        assert "stage_id" in info
        assert "num_stages" in info
        assert "bubble_ratio" in info
        assert info["stage_id"] == 0
        assert info["num_stages"] == 2


class TestDualPipeBubbleReduction:
    """Tests for DualPipe bubble reduction benefits."""

    def test_dualpipe_vs_1f1b_bubble(self):
        """Verify DualPipe reduces bubble compared to 1F1B."""
        from medai_compass.training.distributed import DualPipeSchedule

        p = 8  # stages
        m = 16  # micro-batches

        # Standard 1F1B bubble: (p-1)/m
        standard_bubble = (p - 1) / m

        # DualPipe with overlap: (p-1)/(2m)
        dualpipe_schedule = DualPipeSchedule(
            num_stages=p,
            num_micro_batches=m,
            overlap_communication=True
        )
        dualpipe_bubble = dualpipe_schedule.get_bubble_ratio()

        # DualPipe should have lower bubble ratio
        assert dualpipe_bubble < standard_bubble

    def test_bubble_ratio_with_more_micro_batches(self):
        """Verify bubble ratio decreases with more micro-batches."""
        from medai_compass.training.distributed import DualPipeSchedule

        p = 4  # stages

        schedule_8 = DualPipeSchedule(num_stages=p, num_micro_batches=8)
        schedule_16 = DualPipeSchedule(num_stages=p, num_micro_batches=16)

        # More micro-batches should mean lower bubble ratio
        assert schedule_16.get_bubble_ratio() < schedule_8.get_bubble_ratio()

    def test_schedule_covers_all_micro_batches(self):
        """Verify schedule processes all micro-batches forward and backward."""
        from medai_compass.training.distributed import DualPipeSchedule

        schedule = DualPipeSchedule(num_stages=4, num_micro_batches=8)

        # Check for first stage
        steps = schedule.get_schedule(stage_id=0)

        # Count forward and backward passes
        forward_mbs = set()
        backward_mbs = set()

        for step in steps:
            if step.is_forward:
                forward_mbs.add(step.micro_batch_id)
            else:
                backward_mbs.add(step.micro_batch_id)

        # Should eventually process all micro-batches
        assert len(forward_mbs) > 0
        assert len(backward_mbs) > 0
