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
