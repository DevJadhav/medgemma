"""
Tests for Training Strategy Selector.

TDD tests for unified training strategy selection including:
- DeepSpeed ZeRO (stages 1, 2, 3, Infinity)
- Megatron-LM (TP, PP)
- FSDP2
- DualPipe
- 5D Parallelism
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional


class TestTrainingStrategySelector:
    """Tests for TrainingStrategySelector class."""

    def test_selector_creation(self):
        """Verify strategy selector can be created."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        assert selector is not None

    def test_list_available_strategies(self):
        """Verify all strategies are listed."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategies = selector.list_strategies()

        assert "single_gpu" in strategies
        assert "deepspeed_zero1" in strategies
        assert "deepspeed_zero2" in strategies
        assert "deepspeed_zero3" in strategies
        assert "deepspeed_infinity" in strategies
        assert "fsdp" in strategies
        assert "fsdp2" in strategies
        assert "megatron_tp" in strategies
        assert "megatron_pp" in strategies
        assert "megatron_tp_pp" in strategies
        assert "dualpipe" in strategies
        assert "parallelism_5d" in strategies

    def test_select_single_gpu_strategy(self):
        """Verify single GPU strategy selection."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("single_gpu")

        assert strategy is not None
        assert strategy.name == "single_gpu"
        assert strategy.num_gpus == 1

    def test_select_deepspeed_zero3(self):
        """Verify DeepSpeed ZeRO-3 strategy selection."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("deepspeed_zero3", num_gpus=8)

        assert strategy is not None
        assert strategy.name == "deepspeed_zero3"
        assert strategy.num_gpus == 8
        assert hasattr(strategy, "config")
        assert strategy.config.zero_stage == 3

    def test_select_megatron_tp(self):
        """Verify Megatron Tensor Parallel strategy."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("megatron_tp", tensor_parallel_size=4)

        assert strategy is not None
        assert strategy.name == "megatron_tp"
        assert strategy.config.tensor_parallel_size == 4

    def test_select_megatron_pp(self):
        """Verify Megatron Pipeline Parallel strategy."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("megatron_pp", pipeline_parallel_size=2)

        assert strategy is not None
        assert strategy.name == "megatron_pp"
        assert strategy.config.pipeline_parallel_size == 2

    def test_select_dualpipe(self):
        """Verify DualPipe strategy selection."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select(
            "dualpipe",
            num_stages=4,
            num_micro_batches=8
        )

        assert strategy is not None
        assert strategy.name == "dualpipe"
        assert strategy.config.num_stages == 4
        assert strategy.config.num_micro_batches == 8

    def test_select_fsdp2(self):
        """Verify FSDP2 strategy selection."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("fsdp2", num_gpus=4)

        assert strategy is not None
        assert strategy.name == "fsdp2"
        assert strategy.config.per_param_sharding is True

    def test_select_5d_parallelism(self):
        """Verify 5D Parallelism strategy selection."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select(
            "parallelism_5d",
            data_parallel_size=2,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        )

        assert strategy is not None
        assert strategy.name == "parallelism_5d"
        assert strategy.config.data_parallel_size == 2
        assert strategy.config.tensor_parallel_size == 4
        assert strategy.config.pipeline_parallel_size == 2

    def test_auto_select_for_model_size(self):
        """Verify automatic strategy selection based on model size."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()

        # Small model (4B) - should use single GPU or basic DP
        strategy_4b = selector.auto_select(
            model_params=4e9,
            num_gpus=8,
            gpu_memory_gb=80
        )
        assert strategy_4b is not None

        # Large model (27B) - should use ZeRO-3 or TP
        strategy_27b = selector.auto_select(
            model_params=27e9,
            num_gpus=8,
            gpu_memory_gb=80
        )
        assert strategy_27b is not None
        # Should recommend more aggressive parallelism
        assert strategy_27b.name in ["deepspeed_zero3", "megatron_tp", "megatron_tp_pp", "parallelism_5d"]

    def test_invalid_strategy_raises_error(self):
        """Verify invalid strategy name raises error."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()

        with pytest.raises(ValueError, match="Unknown strategy"):
            selector.select("invalid_strategy")

    def test_strategy_returns_trainer_class(self):
        """Verify strategy returns appropriate trainer class."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("deepspeed_zero3", num_gpus=8)

        assert hasattr(strategy, "get_trainer")
        trainer_class = strategy.get_trainer()
        assert trainer_class is not None


class TestTrainingStrategyConfig:
    """Tests for training strategy configurations."""

    def test_deepspeed_config_to_dict(self):
        """Verify DeepSpeed config can be exported to dict."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("deepspeed_zero3", num_gpus=8)

        config_dict = strategy.to_deepspeed_config()
        assert isinstance(config_dict, dict)
        assert "zero_optimization" in config_dict
        assert config_dict["zero_optimization"]["stage"] == 3

    def test_megatron_config_to_dict(self):
        """Verify Megatron config can be exported to dict."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("megatron_tp", tensor_parallel_size=4)

        config_dict = strategy.to_megatron_config()
        assert isinstance(config_dict, dict)
        assert "tensor_model_parallel_size" in config_dict


class TestRayTrainerIntegration:
    """Tests for Ray Trainer integration with strategies."""

    def test_ray_trainer_with_strategy(self):
        """Verify Ray trainer accepts strategy."""
        from medai_compass.training import TrainingStrategySelector
        from medai_compass.training.ray_trainer import TrainingConfig

        selector = TrainingStrategySelector()
        strategy = selector.select("deepspeed_zero3", num_gpus=8)

        config = TrainingConfig(
            model_name="medgemma-4b",
            distributed_strategy="deepspeed_zero3",
        )

        assert config.distributed_strategy == "deepspeed_zero3"

    def test_ray_trainer_strategy_factory_method(self):
        """Verify Ray trainer has strategy factory method."""
        from medai_compass.training.ray_trainer import MedGemmaTrainer, TrainingConfig

        config = TrainingConfig(model_name="medgemma-4b", dry_run=True)
        trainer = MedGemmaTrainer(config)

        assert hasattr(trainer, "get_strategy")

        # Verify get_strategy returns a valid strategy
        strategy = trainer.get_strategy()
        assert strategy is not None
        assert strategy.name is not None

    def test_create_trainer_with_deepspeed(self):
        """Verify trainer creation with DeepSpeed strategy."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select("deepspeed_zero3", num_gpus=8)

        trainer = strategy.create_trainer()
        assert trainer is not None

    def test_create_trainer_with_dualpipe(self):
        """Verify trainer creation with DualPipe strategy."""
        from medai_compass.training import TrainingStrategySelector

        selector = TrainingStrategySelector()
        strategy = selector.select(
            "dualpipe",
            num_stages=4,
            num_micro_batches=8
        )

        trainer = strategy.create_trainer()
        assert trainer is not None


class TestInferenceStrategySelector:
    """Tests for InferenceStrategySelector class."""

    def test_selector_creation(self):
        """Verify inference strategy selector can be created."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()
        assert selector is not None

    def test_list_available_backends(self):
        """Verify all inference backends are listed."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()
        backends = selector.list_backends()

        assert "vllm" in backends
        assert "ray_serve" in backends
        assert "triton" in backends
        assert "hf_pipeline" in backends
        assert "modal" in backends

    def test_select_vllm_backend(self):
        """Verify vLLM backend selection."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()
        backend = selector.select("vllm", tensor_parallel_size=4)

        assert backend is not None
        assert backend.name == "vllm"
        assert backend.config.tensor_parallel_size == 4

    def test_select_ray_serve_backend(self):
        """Verify Ray Serve backend selection."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()
        backend = selector.select("ray_serve", num_replicas=4)

        assert backend is not None
        assert backend.name == "ray_serve"
        assert backend.config.num_replicas == 4

    def test_select_triton_backend(self):
        """Verify Triton backend selection."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()
        backend = selector.select("triton")

        assert backend is not None
        assert backend.name == "triton"

    def test_select_modal_backend(self):
        """Verify Modal backend selection."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()
        backend = selector.select("modal", gpu_type="H100")

        assert backend is not None
        assert backend.name == "modal"
        assert backend.config.gpu_type == "H100"

    def test_auto_select_for_requirements(self):
        """Verify automatic backend selection based on requirements."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()

        # High throughput requirement
        backend = selector.auto_select(
            priority="throughput",
            max_latency_ms=200,
            batch_size=64
        )
        assert backend is not None
        assert backend.name in ["vllm", "triton"]

        # Low latency requirement
        backend_low_latency = selector.auto_select(
            priority="latency",
            max_latency_ms=50,
        )
        assert backend_low_latency is not None

    def test_backend_returns_engine(self):
        """Verify backend returns inference engine."""
        from medai_compass.inference import InferenceStrategySelector

        selector = InferenceStrategySelector()
        backend = selector.select("hf_pipeline")

        assert hasattr(backend, "create_engine")


class TestStrategyIntegrationWithPipeline:
    """Tests for strategy integration with training/inference pipelines."""

    def test_end_to_end_training_strategy_flow(self):
        """Verify complete training strategy flow."""
        from medai_compass.training import TrainingStrategySelector

        # 1. Create selector
        selector = TrainingStrategySelector()

        # 2. Auto-select strategy for model
        strategy = selector.auto_select(
            model_params=27e9,
            num_gpus=8,
            gpu_memory_gb=80
        )

        # 3. Get configuration
        assert strategy.config is not None

        # 4. Strategy should be usable
        assert strategy.name is not None
        assert strategy.is_valid()

    def test_end_to_end_inference_strategy_flow(self):
        """Verify complete inference strategy flow."""
        from medai_compass.inference import InferenceStrategySelector

        # 1. Create selector
        selector = InferenceStrategySelector()

        # 2. Select backend
        backend = selector.select("vllm", tensor_parallel_size=4)

        # 3. Get configuration
        assert backend.config is not None

        # 4. Backend should be usable
        assert backend.name == "vllm"
        assert backend.is_valid()
