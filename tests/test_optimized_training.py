"""
Tests for Optimized Training Pipeline for H100 GPUs.

TDD approach: Tests written first for all optimizations:
- Ray Data integration for efficient data loading
- Optimized DICOM preprocessing pipeline
- Flash Attention 2 for training
- Gradient checkpointing optimizations
- Mixed precision with H100 FP8 support
- FSDP with activation checkpointing
- Memory-efficient attention
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


# =============================================================================
# Ray Data Pipeline Tests
# =============================================================================

class TestRayDataPipeline:
    """Tests for Ray Data optimized data loading."""

    def test_ray_data_dicom_loader(self):
        """Verify Ray Data DICOM loader creation."""
        from medai_compass.training.optimized import RayDataDICOMLoader

        loader = RayDataDICOMLoader()
        assert loader is not None
        assert hasattr(loader, "load_dataset")

    def test_ray_data_parallel_read(self):
        """Verify parallel read configuration."""
        from medai_compass.training.optimized import RayDataDICOMLoader

        loader = RayDataDICOMLoader(read_parallelism=200)
        assert loader.read_parallelism == 200

    def test_ray_data_prefetching(self):
        """Verify prefetching is enabled."""
        from medai_compass.training.optimized import RayDataDICOMLoader

        loader = RayDataDICOMLoader(prefetch_batches=4)
        assert loader.prefetch_batches == 4

    def test_ray_data_streaming(self):
        """Verify streaming mode for large datasets."""
        from medai_compass.training.optimized import RayDataDICOMLoader

        loader = RayDataDICOMLoader(streaming=True)
        assert loader.streaming is True

    def test_ray_data_dicom_preprocessing(self):
        """Verify DICOM preprocessing in Ray Data pipeline."""
        from medai_compass.training.optimized import RayDataDICOMLoader

        loader = RayDataDICOMLoader()
        assert hasattr(loader, "preprocess_batch")
        assert hasattr(loader, "apply_windowing")


class TestRayDataDICOMDataset:
    """Tests for Ray Data DICOM dataset."""

    def test_dataset_from_directory(self):
        """Verify dataset creation from DICOM directory."""
        from medai_compass.training.optimized import RayDICOMDataset

        dataset = RayDICOMDataset(data_dir="data/mimic_cxr")
        assert dataset is not None
        assert hasattr(dataset, "to_ray_dataset")

    def test_dataset_with_labels(self):
        """Verify dataset with label loading."""
        from medai_compass.training.optimized import RayDICOMDataset

        dataset = RayDICOMDataset(
            data_dir="data/mimic_cxr",
            labels_path="data/mimic_cxr/labels.csv"
        )
        assert dataset.has_labels is True

    def test_dataset_multimodal_format(self):
        """Verify multimodal format (image + text)."""
        from medai_compass.training.optimized import RayDICOMDataset

        dataset = RayDICOMDataset(multimodal=True)
        assert dataset.multimodal is True
        assert hasattr(dataset, "format_for_training")

    def test_dataset_shuffling(self):
        """Verify dataset shuffling."""
        from medai_compass.training.optimized import RayDICOMDataset

        dataset = RayDICOMDataset(shuffle=True, shuffle_buffer_size=10000)
        assert dataset.shuffle is True
        assert dataset.shuffle_buffer_size == 10000

    def test_dataset_split_creation(self):
        """Verify train/val split creation."""
        from medai_compass.training.optimized import RayDICOMDataset

        dataset = RayDICOMDataset()
        assert hasattr(dataset, "split")
        # Should support train_split, val_split


# =============================================================================
# Optimized Training Config Tests
# =============================================================================

class TestOptimizedTrainingConfig:
    """Tests for H100-optimized training configuration."""

    def test_h100_training_config(self):
        """Verify H100-specific training config."""
        from medai_compass.training.optimized import H100TrainingConfig

        config = H100TrainingConfig()
        assert config is not None
        assert config.use_flash_attention_2 is True

    def test_config_mixed_precision(self):
        """Verify mixed precision settings for H100."""
        from medai_compass.training.optimized import H100TrainingConfig

        config = H100TrainingConfig()
        assert config.bf16 is True
        assert config.tf32 is True  # H100 TF32 support

    def test_config_fp8_training(self):
        """Verify FP8 training support for H100."""
        from medai_compass.training.optimized import H100TrainingConfig

        config = H100TrainingConfig(use_fp8=True)
        assert config.use_fp8 is True
        assert hasattr(config, "fp8_recipe")

    def test_config_gradient_checkpointing(self):
        """Verify gradient checkpointing settings."""
        from medai_compass.training.optimized import H100TrainingConfig

        config = H100TrainingConfig()
        assert config.gradient_checkpointing is True
        assert hasattr(config, "gradient_checkpointing_kwargs")

    def test_config_fsdp_settings(self):
        """Verify FSDP settings for distributed training."""
        from medai_compass.training.optimized import H100TrainingConfig

        config = H100TrainingConfig.for_distributed(num_gpus=8)
        assert config.use_fsdp is True
        assert config.fsdp_sharding_strategy == "FULL_SHARD"

    def test_config_for_model_size(self):
        """Verify config adjusts for model size."""
        from medai_compass.training.optimized import H100TrainingConfig

        config_4b = H100TrainingConfig.for_model("medgemma-4b")
        config_27b = H100TrainingConfig.for_model("medgemma-27b")

        assert config_4b.batch_size > config_27b.batch_size
        assert config_27b.gradient_accumulation_steps > config_4b.gradient_accumulation_steps


# =============================================================================
# Flash Attention Training Tests
# =============================================================================

class TestFlashAttentionTraining:
    """Tests for Flash Attention 2 in training."""

    def test_training_with_flash_attention(self):
        """Verify training uses Flash Attention 2."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(use_flash_attention_2=True)
        assert trainer.use_flash_attention_2 is True

    def test_flash_attention_memory_savings(self):
        """Verify Flash Attention reduces memory usage."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(use_flash_attention_2=True)
        assert hasattr(trainer, "estimate_memory_savings")
        # Should show ~2-4x memory reduction for long sequences


# =============================================================================
# FSDP and Distributed Training Tests
# =============================================================================

class TestFSDPTraining:
    """Tests for FSDP distributed training."""

    def test_fsdp_trainer_creation(self):
        """Verify FSDP trainer can be created."""
        from medai_compass.training.optimized import FSDPTrainer

        trainer = FSDPTrainer(num_gpus=8)
        assert trainer is not None
        assert trainer.num_gpus == 8

    def test_fsdp_sharding_strategy(self):
        """Verify FSDP sharding strategy."""
        from medai_compass.training.optimized import FSDPTrainer

        trainer = FSDPTrainer(sharding_strategy="FULL_SHARD")
        assert trainer.sharding_strategy == "FULL_SHARD"

    def test_fsdp_activation_checkpointing(self):
        """Verify FSDP with activation checkpointing."""
        from medai_compass.training.optimized import FSDPTrainer

        trainer = FSDPTrainer(activation_checkpointing=True)
        assert trainer.activation_checkpointing is True

    def test_fsdp_offload_params(self):
        """Verify CPU offloading option."""
        from medai_compass.training.optimized import FSDPTrainer

        trainer = FSDPTrainer(cpu_offload=True)
        assert trainer.cpu_offload is True

    def test_fsdp_mixed_precision_policy(self):
        """Verify FSDP mixed precision policy."""
        from medai_compass.training.optimized import FSDPTrainer

        trainer = FSDPTrainer(mixed_precision="bf16")
        assert trainer.mixed_precision == "bf16"


# =============================================================================
# Memory Optimization Tests
# =============================================================================

class TestMemoryOptimization:
    """Tests for memory optimization strategies."""

    def test_gradient_accumulation_optimization(self):
        """Verify optimized gradient accumulation."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(
            gradient_accumulation_steps=8,
            gradient_accumulation_kwargs={"sync_each_step": False}
        )
        assert trainer.gradient_accumulation_steps == 8

    def test_memory_efficient_attention(self):
        """Verify memory-efficient attention options."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(use_memory_efficient_attention=True)
        assert trainer.use_memory_efficient_attention is True

    def test_selective_activation_checkpointing(self):
        """Verify selective activation checkpointing."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(
            gradient_checkpointing=True,
            checkpoint_every_n_layers=2
        )
        assert trainer.checkpoint_every_n_layers == 2

    def test_optimizer_state_offload(self):
        """Verify optimizer state CPU offload."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(offload_optimizer_state=True)
        assert trainer.offload_optimizer_state is True


# =============================================================================
# DICOM Training Pipeline Tests
# =============================================================================

class TestDICOMTrainingPipeline:
    """Tests for DICOM-optimized training pipeline."""

    def test_dicom_training_pipeline_creation(self):
        """Verify DICOM training pipeline can be created."""
        from medai_compass.training.optimized import DICOMTrainingPipeline

        pipeline = DICOMTrainingPipeline(
            data_dir="data/mimic_cxr",
            model_name="medgemma-4b"
        )
        assert pipeline is not None

    def test_dicom_preprocessing_caching(self):
        """Verify DICOM preprocessing is cached."""
        from medai_compass.training.optimized import DICOMTrainingPipeline

        pipeline = DICOMTrainingPipeline(
            cache_preprocessed=True,
            cache_dir="/tmp/dicom_cache"
        )
        assert pipeline.cache_preprocessed is True

    def test_dicom_augmentation(self):
        """Verify DICOM augmentation support."""
        from medai_compass.training.optimized import DICOMTrainingPipeline

        pipeline = DICOMTrainingPipeline(
            augmentations=["rotation", "flip", "brightness"]
        )
        assert "rotation" in pipeline.augmentations

    def test_dicom_multiview_training(self):
        """Verify multi-view DICOM training."""
        from medai_compass.training.optimized import DICOMTrainingPipeline

        pipeline = DICOMTrainingPipeline(multiview=True)
        assert pipeline.multiview is True

    def test_dicom_series_handling(self):
        """Verify 3D DICOM series handling."""
        from medai_compass.training.optimized import DICOMTrainingPipeline

        pipeline = DICOMTrainingPipeline(handle_3d_series=True)
        assert pipeline.handle_3d_series is True
        assert hasattr(pipeline, "extract_slices")


# =============================================================================
# Optimized Trainer Tests
# =============================================================================

class TestOptimizedTrainer:
    """Tests for the complete optimized trainer."""

    def test_trainer_creation(self):
        """Verify optimized trainer can be created."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_trainer_auto_optimization(self):
        """Verify trainer auto-detects and applies optimizations."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(auto_optimize=True)
        assert trainer.auto_optimize is True
        assert hasattr(trainer, "detected_optimizations")

    def test_trainer_ray_data_integration(self):
        """Verify Ray Data integration."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(use_ray_data=True)
        assert trainer.use_ray_data is True

    def test_trainer_train_method(self):
        """Verify train method exists and configurable."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer()
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "evaluate")

    def test_trainer_checkpoint_management(self):
        """Verify checkpoint management."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer()
        assert hasattr(trainer, "save_checkpoint")
        assert hasattr(trainer, "load_checkpoint")

    def test_trainer_metrics_logging(self):
        """Verify metrics logging."""
        from medai_compass.training.optimized import OptimizedTrainer

        trainer = OptimizedTrainer(log_to_mlflow=True)
        assert trainer.log_to_mlflow is True
        assert hasattr(trainer, "get_training_metrics")


# =============================================================================
# Training Throughput Tests
# =============================================================================

class TestTrainingThroughput:
    """Tests for training throughput benchmarking."""

    def test_throughput_tracker(self):
        """Verify throughput tracking."""
        from medai_compass.training.optimized import ThroughputTracker

        tracker = ThroughputTracker()
        assert tracker is not None
        assert hasattr(tracker, "log_step")

    def test_samples_per_second(self):
        """Verify samples per second measurement."""
        from medai_compass.training.optimized import ThroughputTracker

        tracker = ThroughputTracker()
        assert hasattr(tracker, "samples_per_second")

    def test_tokens_per_second_training(self):
        """Verify tokens per second for training."""
        from medai_compass.training.optimized import ThroughputTracker

        tracker = ThroughputTracker()
        assert hasattr(tracker, "tokens_per_second")

    def test_gpu_utilization_tracking(self):
        """Verify GPU utilization tracking."""
        from medai_compass.training.optimized import ThroughputTracker

        tracker = ThroughputTracker()
        assert hasattr(tracker, "gpu_utilization")
        assert hasattr(tracker, "gpu_memory_used")

    def test_throughput_comparison(self):
        """Verify throughput comparison with baseline."""
        from medai_compass.training.optimized import ThroughputTracker

        tracker = ThroughputTracker()
        assert hasattr(tracker, "compare_with_baseline")


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndOptimizedPipeline:
    """Tests for end-to-end optimized training pipeline."""

    def test_full_pipeline_creation(self):
        """Verify full pipeline can be created."""
        from medai_compass.training.optimized import OptimizedTrainingPipeline

        pipeline = OptimizedTrainingPipeline(
            model_name="medgemma-4b",
            data_dir="data/mimic_cxr"
        )
        assert pipeline is not None

    def test_pipeline_with_ray_cluster(self):
        """Verify pipeline works with Ray cluster."""
        from medai_compass.training.optimized import OptimizedTrainingPipeline

        pipeline = OptimizedTrainingPipeline(
            use_ray=True,
            num_workers=8
        )
        assert pipeline.use_ray is True
        assert pipeline.num_workers == 8

    def test_pipeline_run_method(self):
        """Verify pipeline run method."""
        from medai_compass.training.optimized import OptimizedTrainingPipeline

        pipeline = OptimizedTrainingPipeline()
        assert hasattr(pipeline, "run")
        assert hasattr(pipeline, "run_async")

    def test_pipeline_validation(self):
        """Verify pipeline configuration validation."""
        from medai_compass.training.optimized import OptimizedTrainingPipeline

        pipeline = OptimizedTrainingPipeline()
        assert hasattr(pipeline, "validate_config")

    def test_pipeline_dry_run(self):
        """Verify pipeline dry run mode."""
        from medai_compass.training.optimized import OptimizedTrainingPipeline

        pipeline = OptimizedTrainingPipeline()
        assert hasattr(pipeline, "dry_run")


# =============================================================================
# H100-Specific Optimization Tests
# =============================================================================

class TestH100Optimizations:
    """Tests for H100-specific optimizations."""

    def test_h100_fp8_matmul(self):
        """Verify FP8 matrix multiplication on H100."""
        from medai_compass.training.optimized import H100Optimizer

        optimizer = H100Optimizer()
        assert hasattr(optimizer, "enable_fp8_matmul")

    def test_h100_transformer_engine(self):
        """Verify Transformer Engine integration."""
        from medai_compass.training.optimized import H100Optimizer

        optimizer = H100Optimizer(use_transformer_engine=True)
        assert optimizer.use_transformer_engine is True

    def test_h100_nvlink_optimization(self):
        """Verify NVLink optimization for multi-GPU."""
        from medai_compass.training.optimized import H100Optimizer

        optimizer = H100Optimizer()
        assert hasattr(optimizer, "optimize_nvlink_comm")

    def test_h100_memory_bandwidth(self):
        """Verify HBM3 bandwidth optimization."""
        from medai_compass.training.optimized import H100Optimizer

        optimizer = H100Optimizer()
        assert hasattr(optimizer, "optimize_memory_access")
