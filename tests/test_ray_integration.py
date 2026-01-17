"""Tests for Ray training orchestration integration.

These tests verify Ray cluster functionality for distributed training
of MedGemma models.
"""

import os
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, AsyncMock
import yaml

# Check for optional dependencies
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Ray Configuration Tests
# =============================================================================

class TestRayTrainingConfig:
    """Test Ray training configurations."""
    
    @pytest.fixture
    def training_config_path(self) -> Path:
        """Path to training configuration."""
        return Path(__file__).parent.parent / "config" / "training.yaml"
    
    def test_training_config_exists(self, training_config_path: Path):
        """Training configuration file must exist."""
        assert training_config_path.exists(), \
            f"Training config not found at {training_config_path}"
    
    def test_training_config_has_ray_settings(self, training_config_path: Path):
        """Training config must have Ray-specific settings."""
        with open(training_config_path) as f:
            config = yaml.safe_load(f)
        
        ray_settings = config.get("ray", {})
        assert "num_workers" in ray_settings, "Missing num_workers setting"
        assert "use_gpu" in ray_settings, "Missing use_gpu setting"
        assert "resources_per_worker" in ray_settings, "Missing resources_per_worker"
    
    def test_training_config_has_deepspeed(self, training_config_path: Path):
        """Training config should include DeepSpeed settings for 27B model."""
        with open(training_config_path) as f:
            config = yaml.safe_load(f)
        
        assert "deepspeed" in config, "DeepSpeed configuration required for 27B model"
        ds_config = config["deepspeed"]
        assert "zero_optimization" in ds_config, "ZeRO optimization required"


class TestRayTrainerModule:
    """Test Ray trainer module structure."""
    
    @pytest.fixture
    def trainer_module_path(self) -> Path:
        """Path to Ray trainer module."""
        return Path(__file__).parent.parent / "medai_compass" / "training" / "ray_trainer.py"
    
    def test_ray_trainer_module_exists(self, trainer_module_path: Path):
        """Ray trainer module must exist."""
        assert trainer_module_path.exists(), \
            f"Ray trainer module not found at {trainer_module_path}"
    
    def test_ray_trainer_has_required_classes(self, trainer_module_path: Path):
        """Ray trainer module must define required classes."""
        content = trainer_module_path.read_text()
        
        required_classes = [
            "MedGemmaTrainer",
            "TrainingConfig",
            "TrainingResult",
        ]
        
        for cls in required_classes:
            assert f"class {cls}" in content, f"Missing class: {cls}"
    
    def test_ray_trainer_has_model_selection(self, trainer_module_path: Path):
        """Ray trainer must support model selection."""
        content = trainer_module_path.read_text()
        
        # Should have model parameter
        assert "model_name" in content or "model_id" in content, \
            "Trainer should accept model selection parameter"
        
        # Should reference both models
        assert "medgemma-4b" in content.lower() or "medgemma_4b" in content.lower(), \
            "Should reference MedGemma 4B model"


class TestRayDataPipeline:
    """Test Ray data pipeline for training."""
    
    @pytest.fixture
    def data_pipeline_path(self) -> Path:
        """Path to data pipeline module."""
        return Path(__file__).parent.parent / "medai_compass" / "training" / "data_pipeline.py"
    
    def test_data_pipeline_module_exists(self, data_pipeline_path: Path):
        """Data pipeline module must exist."""
        assert data_pipeline_path.exists(), \
            f"Data pipeline module not found at {data_pipeline_path}"
    
    def test_data_pipeline_uses_ray_data(self, data_pipeline_path: Path):
        """Data pipeline should use Ray Data for distributed processing."""
        content = data_pipeline_path.read_text()
        assert "ray.data" in content or "from ray import data" in content, \
            "Data pipeline should use Ray Data"


# =============================================================================
# Ray Actor Tests
# =============================================================================

class TestRayActors:
    """Test Ray actor definitions for training."""
    
    @pytest.mark.skipif(not HAS_RAY, reason="Ray not installed")
    def test_training_actor_creation(self):
        """Test training actor can be created with mocked Ray."""
        with patch("ray.remote") as mock_remote:
            mock_remote.return_value = lambda cls: cls
            
            # Import should work with mocked Ray
            from medai_compass.training.ray_trainer import MedGemmaTrainer
            
            # Should be able to instantiate config
            trainer = MedGemmaTrainer.__new__(MedGemmaTrainer)
            assert trainer is not None


# =============================================================================
# Checkpoint Management Tests
# =============================================================================

class TestCheckpointManagement:
    """Test checkpoint saving and loading for training."""
    
    @pytest.fixture
    def checkpoint_manager_path(self) -> Path:
        """Path to checkpoint manager module."""
        return Path(__file__).parent.parent / "medai_compass" / "training" / "checkpoint_manager.py"
    
    def test_checkpoint_manager_exists(self, checkpoint_manager_path: Path):
        """Checkpoint manager module must exist."""
        assert checkpoint_manager_path.exists(), \
            f"Checkpoint manager not found at {checkpoint_manager_path}"
    
    def test_checkpoint_manager_has_required_functions(self, checkpoint_manager_path: Path):
        """Checkpoint manager must have save/load functions."""
        content = checkpoint_manager_path.read_text()
        
        required_functions = [
            "save_checkpoint",
            "load_checkpoint",
            "get_latest_checkpoint",
        ]
        
        for func in required_functions:
            assert f"def {func}" in content, f"Missing function: {func}"
    
    def test_checkpoint_manager_supports_minio(self, checkpoint_manager_path: Path):
        """Checkpoint manager should support MinIO/S3 storage."""
        content = checkpoint_manager_path.read_text()
        assert "s3" in content.lower() or "minio" in content.lower(), \
            "Checkpoint manager should support S3/MinIO storage"


# =============================================================================
# Distributed Training Tests
# =============================================================================

class TestDistributedTraining:
    """Test distributed training configuration."""
    
    @pytest.fixture
    def distributed_config_path(self) -> Path:
        """Path to distributed training config."""
        return Path(__file__).parent.parent / "config" / "training.yaml"
    
    def test_distributed_config_for_27b(self, distributed_config_path: Path):
        """27B model should have proper distributed config."""
        with open(distributed_config_path) as f:
            config = yaml.safe_load(f)
        
        model_27b_config = config.get("models", {}).get("medgemma_27b", {})
        
        # Should use model parallelism or ZeRO-3
        assert model_27b_config.get("distributed_strategy") in [
            "deepspeed_zero3",
            "fsdp",
            "model_parallel"
        ], "27B model requires advanced distributed strategy"
    
    def test_distributed_config_for_4b(self, distributed_config_path: Path):
        """4B model can use simpler distributed config."""
        with open(distributed_config_path) as f:
            config = yaml.safe_load(f)
        
        model_4b_config = config.get("models", {}).get("medgemma_4b", {})
        
        # 4B can use DDP or single GPU
        assert model_4b_config.get("distributed_strategy") in [
            "ddp",
            "single_gpu",
            "deepspeed_zero2",
            None
        ], "4B model should use appropriate strategy"


# =============================================================================
# Training Metrics Tests
# =============================================================================

class TestTrainingMetrics:
    """Test training metrics collection and logging."""
    
    @pytest.mark.skipif(not HAS_MLFLOW, reason="MLflow not installed")
    def test_metrics_logged_to_mlflow(self):
        """Test that training metrics are logged to MLflow."""
        with patch("mlflow.log_metrics") as mock_log:
            from medai_compass.training.metrics import log_training_step
            
            metrics = {
                "loss": 0.5,
                "learning_rate": 1e-4,
                "epoch": 1,
            }
            
            log_training_step(metrics, step=100)
            
            mock_log.assert_called_once()
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_gpu_metrics_collected(self):
        """Test GPU utilization metrics are collected."""
        with patch("torch.cuda.is_available") as mock_avail, \
             patch("torch.cuda.memory_allocated") as mock_mem:
            mock_avail.return_value = True
            mock_mem.return_value = 1024 * 1024 * 1024  # 1GB
            
            from medai_compass.training.metrics import get_gpu_metrics
            
            metrics = get_gpu_metrics()
            
            # When CUDA is not actually available, function returns empty dict
            # This is expected behavior
            if not torch.cuda.is_available():
                assert metrics == {} or "gpu_memory_used_gb" in metrics
            else:
                assert "gpu_memory_used_gb" in metrics
                assert metrics["gpu_memory_used_gb"] > 0


# =============================================================================
# Integration Tests (Require Ray Cluster)
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestRayClusterIntegration:
    """Integration tests requiring a running Ray cluster."""
    
    @pytest.fixture
    def ray_address(self) -> str:
        """Get Ray cluster address."""
        return os.environ.get("RAY_ADDRESS", "ray://localhost:10001")
    
    def test_submit_training_job(self, ray_address: str):
        """Test submitting a training job to Ray cluster."""
        ray = pytest.importorskip("ray")
        
        try:
            ray.init(address=ray_address, ignore_reinit_error=True)
            
            from medai_compass.training.ray_trainer import MedGemmaTrainer, TrainingConfig
            
            # Create minimal config for test
            config = TrainingConfig(
                model_name="medgemma-4b",
                max_steps=1,
                batch_size=1,
                dry_run=True,  # Don't actually train
            )
            
            # Submit job
            trainer = MedGemmaTrainer(config)
            result = trainer.train()
            
            assert result is not None
            assert result.status in ["completed", "dry_run"]
            
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    @pytest.mark.gpu
    def test_gpu_allocation_in_ray(self, ray_address: str):
        """Test GPU resource allocation in Ray cluster."""
        ray = pytest.importorskip("ray")
        
        try:
            ray.init(address=ray_address, ignore_reinit_error=True)
            
            # Check GPU resources
            resources = ray.cluster_resources()
            gpu_count = resources.get("GPU", 0)
            
            assert gpu_count > 0, "Ray cluster should have GPU resources"
            
            # Try to allocate GPU task
            @ray.remote(num_gpus=1)
            def gpu_task():
                import torch
                return torch.cuda.is_available()
            
            result = ray.get(gpu_task.remote())
            assert result is True, "GPU task should have CUDA available"
            
        finally:
            if ray.is_initialized():
                ray.shutdown()


# =============================================================================
# Training Pipeline End-to-End Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestTrainingPipelineE2E:
    """End-to-end tests for the training pipeline."""
    
    def test_full_training_pipeline_4b(self):
        """Test full training pipeline for 4B model (dry run)."""
        from medai_compass.training.pipeline import run_training_pipeline
        
        result = run_training_pipeline(
            model_name="medgemma-4b",
            dataset_name="test_dataset",
            max_steps=1,
            dry_run=True,
        )
        
        assert result["status"] == "success"
        assert "run_id" in result
    
    def test_full_training_pipeline_27b(self):
        """Test full training pipeline for 27B model (dry run)."""
        from medai_compass.training.pipeline import run_training_pipeline
        
        result = run_training_pipeline(
            model_name="medgemma-27b",
            dataset_name="test_dataset",
            max_steps=1,
            dry_run=True,
        )
        
        assert result["status"] == "success"
        assert result["distributed_strategy"] == "deepspeed_zero3"
