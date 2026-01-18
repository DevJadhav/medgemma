"""TDD Tests for Phase 3: Training Pipeline Development.

Tests cover all Phase 3 tasks:
- 3.1: Ray Train integration
- 3.2: LoRA/QLoRA training module
- 3.3: Distributed checkpointing
- 3.4: MLflow experiment tracking
- 3.5: Training monitoring dashboard
- 3.6: Gradient accumulation
- 3.7: Mixed precision training

Deliverables tested:
- medai_compass/pipelines/training_pipeline.py
- medai_compass/pipelines/lora_trainer.py
- medai_compass/pipelines/mlflow_integration.py
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Task 3.1: Ray Train Integration Tests
# =============================================================================

class TestRayTrainIntegration:
    """Test Ray Train integration for distributed training."""
    
    def test_training_pipeline_module_exists(self):
        """Training pipeline module should exist."""
        from medai_compass.pipelines import training_pipeline
        assert training_pipeline is not None
    
    def test_training_pipeline_has_required_classes(self):
        """Training pipeline should have required classes."""
        from medai_compass.pipelines.training_pipeline import (
            TrainingPipelineConfig,
            TrainingPipelineOrchestrator,
        )
        assert TrainingPipelineConfig is not None
        assert TrainingPipelineOrchestrator is not None
    
    def test_create_training_config_for_4b(self):
        """Should create training config for MedGemma 4B."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineConfig
        
        config = TrainingPipelineConfig(model_name="medgemma-4b")
        
        assert config.model_name == "medgemma-4b"
        assert config.distributed_strategy == "single_gpu"
        assert config.num_workers == 1
    
    def test_create_training_config_for_27b(self):
        """Should create training config for MedGemma 27B."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineConfig
        
        config = TrainingPipelineConfig(model_name="medgemma-27b")
        
        assert config.model_name == "medgemma-27b"
        assert config.distributed_strategy == "deepspeed_zero3"
        assert config.num_workers >= 4  # Multi-GPU required
    
    def test_pipeline_orchestrator_has_run_method(self):
        """Orchestrator should have run method for Airflow compatibility."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineOrchestrator
        
        assert hasattr(TrainingPipelineOrchestrator, "run")
        assert hasattr(TrainingPipelineOrchestrator, "prepare_data")
        assert hasattr(TrainingPipelineOrchestrator, "train")
        assert hasattr(TrainingPipelineOrchestrator, "evaluate")


class TestRayTrainScaling:
    """Test Ray Train scaling configuration."""
    
    def test_scaling_config_single_gpu(self):
        """Should configure single GPU for 4B model."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineConfig
        
        config = TrainingPipelineConfig(model_name="medgemma-4b")
        scaling = config.get_scaling_config()
        
        assert scaling["num_workers"] == 1
        assert scaling["use_gpu"] is True
    
    def test_scaling_config_multi_gpu(self):
        """Should configure multi-GPU for 27B model."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineConfig
        
        config = TrainingPipelineConfig(model_name="medgemma-27b")
        scaling = config.get_scaling_config()
        
        assert scaling["num_workers"] >= 4
        assert scaling["use_gpu"] is True


# =============================================================================
# Task 3.2: LoRA/QLoRA Training Module Tests
# =============================================================================

class TestLoRATrainerModuleStructure:
    """Test LoRA trainer module structure."""
    
    def test_lora_trainer_module_exists(self):
        """LoRA trainer module should exist."""
        from medai_compass.pipelines import lora_trainer
        assert lora_trainer is not None
    
    def test_lora_trainer_has_required_classes(self):
        """LoRA trainer should have LoRATrainer and QLoRATrainer."""
        from medai_compass.pipelines.lora_trainer import (
            LoRATrainer,
            QLoRATrainer,
            LoRAConfig,
        )
        assert LoRATrainer is not None
        assert QLoRATrainer is not None
        assert LoRAConfig is not None
    
    def test_lora_trainer_supports_model_selection(self):
        """LoRA trainer should support both 4B and 27B models."""
        from medai_compass.pipelines.lora_trainer import LoRATrainer
        
        assert hasattr(LoRATrainer, "from_model_name")


class TestLoRAConfig:
    """Test LoRA configuration."""
    
    def test_create_lora_config_4b(self):
        """Should create LoRA config for 4B model."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.target_modules
    
    def test_create_lora_config_27b(self):
        """Should create LoRA config for 27B model."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-27b")
        
        assert config.r == 64  # Larger for 27B
        assert config.lora_alpha == 128
    
    def test_lora_config_has_target_modules(self):
        """LoRA config should have target modules."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for module in expected_modules:
            assert module in config.target_modules


class TestQLoRATrainer:
    """Test QLoRA trainer with quantization."""
    
    def test_qlora_trainer_has_quantization_config(self):
        """QLoRA trainer should support 4-bit quantization."""
        from medai_compass.pipelines.lora_trainer import QLoRATrainer
        
        assert hasattr(QLoRATrainer, "get_quantization_config")
    
    def test_qlora_quantization_config(self):
        """Should create 4-bit quantization config."""
        from medai_compass.pipelines.lora_trainer import QLoRATrainer
        
        quant_config = QLoRATrainer.get_quantization_config()
        
        assert quant_config["load_in_4bit"] is True
        assert quant_config["bnb_4bit_compute_dtype"] == "bfloat16"
        assert quant_config["bnb_4bit_quant_type"] == "nf4"


class TestLoRATrainerTraining:
    """Test LoRA trainer training functionality."""
    
    def test_trainer_has_train_method(self):
        """Trainer should have train method."""
        from medai_compass.pipelines.lora_trainer import LoRATrainer
        
        assert hasattr(LoRATrainer, "train")
    
    def test_trainer_has_save_method(self):
        """Trainer should have save method for LoRA adapters."""
        from medai_compass.pipelines.lora_trainer import LoRATrainer
        
        assert hasattr(LoRATrainer, "save_adapter")
    
    def test_trainer_has_merge_method(self):
        """Trainer should have method to merge LoRA weights."""
        from medai_compass.pipelines.lora_trainer import LoRATrainer
        
        assert hasattr(LoRATrainer, "merge_and_save")


# =============================================================================
# Task 3.3: Distributed Checkpointing Tests
# =============================================================================

class TestDistributedCheckpointing:
    """Test distributed checkpointing functionality."""
    
    def test_checkpoint_manager_in_pipeline(self):
        """Training pipeline should integrate checkpoint manager."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineOrchestrator
        
        assert hasattr(TrainingPipelineOrchestrator, "setup_checkpointing")
    
    def test_checkpoint_config_in_training_config(self):
        """Training config should have checkpoint settings."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineConfig
        
        config = TrainingPipelineConfig(model_name="medgemma-4b")
        
        assert hasattr(config, "checkpoint_dir")
        assert hasattr(config, "save_steps")
        assert hasattr(config, "save_total_limit")
    
    def test_checkpoint_resume_capability(self):
        """Should support resuming from checkpoint."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineConfig
        
        config = TrainingPipelineConfig(
            model_name="medgemma-4b",
            resume_from_checkpoint="/path/to/checkpoint"
        )
        
        assert config.resume_from_checkpoint == "/path/to/checkpoint"


# =============================================================================
# Task 3.4: MLflow Experiment Tracking Tests
# =============================================================================

class TestMLflowIntegrationModule:
    """Test MLflow integration module structure."""
    
    def test_mlflow_integration_module_exists(self):
        """MLflow integration module should exist."""
        from medai_compass.pipelines import mlflow_integration
        assert mlflow_integration is not None
    
    def test_mlflow_integration_has_required_classes(self):
        """MLflow integration should have required classes."""
        from medai_compass.pipelines.mlflow_integration import (
            MLflowTracker,
            ExperimentConfig,
        )
        assert MLflowTracker is not None
        assert ExperimentConfig is not None


class TestMLflowTracker:
    """Test MLflow tracker functionality."""
    
    def test_create_mlflow_tracker(self):
        """Should create MLflow tracker."""
        from medai_compass.pipelines.mlflow_integration import MLflowTracker
        
        tracker = MLflowTracker(
            tracking_uri="http://localhost:5000",
            experiment_name="test-experiment"
        )
        
        assert tracker.tracking_uri == "http://localhost:5000"
        assert tracker.experiment_name == "test-experiment"
    
    def test_tracker_has_logging_methods(self):
        """Tracker should have metric logging methods."""
        from medai_compass.pipelines.mlflow_integration import MLflowTracker
        
        assert hasattr(MLflowTracker, "log_metrics")
        assert hasattr(MLflowTracker, "log_params")
        assert hasattr(MLflowTracker, "log_artifact")
    
    def test_tracker_has_run_management(self):
        """Tracker should have run management methods."""
        from medai_compass.pipelines.mlflow_integration import MLflowTracker
        
        assert hasattr(MLflowTracker, "start_run")
        assert hasattr(MLflowTracker, "end_run")
    
    def test_tracker_has_model_registry_methods(self):
        """Tracker should have model registry methods."""
        from medai_compass.pipelines.mlflow_integration import MLflowTracker
        
        assert hasattr(MLflowTracker, "register_model")
        assert hasattr(MLflowTracker, "transition_model_stage")


class TestMLflowExperimentConfig:
    """Test MLflow experiment configuration."""
    
    def test_create_experiment_config(self):
        """Should create experiment config."""
        from medai_compass.pipelines.mlflow_integration import ExperimentConfig
        
        config = ExperimentConfig(
            name="medgemma-finetuning",
            tags={"model": "medgemma-4b", "task": "medical-qa"}
        )
        
        assert config.name == "medgemma-finetuning"
        assert config.tags["model"] == "medgemma-4b"
    
    def test_experiment_config_default_tags(self):
        """Experiment config should have default tags."""
        from medai_compass.pipelines.mlflow_integration import ExperimentConfig
        
        config = ExperimentConfig(name="test")
        
        # Should have framework info
        assert hasattr(config, "get_default_tags")


# =============================================================================
# Task 3.5: Training Monitoring Dashboard Tests
# =============================================================================

class TestTrainingDashboard:
    """Test Grafana training dashboard configuration."""
    
    def test_dashboard_config_exists(self):
        """Grafana dashboard config should exist."""
        dashboard_path = Path("docker/grafana/dashboards/training_metrics.json")
        assert dashboard_path.exists(), "Training metrics dashboard should exist"
    
    def test_dashboard_has_required_panels(self):
        """Dashboard should have required panels."""
        dashboard_path = Path("docker/grafana/dashboards/training_metrics.json")
        
        with open(dashboard_path) as f:
            dashboard = json.load(f)
        
        panel_titles = [p.get("title", "") for p in dashboard.get("panels", [])]
        
        # Check for essential panels
        assert any("Loss" in t for t in panel_titles), "Should have loss panel"
        assert any("Learning Rate" in t or "LR" in t for t in panel_titles), "Should have LR panel"
    
    def test_dashboard_has_prometheus_datasource(self):
        """Dashboard should use Prometheus datasource."""
        dashboard_path = Path("docker/grafana/dashboards/training_metrics.json")
        
        with open(dashboard_path) as f:
            dashboard = json.load(f)
        
        # Check templating for datasource
        assert "templating" in dashboard or any(
            p.get("datasource", {}).get("type") == "prometheus" 
            for p in dashboard.get("panels", [])
        )


# =============================================================================
# Task 3.6: Gradient Accumulation Tests
# =============================================================================

class TestGradientAccumulation:
    """Test gradient accumulation support."""
    
    def test_lora_config_has_gradient_accumulation(self):
        """LoRA config should support gradient accumulation."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        assert hasattr(config, "gradient_accumulation_steps")
        assert config.gradient_accumulation_steps >= 1
    
    def test_gradient_accumulation_4b_default(self):
        """4B model should have appropriate gradient accumulation."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        assert config.gradient_accumulation_steps == 4
    
    def test_gradient_accumulation_27b_default(self):
        """27B model should have higher gradient accumulation."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-27b")
        
        assert config.gradient_accumulation_steps >= 8  # Higher for 27B


class TestEffectiveBatchSize:
    """Test effective batch size calculation."""
    
    def test_effective_batch_size_calculation(self):
        """Should calculate effective batch size correctly."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        assert effective_batch == 16  # 4 * 4
    
    def test_effective_batch_size_27b(self):
        """Should calculate effective batch size for 27B."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-27b")
        
        # 27B uses smaller batch with more accumulation
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        assert effective_batch >= 8


# =============================================================================
# Task 3.7: Mixed Precision Training Tests
# =============================================================================

class TestMixedPrecisionTraining:
    """Test mixed precision training support."""
    
    def test_lora_config_has_mixed_precision(self):
        """LoRA config should have mixed precision settings."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        assert hasattr(config, "mixed_precision")
        assert hasattr(config, "bf16")
        assert hasattr(config, "fp16")
    
    def test_default_mixed_precision_bf16(self):
        """Default mixed precision should be bf16."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        assert config.bf16 is True
        assert config.fp16 is False
    
    def test_mixed_precision_in_training_args(self):
        """Mixed precision should be in training arguments."""
        from medai_compass.pipelines.lora_trainer import LoRATrainer
        
        trainer = LoRATrainer(model_name="medgemma-4b")
        training_args = trainer.get_training_arguments()
        
        assert training_args.get("bf16") is True or training_args.get("fp16") is True


class TestFlashAttention:
    """Test Flash Attention integration."""
    
    def test_lora_config_has_flash_attention_option(self):
        """LoRA config should have Flash Attention option."""
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        assert hasattr(config, "use_flash_attention")
    
    def test_flash_attention_default_depends_on_platform(self):
        """Flash Attention default should depend on platform."""
        import sys
        from medai_compass.pipelines.lora_trainer import LoRAConfig
        
        config = LoRAConfig.for_model("medgemma-4b")
        
        # Flash Attention only available on Linux with CUDA
        if sys.platform == "linux":
            # May or may not be enabled depending on CUDA availability
            assert isinstance(config.use_flash_attention, bool)
        else:
            # Should be disabled on non-Linux
            assert config.use_flash_attention is False


# =============================================================================
# Training Callbacks Tests
# =============================================================================

class TestTrainingCallbacks:
    """Test custom training callbacks."""
    
    def test_callbacks_module_exists(self):
        """Callbacks module should exist."""
        from medai_compass.training import callbacks
        assert callbacks is not None
    
    def test_mlflow_callback_exists(self):
        """MLflow logging callback should exist."""
        from medai_compass.training.callbacks import MLflowCallback
        assert MLflowCallback is not None
    
    def test_early_stopping_callback_exists(self):
        """Early stopping callback should exist."""
        from medai_compass.training.callbacks import EarlyStoppingCallback
        assert EarlyStoppingCallback is not None
    
    def test_gradient_accumulation_callback_exists(self):
        """Gradient accumulation tracking callback should exist."""
        from medai_compass.training.callbacks import GradientAccumulationCallback
        assert GradientAccumulationCallback is not None


class TestMLflowCallback:
    """Test MLflow callback functionality."""
    
    def test_mlflow_callback_logs_metrics(self):
        """MLflow callback should log metrics."""
        from medai_compass.training.callbacks import MLflowCallback
        
        callback = MLflowCallback()
        
        assert hasattr(callback, "on_log")
        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_train_end")


class TestEarlyStoppingCallback:
    """Test early stopping callback."""
    
    def test_early_stopping_has_patience(self):
        """Early stopping should have patience parameter."""
        from medai_compass.training.callbacks import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(patience=3)
        
        assert callback.patience == 3
    
    def test_early_stopping_monitors_metric(self):
        """Early stopping should monitor a metric."""
        from medai_compass.training.callbacks import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(
            patience=3,
            metric="eval_loss",
            mode="min"
        )
        
        assert callback.metric == "eval_loss"
        assert callback.mode == "min"


# =============================================================================
# Integration Tests
# =============================================================================

class TestTrainingPipelineIntegration:
    """Test training pipeline integration."""
    
    def test_pipeline_uses_lora_trainer(self):
        """Training pipeline should use LoRA trainer."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineOrchestrator
        
        orchestrator = TrainingPipelineOrchestrator(model_name="medgemma-4b")
        
        assert hasattr(orchestrator, "trainer")
    
    def test_pipeline_uses_mlflow_tracker(self):
        """Training pipeline should use MLflow tracker."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineOrchestrator
        
        orchestrator = TrainingPipelineOrchestrator(model_name="medgemma-4b")
        
        assert hasattr(orchestrator, "mlflow_tracker")
    
    def test_pipeline_supports_both_models(self):
        """Pipeline should support both 4B and 27B models."""
        from medai_compass.pipelines.training_pipeline import TrainingPipelineOrchestrator
        
        # Test 4B
        orchestrator_4b = TrainingPipelineOrchestrator(model_name="medgemma-4b")
        assert orchestrator_4b.model_name == "medgemma-4b"
        
        # Test 27B
        orchestrator_27b = TrainingPipelineOrchestrator(model_name="medgemma-27b")
        assert orchestrator_27b.model_name == "medgemma-27b"


class TestAirflowCompatibility:
    """Test Airflow DAG compatibility."""
    
    def test_pipeline_has_dag_functions(self):
        """Pipeline should have Airflow-compatible DAG functions."""
        from medai_compass.pipelines import training_pipeline
        
        # Should have callable functions for Airflow tasks
        assert hasattr(training_pipeline, "prepare_data_task")
        assert hasattr(training_pipeline, "train_model_task")
        assert hasattr(training_pipeline, "evaluate_model_task")
        assert hasattr(training_pipeline, "register_model_task")
    
    def test_dag_functions_are_callable(self):
        """DAG functions should be callable."""
        from medai_compass.pipelines.training_pipeline import (
            prepare_data_task,
            train_model_task,
            evaluate_model_task,
            register_model_task,
        )
        
        assert callable(prepare_data_task)
        assert callable(train_model_task)
        assert callable(evaluate_model_task)
        assert callable(register_model_task)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestDeepSpeedConfigs:
    """Test DeepSpeed configuration files."""
    
    def test_zero2_config_exists(self):
        """DeepSpeed ZeRO-2 config should exist."""
        config_path = Path("config/deepspeed/ds_config_zero2.json")
        assert config_path.exists()
    
    def test_zero3_config_exists(self):
        """DeepSpeed ZeRO-3 config should exist."""
        config_path = Path("config/deepspeed/ds_config_zero3.json")
        assert config_path.exists()
    
    def test_zero2_config_valid(self):
        """DeepSpeed ZeRO-2 config should be valid JSON."""
        config_path = Path("config/deepspeed/ds_config_zero2.json")
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["zero_optimization"]["stage"] == 2
        assert config["bf16"]["enabled"] is True
    
    def test_zero3_config_valid(self):
        """DeepSpeed ZeRO-3 config should be valid JSON."""
        config_path = Path("config/deepspeed/ds_config_zero3.json")
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["zero_optimization"]["stage"] == 3
        assert "offload_optimizer" in config["zero_optimization"]
        assert config["bf16"]["enabled"] is True


class TestTrainingModuleExports:
    """Test training module exports."""
    
    def test_training_module_exports(self):
        """Training module should export required classes."""
        from medai_compass.training import (
            MedGemmaTrainer,
            TrainingConfig,
            TrainingResult,
            select_model,
            get_training_config,
            MODEL_PROFILES,
        )
        
        assert MedGemmaTrainer is not None
        assert TrainingConfig is not None
        assert MODEL_PROFILES is not None
    
    def test_model_profiles_have_both_models(self):
        """Model profiles should have both 4B and 27B."""
        from medai_compass.training import MODEL_PROFILES
        
        assert "medgemma-4b" in MODEL_PROFILES
        assert "medgemma-27b" in MODEL_PROFILES
