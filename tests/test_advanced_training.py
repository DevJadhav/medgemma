"""
Tests for Advanced Training Algorithms.

TDD approach: Tests written first for all advanced training methods:
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- Adapter Modules (Houlsby/Pfeiffer)
- IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- RLHF/PPO (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- KTO (Kahneman-Tversky Optimization)
- GRPO (Group Relative Policy Optimization)
- mHC (Manifold-Constrained Hyper-Connections)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any, List


# =============================================================================
# Training Algorithm Registry Tests
# =============================================================================

class TestTrainingAlgorithmRegistry:
    """Tests for training algorithm registry and selection."""

    def test_registry_contains_all_algorithms(self):
        """Verify registry contains all supported algorithms."""
        from medai_compass.training.algorithms import ALGORITHM_REGISTRY

        expected_algorithms = [
            "lora", "qlora", "dora", "adapter_houlsby", "adapter_pfeiffer",
            "ia3", "rlhf_ppo", "dpo", "kto", "grpo", "mhc"
        ]

        for algo in expected_algorithms:
            assert algo in ALGORITHM_REGISTRY, f"Missing algorithm: {algo}"

    def test_get_algorithm_by_name(self):
        """Verify algorithm can be retrieved by name."""
        from medai_compass.training.algorithms import get_algorithm

        algo = get_algorithm("lora")
        assert algo is not None
        assert hasattr(algo, "get_config")
        assert hasattr(algo, "get_trainer")

    def test_list_available_algorithms(self):
        """Verify listing all available algorithms."""
        from medai_compass.training.algorithms import list_algorithms

        algorithms = list_algorithms()
        assert len(algorithms) >= 11
        assert "lora" in algorithms
        assert "dpo" in algorithms


# =============================================================================
# DoRA (Weight-Decomposed Low-Rank Adaptation) Tests
# =============================================================================

class TestDoRAConfig:
    """Tests for DoRA configuration."""

    def test_dora_config_creation(self):
        """Verify DoRA config can be created."""
        from medai_compass.training.algorithms import DoRAConfig

        config = DoRAConfig(model_name="medgemma-4b")
        assert config is not None
        assert config.use_dora is True

    def test_dora_has_magnitude_vector(self):
        """Verify DoRA config includes magnitude vector settings."""
        from medai_compass.training.algorithms import DoRAConfig

        config = DoRAConfig(model_name="medgemma-4b")
        assert hasattr(config, "use_dora")
        assert config.use_dora is True

    def test_dora_for_4b_model(self):
        """Verify DoRA config for 4B model."""
        from medai_compass.training.algorithms import DoRAConfig

        config = DoRAConfig.for_model("medgemma-4b")
        assert config.r == 16
        assert config.use_dora is True

    def test_dora_for_27b_model(self):
        """Verify DoRA config for 27B model."""
        from medai_compass.training.algorithms import DoRAConfig

        config = DoRAConfig.for_model("medgemma-27b")
        assert config.r == 64
        assert config.use_dora is True


class TestDoRATrainer:
    """Tests for DoRA trainer."""

    def test_dora_trainer_creation(self):
        """Verify DoRA trainer can be created."""
        from medai_compass.training.algorithms import DoRATrainer

        trainer = DoRATrainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_dora_trainer_has_required_methods(self):
        """Verify DoRA trainer has required methods."""
        from medai_compass.training.algorithms import DoRATrainer

        trainer = DoRATrainer(model_name="medgemma-4b")
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "save_adapter")
        assert hasattr(trainer, "get_training_arguments")


# =============================================================================
# Adapter Module Tests (Houlsby/Pfeiffer)
# =============================================================================

class TestAdapterConfig:
    """Tests for Adapter module configurations."""

    def test_houlsby_adapter_config(self):
        """Verify Houlsby adapter config."""
        from medai_compass.training.algorithms import AdapterConfig

        config = AdapterConfig(
            model_name="medgemma-4b",
            adapter_type="houlsby"
        )
        assert config.adapter_type == "houlsby"
        assert config.reduction_factor > 0

    def test_pfeiffer_adapter_config(self):
        """Verify Pfeiffer adapter config."""
        from medai_compass.training.algorithms import AdapterConfig

        config = AdapterConfig(
            model_name="medgemma-4b",
            adapter_type="pfeiffer"
        )
        assert config.adapter_type == "pfeiffer"

    def test_adapter_bottleneck_dimension(self):
        """Verify adapter has bottleneck dimension setting."""
        from medai_compass.training.algorithms import AdapterConfig

        config = AdapterConfig(model_name="medgemma-4b")
        assert hasattr(config, "bottleneck_dim")
        assert config.bottleneck_dim > 0

    def test_adapter_config_for_model(self):
        """Verify adapter config factory method."""
        from medai_compass.training.algorithms import AdapterConfig

        config = AdapterConfig.for_model("medgemma-4b", adapter_type="houlsby")
        assert config is not None
        assert config.adapter_type == "houlsby"


class TestAdapterTrainer:
    """Tests for Adapter trainer."""

    def test_adapter_trainer_creation(self):
        """Verify adapter trainer can be created."""
        from medai_compass.training.algorithms import AdapterTrainer

        trainer = AdapterTrainer(
            model_name="medgemma-4b",
            adapter_type="houlsby"
        )
        assert trainer is not None

    def test_adapter_trainer_pfeiffer(self):
        """Verify Pfeiffer adapter trainer."""
        from medai_compass.training.algorithms import AdapterTrainer

        trainer = AdapterTrainer(
            model_name="medgemma-4b",
            adapter_type="pfeiffer"
        )
        assert trainer.adapter_type == "pfeiffer"


# =============================================================================
# IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) Tests
# =============================================================================

class TestIA3Config:
    """Tests for IA3 configuration."""

    def test_ia3_config_creation(self):
        """Verify IA3 config can be created."""
        from medai_compass.training.algorithms import IA3Config

        config = IA3Config(model_name="medgemma-4b")
        assert config is not None

    def test_ia3_target_modules(self):
        """Verify IA3 has target modules for scaling."""
        from medai_compass.training.algorithms import IA3Config

        config = IA3Config(model_name="medgemma-4b")
        assert hasattr(config, "target_modules")
        assert len(config.target_modules) > 0

    def test_ia3_feedforward_modules(self):
        """Verify IA3 includes feedforward modules."""
        from medai_compass.training.algorithms import IA3Config

        config = IA3Config(model_name="medgemma-4b")
        assert hasattr(config, "feedforward_modules")

    def test_ia3_for_model(self):
        """Verify IA3 config factory method."""
        from medai_compass.training.algorithms import IA3Config

        config = IA3Config.for_model("medgemma-4b")
        assert config is not None


class TestIA3Trainer:
    """Tests for IA3 trainer."""

    def test_ia3_trainer_creation(self):
        """Verify IA3 trainer can be created."""
        from medai_compass.training.algorithms import IA3Trainer

        trainer = IA3Trainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_ia3_trainer_has_required_methods(self):
        """Verify IA3 trainer has required methods."""
        from medai_compass.training.algorithms import IA3Trainer

        trainer = IA3Trainer(model_name="medgemma-4b")
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "save_adapter")


# =============================================================================
# RLHF/PPO (Reinforcement Learning from Human Feedback) Tests
# =============================================================================

class TestRLHFConfig:
    """Tests for RLHF/PPO configuration."""

    def test_rlhf_config_creation(self):
        """Verify RLHF config can be created."""
        from medai_compass.training.algorithms import RLHFConfig

        config = RLHFConfig(model_name="medgemma-4b")
        assert config is not None

    def test_rlhf_has_ppo_params(self):
        """Verify RLHF config has PPO parameters."""
        from medai_compass.training.algorithms import RLHFConfig

        config = RLHFConfig(model_name="medgemma-4b")
        assert hasattr(config, "ppo_epochs")
        assert hasattr(config, "clip_range")
        assert hasattr(config, "value_coefficient")
        assert hasattr(config, "entropy_coefficient")

    def test_rlhf_has_reward_model_config(self):
        """Verify RLHF config includes reward model settings."""
        from medai_compass.training.algorithms import RLHFConfig

        config = RLHFConfig(model_name="medgemma-4b")
        assert hasattr(config, "reward_model_name")
        assert hasattr(config, "kl_penalty")

    def test_rlhf_for_model(self):
        """Verify RLHF config factory method."""
        from medai_compass.training.algorithms import RLHFConfig

        config = RLHFConfig.for_model("medgemma-4b")
        assert config is not None


class TestRLHFTrainer:
    """Tests for RLHF/PPO trainer."""

    def test_rlhf_trainer_creation(self):
        """Verify RLHF trainer can be created."""
        from medai_compass.training.algorithms import RLHFTrainer

        trainer = RLHFTrainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_rlhf_trainer_has_ppo_methods(self):
        """Verify RLHF trainer has PPO training methods."""
        from medai_compass.training.algorithms import RLHFTrainer

        trainer = RLHFTrainer(model_name="medgemma-4b")
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "compute_rewards")
        assert hasattr(trainer, "ppo_step")


# =============================================================================
# DPO (Direct Preference Optimization) Tests
# =============================================================================

class TestDPOConfig:
    """Tests for DPO configuration."""

    def test_dpo_config_creation(self):
        """Verify DPO config can be created."""
        from medai_compass.training.algorithms import DPOConfig

        config = DPOConfig(model_name="medgemma-4b")
        assert config is not None

    def test_dpo_has_beta_param(self):
        """Verify DPO config has beta parameter."""
        from medai_compass.training.algorithms import DPOConfig

        config = DPOConfig(model_name="medgemma-4b")
        assert hasattr(config, "beta")
        assert config.beta > 0

    def test_dpo_has_reference_model_config(self):
        """Verify DPO config includes reference model settings."""
        from medai_compass.training.algorithms import DPOConfig

        config = DPOConfig(model_name="medgemma-4b")
        assert hasattr(config, "reference_free")
        assert hasattr(config, "label_smoothing")

    def test_dpo_loss_type(self):
        """Verify DPO config has loss type setting."""
        from medai_compass.training.algorithms import DPOConfig

        config = DPOConfig(model_name="medgemma-4b")
        assert hasattr(config, "loss_type")
        assert config.loss_type in ["sigmoid", "hinge", "ipo"]


class TestDPOTrainer:
    """Tests for DPO trainer."""

    def test_dpo_trainer_creation(self):
        """Verify DPO trainer can be created."""
        from medai_compass.training.algorithms import DPOTrainer

        trainer = DPOTrainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_dpo_trainer_has_required_methods(self):
        """Verify DPO trainer has required methods."""
        from medai_compass.training.algorithms import DPOTrainer

        trainer = DPOTrainer(model_name="medgemma-4b")
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "compute_dpo_loss")


# =============================================================================
# KTO (Kahneman-Tversky Optimization) Tests
# =============================================================================

class TestKTOConfig:
    """Tests for KTO configuration."""

    def test_kto_config_creation(self):
        """Verify KTO config can be created."""
        from medai_compass.training.algorithms import KTOConfig

        config = KTOConfig(model_name="medgemma-4b")
        assert config is not None

    def test_kto_has_loss_aversion_param(self):
        """Verify KTO config has loss aversion parameter (lambda)."""
        from medai_compass.training.algorithms import KTOConfig

        config = KTOConfig(model_name="medgemma-4b")
        assert hasattr(config, "loss_aversion")
        assert config.loss_aversion > 0

    def test_kto_desirable_weight(self):
        """Verify KTO config has desirable/undesirable weights."""
        from medai_compass.training.algorithms import KTOConfig

        config = KTOConfig(model_name="medgemma-4b")
        assert hasattr(config, "desirable_weight")
        assert hasattr(config, "undesirable_weight")

    def test_kto_for_model(self):
        """Verify KTO config factory method."""
        from medai_compass.training.algorithms import KTOConfig

        config = KTOConfig.for_model("medgemma-4b")
        assert config is not None


class TestKTOTrainer:
    """Tests for KTO trainer."""

    def test_kto_trainer_creation(self):
        """Verify KTO trainer can be created."""
        from medai_compass.training.algorithms import KTOTrainer

        trainer = KTOTrainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_kto_trainer_has_required_methods(self):
        """Verify KTO trainer has required methods."""
        from medai_compass.training.algorithms import KTOTrainer

        trainer = KTOTrainer(model_name="medgemma-4b")
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "compute_kto_loss")


# =============================================================================
# GRPO (Group Relative Policy Optimization) Tests
# =============================================================================

class TestGRPOConfig:
    """Tests for GRPO configuration."""

    def test_grpo_config_creation(self):
        """Verify GRPO config can be created."""
        from medai_compass.training.algorithms import GRPOConfig

        config = GRPOConfig(model_name="medgemma-4b")
        assert config is not None

    def test_grpo_has_group_size(self):
        """Verify GRPO config has group size parameter."""
        from medai_compass.training.algorithms import GRPOConfig

        config = GRPOConfig(model_name="medgemma-4b")
        assert hasattr(config, "group_size")
        assert config.group_size > 0

    def test_grpo_has_sampling_params(self):
        """Verify GRPO config has sampling parameters."""
        from medai_compass.training.algorithms import GRPOConfig

        config = GRPOConfig(model_name="medgemma-4b")
        assert hasattr(config, "num_generations")
        assert hasattr(config, "temperature")

    def test_grpo_for_model(self):
        """Verify GRPO config factory method."""
        from medai_compass.training.algorithms import GRPOConfig

        config = GRPOConfig.for_model("medgemma-4b")
        assert config is not None


class TestGRPOTrainer:
    """Tests for GRPO trainer."""

    def test_grpo_trainer_creation(self):
        """Verify GRPO trainer can be created."""
        from medai_compass.training.algorithms import GRPOTrainer

        trainer = GRPOTrainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_grpo_trainer_has_required_methods(self):
        """Verify GRPO trainer has required methods."""
        from medai_compass.training.algorithms import GRPOTrainer

        trainer = GRPOTrainer(model_name="medgemma-4b")
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "compute_group_rewards")
        assert hasattr(trainer, "generate_responses")


# =============================================================================
# mHC (Manifold-Constrained Hyper-Connections) Tests
# =============================================================================

class TestMHCConfig:
    """Tests for mHC configuration."""

    def test_mhc_config_creation(self):
        """Verify mHC config can be created."""
        from medai_compass.training.algorithms import MHCConfig

        config = MHCConfig(model_name="medgemma-4b")
        assert config is not None

    def test_mhc_has_manifold_params(self):
        """Verify mHC config has manifold constraint parameters."""
        from medai_compass.training.algorithms import MHCConfig

        config = MHCConfig(model_name="medgemma-4b")
        assert hasattr(config, "manifold_dim")
        assert hasattr(config, "constraint_strength")

    def test_mhc_has_hyper_connection_config(self):
        """Verify mHC config has hyper-connection settings."""
        from medai_compass.training.algorithms import MHCConfig

        config = MHCConfig(model_name="medgemma-4b")
        assert hasattr(config, "hyper_connection_type")
        assert hasattr(config, "connection_rank")

    def test_mhc_for_model(self):
        """Verify mHC config factory method."""
        from medai_compass.training.algorithms import MHCConfig

        config = MHCConfig.for_model("medgemma-4b")
        assert config is not None


class TestMHCTrainer:
    """Tests for mHC trainer."""

    def test_mhc_trainer_creation(self):
        """Verify mHC trainer can be created."""
        from medai_compass.training.algorithms import MHCTrainer

        trainer = MHCTrainer(model_name="medgemma-4b")
        assert trainer is not None

    def test_mhc_trainer_has_required_methods(self):
        """Verify mHC trainer has required methods."""
        from medai_compass.training.algorithms import MHCTrainer

        trainer = MHCTrainer(model_name="medgemma-4b")
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "compute_manifold_loss")


# =============================================================================
# Algorithm Selection and Switching Tests
# =============================================================================

class TestAlgorithmSelection:
    """Tests for algorithm selection in training pipeline."""

    def test_create_trainer_by_algorithm_name(self):
        """Verify trainer can be created by algorithm name."""
        from medai_compass.training.algorithms import create_trainer

        algorithms = ["lora", "qlora", "dora", "ia3", "dpo", "kto", "grpo"]

        for algo in algorithms:
            trainer = create_trainer(algo, model_name="medgemma-4b")
            assert trainer is not None, f"Failed to create trainer for {algo}"

    def test_switch_algorithm_at_runtime(self):
        """Verify algorithm can be switched at runtime."""
        from medai_compass.training.algorithms import TrainingPipelineManager

        manager = TrainingPipelineManager(model_name="medgemma-4b")

        # Start with LoRA
        manager.set_algorithm("lora")
        assert manager.current_algorithm == "lora"

        # Switch to DPO
        manager.set_algorithm("dpo")
        assert manager.current_algorithm == "dpo"

    def test_algorithm_compatibility_check(self):
        """Verify algorithm compatibility checking."""
        from medai_compass.training.algorithms import check_algorithm_compatibility

        # RLHF requires reward model
        compat = check_algorithm_compatibility("rlhf_ppo", model_name="medgemma-4b")
        assert "requires_reward_model" in compat

        # LoRA should be compatible with basic setup
        compat = check_algorithm_compatibility("lora", model_name="medgemma-4b")
        assert compat["compatible"] is True


# =============================================================================
# Ray Integration Tests
# =============================================================================

class TestRayTrainingIntegration:
    """Tests for Ray distributed training integration."""

    def test_ray_trainer_with_algorithm(self):
        """Verify Ray trainer works with different algorithms."""
        from medai_compass.training.algorithms import create_ray_trainer

        trainer = create_ray_trainer(
            algorithm="lora",
            model_name="medgemma-4b",
            num_workers=2
        )
        assert trainer is not None
        assert hasattr(trainer, "fit")

    def test_ray_trainer_scaling_config(self):
        """Verify Ray trainer has proper scaling config."""
        from medai_compass.training.algorithms import create_ray_trainer

        trainer = create_ray_trainer(
            algorithm="dpo",
            model_name="medgemma-4b",
            num_workers=4,
            use_gpu=True
        )
        assert trainer.scaling_config is not None
        assert trainer.scaling_config["num_workers"] == 4

    def test_ray_trainer_algorithm_switching(self):
        """Verify algorithm can be changed in Ray trainer."""
        from medai_compass.training.algorithms import RayTrainingOrchestrator

        orchestrator = RayTrainingOrchestrator(model_name="medgemma-4b")

        # Set algorithm
        orchestrator.configure_algorithm("lora")
        assert orchestrator.algorithm == "lora"

        # Switch
        orchestrator.configure_algorithm("kto")
        assert orchestrator.algorithm == "kto"


# =============================================================================
# Training Pipeline Configuration Tests
# =============================================================================

class TestTrainingPipelineConfig:
    """Tests for unified training pipeline configuration."""

    def test_unified_config_creation(self):
        """Verify unified config can be created."""
        from medai_compass.training.algorithms import UnifiedTrainingConfig

        config = UnifiedTrainingConfig(
            model_name="medgemma-4b",
            algorithm="dpo"
        )
        assert config is not None
        assert config.algorithm == "dpo"

    def test_config_validation(self):
        """Verify config validation works."""
        from medai_compass.training.algorithms import UnifiedTrainingConfig

        # Valid config
        config = UnifiedTrainingConfig(
            model_name="medgemma-4b",
            algorithm="lora"
        )
        assert config.validate() is True

        # Invalid algorithm should raise
        with pytest.raises(ValueError):
            UnifiedTrainingConfig(
                model_name="medgemma-4b",
                algorithm="invalid_algo"
            )

    def test_config_to_dict(self):
        """Verify config can be serialized to dict."""
        from medai_compass.training.algorithms import UnifiedTrainingConfig

        config = UnifiedTrainingConfig(
            model_name="medgemma-4b",
            algorithm="grpo"
        )
        config_dict = config.to_dict()

        assert "model_name" in config_dict
        assert "algorithm" in config_dict
        assert config_dict["algorithm"] == "grpo"

    def test_config_from_dict(self):
        """Verify config can be loaded from dict."""
        from medai_compass.training.algorithms import UnifiedTrainingConfig

        config_dict = {
            "model_name": "medgemma-27b",
            "algorithm": "mhc",
            "learning_rate": 1e-5
        }

        config = UnifiedTrainingConfig.from_dict(config_dict)
        assert config.model_name == "medgemma-27b"
        assert config.algorithm == "mhc"


# =============================================================================
# Medical Domain Specific Tests
# =============================================================================

class TestMedicalDomainTraining:
    """Tests for medical domain specific training features."""

    def test_medical_safety_callback(self):
        """Verify medical safety callback is available."""
        from medai_compass.training.algorithms import MedicalSafetyCallback

        callback = MedicalSafetyCallback()
        assert callback is not None
        assert hasattr(callback, "on_step_end")

    def test_phi_filtering_during_training(self):
        """Verify PHI filtering is applied during training."""
        from medai_compass.training.algorithms import create_trainer

        trainer = create_trainer(
            "dpo",
            model_name="medgemma-4b",
            apply_phi_filter=True
        )
        assert trainer.apply_phi_filter is True

    def test_medical_evaluation_metrics(self):
        """Verify medical evaluation metrics are available."""
        from medai_compass.training.algorithms import MedicalEvaluationCallback

        callback = MedicalEvaluationCallback()
        assert hasattr(callback, "compute_medical_accuracy")
        assert hasattr(callback, "compute_safety_score")
