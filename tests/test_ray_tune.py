"""
TDD Tests for Ray Tune Hyperparameter Tuning.

Tests ASHA, PBT, and Hyperband schedulers.
"""

import os
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch


class TestASHAScheduler:
    """Test ASHA scheduler implementation."""

    def test_asha_scheduler_creation(self):
        """Test ASHA scheduler can be created with config."""
        from medai_compass.tuning.asha_tuner import create_asha_scheduler

        scheduler = create_asha_scheduler(
            max_t=1000,
            grace_period=100,
            reduction_factor=3,
            brackets=1,
        )
        assert scheduler is not None

    def test_asha_scheduler_config(self):
        """Test ASHA scheduler has correct configuration."""
        from medai_compass.tuning.asha_tuner import create_asha_scheduler

        scheduler = create_asha_scheduler(
            max_t=5000,
            grace_period=100,
            reduction_factor=3,
        )

        # Check scheduler attributes
        assert hasattr(scheduler, "_max_t") or hasattr(scheduler, "max_t")

    def test_asha_tuner_from_hydra_config(self):
        """Test creating ASHA tuner from Hydra config."""
        from medai_compass.tuning.asha_tuner import ASHATuner
        from medai_compass.config.hydra_config import MedGemmaConfig
        from omegaconf import OmegaConf

        config = MedGemmaConfig()
        cfg = OmegaConf.structured(config)

        tuner = ASHATuner(cfg)
        assert tuner is not None
        assert tuner.scheduler is not None


class TestPBTScheduler:
    """Test PBT scheduler implementation."""

    def test_pbt_scheduler_creation(self):
        """Test PBT scheduler can be created with config."""
        from medai_compass.tuning.pbt_tuner import create_pbt_scheduler

        hyperparam_mutations = {
            "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4],
            "weight_decay": [0.001, 0.01, 0.1],
        }

        scheduler = create_pbt_scheduler(
            perturbation_interval=100,
            hyperparam_mutations=hyperparam_mutations,
            quantile_fraction=0.25,
            resample_probability=0.25,
        )
        assert scheduler is not None

    def test_pbt_explore_function(self):
        """Test PBT exploration function works correctly."""
        from medai_compass.tuning.pbt_tuner import create_pbt_explore_fn

        explore_fn = create_pbt_explore_fn(
            learning_rate_range=(1e-6, 1e-3),
            weight_decay_range=(1e-4, 0.1),
        )

        config = {"learning_rate": 1e-4, "weight_decay": 0.01}
        new_config = explore_fn(config)

        # New config should have perturbed values
        assert "learning_rate" in new_config
        assert "weight_decay" in new_config

    def test_pbt_tuner_from_hydra_config(self):
        """Test creating PBT tuner from Hydra config."""
        from medai_compass.tuning.pbt_tuner import PBTTuner
        from medai_compass.config.hydra_config import MedGemmaConfig
        from omegaconf import OmegaConf

        config = MedGemmaConfig()
        cfg = OmegaConf.structured(config)
        cfg.tuning.scheduler = "pbt"

        tuner = PBTTuner(cfg)
        assert tuner is not None
        assert tuner.scheduler is not None


class TestHyperbandScheduler:
    """Test Hyperband scheduler implementation."""

    def test_hyperband_scheduler_creation(self):
        """Test Hyperband scheduler can be created."""
        from medai_compass.tuning.hyperband_tuner import create_hyperband_scheduler

        scheduler = create_hyperband_scheduler(
            max_t=5000,
            reduction_factor=3,
        )
        assert scheduler is not None

    def test_hyperband_bracket_calculation(self):
        """Test Hyperband bracket calculation."""
        from medai_compass.tuning.utils import calculate_hyperband_brackets

        brackets = calculate_hyperband_brackets(
            max_t=5000,
            reduction_factor=3,
        )

        assert "brackets" in brackets
        assert "total_configs" in brackets
        assert len(brackets["brackets"]) > 0

    def test_hyperband_tuner_from_hydra_config(self):
        """Test creating Hyperband tuner from Hydra config."""
        from medai_compass.tuning.hyperband_tuner import HyperbandTuner
        from medai_compass.config.hydra_config import MedGemmaConfig
        from omegaconf import OmegaConf

        config = MedGemmaConfig()
        cfg = OmegaConf.structured(config)
        cfg.tuning.scheduler = "hyperband"

        tuner = HyperbandTuner(cfg)
        assert tuner is not None
        assert tuner.scheduler is not None


class TestSearchSpaceConversion:
    """Test search space conversion utilities."""

    def test_loguniform_conversion(self):
        """Test loguniform to Ray Tune conversion."""
        from medai_compass.tuning.utils import spec_to_ray_tune
        from ray import tune

        spec = {"type": "loguniform", "lower": 1e-5, "upper": 1e-3}
        result = spec_to_ray_tune(spec)

        # Result should be a Ray Tune sample function
        assert result is not None

    def test_uniform_conversion(self):
        """Test uniform to Ray Tune conversion."""
        from medai_compass.tuning.utils import spec_to_ray_tune

        spec = {"type": "uniform", "lower": 0.0, "upper": 1.0}
        result = spec_to_ray_tune(spec)

        assert result is not None

    def test_choice_conversion(self):
        """Test choice to Ray Tune conversion."""
        from medai_compass.tuning.utils import spec_to_ray_tune

        spec = {"type": "choice", "categories": [8, 16, 32, 64]}
        result = spec_to_ray_tune(spec)

        assert result is not None

    def test_full_search_space_conversion(self):
        """Test converting full search space config."""
        from medai_compass.tuning.utils import config_to_search_space
        from medai_compass.config.hydra_config import SearchSpaceConfig

        config = SearchSpaceConfig()
        search_space = config_to_search_space(config)

        assert "learning_rate" in search_space
        assert "lora_r" in search_space


class TestTrainableClass:
    """Test unified Trainable class for Ray Tune."""

    @pytest.mark.skip(reason="Requires GPU or model")
    def test_trainable_setup(self):
        """Test Trainable setup method."""
        from medai_compass.tuning.trainable import MedGemmaTrainable

        config = {
            "model_name": "google/medgemma-4b-it",
            "learning_rate": 2e-4,
            "lora_r": 64,
            "output_dir": "/tmp/test",
        }

        trainable = MedGemmaTrainable(config)
        assert trainable is not None

    def test_trainable_config_validation(self):
        """Test Trainable config validation."""
        from medai_compass.tuning.trainable import validate_trainable_config

        valid_config = {
            "model_name": "test-model",
            "learning_rate": 1e-4,
            "output_dir": "/tmp",
        }
        assert validate_trainable_config(valid_config) is True

        invalid_config = {"learning_rate": -1}
        with pytest.raises(ValueError):
            validate_trainable_config(invalid_config)


class TestTuningRunner:
    """Test unified tuning runner."""

    def test_scheduler_selection(self):
        """Test scheduler selection logic."""
        from medai_compass.tuning.utils import suggest_scheduler

        # Large space, limited time -> ASHA
        scheduler = suggest_scheduler(
            search_space_size=100,
            max_training_hours=6,
            gpu_budget=8,
        )
        assert scheduler == "asha"

        # Long training, good resources -> PBT
        scheduler = suggest_scheduler(
            search_space_size=20,
            max_training_hours=48,
            gpu_budget=16,
            is_long_running=True,
        )
        assert scheduler == "pbt"

    def test_cost_estimation(self):
        """Test tuning cost estimation."""
        from medai_compass.tuning.utils import estimate_tuning_cost

        cost = estimate_tuning_cost(
            scheduler="asha",
            num_samples=50,
            cost_per_gpu_hour=3.0,
            avg_trial_hours=2.0,
            num_gpus_per_trial=1,
            max_concurrent=8,
        )

        assert "total_gpu_hours" in cost
        assert "estimated_cost_usd" in cost
        assert cost["estimated_cost_usd"] > 0


class TestTuningIntegration:
    """Integration tests for tuning system."""

    def test_tuning_with_hydra_config(self):
        """Test tuning can be initialized from Hydra config."""
        from medai_compass.tuning.runner import TuningRunner
        from medai_compass.config.hydra_config import MedGemmaConfig
        from omegaconf import OmegaConf

        config = MedGemmaConfig()
        cfg = OmegaConf.structured(config)

        runner = TuningRunner(cfg)
        assert runner is not None
        assert runner.scheduler_type in ["asha", "pbt", "hyperband"]

    def test_tuning_config_export(self):
        """Test exporting tuning config for Ray."""
        from medai_compass.tuning.runner import TuningRunner
        from medai_compass.config.hydra_config import MedGemmaConfig
        from omegaconf import OmegaConf

        config = MedGemmaConfig()
        cfg = OmegaConf.structured(config)

        runner = TuningRunner(cfg)
        ray_config = runner.get_ray_config()

        assert "param_space" in ray_config
        assert "tune_config" in ray_config
        assert "run_config" in ray_config
