"""
TDD Tests for Hydra Configuration System.

These tests are written FIRST (TDD approach) to define the expected behavior
of the Hydra configuration infrastructure.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


class TestHydraConfigSchemas:
    """Test Hydra configuration dataclass schemas."""

    def test_model_config_defaults(self):
        """Test ModelConfig has correct defaults."""
        from medai_compass.config.hydra_config import ModelConfig

        config = ModelConfig()
        assert config.name == "google/medgemma-4b-it"
        assert config.torch_dtype == "bfloat16"
        assert config.trust_remote_code is True
        assert config.attn_implementation == "flash_attention_2"

    def test_model_config_custom_values(self):
        """Test ModelConfig accepts custom values."""
        from medai_compass.config.hydra_config import ModelConfig

        config = ModelConfig(
            name="google/medgemma-27b-text-it",
            torch_dtype="float16",
            trust_remote_code=False,
        )
        assert config.name == "google/medgemma-27b-text-it"
        assert config.torch_dtype == "float16"
        assert config.trust_remote_code is False

    def test_lora_config_defaults(self):
        """Test LoRAConfig has correct defaults."""
        from medai_compass.config.hydra_config import LoRAConfig

        config = LoRAConfig()
        assert config.r == 64
        assert config.lora_alpha == 128
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_training_args_config_defaults(self):
        """Test TrainingArgsConfig has correct defaults."""
        from medai_compass.config.hydra_config import TrainingArgsConfig

        config = TrainingArgsConfig()
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 2e-4
        assert config.bf16 is True
        assert config.gradient_checkpointing is True

    def test_compute_config_defaults(self):
        """Test ComputeConfig has correct defaults."""
        from medai_compass.config.hydra_config import ComputeConfig

        config = ComputeConfig()
        assert config.backend == "modal"
        assert config.gpu == "H100"
        assert config.gpu_count == 8

    def test_tuning_config_defaults(self):
        """Test TuningConfig has correct defaults."""
        from medai_compass.config.hydra_config import TuningConfig

        config = TuningConfig()
        assert config.scheduler == "asha"
        assert config.num_samples == 20
        assert config.metric == "eval_loss"
        assert config.mode == "min"

    def test_medgemma_config_composition(self):
        """Test MedGemmaConfig composes all sub-configs."""
        from medai_compass.config.hydra_config import MedGemmaConfig

        config = MedGemmaConfig()
        assert hasattr(config, "project")
        assert hasattr(config, "model")
        assert hasattr(config, "training")
        assert hasattr(config, "compute")
        assert hasattr(config, "tuning")
        assert config.project.name == "medgemma-training"
        assert config.model.name == "google/medgemma-4b-it"


class TestHydraConfigLoader:
    """Test Hydra configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        from medai_compass.config.hydra_config import load_config

        cfg = load_config()
        assert cfg is not None
        assert "model" in cfg or hasattr(cfg, "model")

    def test_load_config_with_overrides(self):
        """Test loading config with command-line overrides."""
        from medai_compass.config.hydra_config import load_config_with_overrides

        # Use ++ prefix to override existing keys in struct config
        cfg = load_config_with_overrides(["++model.name=test-model"])
        assert cfg.model.name == "test-model"

    def test_config_to_training_args(self):
        """Test converting config to HuggingFace TrainingArguments dict."""
        from medai_compass.config.hydra_config import (
            MedGemmaConfig,
            to_training_args,
        )
        from omegaconf import OmegaConf

        config = MedGemmaConfig()
        cfg = OmegaConf.structured(config)
        args = to_training_args(cfg)

        assert "learning_rate" in args
        assert "output_dir" in args
        assert "seed" in args
        assert args["learning_rate"] == 2e-4

    def test_config_to_lora_config(self):
        """Test converting config to PEFT LoraConfig dict."""
        from medai_compass.config.hydra_config import (
            MedGemmaConfig,
            to_lora_config,
        )
        from omegaconf import OmegaConf

        config = MedGemmaConfig()
        cfg = OmegaConf.structured(config)
        lora_dict = to_lora_config(cfg)

        assert "r" in lora_dict
        assert "lora_alpha" in lora_dict
        assert "target_modules" in lora_dict
        assert lora_dict["r"] == 64


class TestHydraSearchSpace:
    """Test search space configuration for hyperparameter tuning."""

    def test_search_space_to_ray_tune(self):
        """Test converting search space config to Ray Tune format."""
        from medai_compass.config.hydra_config import SearchSpaceConfig
        from medai_compass.tuning.utils import config_to_search_space

        config = SearchSpaceConfig()
        search_space = config_to_search_space(config)

        assert "learning_rate" in search_space
        assert "lora_r" in search_space

    def test_search_space_loguniform(self):
        """Test loguniform search space conversion."""
        from medai_compass.tuning.utils import spec_to_ray_tune

        spec = {"type": "loguniform", "lower": 1e-5, "upper": 1e-3}
        ray_spec = spec_to_ray_tune(spec)

        # Should return a Ray Tune sample function
        assert callable(getattr(ray_spec, "sample", None)) or hasattr(ray_spec, "lower")

    def test_search_space_choice(self):
        """Test choice search space conversion."""
        from medai_compass.tuning.utils import spec_to_ray_tune

        spec = {"type": "choice", "categories": [8, 16, 32, 64]}
        ray_spec = spec_to_ray_tune(spec)

        assert ray_spec is not None


class TestHydraYAMLConfigs:
    """Test YAML configuration files."""

    def test_config_directory_exists(self):
        """Test that config directory structure exists."""
        config_dir = Path(__file__).parent.parent / "config" / "hydra"
        assert config_dir.exists(), f"Config directory {config_dir} should exist"

    def test_main_config_exists(self):
        """Test that main config.yaml exists."""
        config_file = Path(__file__).parent.parent / "config" / "hydra" / "config.yaml"
        assert config_file.exists(), f"Main config {config_file} should exist"

    def test_model_configs_exist(self):
        """Test that model config files exist."""
        model_dir = Path(__file__).parent.parent / "config" / "hydra" / "model"
        assert model_dir.exists(), f"Model config dir {model_dir} should exist"

        # Check for specific model configs
        assert (model_dir / "medgemma_4b.yaml").exists()
        assert (model_dir / "medgemma_27b.yaml").exists()

    def test_training_configs_exist(self):
        """Test that training config files exist."""
        training_dir = Path(__file__).parent.parent / "config" / "hydra" / "training"
        assert training_dir.exists()
        assert (training_dir / "lora.yaml").exists()

    def test_compute_configs_exist(self):
        """Test that compute config files exist."""
        compute_dir = Path(__file__).parent.parent / "config" / "hydra" / "compute"
        assert compute_dir.exists()
        assert (compute_dir / "modal_h100.yaml").exists()

    def test_tuning_configs_exist(self):
        """Test that tuning config files exist."""
        tuning_dir = Path(__file__).parent.parent / "config" / "hydra" / "tuning"
        assert tuning_dir.exists()
        assert (tuning_dir / "asha.yaml").exists()
        assert (tuning_dir / "pbt.yaml").exists()
        assert (tuning_dir / "hyperband.yaml").exists()


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_model_config(self):
        """Test model config validation."""
        from medai_compass.config.hydra_config import ModelConfig, validate_model_config

        # Valid config
        config = ModelConfig()
        assert validate_model_config(config) is True

        # Invalid config (empty name)
        config_invalid = ModelConfig(name="")
        with pytest.raises(ValueError):
            validate_model_config(config_invalid)

    def test_validate_training_config(self):
        """Test training config validation."""
        from medai_compass.config.hydra_config import (
            TrainingArgsConfig,
            validate_training_config,
        )

        # Valid config
        config = TrainingArgsConfig()
        assert validate_training_config(config) is True

        # Invalid config (negative learning rate)
        config_invalid = TrainingArgsConfig(learning_rate=-1.0)
        with pytest.raises(ValueError):
            validate_training_config(config_invalid)

    def test_validate_lora_config(self):
        """Test LoRA config validation."""
        from medai_compass.config.hydra_config import LoRAConfig, validate_lora_config

        # Valid config
        config = LoRAConfig()
        assert validate_lora_config(config) is True

        # Invalid config (r=0)
        config_invalid = LoRAConfig(r=0)
        with pytest.raises(ValueError):
            validate_lora_config(config_invalid)


class TestConfigInterpolation:
    """Test OmegaConf interpolation and resolution."""

    def test_env_var_interpolation(self):
        """Test environment variable interpolation."""
        from omegaconf import OmegaConf

        os.environ["TEST_MODEL_NAME"] = "test-model"
        cfg = OmegaConf.create({"model_name": "${oc.env:TEST_MODEL_NAME}"})
        OmegaConf.resolve(cfg)
        assert cfg.model_name == "test-model"
        del os.environ["TEST_MODEL_NAME"]

    def test_config_reference_interpolation(self):
        """Test config reference interpolation."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "project": {"name": "test-project"},
            "output_dir": "/outputs/${project.name}",
        })
        OmegaConf.resolve(cfg)
        assert cfg.output_dir == "/outputs/test-project"


class TestConfigMerging:
    """Test configuration merging and composition."""

    def test_merge_configs(self):
        """Test merging multiple configs."""
        from omegaconf import OmegaConf

        base = OmegaConf.create({"a": 1, "b": 2})
        override = OmegaConf.create({"b": 3, "c": 4})
        merged = OmegaConf.merge(base, override)

        assert merged.a == 1
        assert merged.b == 3
        assert merged.c == 4

    def test_structured_config_merge(self):
        """Test merging with structured configs."""
        from medai_compass.config.hydra_config import ModelConfig
        from omegaconf import OmegaConf

        base = OmegaConf.structured(ModelConfig())
        override = OmegaConf.create({"name": "custom-model"})
        merged = OmegaConf.merge(base, override)

        assert merged.name == "custom-model"
        assert merged.torch_dtype == "bfloat16"  # From base
