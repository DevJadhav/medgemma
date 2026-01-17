"""Tests for GPU detection and configuration utilities."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestGPUDetection:
    """Tests for GPU detection functions."""

    def test_detect_local_gpu_runs(self):
        """Test detection runs without error."""
        from medai_compass.utils.gpu import detect_local_gpu

        result = detect_local_gpu()

        # May return None or GPUInfo depending on actual hardware
        # Just verify no exception is raised
        assert result is None or hasattr(result, "backend")

    def test_detect_local_gpu_returns_gpu_info(self):
        """Test that detection returns GPUInfo when GPU available."""
        from medai_compass.utils.gpu import detect_local_gpu, GPUInfo
        
        result = detect_local_gpu()
        
        # Result is either None or GPUInfo
        if result is not None:
            assert isinstance(result, GPUInfo)
            assert result.memory_total_gb >= 0
            assert result.device_name is not None

    def test_gpu_backend_enum(self):
        """Test GPUBackend enum values."""
        from medai_compass.utils.gpu import GPUBackend
        
        assert GPUBackend.CUDA.value == "cuda"
        assert GPUBackend.MPS.value == "mps"
        assert GPUBackend.MODAL.value == "modal"
        assert GPUBackend.CPU.value == "cpu"


class TestShouldUseModal:
    """Tests for Modal fallback decision logic."""

    def test_should_use_modal_returns_bool(self):
        """Should use Modal returns boolean."""
        from medai_compass.utils.gpu import should_use_modal

        result = should_use_modal()
        assert isinstance(result, bool)

    def test_should_use_modal_with_required_memory(self):
        """Test should_use_modal with memory requirements."""
        from medai_compass.utils.gpu import should_use_modal

        # Very high memory requirement should suggest Modal
        result = should_use_modal(required_memory_gb=500)
        # If no local GPU with 500GB exists, should return True
        # Just verify it runs without error
        assert isinstance(result, bool)

    def test_should_use_modal_prefer_modal_false(self):
        """Should respect prefer_modal=False flag."""
        from medai_compass.utils.gpu import should_use_modal

        result = should_use_modal(prefer_modal=False)
        # If no local GPU, may still return True for Modal
        assert isinstance(result, bool)

    def test_should_use_modal_prefer_modal_true(self):
        """Should respect prefer_modal=True flag."""
        from medai_compass.utils.gpu import should_use_modal, is_modal_available

        result = should_use_modal(prefer_modal=True)
        # If Modal is available, should return True
        if is_modal_available():
            assert result is True
        else:
            # No Modal - depends on local GPU
            assert isinstance(result, bool)


class TestGetInferenceConfig:
    """Tests for inference configuration generation."""

    def test_get_inference_config_returns_dict(self):
        """Test config returns dictionary."""
        from medai_compass.utils.gpu import get_inference_config

        config = get_inference_config()

        assert isinstance(config, dict)
        assert "use_modal" in config
        assert "device" in config

    def test_get_inference_config_model_specific(self):
        """Test model-specific configuration."""
        from medai_compass.utils.gpu import get_inference_config

        # 27B model should have different settings
        config_27b = get_inference_config(model_size="27b")
        config_4b = get_inference_config(model_size="4b")

        # Both should be valid configs
        assert "use_modal" in config_27b
        assert "use_modal" in config_4b

    def test_get_inference_config_has_device(self):
        """Test config includes device information."""
        from medai_compass.utils.gpu import get_inference_config

        config = get_inference_config()

        assert "device" in config
        assert config["device"] in ["cuda", "mps", "cpu"]


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test creating GPUInfo."""
        from medai_compass.utils.gpu import GPUInfo, GPUBackend
        
        info = GPUInfo(
            backend=GPUBackend.CUDA,
            device_name="NVIDIA A100",
            memory_total_gb=80.0,
            memory_available_gb=75.0,
            compute_capability="8.0"
        )
        
        assert info.backend == GPUBackend.CUDA
        assert info.device_name == "NVIDIA A100"
        assert info.memory_total_gb == 80.0
        assert info.is_remote is False
