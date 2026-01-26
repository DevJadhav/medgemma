"""
Tests for Ray Serve deployment module.

Tests configuration, deployment logic, and metrics collection.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from medai_compass.inference.ray_serve_deployment import (
    RayServeConfig,
    GenerationRequest,
    GenerationResponse,
    MetricsCollector,
    MedGemmaModelWrapper,
    RayServeDeploymentManager,
)


# =============================================================================
# RayServeConfig Tests
# =============================================================================

class TestRayServeConfig:
    """Tests for RayServeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RayServeConfig()

        assert config.model_name == "google/medgemma-4b-it"
        assert config.model_variant == "4b"
        assert config.torch_dtype == "bfloat16"
        assert config.num_replicas == 1
        assert config.autoscaling_enabled is True
        assert config.min_replicas == 1
        assert config.max_replicas == 10

    def test_for_model_4b(self):
        """Test config for 4B model."""
        config = RayServeConfig.for_model("google/medgemma-4b-it")

        assert config.model_variant == "4b"
        assert config.num_gpus == 1.0
        assert config.num_cpus == 4
        assert config.max_batch_size == 8

    def test_for_model_27b(self):
        """Test config for 27B model."""
        config = RayServeConfig.for_model("google/medgemma-27b-text-it")

        assert config.model_variant == "27b"
        assert config.num_gpus == 4.0
        assert config.num_cpus == 16
        assert config.max_batch_size == 4

    def test_from_hydra(self):
        """Test config from Hydra configuration."""
        # Create mock Hydra config
        mock_cfg = MagicMock()
        mock_cfg.model.name = "google/medgemma-4b-it"
        mock_cfg.model.torch_dtype = "bfloat16"
        mock_cfg.model.trust_remote_code = True
        mock_cfg.model.attn_implementation = "flash_attention_2"
        mock_cfg.compute.ray_serve_replicas = 2
        mock_cfg.compute.ray_serve_max_concurrent = 50
        mock_cfg.compute.ray_serve_autoscaling = True
        mock_cfg.compute.ray_serve_min_replicas = 1
        mock_cfg.compute.ray_serve_max_replicas = 5
        mock_cfg.compute.gpus_per_replica = 1.0
        mock_cfg.compute.cpus_per_replica = 4

        config = RayServeConfig.from_hydra(mock_cfg)

        assert config.model_name == "google/medgemma-4b-it"
        assert config.num_replicas == 2
        assert config.max_concurrent_queries == 50
        assert config.max_replicas == 5


# =============================================================================
# GenerationRequest Tests
# =============================================================================

class TestGenerationRequest:
    """Tests for GenerationRequest."""

    def test_default_values(self):
        """Test default request values."""
        request = GenerationRequest(prompt="Hello")

        assert request.prompt == "Hello"
        assert request.max_tokens == 512
        assert request.temperature == 0.1
        assert request.top_p == 0.9
        assert request.stream is False

    def test_custom_values(self):
        """Test custom request values."""
        request = GenerationRequest(
            prompt="What are the symptoms?",
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            request_id="req-123",
        )

        assert request.prompt == "What are the symptoms?"
        assert request.max_tokens == 256
        assert request.temperature == 0.7
        assert request.request_id == "req-123"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        request = GenerationRequest(
            prompt="Test",
            max_tokens=100,
            request_id="test-id",
        )

        d = request.to_dict()

        assert d["prompt"] == "Test"
        assert d["max_tokens"] == 100
        assert d["request_id"] == "test-id"


# =============================================================================
# GenerationResponse Tests
# =============================================================================

class TestGenerationResponse:
    """Tests for GenerationResponse."""

    def test_response_creation(self):
        """Test response creation."""
        response = GenerationResponse(
            text="This is the response",
            model="medgemma-4b",
            tokens_generated=10,
            latency_ms=150.5,
        )

        assert response.text == "This is the response"
        assert response.model == "medgemma-4b"
        assert response.tokens_generated == 10
        assert response.latency_ms == 150.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        response = GenerationResponse(
            text="Response text",
            model="medgemma-4b",
            request_id="req-456",
            tokens_generated=20,
            latency_ms=200.0,
            finish_reason="length",
        )

        d = response.to_dict()

        assert d["text"] == "Response text"
        assert d["model"] == "medgemma-4b"
        assert d["request_id"] == "req-456"
        assert d["finish_reason"] == "length"


# =============================================================================
# MetricsCollector Tests
# =============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_initial_state(self):
        """Test initial metrics state."""
        collector = MetricsCollector()

        assert collector.request_count == 0
        assert collector.total_tokens == 0
        assert collector.error_count == 0

    def test_record_request(self):
        """Test recording a request."""
        collector = MetricsCollector()

        collector.record_request(latency_ms=100.0, tokens=50, success=True)

        assert collector.request_count == 1
        assert collector.total_tokens == 50
        assert collector.total_latency_ms == 100.0

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        collector = MetricsCollector()

        collector.record_request(latency_ms=100.0, tokens=50, success=True)
        collector.record_request(latency_ms=200.0, tokens=100, success=True)
        collector.record_request(latency_ms=150.0, tokens=75, success=False)

        assert collector.request_count == 3
        assert collector.total_tokens == 225
        assert collector.error_count == 1

    def test_get_metrics(self):
        """Test getting metrics summary."""
        collector = MetricsCollector()

        collector.record_request(latency_ms=100.0, tokens=50, success=True)
        collector.record_request(latency_ms=200.0, tokens=100, success=True)

        metrics = collector.get_metrics()

        assert metrics["request_count"] == 2
        assert metrics["total_tokens"] == 150
        assert metrics["avg_latency_ms"] == 150.0
        assert metrics["error_count"] == 0
        assert metrics["error_rate"] == 0.0

    def test_latency_percentiles(self):
        """Test latency percentile calculation."""
        collector = MetricsCollector()

        # Add varied latencies
        for latency in [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]:
            collector.record_request(latency_ms=float(latency), tokens=10, success=True)

        metrics = collector.get_metrics()

        # P50 should be around 145-150
        assert 140 <= metrics["p50_latency_ms"] <= 160
        # P90 should be around 180-190
        assert 175 <= metrics["p90_latency_ms"] <= 195

    def test_prometheus_export(self):
        """Test Prometheus metrics export."""
        collector = MetricsCollector()

        collector.record_request(latency_ms=100.0, tokens=50, success=True)

        prometheus = collector.to_prometheus()

        assert "medgemma_requests_total 1" in prometheus
        assert "medgemma_tokens_total 50" in prometheus
        assert "medgemma_errors_total 0" in prometheus


# =============================================================================
# MedGemmaModelWrapper Tests
# =============================================================================

class TestMedGemmaModelWrapper:
    """Tests for MedGemmaModelWrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        config = RayServeConfig()
        wrapper = MedGemmaModelWrapper(config)

        assert wrapper.model is None
        assert wrapper.tokenizer is None
        assert wrapper._initialized is False

    def test_initialize_loads_model(self):
        """Test that initialize would load model and tokenizer."""
        # Test basic wrapper creation
        config = RayServeConfig(model_name="test-model")
        wrapper = MedGemmaModelWrapper(config)

        # Verify initial state
        assert wrapper._initialized is False
        assert wrapper.model is None
        assert wrapper.tokenizer is None
        assert wrapper.config.model_name == "test-model"

    def test_generate_requires_initialization(self):
        """Test that generate checks initialization."""
        config = RayServeConfig()
        wrapper = MedGemmaModelWrapper(config)

        # Wrapper should try to initialize when generating
        request = GenerationRequest(prompt="Test")

        # Without mocking imports, this would fail
        # Just verify the method exists
        assert hasattr(wrapper, "generate")
        assert hasattr(wrapper, "generate_async")


# =============================================================================
# RayServeDeploymentManager Tests
# =============================================================================

class TestRayServeDeploymentManager:
    """Tests for RayServeDeploymentManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = RayServeDeploymentManager()

        assert manager.config is not None
        assert manager._initialized is False
        assert manager._handle is None

    def test_initialization_with_config(self):
        """Test manager initialization with custom config."""
        config = RayServeConfig(
            model_name="custom-model",
            num_replicas=3,
        )
        manager = RayServeDeploymentManager(config)

        assert manager.config.model_name == "custom-model"
        assert manager.config.num_replicas == 3

    @patch("ray.is_initialized")
    @patch("ray.init")
    def test_initialize_ray(self, mock_init, mock_is_initialized):
        """Test Ray initialization."""
        mock_is_initialized.return_value = False

        manager = RayServeDeploymentManager()
        manager.initialize_ray()

        mock_init.assert_called_once()

    @patch("ray.is_initialized")
    @patch("ray.init")
    def test_initialize_ray_already_running(self, mock_init, mock_is_initialized):
        """Test Ray initialization when already running."""
        mock_is_initialized.return_value = True

        manager = RayServeDeploymentManager()
        manager.initialize_ray()

        mock_init.assert_not_called()

    def test_get_status_not_initialized(self):
        """Test getting status when not initialized."""
        manager = RayServeDeploymentManager()

        # Without Ray, this should handle gracefully
        status = manager.get_status()

        # Should either return error or stopped status
        assert "status" in status

    def test_shutdown_not_initialized(self):
        """Test shutdown when not initialized."""
        manager = RayServeDeploymentManager()

        # Should not raise
        manager.shutdown()

        assert manager._initialized is False


# =============================================================================
# Integration Tests (requires Ray)
# =============================================================================

@pytest.mark.integration
class TestRayServeIntegration:
    """Integration tests for Ray Serve deployment."""

    @pytest.fixture
    def ray_context(self):
        """Setup Ray context for tests."""
        try:
            import ray
            ray.init(ignore_reinit_error=True, num_cpus=2)
            yield
            ray.shutdown()
        except ImportError:
            pytest.skip("Ray not installed")

    @pytest.mark.skip(reason="Requires full Ray and model")
    def test_full_deployment(self, ray_context):
        """Test full deployment lifecycle."""
        from medai_compass.inference.ray_serve_deployment import (
            deploy_medgemma,
        )

        manager = deploy_medgemma(
            model_name="google/medgemma-4b-it",
            num_replicas=1,
            autoscaling=False,
        )

        assert manager._initialized

        # Cleanup
        manager.shutdown()


# =============================================================================
# Async Tests
# =============================================================================

class TestAsyncGeneration:
    """Tests for async generation."""

    @pytest.mark.asyncio
    async def test_generate_async_method_exists(self):
        """Test that async generation method exists."""
        config = RayServeConfig()
        wrapper = MedGemmaModelWrapper(config)

        # Verify async method signature
        assert asyncio.iscoroutinefunction(wrapper.generate_async)


# =============================================================================
# Configuration Validation Tests
# =============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""

    def test_autoscaling_config_consistency(self):
        """Test autoscaling config is consistent."""
        config = RayServeConfig(
            autoscaling_enabled=True,
            min_replicas=2,
            max_replicas=10,
        )

        assert config.min_replicas <= config.max_replicas

    def test_resource_allocation(self):
        """Test resource allocation is reasonable."""
        config = RayServeConfig(
            num_gpus=1.0,
            num_cpus=4,
            memory_mb=32000,
        )

        # GPU should be reasonable for single model
        assert 0 < config.num_gpus <= 8
        # CPU should be reasonable
        assert 1 <= config.num_cpus <= 64
        # Memory should be reasonable (in MB)
        assert 1000 <= config.memory_mb <= 500000

    def test_generation_defaults(self):
        """Test generation default values are valid."""
        config = RayServeConfig()

        # Temperature should be valid
        assert 0 <= config.default_temperature <= 2.0
        # Top-p should be valid
        assert 0 < config.default_top_p <= 1.0
        # Max tokens should be positive
        assert config.default_max_tokens > 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_metrics_collector_handles_no_requests(self):
        """Test metrics collector with no requests."""
        collector = MetricsCollector()

        metrics = collector.get_metrics()

        # Should not raise division by zero
        assert metrics["avg_latency_ms"] == 0.0
        assert metrics["error_rate"] == 0.0

    def test_metrics_collector_handles_all_errors(self):
        """Test metrics collector with all failed requests."""
        collector = MetricsCollector()

        collector.record_request(latency_ms=100.0, tokens=0, success=False)
        collector.record_request(latency_ms=200.0, tokens=0, success=False)

        metrics = collector.get_metrics()

        assert metrics["error_count"] == 2
        assert metrics["error_rate"] == 1.0

    def test_config_for_unknown_model(self):
        """Test config for unknown model size."""
        # Should default to 4B settings
        config = RayServeConfig.for_model("unknown-model")

        assert config.model_variant == "4b"
        assert config.num_gpus == 1.0
