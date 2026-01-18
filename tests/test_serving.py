"""
Tests for serving module (Phase 7: Deployment Pipeline).

TDD tests for Ray Serve deployment, health checks, and model serving.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
import asyncio


# ============================================================================
# Test: Ray Serve Deployment Configuration
# ============================================================================

class TestRayServeConfig:
    """Test Ray Serve deployment configuration."""
    
    def test_deployment_config_defaults(self):
        """Test default deployment configuration values."""
        from medai_compass.serving.ray_serve_app import DeploymentConfig
        
        config = DeploymentConfig()
        
        assert config.model_name == "medgemma-27b-it"
        assert config.num_replicas == 1
        assert config.max_concurrent_queries == 100
        assert config.ray_actor_options is not None
        assert config.autoscaling_config is not None
    
    def test_deployment_config_4b_model(self):
        """Test configuration for 4B model."""
        from medai_compass.serving.ray_serve_app import DeploymentConfig
        
        config = DeploymentConfig(model_name="medgemma-4b-it")
        
        assert config.model_name == "medgemma-4b-it"
        assert config.ray_actor_options["num_gpus"] == 1
    
    def test_deployment_config_27b_model(self):
        """Test configuration for 27B model (default)."""
        from medai_compass.serving.ray_serve_app import DeploymentConfig
        
        config = DeploymentConfig(model_name="medgemma-27b-it")
        
        assert config.model_name == "medgemma-27b-it"
        # 27B requires more GPU resources
        assert config.ray_actor_options["num_gpus"] >= 1
    
    def test_deployment_config_autoscaling(self):
        """Test autoscaling configuration."""
        from medai_compass.serving.ray_serve_app import DeploymentConfig
        
        config = DeploymentConfig(
            min_replicas=1,
            max_replicas=4,
            target_num_ongoing_requests=10
        )
        
        assert config.autoscaling_config["min_replicas"] == 1
        assert config.autoscaling_config["max_replicas"] == 4
        assert config.autoscaling_config["target_num_ongoing_requests_per_replica"] == 10
    
    def test_deployment_config_custom_options(self):
        """Test custom Ray actor options."""
        from medai_compass.serving.ray_serve_app import DeploymentConfig
        
        config = DeploymentConfig(
            num_gpus=2,
            num_cpus=8,
            memory=32 * 1024 ** 3  # 32GB
        )
        
        assert config.ray_actor_options["num_gpus"] == 2
        assert config.ray_actor_options["num_cpus"] == 8
        assert config.ray_actor_options["memory"] == 32 * 1024 ** 3


# ============================================================================
# Test: MedGemma Deployment
# ============================================================================

class TestMedGemmaDeployment:
    """Test MedGemma Ray Serve deployment."""
    
    @patch("medai_compass.serving.ray_serve_app.load_model")
    def test_deployment_initialization(self, mock_load_model):
        """Test deployment initializes with model."""
        from medai_compass.serving.ray_serve_app import MedGemmaDeployment
        
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        deployment = MedGemmaDeployment(model_name="medgemma-27b-it")
        
        assert deployment.model is not None
        assert deployment.model_name == "medgemma-27b-it"
        mock_load_model.assert_called_once()
    
    @patch("medai_compass.serving.ray_serve_app.load_model")
    def test_deployment_model_selection_27b(self, mock_load_model):
        """Test 27B model selection (default)."""
        from medai_compass.serving.ray_serve_app import MedGemmaDeployment
        
        deployment = MedGemmaDeployment(model_name="medgemma-27b-it")
        
        call_args = mock_load_model.call_args
        assert "27b" in call_args[1].get("model_name", "") or "27b" in str(call_args)
    
    @patch("medai_compass.serving.ray_serve_app.load_model")
    def test_deployment_model_selection_4b(self, mock_load_model):
        """Test 4B model selection."""
        from medai_compass.serving.ray_serve_app import MedGemmaDeployment
        
        deployment = MedGemmaDeployment(model_name="medgemma-4b-it")
        
        call_args = mock_load_model.call_args
        assert "4b" in call_args[1].get("model_name", "") or "4b" in str(call_args)
    
    @pytest.mark.asyncio
    @patch("medai_compass.serving.ray_serve_app.load_model")
    async def test_deployment_generate(self, mock_load_model):
        """Test text generation endpoint."""
        from medai_compass.serving.ray_serve_app import MedGemmaDeployment
        
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Generated response")
        mock_load_model.return_value = mock_model
        
        deployment = MedGemmaDeployment(model_name="medgemma-27b-it")
        result = await deployment.generate("Test prompt")
        
        assert result is not None
        assert "response" in result or isinstance(result, str)
    
    @pytest.mark.asyncio
    @patch("medai_compass.serving.ray_serve_app.load_model")
    async def test_deployment_generate_with_params(self, mock_load_model):
        """Test generation with custom parameters."""
        from medai_compass.serving.ray_serve_app import MedGemmaDeployment
        
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Generated response")
        mock_load_model.return_value = mock_model
        
        deployment = MedGemmaDeployment(model_name="medgemma-27b-it")
        result = await deployment.generate(
            "Test prompt",
            max_tokens=512,
            temperature=0.7
        )
        
        assert result is not None


# ============================================================================
# Test: Health Checks
# ============================================================================

class TestHealthChecks:
    """Test deployment health check functionality."""
    
    def test_health_check_result_dataclass(self):
        """Test HealthCheckResult dataclass."""
        from medai_compass.serving.health import HealthCheckResult, HealthStatus
        
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            model_loaded=True,
            gpu_available=True,
            memory_usage_percent=45.0,
            latency_ms=50.0
        )
        
        assert result.status == HealthStatus.HEALTHY
        assert result.model_loaded is True
        assert result.gpu_available is True
        assert result.memory_usage_percent == 45.0
        assert result.latency_ms == 50.0
    
    def test_health_status_enum(self):
        """Test HealthStatus enum values."""
        from medai_compass.serving.health import HealthStatus
        
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
    
    @patch("medai_compass.serving.health.check_gpu_available")
    @patch("medai_compass.serving.health.check_model_loaded")
    def test_health_checker_healthy(self, mock_model, mock_gpu):
        """Test health checker returns healthy status."""
        from medai_compass.serving.health import HealthChecker, HealthStatus
        
        mock_model.return_value = True
        mock_gpu.return_value = True
        
        checker = HealthChecker()
        result = checker.check()
        
        assert result.status == HealthStatus.HEALTHY
    
    @patch("medai_compass.serving.health.check_gpu_available")
    @patch("medai_compass.serving.health.check_model_loaded")
    def test_health_checker_degraded_no_gpu(self, mock_model, mock_gpu):
        """Test health checker returns degraded when GPU unavailable."""
        from medai_compass.serving.health import HealthChecker, HealthStatus
        
        mock_model.return_value = True
        mock_gpu.return_value = False
        
        checker = HealthChecker()
        result = checker.check()
        
        assert result.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
    
    @patch("medai_compass.serving.health.check_gpu_available")
    @patch("medai_compass.serving.health.check_model_loaded")
    def test_health_checker_unhealthy_no_model(self, mock_model, mock_gpu):
        """Test health checker returns unhealthy when model not loaded."""
        from medai_compass.serving.health import HealthChecker, HealthStatus
        
        mock_model.return_value = False
        mock_gpu.return_value = True
        
        checker = HealthChecker()
        result = checker.check()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    def test_health_checker_to_dict(self):
        """Test health check result serialization."""
        from medai_compass.serving.health import HealthCheckResult, HealthStatus
        
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            model_loaded=True,
            gpu_available=True,
            memory_usage_percent=45.0,
            latency_ms=50.0
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["status"] == "healthy"
        assert result_dict["model_loaded"] is True
        assert result_dict["gpu_available"] is True


# ============================================================================
# Test: Serving Application
# ============================================================================

class TestServingApplication:
    """Test Ray Serve application setup."""
    
    @patch("ray.serve.deployment")
    def test_create_deployment(self, mock_deployment):
        """Test creating Ray Serve deployment."""
        from medai_compass.serving.ray_serve_app import create_deployment
        
        deployment = create_deployment(model_name="medgemma-27b-it")
        
        assert deployment is not None
    
    @patch("ray.serve.deployment")
    def test_create_deployment_with_config(self, mock_deployment):
        """Test creating deployment with custom config."""
        from medai_compass.serving.ray_serve_app import (
            create_deployment,
            DeploymentConfig
        )
        
        config = DeploymentConfig(
            model_name="medgemma-4b-it",
            num_replicas=2
        )
        
        deployment = create_deployment(config=config)
        
        assert deployment is not None
    
    @patch("medai_compass.serving.ray_serve_app.create_deployment")
    @patch("ray.serve.run")
    @patch("ray.init")
    def test_serve_application(self, mock_init, mock_run, mock_create):
        """Test serving application startup."""
        from medai_compass.serving.ray_serve_app import serve_application
        
        mock_create.return_value = Mock()
        
        serve_application(model_name="medgemma-27b-it", port=8000)
        
        mock_init.assert_called_once()
        mock_run.assert_called_once()


# ============================================================================
# Test: Model Registry Integration
# ============================================================================

class TestModelRegistryIntegration:
    """Test MLflow model registry integration for serving."""
    
    @patch("mlflow.pyfunc.load_model")
    def test_load_model_from_registry(self, mock_load):
        """Test loading model from MLflow registry."""
        from medai_compass.serving.ray_serve_app import load_model_from_registry
        
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        model = load_model_from_registry(
            model_name="medgemma-27b-it",
            version="production"
        )
        
        assert model is not None
        mock_load.assert_called_once()
    
    @patch("mlflow.pyfunc.load_model")
    def test_load_model_specific_version(self, mock_load):
        """Test loading specific model version."""
        from medai_compass.serving.ray_serve_app import load_model_from_registry
        
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        model = load_model_from_registry(
            model_name="medgemma-27b-it",
            version="v1.2.0"
        )
        
        assert model is not None
    
    @patch("mlflow.pyfunc.load_model")
    def test_load_model_fallback_to_local(self, mock_load):
        """Test fallback to local model if registry fails."""
        from medai_compass.serving.ray_serve_app import load_model_from_registry
        
        mock_load.side_effect = Exception("Registry unavailable")
        
        # Should handle gracefully and potentially use local fallback
        with pytest.raises(Exception):
            load_model_from_registry(
                model_name="medgemma-27b-it",
                version="production",
                fallback_to_local=False
            )


# ============================================================================
# Test: Request/Response Models
# ============================================================================

class TestRequestResponseModels:
    """Test request and response data models."""
    
    def test_generate_request(self):
        """Test GenerateRequest model."""
        from medai_compass.serving.models import GenerateRequest
        
        request = GenerateRequest(
            prompt="What are the symptoms of diabetes?",
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        
        assert request.prompt == "What are the symptoms of diabetes?"
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.9
    
    def test_generate_request_defaults(self):
        """Test GenerateRequest default values."""
        from medai_compass.serving.models import GenerateRequest
        
        request = GenerateRequest(prompt="Test prompt")
        
        assert request.max_tokens == 256
        assert request.temperature == 0.1  # Medical: low temperature for safety
        assert request.top_p == 0.95
    
    def test_generate_response(self):
        """Test GenerateResponse model."""
        from medai_compass.serving.models import GenerateResponse
        
        response = GenerateResponse(
            text="The common symptoms of diabetes include...",
            model_name="medgemma-27b-it",
            tokens_generated=100,
            latency_ms=150.0
        )
        
        assert response.text == "The common symptoms of diabetes include..."
        assert response.model_name == "medgemma-27b-it"
        assert response.tokens_generated == 100
        assert response.latency_ms == 150.0
    
    def test_generate_response_with_metadata(self):
        """Test GenerateResponse with additional metadata."""
        from medai_compass.serving.models import GenerateResponse
        
        response = GenerateResponse(
            text="Response text",
            model_name="medgemma-27b-it",
            tokens_generated=50,
            latency_ms=100.0,
            metadata={"version": "v1.0", "replica_id": "replica-1"}
        )
        
        assert response.metadata["version"] == "v1.0"
        assert response.metadata["replica_id"] == "replica-1"


# ============================================================================
# Test: Metrics Collection
# ============================================================================

class TestServingMetrics:
    """Test metrics collection for serving."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        from medai_compass.serving.metrics import ServingMetricsCollector
        
        collector = ServingMetricsCollector()
        
        assert collector is not None
        assert hasattr(collector, 'record_request')
        assert hasattr(collector, 'record_latency')
    
    def test_record_request_count(self):
        """Test recording request counts."""
        from medai_compass.serving.metrics import ServingMetricsCollector
        
        collector = ServingMetricsCollector()
        collector.record_request(
            model_name="medgemma-27b-it",
            status="success"
        )
        
        # Verify metric was recorded
        metrics = collector.get_metrics()
        assert "request_count" in metrics or "requests_total" in metrics
    
    def test_record_latency(self):
        """Test recording latency metrics."""
        from medai_compass.serving.metrics import ServingMetricsCollector
        
        collector = ServingMetricsCollector()
        collector.record_latency(
            model_name="medgemma-27b-it",
            latency_ms=150.0
        )
        
        metrics = collector.get_metrics()
        assert "latency" in metrics or "latency_ms" in metrics
    
    def test_record_tokens_generated(self):
        """Test recording tokens generated."""
        from medai_compass.serving.metrics import ServingMetricsCollector
        
        collector = ServingMetricsCollector()
        collector.record_tokens(
            model_name="medgemma-27b-it",
            tokens=256
        )
        
        metrics = collector.get_metrics()
        assert "tokens" in metrics or "tokens_generated" in metrics
    
    def test_prometheus_format(self):
        """Test metrics in Prometheus format."""
        from medai_compass.serving.metrics import ServingMetricsCollector
        
        collector = ServingMetricsCollector()
        collector.record_request(model_name="medgemma-27b-it", status="success")
        
        prometheus_output = collector.to_prometheus()
        
        assert isinstance(prometheus_output, str)
        assert "medgemma" in prometheus_output.lower() or "request" in prometheus_output.lower()


# ============================================================================
# Test: GPU Resource Management
# ============================================================================

class TestGPUResourceManagement:
    """Test GPU resource management for serving."""
    
    @patch("torch.cuda.is_available")
    def test_check_gpu_availability(self, mock_cuda):
        """Test checking GPU availability."""
        from medai_compass.serving.health import check_gpu_available
        
        mock_cuda.return_value = True
        
        result = check_gpu_available()
        
        assert result is True
    
    @patch("torch.cuda.is_available")
    def test_check_gpu_unavailable(self, mock_cuda):
        """Test when GPU is unavailable."""
        from medai_compass.serving.health import check_gpu_available
        
        mock_cuda.return_value = False
        
        result = check_gpu_available()
        
        assert result is False
    
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.is_available")
    def test_get_gpu_memory_usage(self, mock_available, mock_allocated, mock_max, mock_props):
        """Test getting GPU memory usage."""
        from medai_compass.serving.health import get_gpu_memory_usage
        
        mock_available.return_value = True
        mock_allocated.return_value = 8 * 1024 ** 3  # 8GB
        mock_max.return_value = 16 * 1024 ** 3  # 16GB
        mock_props.return_value = Mock(total_memory=16 * 1024 ** 3)
        
        usage = get_gpu_memory_usage()
        
        assert usage["allocated_gb"] == 8.0
        assert usage["max_gb"] == 16.0
        assert usage["utilization_percent"] == 50.0
