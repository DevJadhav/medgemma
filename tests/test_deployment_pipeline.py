"""
Tests for deployment pipeline (Phase 7: Deployment Pipeline).

TDD tests for canary deployment, A/B testing, rollback, and deployment orchestration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from datetime import datetime
import asyncio


# ============================================================================
# Test: Deployment Pipeline Configuration
# ============================================================================

class TestDeploymentPipelineConfig:
    """Test deployment pipeline configuration."""
    
    def test_deployment_config_defaults(self):
        """Test default deployment configuration."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentPipelineConfig
        
        config = DeploymentPipelineConfig()
        
        assert config.model_name == "medgemma-27b-it"
        assert config.deployment_strategy == "canary"
        assert config.canary_percentage == 10
        assert config.health_check_interval == 30
        assert config.rollback_on_failure is True
    
    def test_deployment_config_4b_model(self):
        """Test configuration for 4B model."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentPipelineConfig
        
        config = DeploymentPipelineConfig(model_name="medgemma-4b-it")
        
        assert config.model_name == "medgemma-4b-it"
    
    def test_deployment_config_blue_green(self):
        """Test blue-green deployment strategy."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentPipelineConfig
        
        config = DeploymentPipelineConfig(deployment_strategy="blue-green")
        
        assert config.deployment_strategy == "blue-green"
    
    def test_deployment_config_rolling(self):
        """Test rolling deployment strategy."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentPipelineConfig
        
        config = DeploymentPipelineConfig(deployment_strategy="rolling")
        
        assert config.deployment_strategy == "rolling"
    
    def test_deployment_config_custom_canary(self):
        """Test custom canary percentage."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentPipelineConfig
        
        config = DeploymentPipelineConfig(canary_percentage=25)
        
        assert config.canary_percentage == 25


# ============================================================================
# Test: Canary Deployment
# ============================================================================

class TestCanaryDeployment:
    """Test canary deployment functionality."""
    
    def test_canary_manager_initialization(self):
        """Test canary manager initialization."""
        from medai_compass.serving.canary import CanaryManager
        
        manager = CanaryManager(
            current_version="v1.0",
            canary_version="v1.1"
        )
        
        assert manager.current_version == "v1.0"
        assert manager.canary_version == "v1.1"
        assert manager.canary_percentage == 0
    
    def test_canary_start_rollout(self):
        """Test starting canary rollout."""
        from medai_compass.serving.canary import CanaryManager
        
        manager = CanaryManager(
            current_version="v1.0",
            canary_version="v1.1"
        )
        
        manager.start_rollout(percentage=10)
        
        assert manager.canary_percentage == 10
        assert manager.is_active is True
    
    def test_canary_increase_traffic(self):
        """Test increasing canary traffic."""
        from medai_compass.serving.canary import CanaryManager
        
        manager = CanaryManager(
            current_version="v1.0",
            canary_version="v1.1"
        )
        manager.start_rollout(percentage=10)
        
        manager.increase_traffic(percentage=50)
        
        assert manager.canary_percentage == 50
    
    def test_canary_complete_rollout(self):
        """Test completing canary rollout."""
        from medai_compass.serving.canary import CanaryManager
        
        manager = CanaryManager(
            current_version="v1.0",
            canary_version="v1.1"
        )
        manager.start_rollout(percentage=10)
        manager.increase_traffic(percentage=100)
        
        manager.complete_rollout()
        
        assert manager.canary_percentage == 100
        assert manager.is_active is False
        assert manager.current_version == "v1.1"
    
    def test_canary_rollback(self):
        """Test canary rollback."""
        from medai_compass.serving.canary import CanaryManager
        
        manager = CanaryManager(
            current_version="v1.0",
            canary_version="v1.1"
        )
        manager.start_rollout(percentage=10)
        
        manager.rollback()
        
        assert manager.canary_percentage == 0
        assert manager.is_active is False
        assert manager.current_version == "v1.0"
    
    def test_canary_route_request(self):
        """Test routing requests based on canary percentage."""
        from medai_compass.serving.canary import CanaryManager
        
        manager = CanaryManager(
            current_version="v1.0",
            canary_version="v1.1"
        )
        manager.start_rollout(percentage=50)
        
        # Run multiple routing decisions
        canary_count = 0
        total_requests = 1000
        
        for _ in range(total_requests):
            if manager.should_route_to_canary():
                canary_count += 1
        
        # Should be roughly 50% (allow some variance)
        canary_ratio = canary_count / total_requests
        assert 0.4 <= canary_ratio <= 0.6
    
    def test_canary_status(self):
        """Test getting canary deployment status."""
        from medai_compass.serving.canary import CanaryManager, CanaryStatus
        
        manager = CanaryManager(
            current_version="v1.0",
            canary_version="v1.1"
        )
        manager.start_rollout(percentage=25)
        
        status = manager.get_status()
        
        assert isinstance(status, CanaryStatus)
        assert status.current_version == "v1.0"
        assert status.canary_version == "v1.1"
        assert status.canary_percentage == 25
        assert status.is_active is True


# ============================================================================
# Test: A/B Testing
# ============================================================================

class TestABTesting:
    """Test A/B testing functionality."""
    
    def test_ab_test_creation(self):
        """Test creating an A/B test."""
        from medai_compass.serving.ab_testing import ABTest, ABTestConfig
        
        config = ABTestConfig(
            name="model-comparison",
            variant_a="medgemma-27b-it",
            variant_b="medgemma-4b-it",
            traffic_split=0.5
        )
        
        test = ABTest(config)
        
        assert test.name == "model-comparison"
        assert test.variant_a == "medgemma-27b-it"
        assert test.variant_b == "medgemma-4b-it"
        assert test.traffic_split == 0.5
    
    def test_ab_test_assign_variant(self):
        """Test assigning users to variants."""
        from medai_compass.serving.ab_testing import ABTest, ABTestConfig
        
        config = ABTestConfig(
            name="model-comparison",
            variant_a="medgemma-27b-it",
            variant_b="medgemma-4b-it",
            traffic_split=0.5
        )
        
        test = ABTest(config)
        
        # Same user should always get same variant
        variant1 = test.assign_variant(user_id="user-123")
        variant2 = test.assign_variant(user_id="user-123")
        
        assert variant1 == variant2
        assert variant1 in ["medgemma-27b-it", "medgemma-4b-it"]
    
    def test_ab_test_record_outcome(self):
        """Test recording A/B test outcomes."""
        from medai_compass.serving.ab_testing import ABTest, ABTestConfig
        
        config = ABTestConfig(
            name="model-comparison",
            variant_a="medgemma-27b-it",
            variant_b="medgemma-4b-it",
            traffic_split=0.5
        )
        
        test = ABTest(config)
        variant = test.assign_variant(user_id="user-123")
        
        test.record_outcome(
            user_id="user-123",
            metric_name="latency_ms",
            value=150.0
        )
        
        outcomes = test.get_outcomes()
        assert len(outcomes) >= 1
    
    def test_ab_test_compute_results(self):
        """Test computing A/B test results."""
        from medai_compass.serving.ab_testing import ABTest, ABTestConfig
        
        config = ABTestConfig(
            name="model-comparison",
            variant_a="medgemma-27b-it",
            variant_b="medgemma-4b-it",
            traffic_split=0.5
        )
        
        test = ABTest(config)
        
        # Simulate some test data
        for i in range(100):
            variant = test.assign_variant(user_id=f"user-{i}")
            test.record_outcome(
                user_id=f"user-{i}",
                metric_name="latency_ms",
                value=100.0 + (i % 50)
            )
        
        results = test.compute_results()
        
        assert "variant_a" in results
        assert "variant_b" in results
        assert "statistical_significance" in results
    
    def test_ab_test_manager(self):
        """Test A/B test manager."""
        from medai_compass.serving.ab_testing import ABTestManager, ABTestConfig
        
        manager = ABTestManager()
        
        config = ABTestConfig(
            name="test-1",
            variant_a="medgemma-27b-it",
            variant_b="medgemma-4b-it",
            traffic_split=0.5
        )
        
        manager.create_test(config)
        
        assert manager.get_test("test-1") is not None
    
    def test_ab_test_manager_multiple_tests(self):
        """Test managing multiple A/B tests."""
        from medai_compass.serving.ab_testing import ABTestManager, ABTestConfig
        
        manager = ABTestManager()
        
        manager.create_test(ABTestConfig(
            name="test-1",
            variant_a="medgemma-27b-it",
            variant_b="medgemma-4b-it"
        ))
        manager.create_test(ABTestConfig(
            name="test-2",
            variant_a="v1.0",
            variant_b="v1.1"
        ))
        
        tests = manager.list_tests()
        
        assert len(tests) == 2
        assert "test-1" in [t.name for t in tests]
        assert "test-2" in [t.name for t in tests]


# ============================================================================
# Test: Rollback Mechanism
# ============================================================================

class TestRollbackMechanism:
    """Test rollback mechanism."""
    
    def test_rollback_manager_initialization(self):
        """Test rollback manager initialization."""
        from medai_compass.serving.rollback import RollbackManager
        
        manager = RollbackManager()
        
        assert manager is not None
        assert manager.deployment_history == []
    
    def test_record_deployment(self):
        """Test recording deployment in history."""
        from medai_compass.serving.rollback import RollbackManager, DeploymentRecord
        
        manager = RollbackManager()
        
        record = manager.record_deployment(
            version="v1.0",
            model_name="medgemma-27b-it",
            config={"replicas": 2}
        )
        
        assert isinstance(record, DeploymentRecord)
        assert record.version == "v1.0"
        assert len(manager.deployment_history) == 1
    
    def test_get_rollback_target(self):
        """Test getting rollback target version."""
        from medai_compass.serving.rollback import RollbackManager
        
        manager = RollbackManager()
        
        manager.record_deployment(version="v1.0", model_name="medgemma-27b-it")
        manager.record_deployment(version="v1.1", model_name="medgemma-27b-it")
        manager.record_deployment(version="v1.2", model_name="medgemma-27b-it")
        
        # Current is v1.2, rollback target should be v1.1
        target = manager.get_rollback_target()
        
        assert target.version == "v1.1"
    
    def test_get_rollback_target_multiple_steps(self):
        """Test getting rollback target multiple steps back."""
        from medai_compass.serving.rollback import RollbackManager
        
        manager = RollbackManager()
        
        manager.record_deployment(version="v1.0", model_name="medgemma-27b-it")
        manager.record_deployment(version="v1.1", model_name="medgemma-27b-it")
        manager.record_deployment(version="v1.2", model_name="medgemma-27b-it")
        
        target = manager.get_rollback_target(steps_back=2)
        
        assert target.version == "v1.0"
    
    @patch("medai_compass.serving.rollback.deploy_version")
    def test_execute_rollback(self, mock_deploy):
        """Test executing rollback."""
        from medai_compass.serving.rollback import RollbackManager
        
        mock_deploy.return_value = True
        
        manager = RollbackManager()
        manager.record_deployment(version="v1.0", model_name="medgemma-27b-it")
        manager.record_deployment(version="v1.1", model_name="medgemma-27b-it")
        
        result = manager.execute_rollback()
        
        assert result.success is True
        assert result.rolled_back_to == "v1.0"
        mock_deploy.assert_called_once()
    
    def test_rollback_no_history(self):
        """Test rollback with no deployment history."""
        from medai_compass.serving.rollback import RollbackManager, RollbackError
        
        manager = RollbackManager()
        
        with pytest.raises(RollbackError):
            manager.execute_rollback()
    
    def test_automatic_rollback_trigger(self):
        """Test automatic rollback on health check failure."""
        from medai_compass.serving.rollback import RollbackManager, RollbackTrigger
        
        manager = RollbackManager()
        manager.record_deployment(version="v1.0", model_name="medgemma-27b-it")
        manager.record_deployment(version="v1.1", model_name="medgemma-27b-it")
        
        trigger = RollbackTrigger(
            manager=manager,
            health_threshold=0.95,
            latency_threshold_ms=500
        )
        
        # Simulate health degradation - need 3 consecutive failures
        trigger.evaluate(health_score=0.80, latency_p95_ms=600)  # 1st
        trigger.evaluate(health_score=0.80, latency_p95_ms=600)  # 2nd
        should_rollback = trigger.evaluate(health_score=0.80, latency_p95_ms=600)  # 3rd
        
        assert should_rollback is True


# ============================================================================
# Test: Deployment Pipeline Orchestration
# ============================================================================

class TestDeploymentPipelineOrchestration:
    """Test deployment pipeline orchestration."""
    
    def test_deployment_pipeline_initialization(self):
        """Test deployment pipeline initialization."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentPipeline
        
        pipeline = DeploymentPipeline()
        
        assert pipeline is not None
    
    @patch("medai_compass.pipelines.deployment_pipeline.validate_model")
    @patch("medai_compass.pipelines.deployment_pipeline.deploy_to_ray_serve")
    def test_deployment_pipeline_run(self, mock_deploy, mock_validate):
        """Test running deployment pipeline."""
        from medai_compass.pipelines.deployment_pipeline import (
            DeploymentPipeline,
            DeploymentPipelineConfig
        )
        
        mock_validate.return_value = True
        mock_deploy.return_value = {"status": "success"}
        
        config = DeploymentPipelineConfig(
            model_name="medgemma-27b-it",
            deployment_strategy="canary"
        )
        
        pipeline = DeploymentPipeline()
        result = pipeline.run(config)
        
        assert result.success is True
        mock_validate.assert_called_once()
        mock_deploy.assert_called_once()
    
    @patch("medai_compass.pipelines.deployment_pipeline.validate_model")
    def test_deployment_pipeline_validation_failure(self, mock_validate):
        """Test pipeline handles validation failure."""
        from medai_compass.pipelines.deployment_pipeline import (
            DeploymentPipeline,
            DeploymentPipelineConfig
        )
        
        mock_validate.return_value = False
        
        config = DeploymentPipelineConfig(model_name="medgemma-27b-it")
        
        pipeline = DeploymentPipeline()
        result = pipeline.run(config)
        
        assert result.success is False
        assert "validation" in result.error.lower()
    
    def test_deployment_stages(self):
        """Test deployment pipeline stages."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentStage
        
        assert DeploymentStage.VALIDATE.value == "validate"
        assert DeploymentStage.PACKAGE.value == "package"
        assert DeploymentStage.DEPLOY.value == "deploy"
        assert DeploymentStage.VERIFY.value == "verify"
        assert DeploymentStage.PROMOTE.value == "promote"
    
    def test_deployment_result(self):
        """Test deployment result dataclass."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentResult
        
        result = DeploymentResult(
            success=True,
            version="v1.0",
            model_name="medgemma-27b-it",
            deployment_time_seconds=45.0,
            endpoint_url="http://localhost:8000"
        )
        
        assert result.success is True
        assert result.version == "v1.0"
        assert result.endpoint_url == "http://localhost:8000"


# ============================================================================
# Test: Model Packaging
# ============================================================================

class TestModelPackaging:
    """Test model packaging for deployment."""
    
    @patch("mlflow.pyfunc.load_model")
    def test_fetch_model_from_registry(self, mock_load):
        """Test fetching model from MLflow registry."""
        from medai_compass.pipelines.deployment_pipeline import ModelPackager
        
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        packager = ModelPackager()
        model = packager.fetch_from_registry(
            model_name="medgemma-27b-it",
            version="production"
        )
        
        assert model is not None
    
    def test_package_for_ray_serve(self):
        """Test packaging model for Ray Serve."""
        from medai_compass.pipelines.deployment_pipeline import ModelPackager
        
        packager = ModelPackager()
        
        mock_model = Mock()
        package = packager.package_for_ray_serve(
            model=mock_model,
            model_name="medgemma-27b-it"
        )
        
        assert package is not None
        assert "config" in package
        assert "model_path" in package or "model" in package
    
    def test_validate_model_before_deployment(self):
        """Test model validation before deployment."""
        from medai_compass.pipelines.deployment_pipeline import ModelPackager
        
        packager = ModelPackager()
        
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Test response")
        
        is_valid = packager.validate_model(mock_model)
        
        assert is_valid is True
    
    def test_validate_model_failure(self):
        """Test model validation failure."""
        from medai_compass.pipelines.deployment_pipeline import ModelPackager
        
        packager = ModelPackager()
        
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=Exception("Model error"))
        
        is_valid = packager.validate_model(mock_model)
        
        assert is_valid is False


# ============================================================================
# Test: Deployment Monitoring
# ============================================================================

class TestDeploymentMonitoring:
    """Test deployment monitoring functionality."""
    
    def test_deployment_monitor_initialization(self):
        """Test deployment monitor initialization."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentMonitor
        
        monitor = DeploymentMonitor()
        
        assert monitor is not None
    
    def test_monitor_health_check(self):
        """Test monitoring health checks."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentMonitor
        
        monitor = DeploymentMonitor()
        
        health = monitor.check_health(endpoint="http://localhost:8000/health")
        
        # Returns a health status dict
        assert isinstance(health, dict)
        assert "status" in health
    
    def test_monitor_latency(self):
        """Test monitoring latency metrics."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentMonitor
        
        monitor = DeploymentMonitor()
        
        metrics = monitor.get_latency_metrics(
            model_name="medgemma-27b-it"
        )
        
        assert "p50_ms" in metrics
        assert "p95_ms" in metrics
        assert "p99_ms" in metrics
    
    def test_monitor_alerts(self):
        """Test deployment alerts."""
        from medai_compass.pipelines.deployment_pipeline import (
            DeploymentMonitor,
            AlertConfig
        )
        
        monitor = DeploymentMonitor()
        
        alert_config = AlertConfig(
            latency_threshold_ms=500,
            error_rate_threshold=0.01,
            health_threshold=0.95
        )
        
        monitor.configure_alerts(alert_config)
        
        # Simulate metrics that should trigger alert
        alerts = monitor.check_alerts(
            latency_p95_ms=600,
            error_rate=0.02,
            health_score=0.90
        )
        
        assert len(alerts) > 0


# ============================================================================
# Test: Deployment Configuration Validation
# ============================================================================

class TestDeploymentConfigValidation:
    """Test deployment configuration validation."""
    
    def test_validate_model_name(self):
        """Test model name validation."""
        from medai_compass.pipelines.deployment_pipeline import validate_config
        
        # Valid model names
        assert validate_config({"model_name": "medgemma-27b-it"})["valid"] is True
        assert validate_config({"model_name": "medgemma-4b-it"})["valid"] is True
        
        # Invalid model name
        result = validate_config({"model_name": "invalid-model"})
        assert result["valid"] is False
    
    def test_validate_deployment_strategy(self):
        """Test deployment strategy validation."""
        from medai_compass.pipelines.deployment_pipeline import validate_config
        
        # Valid strategies
        assert validate_config({"deployment_strategy": "canary"})["valid"] is True
        assert validate_config({"deployment_strategy": "blue-green"})["valid"] is True
        assert validate_config({"deployment_strategy": "rolling"})["valid"] is True
        
        # Invalid strategy
        result = validate_config({"deployment_strategy": "invalid"})
        assert result["valid"] is False
    
    def test_validate_canary_percentage(self):
        """Test canary percentage validation."""
        from medai_compass.pipelines.deployment_pipeline import validate_config
        
        # Valid percentages
        assert validate_config({"canary_percentage": 10})["valid"] is True
        assert validate_config({"canary_percentage": 50})["valid"] is True
        
        # Invalid percentages
        result = validate_config({"canary_percentage": -1})
        assert result["valid"] is False
        
        result = validate_config({"canary_percentage": 101})
        assert result["valid"] is False


# ============================================================================
# Test: Integration with Existing Infrastructure
# ============================================================================

class TestExistingInfrastructureIntegration:
    """Test integration with existing infrastructure."""
    
    def test_prometheus_integration(self):
        """Test Prometheus metrics integration."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentMetrics
        
        metrics = DeploymentMetrics()
        
        metrics.record_deployment(
            model_name="medgemma-27b-it",
            version="v1.0",
            duration_seconds=45.0
        )
        
        # Verify deployment was recorded in internal metrics
        stats = metrics.get_stats()
        assert stats["total_deployments"] == 1
        assert stats["avg_duration_seconds"] == 45.0
    
    @patch("mlflow.log_metric")
    @patch("mlflow.start_run")
    def test_mlflow_integration(self, mock_start, mock_log):
        """Test MLflow integration for deployment tracking."""
        from medai_compass.pipelines.deployment_pipeline import DeploymentPipeline
        
        mock_start.return_value.__enter__ = Mock(return_value=Mock(info=Mock(run_id="test-run")))
        mock_start.return_value.__exit__ = Mock(return_value=False)
        
        pipeline = DeploymentPipeline()
        pipeline.log_deployment_to_mlflow(
            model_name="medgemma-27b-it",
            version="v1.0",
            metrics={"latency_p95": 150.0}
        )
        
        mock_log.assert_called()


# ============================================================================
# Test: Deployment YAML Configuration
# ============================================================================

class TestDeploymentYAMLConfig:
    """Test YAML configuration for deployments."""
    
    def test_load_deployment_config(self):
        """Test loading deployment config from YAML."""
        from medai_compass.pipelines.deployment_pipeline import load_deployment_config
        
        config_yaml = """
        model_name: medgemma-27b-it
        deployment_strategy: canary
        canary_percentage: 10
        replicas: 2
        """
        
        config = load_deployment_config(config_yaml)
        
        assert config.model_name == "medgemma-27b-it"
        assert config.deployment_strategy == "canary"
        assert config.canary_percentage == 10
    
    def test_load_deployment_config_with_resources(self):
        """Test loading config with resource specifications."""
        from medai_compass.pipelines.deployment_pipeline import load_deployment_config
        
        config_yaml = """
        model_name: medgemma-27b-it
        resources:
          num_gpus: 2
          memory_gb: 64
          num_cpus: 16
        """
        
        config = load_deployment_config(config_yaml)
        
        assert config.resources["num_gpus"] == 2
        assert config.resources["memory_gb"] == 64
