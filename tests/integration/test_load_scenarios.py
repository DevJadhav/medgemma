"""
Load testing scenarios for MedAI Compass (Phase 9).

Tests system behavior under various load conditions:
- Normal load
- Peak load
- Stress testing
- Spike testing
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# Load Test Configuration Tests
# =============================================================================

class TestLoadTestConfiguration:
    """Test load test configuration."""
    
    def test_load_test_module_imports(self):
        """Test load testing module can be imported."""
        from medai_compass.benchmarking import (
            LoadTestConfig,
            LoadTestScenario,
            LoadTestRunner,
        )
        
        assert LoadTestConfig is not None
        assert LoadTestScenario is not None
        assert LoadTestRunner is not None
    
    def test_load_test_config_defaults(self):
        """Test load test config defaults."""
        from medai_compass.benchmarking import LoadTestConfig
        
        config = LoadTestConfig()
        assert config.duration_seconds > 0
        assert config.users > 0
        assert config.ramp_up_seconds >= 0
    
    def test_load_test_config_custom(self):
        """Test custom load test configuration."""
        from medai_compass.benchmarking import LoadTestConfig
        
        config = LoadTestConfig(
            duration_seconds=60,
            users=100,
            ramp_up_seconds=10,
            requests_per_second=50,
        )
        
        assert config.duration_seconds == 60
        assert config.users == 100
        assert config.ramp_up_seconds == 10


class TestLoadTestScenarios:
    """Test different load test scenarios."""
    
    def test_normal_load_scenario(self):
        """Test normal load scenario definition."""
        from medai_compass.benchmarking import LoadTestScenario
        
        scenario = LoadTestScenario.normal_load()
        
        assert scenario.name == "normal_load"
        assert scenario.users <= 50
        assert scenario.duration_seconds >= 60
    
    def test_peak_load_scenario(self):
        """Test peak load scenario definition."""
        from medai_compass.benchmarking import LoadTestScenario
        
        scenario = LoadTestScenario.peak_load()
        
        assert scenario.name == "peak_load"
        assert scenario.users >= 100
    
    def test_stress_test_scenario(self):
        """Test stress test scenario definition."""
        from medai_compass.benchmarking import LoadTestScenario
        
        scenario = LoadTestScenario.stress_test()
        
        assert scenario.name == "stress_test"
        assert scenario.users >= 200
    
    def test_spike_test_scenario(self):
        """Test spike test scenario definition."""
        from medai_compass.benchmarking import LoadTestScenario
        
        scenario = LoadTestScenario.spike_test()
        
        assert scenario.name == "spike_test"
        assert scenario.ramp_up_seconds <= 5  # Fast ramp-up


# =============================================================================
# Load Test Execution Tests
# =============================================================================

class TestLoadTestExecution:
    """Test load test execution."""
    
    @pytest.fixture
    def mock_api_client(self):
        """Mock API client for load testing."""
        client = MagicMock()
        client.get.return_value = MagicMock(status_code=200)
        client.post.return_value = MagicMock(status_code=200)
        return client
    
    def test_load_test_runs(self, mock_api_client):
        """Test load test executes."""
        from medai_compass.benchmarking import LoadTestRunner, LoadTestConfig
        
        config = LoadTestConfig(
            duration_seconds=1,
            users=2,
            ramp_up_seconds=0,
        )
        
        runner = LoadTestRunner(
            config=config,
            api_client=mock_api_client,
        )
        
        result = runner.run()
        assert result is not None
    
    def test_load_test_measures_response_times(self, mock_api_client):
        """Test load test measures response times."""
        from medai_compass.benchmarking import LoadTestRunner, LoadTestConfig
        
        config = LoadTestConfig(
            duration_seconds=1,
            users=2,
        )
        
        runner = LoadTestRunner(
            config=config,
            api_client=mock_api_client,
        )
        
        result = runner.run()
        assert result.response_times is not None
        assert len(result.response_times) > 0
    
    def test_load_test_tracks_errors(self, mock_api_client):
        """Test load test tracks error rate."""
        from medai_compass.benchmarking import LoadTestRunner, LoadTestConfig
        
        # Some requests fail
        call_count = [0]
        def sometimes_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return MagicMock(status_code=500)
            return MagicMock(status_code=200)
        
        mock_api_client.get.side_effect = sometimes_fail
        
        config = LoadTestConfig(
            duration_seconds=1,
            users=3,
        )
        
        runner = LoadTestRunner(
            config=config,
            api_client=mock_api_client,
        )
        
        result = runner.run()
        assert result.failed_requests >= 0


class TestConcurrentUserSimulation:
    """Test concurrent user simulation."""
    
    @pytest.fixture
    def mock_endpoint(self):
        """Mock endpoint for concurrent testing."""
        def endpoint():
            time.sleep(0.01)  # 10ms response
            return {"status": "ok"}
        return endpoint
    
    def test_concurrent_users_simulated(self, mock_endpoint):
        """Test concurrent users are simulated."""
        from medai_compass.benchmarking import ConcurrentUserSimulator
        
        simulator = ConcurrentUserSimulator(
            endpoint=mock_endpoint,
            num_users=5,
            duration_seconds=1,
        )
        
        result = simulator.run()
        assert result.total_requests > 0
        assert result.concurrent_users == 5
    
    def test_ramp_up_applied(self, mock_endpoint):
        """Test ramp-up is applied correctly."""
        from medai_compass.benchmarking import ConcurrentUserSimulator
        
        simulator = ConcurrentUserSimulator(
            endpoint=mock_endpoint,
            num_users=10,
            duration_seconds=2,
            ramp_up_seconds=1,
        )
        
        result = simulator.run()
        assert result is not None
    
    def test_steady_state_maintained(self, mock_endpoint):
        """Test steady state is maintained after ramp-up."""
        from medai_compass.benchmarking import ConcurrentUserSimulator
        
        simulator = ConcurrentUserSimulator(
            endpoint=mock_endpoint,
            num_users=5,
            duration_seconds=3,
            ramp_up_seconds=1,
        )
        
        result = simulator.run()
        # Should have requests throughout duration
        assert result.total_requests > 0


# =============================================================================
# Load Test Result Analysis Tests
# =============================================================================

class TestLoadTestResults:
    """Test load test result analysis."""
    
    def test_load_test_result_creation(self):
        """Test load test result creation."""
        from medai_compass.benchmarking import LoadTestResult
        
        result = LoadTestResult(
            scenario_name="test",
            total_requests=1000,
            successful_requests=980,
            failed_requests=20,
            error_rate=0.02,
            response_times=[100, 150, 200],
            requests_per_second=50.0,
            mean_response_time_ms=150.0,
            p50_response_time_ms=140.0,
            p95_response_time_ms=200.0,
            p99_response_time_ms=250.0,
            concurrent_users=10,
            duration_seconds=20,
        )
        
        assert result.total_requests == 1000
        assert result.error_rate == 0.02
    
    def test_load_test_result_percentiles(self):
        """Test load test result percentile calculation."""
        from medai_compass.benchmarking import LoadTestResult
        
        response_times = list(range(1, 101))  # 1-100ms
        
        result = LoadTestResult.from_response_times(
            scenario_name="test",
            response_times=response_times,
            errors=5,
            duration_seconds=10,
            concurrent_users=5,
        )
        
        assert result.p50_response_time_ms == 50.0 or abs(result.p50_response_time_ms - 50.5) < 1
        assert result.p95_response_time_ms >= 95
    
    def test_load_test_passes_sla(self):
        """Test load test SLA check."""
        from medai_compass.benchmarking import LoadTestResult, SLATargets
        
        result = LoadTestResult(
            scenario_name="test",
            total_requests=1000,
            successful_requests=999,
            failed_requests=1,
            error_rate=0.001,
            response_times=[100] * 1000,
            requests_per_second=100.0,
            mean_response_time_ms=100.0,
            p50_response_time_ms=100.0,
            p95_response_time_ms=100.0,
            p99_response_time_ms=100.0,
            concurrent_users=10,
            duration_seconds=10,
        )
        
        targets = SLATargets()
        assert result.passes_sla(targets)


# =============================================================================
# API Load Test Scenarios
# =============================================================================

class TestAPILoadScenarios:
    """Test API-specific load scenarios."""
    
    @pytest.fixture
    def mock_fastapi_app(self):
        """Create test FastAPI app."""
        from fastapi.testclient import TestClient
        from medai_compass.api.main import app
        return TestClient(app)
    
    def test_health_endpoint_load(self, mock_fastapi_app):
        """Test health endpoint under load."""
        from medai_compass.benchmarking import APILoadTest
        
        test = APILoadTest(
            client=mock_fastapi_app,
            endpoint="/health",
            method="GET",
            concurrent_users=10,
            duration_seconds=1,
        )
        
        result = test.run()
        assert result.total_requests > 0
    
    def test_communication_endpoint_load(self, mock_fastapi_app):
        """Test communication endpoint under load."""
        from medai_compass.benchmarking import APILoadTest
        
        test = APILoadTest(
            client=mock_fastapi_app,
            endpoint="/api/v1/communication/message",
            method="POST",
            payload={
                "message": "Test health question",
                "patient_id": "load-test-patient",
            },
            concurrent_users=5,
            duration_seconds=1,
        )
        
        result = test.run()
        assert result is not None
    
    def test_mixed_endpoint_load(self, mock_fastapi_app):
        """Test mixed endpoint load."""
        from medai_compass.benchmarking import MixedLoadTest
        
        endpoints = [
            {"endpoint": "/health", "method": "GET", "weight": 50},
            {"endpoint": "/health/ready", "method": "GET", "weight": 30},
            {"endpoint": "/health/live", "method": "GET", "weight": 20},
        ]
        
        test = MixedLoadTest(
            client=mock_fastapi_app,
            endpoints=endpoints,
            concurrent_users=10,
            duration_seconds=1,
        )
        
        result = test.run()
        assert result.total_requests > 0


# =============================================================================
# Load Test Report Tests
# =============================================================================

class TestLoadTestReport:
    """Test load test reporting."""
    
    def test_load_test_report_generation(self):
        """Test load test report generation."""
        from medai_compass.benchmarking import LoadTestReport, LoadTestResult
        
        results = [
            LoadTestResult(
                scenario_name="normal",
                total_requests=1000,
                successful_requests=990,
                failed_requests=10,
                error_rate=0.01,
                response_times=[100] * 1000,
                requests_per_second=100.0,
                mean_response_time_ms=100.0,
                p50_response_time_ms=100.0,
                p95_response_time_ms=100.0,
                p99_response_time_ms=100.0,
                concurrent_users=10,
                duration_seconds=10,
            ),
        ]
        
        report = LoadTestReport(results=results)
        assert report is not None
    
    def test_load_test_report_to_json(self):
        """Test load test report JSON export."""
        from medai_compass.benchmarking import LoadTestReport, LoadTestResult
        
        results = [
            LoadTestResult(
                scenario_name="test",
                total_requests=100,
                successful_requests=100,
                failed_requests=0,
                error_rate=0.0,
                response_times=[100] * 100,
                requests_per_second=10.0,
                mean_response_time_ms=100.0,
                p50_response_time_ms=100.0,
                p95_response_time_ms=100.0,
                p99_response_time_ms=100.0,
                concurrent_users=5,
                duration_seconds=10,
            ),
        ]
        
        report = LoadTestReport(results=results)
        json_data = report.to_json()
        
        assert isinstance(json_data, str)
        assert "test" in json_data
    
    def test_load_test_report_summary(self):
        """Test load test report summary."""
        from medai_compass.benchmarking import LoadTestReport, LoadTestResult
        
        results = [
            LoadTestResult(
                scenario_name="normal",
                total_requests=1000,
                successful_requests=990,
                failed_requests=10,
                error_rate=0.01,
                response_times=[100] * 1000,
                requests_per_second=100.0,
                mean_response_time_ms=100.0,
                p50_response_time_ms=100.0,
                p95_response_time_ms=100.0,
                p99_response_time_ms=100.0,
                concurrent_users=10,
                duration_seconds=10,
            ),
        ]
        
        report = LoadTestReport(results=results)
        summary = report.get_summary()
        
        assert "normal" in summary.lower() or "requests" in summary.lower()


# =============================================================================
# Locust Integration Tests
# =============================================================================

class TestLocustIntegration:
    """Test Locust load testing integration."""
    
    def test_locustfile_exists(self):
        """Test locustfile exists."""
        import os
        locustfile = "/Users/dev/Downloads/Projects/MedGemma/tests/load/locustfile.py"
        assert os.path.exists(locustfile)
    
    def test_locust_user_defined(self):
        """Test Locust user class is defined."""
        import ast
        import os
        
        locustfile = "/Users/dev/Downloads/Projects/MedGemma/tests/load/locustfile.py"
        
        with open(locustfile, "r") as f:
            content = f.read()
        
        # Parse the file and look for class definition
        tree = ast.parse(content)
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Should have a user class
        assert any("User" in name for name in class_names), f"Expected User class, found: {class_names}"
    
    def test_locust_tasks_defined(self):
        """Test Locust tasks are defined."""
        import os
        
        locustfile = "/Users/dev/Downloads/Projects/MedGemma/tests/load/locustfile.py"
        
        with open(locustfile, "r") as f:
            content = f.read()
        
        # Should have @task decorator
        assert "@task" in content, "Expected @task decorator in locustfile"


# =============================================================================
# CI Load Test Configuration
# =============================================================================

class TestCILoadTests:
    """Test CI-friendly load test configuration."""
    
    def test_smoke_load_test_config(self):
        """Test smoke load test for CI."""
        from medai_compass.benchmarking import LoadTestConfig
        
        # Short duration, few users for CI
        config = LoadTestConfig.smoke_test()
        
        assert config.duration_seconds <= 30
        assert config.users <= 10
    
    def test_ci_load_test_passes_quickly(self):
        """Test CI load tests complete quickly."""
        from medai_compass.benchmarking import LoadTestConfig
        
        config = LoadTestConfig.ci_config()
        
        # Should complete in under 60 seconds
        assert config.duration_seconds <= 60
        assert config.ramp_up_seconds <= 5
    
    def test_full_load_test_config(self):
        """Test full load test for scheduled runs."""
        from medai_compass.benchmarking import LoadTestConfig
        
        config = LoadTestConfig.full_test()
        
        # Longer duration, more users
        assert config.duration_seconds >= 300
        assert config.users >= 50
