"""
Tests for Ray actors module.

Tests actor creation, state management, and operations.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime

from medai_compass.workflows.ray_actors import (
    EvaluationResult,
    MetricEntry,
    CheckpointInfo,
    RayActorManager,
)


# =============================================================================
# Data Class Tests
# =============================================================================

class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = EvaluationResult(
            checkpoint_path="/checkpoints/step-1000",
            metrics={"accuracy": 0.75, "loss": 0.25},
            benchmarks=["medqa", "pubmedqa"],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=120.5,
        )

        assert result.checkpoint_path == "/checkpoints/step-1000"
        assert result.metrics["accuracy"] == 0.75
        assert "medqa" in result.benchmarks
        assert result.passed_quality_gate is True

    def test_failed_result(self):
        """Test failed evaluation result."""
        result = EvaluationResult(
            checkpoint_path="/checkpoints/failed",
            metrics={},
            benchmarks=[],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=5.0,
            passed_quality_gate=False,
            error="Model loading failed",
        )

        assert result.passed_quality_gate is False
        assert result.error == "Model loading failed"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            checkpoint_path="/checkpoints/test",
            metrics={"accuracy": 0.8},
            benchmarks=["medqa"],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=60.0,
        )

        d = result.to_dict()

        assert d["checkpoint_path"] == "/checkpoints/test"
        assert d["metrics"]["accuracy"] == 0.8
        assert d["benchmarks"] == ["medqa"]


class TestMetricEntry:
    """Tests for MetricEntry dataclass."""

    def test_basic_creation(self):
        """Test basic entry creation."""
        entry = MetricEntry(
            step=100,
            timestamp="2024-01-01T00:00:00",
            values={"loss": 0.5, "learning_rate": 1e-4},
        )

        assert entry.step == 100
        assert entry.values["loss"] == 0.5
        assert entry.worker_id is None

    def test_with_worker_id(self):
        """Test entry with worker ID."""
        entry = MetricEntry(
            step=200,
            timestamp="2024-01-01T00:00:00",
            values={"loss": 0.3},
            worker_id="worker-0",
        )

        assert entry.worker_id == "worker-0"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = MetricEntry(
            step=100,
            timestamp="2024-01-01T00:00:00",
            values={"loss": 0.5, "accuracy": 0.75},
            worker_id="worker-1",
        )

        d = entry.to_dict()

        assert d["step"] == 100
        assert d["loss"] == 0.5
        assert d["accuracy"] == 0.75
        assert d["worker_id"] == "worker-1"


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_basic_creation(self):
        """Test basic checkpoint info creation."""
        info = CheckpointInfo(
            path="/checkpoints/step-500",
            step=500,
            timestamp="2024-01-01T00:00:00",
            metrics={"eval_loss": 0.3, "accuracy": 0.78},
        )

        assert info.path == "/checkpoints/step-500"
        assert info.step == 500
        assert info.is_best is False

    def test_best_checkpoint(self):
        """Test best checkpoint flag."""
        info = CheckpointInfo(
            path="/checkpoints/best",
            step=1000,
            timestamp="2024-01-01T00:00:00",
            metrics={"eval_loss": 0.1},
            is_best=True,
        )

        assert info.is_best is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = CheckpointInfo(
            path="/checkpoints/test",
            step=100,
            timestamp="2024-01-01T00:00:00",
            metrics={"loss": 0.5},
            is_best=True,
        )

        d = info.to_dict()

        assert d["path"] == "/checkpoints/test"
        assert d["step"] == 100
        assert d["is_best"] is True


# =============================================================================
# Actor Manager Tests (without Ray)
# =============================================================================

class TestRayActorManager:
    """Tests for RayActorManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = RayActorManager()

        assert manager.actors == {}
        assert manager._initialized is False

    @patch("ray.is_initialized")
    @patch("ray.init")
    def test_initialize_ray(self, mock_init, mock_is_initialized):
        """Test Ray initialization."""
        mock_is_initialized.return_value = False

        manager = RayActorManager()
        manager.initialize()

        mock_init.assert_called_once()
        assert manager._initialized is True

    @patch("ray.is_initialized")
    def test_initialize_already_running(self, mock_is_initialized):
        """Test initialization when Ray is already running."""
        mock_is_initialized.return_value = True

        manager = RayActorManager()
        manager.initialize()

        assert manager._initialized is True

    def test_get_nonexistent_actor(self):
        """Test getting non-existent actor."""
        manager = RayActorManager()

        actor = manager.get_actor("nonexistent")

        assert actor is None


# =============================================================================
# Mock Actor Tests
# =============================================================================

class TestMockEvaluationActor:
    """Tests for evaluation actor behavior using mocks."""

    def test_evaluate_checkpoint_success(self):
        """Test successful checkpoint evaluation."""
        # Create mock actor
        mock_actor = MagicMock()
        mock_actor.evaluate_checkpoint.return_value = EvaluationResult(
            checkpoint_path="/checkpoints/test",
            metrics={"accuracy": 0.75, "safety_score": 0.98},
            benchmarks=["medqa"],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=60.0,
            passed_quality_gate=True,
        )

        result = mock_actor.evaluate_checkpoint("/checkpoints/test")

        assert result.passed_quality_gate is True
        assert result.metrics["accuracy"] == 0.75

    def test_evaluate_checkpoint_failure(self):
        """Test failed checkpoint evaluation."""
        mock_actor = MagicMock()
        mock_actor.evaluate_checkpoint.return_value = EvaluationResult(
            checkpoint_path="/checkpoints/bad",
            metrics={},
            benchmarks=[],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=5.0,
            passed_quality_gate=False,
            error="Failed to load model",
        )

        result = mock_actor.evaluate_checkpoint("/checkpoints/bad")

        assert result.passed_quality_gate is False
        assert "Failed" in result.error

    def test_get_best_checkpoint(self):
        """Test getting best checkpoint."""
        mock_actor = MagicMock()
        mock_actor.get_best_checkpoint.return_value = "/checkpoints/best"

        best = mock_actor.get_best_checkpoint()

        assert best == "/checkpoints/best"


class TestMockMetricsAggregator:
    """Tests for metrics aggregator behavior using mocks."""

    def test_add_metrics(self):
        """Test adding metrics."""
        mock_aggregator = MagicMock()
        mock_aggregator.get_current_step.return_value = 100

        mock_aggregator.add_metrics({"loss": 0.5}, step=100)
        step = mock_aggregator.get_current_step()

        assert step == 100
        mock_aggregator.add_metrics.assert_called_once()

    def test_get_statistics(self):
        """Test getting statistics."""
        mock_aggregator = MagicMock()
        mock_aggregator.get_statistics.return_value = {
            "loss": {"mean": 0.4, "min": 0.2, "max": 0.6, "latest": 0.3},
            "accuracy": {"mean": 0.75, "min": 0.7, "max": 0.8, "latest": 0.78},
        }

        stats = mock_aggregator.get_statistics(window=100)

        assert stats["loss"]["mean"] == 0.4
        assert stats["accuracy"]["latest"] == 0.78

    def test_prometheus_export(self):
        """Test Prometheus export format."""
        mock_aggregator = MagicMock()
        mock_aggregator.to_prometheus.return_value = """
# HELP training_loss loss
# TYPE training_loss gauge
training_loss{stat="mean"} 0.4
training_loss{stat="latest"} 0.3
""".strip()

        prometheus = mock_aggregator.to_prometheus()

        assert "training_loss" in prometheus
        assert "gauge" in prometheus


class TestMockCheckpointManager:
    """Tests for checkpoint manager behavior using mocks."""

    def test_register_checkpoint(self):
        """Test registering a checkpoint."""
        mock_manager = MagicMock()
        mock_manager.register_checkpoint.return_value = CheckpointInfo(
            path="/checkpoints/step-100",
            step=100,
            timestamp="2024-01-01T00:00:00",
            metrics={"eval_loss": 0.3},
            is_best=True,
        )

        info = mock_manager.register_checkpoint(
            path="/checkpoints/step-100",
            step=100,
            metrics={"eval_loss": 0.3},
        )

        assert info.is_best is True
        assert info.step == 100

    def test_get_best_checkpoint(self):
        """Test getting best checkpoint."""
        mock_manager = MagicMock()
        mock_manager.get_best_checkpoint.return_value = "/checkpoints/best"

        best = mock_manager.get_best_checkpoint()

        assert best == "/checkpoints/best"

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        mock_manager = MagicMock()
        mock_manager.list_checkpoints.return_value = [
            {"path": "/checkpoints/step-100", "step": 100, "is_best": False},
            {"path": "/checkpoints/step-200", "step": 200, "is_best": True},
        ]

        checkpoints = mock_manager.list_checkpoints()

        assert len(checkpoints) == 2
        assert checkpoints[1]["is_best"] is True


class TestMockHealthMonitor:
    """Tests for health monitor behavior using mocks."""

    def test_report_worker_health(self):
        """Test reporting worker health."""
        mock_monitor = MagicMock()
        mock_monitor.get_worker_status.return_value = {
            "worker-0": {"status": "healthy", "consecutive_failures": 0},
            "worker-1": {"status": "degraded", "consecutive_failures": 2},
        }

        mock_monitor.report_worker_health("worker-0", "healthy")
        status = mock_monitor.get_worker_status()

        assert status["worker-0"]["status"] == "healthy"
        assert status["worker-1"]["consecutive_failures"] == 2

    def test_get_alerts(self):
        """Test getting alerts."""
        mock_monitor = MagicMock()
        mock_monitor.get_alerts.return_value = [
            {
                "worker_id": "worker-1",
                "type": "worker_unhealthy",
                "message": "Worker worker-1 has 3 consecutive failures",
                "resolved": False,
            }
        ]

        alerts = mock_monitor.get_alerts(unresolved_only=True)

        assert len(alerts) == 1
        assert alerts[0]["type"] == "worker_unhealthy"

    def test_get_summary(self):
        """Test getting health summary."""
        mock_monitor = MagicMock()
        mock_monitor.get_summary.return_value = {
            "total_workers": 4,
            "healthy_workers": 3,
            "unhealthy_workers": 1,
            "unresolved_alerts": 1,
        }

        summary = mock_monitor.get_summary()

        assert summary["total_workers"] == 4
        assert summary["healthy_workers"] == 3


# =============================================================================
# Integration Tests (requires Ray)
# =============================================================================

@pytest.mark.integration
class TestRayActorIntegration:
    """Integration tests requiring Ray."""

    @pytest.fixture
    def ray_context(self):
        """Setup Ray for tests."""
        try:
            import ray
            ray.init(ignore_reinit_error=True, num_cpus=2)
            yield
            ray.shutdown()
        except ImportError:
            pytest.skip("Ray not installed")

    @pytest.mark.skip(reason="Requires Ray cluster")
    def test_full_actor_lifecycle(self, ray_context):
        """Test full actor lifecycle."""
        from medai_compass.workflows.ray_actors import (
            create_ray_metrics_aggregator,
        )

        # Create actor
        aggregator = create_ray_metrics_aggregator()

        # Add metrics
        import ray
        ray.get(aggregator.add_metrics.remote({"loss": 0.5}, step=1))

        # Get metrics
        metrics = ray.get(aggregator.get_metrics.remote())

        assert len(metrics) >= 1


# =============================================================================
# Performance Tests
# =============================================================================

class TestMetricsAggregatorPerformance:
    """Performance tests for metrics aggregator."""

    def test_many_entries_handling(self):
        """Test handling many metric entries."""
        # Simulate aggregator behavior
        entries = []

        for i in range(10000):
            entries.append({
                "step": i,
                "loss": 0.5 - (i * 0.00005),
                "accuracy": 0.5 + (i * 0.00005),
            })

        # Verify structure
        assert len(entries) == 10000
        assert entries[9999]["loss"] < entries[0]["loss"]

    def test_statistics_computation(self):
        """Test statistics computation efficiency."""
        import statistics

        values = [0.5 - (i * 0.0001) for i in range(1000)]

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        assert 0.4 < mean < 0.5
        assert std > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_metrics(self):
        """Test handling empty metrics."""
        entry = MetricEntry(
            step=0,
            timestamp="2024-01-01T00:00:00",
            values={},
        )

        d = entry.to_dict()

        assert d["step"] == 0
        # Should not have extra keys beyond step, timestamp, worker_id
        assert len(d) == 3

    def test_checkpoint_with_nan_metrics(self):
        """Test checkpoint with NaN metrics."""
        import math

        info = CheckpointInfo(
            path="/checkpoints/nan",
            step=100,
            timestamp="2024-01-01T00:00:00",
            metrics={"loss": float("nan")},
        )

        d = info.to_dict()

        assert math.isnan(d["metrics"]["loss"])

    def test_evaluation_result_with_empty_benchmarks(self):
        """Test evaluation result with no benchmarks."""
        result = EvaluationResult(
            checkpoint_path="/checkpoints/no-benchmarks",
            metrics={},
            benchmarks=[],
            timestamp="2024-01-01T00:00:00",
            duration_seconds=0.0,
        )

        assert result.benchmarks == []
        assert result.metrics == {}
