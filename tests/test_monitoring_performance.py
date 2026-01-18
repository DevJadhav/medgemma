"""
Tests for performance monitoring module (Phase 8: Monitoring & Observability).

TDD tests for latency tracking, throughput monitoring, resource utilization,
and quality metrics tracking.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio


# ============================================================================
# Test: Performance Metrics Configuration
# ============================================================================

class TestPerformanceConfig:
    """Test performance monitoring configuration."""
    
    def test_performance_config_defaults(self):
        """Test default performance configuration."""
        from medai_compass.monitoring.performance_monitor import PerformanceConfig
        
        config = PerformanceConfig()
        
        assert config.model_name == "medgemma-27b-it"
        assert config.latency_p95_threshold_ms == 500
        assert config.latency_p99_threshold_ms == 1000
        assert config.min_throughput_rps == 10
        assert config.collection_interval_seconds == 60
    
    def test_performance_config_4b_model(self):
        """Test configuration for 4B model (faster)."""
        from medai_compass.monitoring.performance_monitor import PerformanceConfig
        
        config = PerformanceConfig(
            model_name="medgemma-4b-it",
            latency_p95_threshold_ms=250  # Faster model
        )
        
        assert config.model_name == "medgemma-4b-it"
        assert config.latency_p95_threshold_ms == 250


# ============================================================================
# Test: Latency Tracker
# ============================================================================

class TestLatencyTracker:
    """Test latency tracking functionality."""
    
    def test_latency_tracker_initialization(self):
        """Test latency tracker initialization."""
        from medai_compass.monitoring.performance_monitor import LatencyTracker
        
        tracker = LatencyTracker()
        
        assert tracker.latencies == []
    
    def test_record_latency(self):
        """Test recording latency measurements."""
        from medai_compass.monitoring.performance_monitor import LatencyTracker
        
        tracker = LatencyTracker()
        
        tracker.record(100.5)
        tracker.record(150.3)
        tracker.record(120.7)
        
        assert len(tracker.latencies) == 3
    
    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        from medai_compass.monitoring.performance_monitor import LatencyTracker
        
        tracker = LatencyTracker()
        
        # Add 100 latencies
        for i in range(100):
            tracker.record(float(i + 1))
        
        p50 = tracker.get_percentile(50)
        p95 = tracker.get_percentile(95)
        p99 = tracker.get_percentile(99)
        
        assert 48 <= p50 <= 52
        assert 93 <= p95 <= 97
        assert 98 <= p99 <= 100
    
    def test_get_stats(self):
        """Test getting latency statistics."""
        from medai_compass.monitoring.performance_monitor import LatencyTracker
        
        tracker = LatencyTracker()
        
        latencies = [100, 150, 120, 180, 90]
        for l in latencies:
            tracker.record(float(l))
        
        stats = tracker.get_stats()
        
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(128.0, rel=0.01)
        assert stats["min"] == 90.0
        assert stats["max"] == 180.0
    
    def test_latency_sliding_window(self):
        """Test sliding window for latency tracking."""
        from medai_compass.monitoring.performance_monitor import LatencyTracker
        
        tracker = LatencyTracker(window_size=5)
        
        for i in range(10):
            tracker.record(float(i * 10))
        
        # Should only keep last 5
        assert len(tracker.latencies) == 5
        assert tracker.latencies[0] == 50.0


# ============================================================================
# Test: Throughput Monitor
# ============================================================================

class TestThroughputMonitor:
    """Test throughput monitoring functionality."""
    
    def test_throughput_monitor_initialization(self):
        """Test throughput monitor initialization."""
        from medai_compass.monitoring.performance_monitor import ThroughputMonitor
        
        monitor = ThroughputMonitor()
        
        assert monitor.request_count == 0
    
    def test_increment_request_count(self):
        """Test incrementing request count."""
        from medai_compass.monitoring.performance_monitor import ThroughputMonitor
        
        monitor = ThroughputMonitor()
        
        monitor.increment()
        monitor.increment()
        monitor.increment()
        
        assert monitor.request_count == 3
    
    def test_calculate_rps(self):
        """Test calculating requests per second."""
        from medai_compass.monitoring.performance_monitor import ThroughputMonitor
        
        monitor = ThroughputMonitor()
        
        # Simulate 100 requests over 10 seconds
        monitor.request_count = 100
        monitor.window_start = datetime.now() - timedelta(seconds=10)
        
        rps = monitor.get_rps()
        
        assert rps == pytest.approx(10.0, rel=0.1)
    
    def test_reset_window(self):
        """Test resetting the monitoring window."""
        from medai_compass.monitoring.performance_monitor import ThroughputMonitor
        
        monitor = ThroughputMonitor()
        
        monitor.request_count = 50
        monitor.reset()
        
        assert monitor.request_count == 0


# ============================================================================
# Test: Resource Monitor
# ============================================================================

class TestResourceMonitor:
    """Test resource utilization monitoring."""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        from medai_compass.monitoring.performance_monitor import ResourceMonitor
        
        monitor = ResourceMonitor()
        
        assert monitor is not None
    
    @patch('psutil.cpu_percent')
    def test_get_cpu_usage(self, mock_cpu):
        """Test getting CPU usage."""
        from medai_compass.monitoring.performance_monitor import ResourceMonitor
        
        mock_cpu.return_value = 45.5
        
        monitor = ResourceMonitor()
        cpu = monitor.get_cpu_usage()
        
        assert cpu == 45.5
    
    @patch('psutil.virtual_memory')
    def test_get_memory_usage(self, mock_memory):
        """Test getting memory usage."""
        from medai_compass.monitoring.performance_monitor import ResourceMonitor
        
        mock_memory.return_value = Mock(percent=65.3)
        
        monitor = ResourceMonitor()
        memory = monitor.get_memory_usage()
        
        assert memory == 65.3
    
    def test_get_gpu_usage(self):
        """Test getting GPU usage (mocked)."""
        from medai_compass.monitoring.performance_monitor import ResourceMonitor
        
        monitor = ResourceMonitor()
        
        with patch.object(monitor, '_get_gpu_stats') as mock_gpu:
            mock_gpu.return_value = {"gpu_0": {"utilization": 80.0, "memory": 75.5}}
            
            gpu_stats = monitor.get_gpu_usage()
            
            assert "gpu_0" in gpu_stats


# ============================================================================
# Test: Quality Metrics Tracker
# ============================================================================

class TestQualityMetricsTracker:
    """Test quality metrics tracking."""
    
    def test_quality_tracker_initialization(self):
        """Test quality tracker initialization."""
        from medai_compass.monitoring.performance_monitor import QualityMetricsTracker
        
        tracker = QualityMetricsTracker(model_name="medgemma-27b-it")
        
        assert tracker.model_name == "medgemma-27b-it"
    
    def test_record_accuracy(self):
        """Test recording accuracy metrics."""
        from medai_compass.monitoring.performance_monitor import QualityMetricsTracker
        
        tracker = QualityMetricsTracker(model_name="medgemma-27b-it")
        
        tracker.record_accuracy(0.85)
        tracker.record_accuracy(0.87)
        tracker.record_accuracy(0.82)
        
        assert len(tracker.accuracy_scores) == 3
    
    def test_check_quality_thresholds(self):
        """Test checking quality thresholds."""
        from medai_compass.monitoring.performance_monitor import QualityMetricsTracker
        
        tracker = QualityMetricsTracker(
            model_name="medgemma-27b-it",
            min_accuracy=0.75  # MedQA threshold
        )
        
        # Below threshold
        tracker.record_accuracy(0.70)
        
        assert tracker.is_below_threshold() is True
    
    def test_get_quality_summary(self):
        """Test getting quality summary."""
        from medai_compass.monitoring.performance_monitor import QualityMetricsTracker
        
        tracker = QualityMetricsTracker(model_name="medgemma-27b-it")
        
        for score in [0.80, 0.85, 0.82, 0.88, 0.86]:
            tracker.record_accuracy(score)
        
        summary = tracker.get_summary()
        
        assert "mean_accuracy" in summary
        assert "min_accuracy" in summary
        assert "max_accuracy" in summary


# ============================================================================
# Test: Performance Monitor (Unified Interface)
# ============================================================================

class TestPerformanceMonitor:
    """Test unified performance monitoring."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        from medai_compass.monitoring.performance_monitor import (
            PerformanceMonitor,
            PerformanceConfig
        )
        
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)
        
        assert monitor.latency_tracker is not None
        assert monitor.throughput_monitor is not None
        assert monitor.resource_monitor is not None
    
    def test_record_request(self):
        """Test recording a request."""
        from medai_compass.monitoring.performance_monitor import (
            PerformanceMonitor,
            PerformanceConfig
        )
        
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)
        
        monitor.record_request(latency_ms=150.5)
        
        assert monitor.throughput_monitor.request_count == 1
        assert len(monitor.latency_tracker.latencies) == 1
    
    def test_check_sla_compliance(self):
        """Test SLA compliance checking."""
        from medai_compass.monitoring.performance_monitor import (
            PerformanceMonitor,
            PerformanceConfig
        )
        
        config = PerformanceConfig(
            latency_p95_threshold_ms=200,
            min_throughput_rps=5
        )
        monitor = PerformanceMonitor(config)
        
        # Record requests that meet SLA
        for _ in range(100):
            monitor.record_request(latency_ms=150.0)
        
        # Force time window for RPS calculation
        monitor.throughput_monitor.window_start = datetime.now() - timedelta(seconds=10)
        
        compliance = monitor.check_sla_compliance()
        
        assert "latency_compliant" in compliance
        assert "throughput_compliant" in compliance
    
    def test_get_performance_report(self):
        """Test getting performance report."""
        from medai_compass.monitoring.performance_monitor import (
            PerformanceMonitor,
            PerformanceConfig
        )
        
        config = PerformanceConfig(model_name="medgemma-27b-it")
        monitor = PerformanceMonitor(config)
        
        for latency in [100, 150, 120, 200, 180]:
            monitor.record_request(latency_ms=float(latency))
        
        report = monitor.get_report()
        
        assert "model_name" in report
        assert "latency" in report
        assert "throughput" in report
        assert "timestamp" in report


# ============================================================================
# Test: Prometheus Integration
# ============================================================================

class TestPrometheusIntegration:
    """Test Prometheus metrics integration."""
    
    def test_create_prometheus_metrics(self):
        """Test creating Prometheus metrics."""
        from medai_compass.monitoring.performance_monitor import PrometheusMetricsExporter
        
        exporter = PrometheusMetricsExporter(namespace="medai")
        
        assert exporter.latency_histogram is not None
        assert exporter.request_counter is not None
        assert exporter.active_requests_gauge is not None
    
    def test_record_latency_metric(self):
        """Test recording latency to Prometheus."""
        from medai_compass.monitoring.performance_monitor import PrometheusMetricsExporter
        
        exporter = PrometheusMetricsExporter(namespace="medai")
        
        # Should not raise
        exporter.record_latency(
            latency_ms=150.0,
            model="medgemma-27b-it",
            endpoint="/predict"
        )
    
    def test_increment_request_counter(self):
        """Test incrementing request counter."""
        from medai_compass.monitoring.performance_monitor import PrometheusMetricsExporter
        
        exporter = PrometheusMetricsExporter(namespace="medai")
        
        exporter.increment_requests(
            model="medgemma-27b-it",
            status="success"
        )


# ============================================================================
# Test: Model-Specific Performance Baselines
# ============================================================================

class TestModelPerformanceBaselines:
    """Test model-specific performance baselines."""
    
    def test_get_27b_baseline(self):
        """Test getting 27B model performance baseline."""
        from medai_compass.monitoring.performance_monitor import get_model_baseline
        
        baseline = get_model_baseline("medgemma-27b-it")
        
        assert baseline["latency_p50_ms"] > 0
        assert baseline["latency_p95_ms"] > 0
        assert baseline["throughput_rps"] > 0
    
    def test_get_4b_baseline(self):
        """Test getting 4B model performance baseline."""
        from medai_compass.monitoring.performance_monitor import get_model_baseline
        
        baseline_4b = get_model_baseline("medgemma-4b-it")
        baseline_27b = get_model_baseline("medgemma-27b-it")
        
        # 4B should be faster
        assert baseline_4b["latency_p50_ms"] < baseline_27b["latency_p50_ms"]
    
    def test_compare_to_baseline(self):
        """Test comparing current performance to baseline."""
        from medai_compass.monitoring.performance_monitor import (
            PerformanceMonitor,
            PerformanceConfig
        )
        
        config = PerformanceConfig(model_name="medgemma-27b-it")
        monitor = PerformanceMonitor(config)
        
        for latency in [100, 150, 120, 200, 180] * 20:
            monitor.record_request(latency_ms=float(latency))
        
        comparison = monitor.compare_to_baseline()
        
        assert "latency_diff_percent" in comparison
        assert "meets_baseline" in comparison


# ============================================================================
# Test: Async Performance Monitoring
# ============================================================================

class TestAsyncPerformanceMonitoring:
    """Test async performance monitoring capabilities."""
    
    @pytest.mark.asyncio
    async def test_async_record_request(self):
        """Test async request recording."""
        from medai_compass.monitoring.performance_monitor import (
            AsyncPerformanceMonitor,
            PerformanceConfig
        )
        
        config = PerformanceConfig()
        monitor = AsyncPerformanceMonitor(config)
        
        await monitor.record_request_async(latency_ms=150.0)
        
        assert monitor.throughput_monitor.request_count == 1
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager for timing."""
        from medai_compass.monitoring.performance_monitor import (
            AsyncPerformanceMonitor,
            PerformanceConfig
        )
        
        config = PerformanceConfig()
        monitor = AsyncPerformanceMonitor(config)
        
        async with monitor.measure_latency():
            await asyncio.sleep(0.01)
        
        assert len(monitor.latency_tracker.latencies) == 1
        assert monitor.latency_tracker.latencies[0] >= 10  # At least 10ms
