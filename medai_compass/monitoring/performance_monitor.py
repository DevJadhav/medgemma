"""
Performance Monitoring Module (Phase 8: Monitoring & Observability).

Provides comprehensive performance monitoring including latency tracking,
throughput monitoring, resource utilization, and quality metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from prometheus_client import Counter, Gauge, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring.
    
    Attributes:
        model_name: Name of the model being monitored
        latency_p95_threshold_ms: P95 latency threshold in milliseconds
        latency_p99_threshold_ms: P99 latency threshold in milliseconds
        min_throughput_rps: Minimum throughput in requests per second
        collection_interval_seconds: Interval for metric collection
    """
    
    model_name: str = "medgemma-27b-it"
    latency_p95_threshold_ms: float = 500
    latency_p99_threshold_ms: float = 1000
    min_throughput_rps: float = 10
    collection_interval_seconds: int = 60
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0


# ============================================================================
# Model Baselines
# ============================================================================

def get_model_baseline(model_name: str) -> dict[str, float]:
    """Get performance baseline for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of baseline performance metrics
    """
    baselines = {
        "medgemma-27b-it": {
            "latency_p50_ms": 200,
            "latency_p95_ms": 500,
            "latency_p99_ms": 1000,
            "throughput_rps": 10,
        },
        "medgemma-4b-it": {
            "latency_p50_ms": 100,
            "latency_p95_ms": 250,
            "latency_p99_ms": 500,
            "throughput_rps": 25,
        },
    }
    
    return baselines.get(model_name, baselines["medgemma-27b-it"])


# ============================================================================
# Latency Tracker
# ============================================================================

class LatencyTracker:
    """Tracks request latencies with percentile calculations.
    
    Uses a sliding window to maintain recent latency measurements
    and calculate statistics.
    """
    
    def __init__(self, window_size: Optional[int] = None):
        """Initialize latency tracker.
        
        Args:
            window_size: Maximum number of latencies to keep (None for unlimited)
        """
        self.latencies: list[float] = []
        self.window_size = window_size
    
    def record(self, latency_ms: float) -> None:
        """Record a latency measurement.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.latencies.append(latency_ms)
        
        # Apply sliding window if configured
        if self.window_size and len(self.latencies) > self.window_size:
            self.latencies = self.latencies[-self.window_size:]
    
    def get_percentile(self, percentile: float) -> float:
        """Calculate a specific percentile.
        
        Args:
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Latency at the given percentile
        """
        if not self.latencies:
            return 0.0
        
        return float(np.percentile(self.latencies, percentile))
    
    def get_stats(self) -> dict[str, float]:
        """Get comprehensive latency statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.latencies:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        
        arr = np.array(self.latencies)
        return {
            "count": len(self.latencies),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    
    def clear(self) -> None:
        """Clear all recorded latencies."""
        self.latencies = []


# ============================================================================
# Throughput Monitor
# ============================================================================

class ThroughputMonitor:
    """Monitors request throughput (requests per second)."""
    
    def __init__(self):
        """Initialize throughput monitor."""
        self.request_count: int = 0
        self.window_start: datetime = datetime.now()
    
    def increment(self) -> None:
        """Increment the request counter."""
        self.request_count += 1
    
    def get_rps(self) -> float:
        """Calculate current requests per second.
        
        Returns:
            Requests per second
        """
        elapsed = (datetime.now() - self.window_start).total_seconds()
        if elapsed <= 0:
            return 0.0
        
        return self.request_count / elapsed
    
    def reset(self) -> None:
        """Reset the counter and window."""
        self.request_count = 0
        self.window_start = datetime.now()


# ============================================================================
# Resource Monitor
# ============================================================================

class ResourceMonitor:
    """Monitors system resource utilization."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self._gpu_available = self._check_gpu_available()
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage.
        
        Returns:
            CPU usage percentage
        """
        if not HAS_PSUTIL:
            return 0.0
        
        return psutil.cpu_percent()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage.
        
        Returns:
            Memory usage percentage
        """
        if not HAS_PSUTIL:
            return 0.0
        
        return psutil.virtual_memory().percent
    
    def get_gpu_usage(self) -> dict[str, dict[str, float]]:
        """Get GPU usage statistics.
        
        Returns:
            Dictionary of GPU statistics per device
        """
        return self._get_gpu_stats()
    
    def _get_gpu_stats(self) -> dict[str, dict[str, float]]:
        """Get GPU statistics (internal implementation).
        
        Returns:
            GPU statistics dictionary
        """
        if not self._gpu_available:
            return {}
        
        try:
            import torch
            stats = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                stats[f"gpu_{i}"] = {
                    "utilization": 0.0,  # Would need nvidia-smi for true utilization
                    "memory": (allocated / total) * 100 if total > 0 else 0.0,
                }
            return stats
        except Exception:
            return {}
    
    @staticmethod
    def _check_gpu_available() -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_all(self) -> dict[str, Any]:
        """Get all resource metrics.
        
        Returns:
            Dictionary of all resource metrics
        """
        return {
            "cpu_percent": self.get_cpu_usage(),
            "memory_percent": self.get_memory_usage(),
            "gpu": self.get_gpu_usage(),
        }


# ============================================================================
# Quality Metrics Tracker
# ============================================================================

class QualityMetricsTracker:
    """Tracks model quality metrics over time."""
    
    def __init__(
        self,
        model_name: str,
        min_accuracy: float = 0.75,
        window_size: int = 100,
    ):
        """Initialize quality tracker.
        
        Args:
            model_name: Name of the model
            min_accuracy: Minimum acceptable accuracy
            window_size: Size of sliding window
        """
        self.model_name = model_name
        self.min_accuracy = min_accuracy
        self.window_size = window_size
        self.accuracy_scores: list[float] = []
    
    def record_accuracy(self, accuracy: float) -> None:
        """Record an accuracy measurement.
        
        Args:
            accuracy: Accuracy score (0-1)
        """
        self.accuracy_scores.append(accuracy)
        
        if len(self.accuracy_scores) > self.window_size:
            self.accuracy_scores = self.accuracy_scores[-self.window_size:]
    
    def is_below_threshold(self) -> bool:
        """Check if recent accuracy is below threshold.
        
        Returns:
            True if below threshold
        """
        if not self.accuracy_scores:
            return False
        
        # Check most recent accuracy
        return self.accuracy_scores[-1] < self.min_accuracy
    
    def get_summary(self) -> dict[str, float]:
        """Get quality metrics summary.
        
        Returns:
            Summary dictionary
        """
        if not self.accuracy_scores:
            return {
                "mean_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "count": 0,
            }
        
        arr = np.array(self.accuracy_scores)
        return {
            "mean_accuracy": float(np.mean(arr)),
            "min_accuracy": float(np.min(arr)),
            "max_accuracy": float(np.max(arr)),
            "count": len(self.accuracy_scores),
        }


# ============================================================================
# Prometheus Metrics Exporter
# ============================================================================

# Singleton metrics to avoid duplicate registration
_PROMETHEUS_METRICS: dict = {}


class PrometheusMetricsExporter:
    """Exports metrics to Prometheus format."""
    
    def __init__(self, namespace: str = "medai"):
        """Initialize Prometheus exporter.
        
        Args:
            namespace: Prometheus metric namespace
        """
        self.namespace = namespace
        
        if HAS_PROMETHEUS:
            # Use singleton pattern to avoid duplicate registration
            global _PROMETHEUS_METRICS
            
            latency_key = f"{namespace}_request_latency_ms"
            if latency_key not in _PROMETHEUS_METRICS:
                _PROMETHEUS_METRICS[latency_key] = Histogram(
                    latency_key,
                    "Request latency in milliseconds",
                    ["model", "endpoint"],
                    buckets=[50, 100, 200, 500, 1000, 2000, 5000],
                )
            self.latency_histogram = _PROMETHEUS_METRICS[latency_key]
            
            counter_key = f"{namespace}_requests_total"
            if counter_key not in _PROMETHEUS_METRICS:
                _PROMETHEUS_METRICS[counter_key] = Counter(
                    counter_key,
                    "Total number of requests",
                    ["model", "status"],
                )
            self.request_counter = _PROMETHEUS_METRICS[counter_key]
            
            gauge_key = f"{namespace}_active_requests"
            if gauge_key not in _PROMETHEUS_METRICS:
                _PROMETHEUS_METRICS[gauge_key] = Gauge(
                    gauge_key,
                    "Number of active requests",
                    ["model"],
                )
            self.active_requests_gauge = _PROMETHEUS_METRICS[gauge_key]
        else:
            self.latency_histogram = None
            self.request_counter = None
            self.active_requests_gauge = None
    
    def record_latency(
        self,
        latency_ms: float,
        model: str,
        endpoint: str,
    ) -> None:
        """Record a latency measurement.
        
        Args:
            latency_ms: Latency in milliseconds
            model: Model name
            endpoint: API endpoint
        """
        if self.latency_histogram:
            self.latency_histogram.labels(model=model, endpoint=endpoint).observe(latency_ms)
    
    def increment_requests(self, model: str, status: str) -> None:
        """Increment request counter.
        
        Args:
            model: Model name
            status: Request status (success, error, etc.)
        """
        if self.request_counter:
            self.request_counter.labels(model=model, status=status).inc()


# ============================================================================
# Performance Monitor (Unified Interface)
# ============================================================================

class PerformanceMonitor:
    """Unified performance monitoring interface.
    
    Combines latency tracking, throughput monitoring, resource monitoring,
    and quality metrics into a single interface.
    """
    
    def __init__(self, config: PerformanceConfig):
        """Initialize performance monitor.
        
        Args:
            config: Performance monitoring configuration
        """
        self.config = config
        self.latency_tracker = LatencyTracker(window_size=10000)
        self.throughput_monitor = ThroughputMonitor()
        self.resource_monitor = ResourceMonitor()
        self.quality_tracker = QualityMetricsTracker(
            model_name=config.model_name,
            min_accuracy=0.75,
        )
        self.prometheus_exporter = PrometheusMetricsExporter()
    
    def record_request(
        self,
        latency_ms: float,
        endpoint: str = "/predict",
        status: str = "success",
    ) -> None:
        """Record a request.
        
        Args:
            latency_ms: Request latency in milliseconds
            endpoint: API endpoint
            status: Request status
        """
        self.latency_tracker.record(latency_ms)
        self.throughput_monitor.increment()
        self.prometheus_exporter.record_latency(
            latency_ms, self.config.model_name, endpoint
        )
        self.prometheus_exporter.increment_requests(self.config.model_name, status)
    
    def check_sla_compliance(self) -> dict[str, Any]:
        """Check SLA compliance.
        
        Returns:
            SLA compliance status
        """
        stats = self.latency_tracker.get_stats()
        rps = self.throughput_monitor.get_rps()
        
        p95_compliant = stats["p95"] <= self.config.latency_p95_threshold_ms
        throughput_compliant = rps >= self.config.min_throughput_rps
        
        return {
            "latency_compliant": p95_compliant,
            "throughput_compliant": throughput_compliant,
            "overall_compliant": p95_compliant and throughput_compliant,
            "p95_latency_ms": stats["p95"],
            "p95_threshold_ms": self.config.latency_p95_threshold_ms,
            "current_rps": rps,
            "min_rps": self.config.min_throughput_rps,
        }
    
    def compare_to_baseline(self) -> dict[str, Any]:
        """Compare current performance to model baseline.
        
        Returns:
            Comparison results
        """
        baseline = get_model_baseline(self.config.model_name)
        current_stats = self.latency_tracker.get_stats()
        
        if current_stats["count"] == 0:
            return {
                "latency_diff_percent": 0.0,
                "meets_baseline": True,
                "message": "No data collected yet",
            }
        
        baseline_p95 = baseline["latency_p95_ms"]
        current_p95 = current_stats["p95"]
        
        diff_percent = ((current_p95 - baseline_p95) / baseline_p95) * 100
        meets_baseline = current_p95 <= baseline_p95 * 1.1  # Allow 10% margin
        
        return {
            "latency_diff_percent": diff_percent,
            "meets_baseline": meets_baseline,
            "baseline_p95_ms": baseline_p95,
            "current_p95_ms": current_p95,
        }
    
    def get_report(self) -> dict[str, Any]:
        """Get comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        return {
            "model_name": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "latency": self.latency_tracker.get_stats(),
            "throughput": {
                "rps": self.throughput_monitor.get_rps(),
                "total_requests": self.throughput_monitor.request_count,
            },
            "resources": self.resource_monitor.get_all(),
            "sla_compliance": self.check_sla_compliance(),
        }


# ============================================================================
# Async Performance Monitor
# ============================================================================

class AsyncPerformanceMonitor(PerformanceMonitor):
    """Async-capable performance monitor."""
    
    async def record_request_async(
        self,
        latency_ms: float,
        endpoint: str = "/predict",
        status: str = "success",
    ) -> None:
        """Record a request asynchronously.
        
        Args:
            latency_ms: Request latency
            endpoint: API endpoint
            status: Request status
        """
        self.record_request(latency_ms, endpoint, status)
    
    @asynccontextmanager
    async def measure_latency(self, endpoint: str = "/predict"):
        """Context manager to measure request latency.
        
        Args:
            endpoint: API endpoint being measured
            
        Yields:
            None
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            await self.record_request_async(latency_ms, endpoint)
