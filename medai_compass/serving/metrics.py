"""
Serving metrics collection.

Provides metrics collection for model serving using Prometheus-compatible format.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


# Default metric names
REQUESTS_TOTAL = "medgemma_requests_total"
LATENCY_HISTOGRAM = "medgemma_latency_seconds"
TOKENS_GENERATED = "medgemma_tokens_generated_total"
ERRORS_TOTAL = "medgemma_errors_total"
MODEL_LOADED = "medgemma_model_loaded"


@dataclass
class MetricValue:
    """A single metric value with labels."""
    
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


class ServingMetricsCollector:
    """Metrics collector for model serving.
    
    Collects and exposes metrics in Prometheus format for monitoring
    model serving performance.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._metrics: Dict[str, List[MetricValue]] = {
            "request_count": [],
            "latency": [],
            "tokens": [],
            "errors": [],
        }
        self._counters: Dict[str, float] = {
            "requests_total": 0,
            "tokens_generated": 0,
            "errors_total": 0,
        }
        self._histograms: Dict[str, List[float]] = {
            "latency_ms": [],
        }
    
    def record_request(
        self,
        model_name: str,
        status: str = "success",
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a request.
        
        Args:
            model_name: Name of the model.
            status: Request status (success/error).
            labels: Additional labels.
        """
        self._counters["requests_total"] += 1
        
        metric_labels = {"model": model_name, "status": status}
        if labels:
            metric_labels.update(labels)
        
        self._metrics["request_count"].append(
            MetricValue(
                name=REQUESTS_TOTAL,
                value=1,
                labels=metric_labels,
            )
        )
    
    def record_latency(
        self,
        model_name: str,
        latency_ms: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record request latency.
        
        Args:
            model_name: Name of the model.
            latency_ms: Latency in milliseconds.
            labels: Additional labels.
        """
        self._histograms["latency_ms"].append(latency_ms)
        
        metric_labels = {"model": model_name}
        if labels:
            metric_labels.update(labels)
        
        self._metrics["latency"].append(
            MetricValue(
                name=LATENCY_HISTOGRAM,
                value=latency_ms / 1000.0,  # Convert to seconds
                labels=metric_labels,
            )
        )
    
    def record_tokens(
        self,
        model_name: str,
        tokens: int,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record tokens generated.
        
        Args:
            model_name: Name of the model.
            tokens: Number of tokens generated.
            labels: Additional labels.
        """
        self._counters["tokens_generated"] += tokens
        
        metric_labels = {"model": model_name}
        if labels:
            metric_labels.update(labels)
        
        self._metrics["tokens"].append(
            MetricValue(
                name=TOKENS_GENERATED,
                value=tokens,
                labels=metric_labels,
            )
        )
    
    def record_error(
        self,
        model_name: str,
        error_type: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an error.
        
        Args:
            model_name: Name of the model.
            error_type: Type of error.
            labels: Additional labels.
        """
        self._counters["errors_total"] += 1
        
        metric_labels = {"model": model_name, "error_type": error_type}
        if labels:
            metric_labels.update(labels)
        
        self._metrics["errors"].append(
            MetricValue(
                name=ERRORS_TOTAL,
                value=1,
                labels=metric_labels,
            )
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of current metrics.
        """
        latency_values = self._histograms.get("latency_ms", [])
        
        return {
            "request_count": self._counters["requests_total"],
            "requests_total": self._counters["requests_total"],
            "tokens_generated": self._counters["tokens_generated"],
            "tokens": self._counters["tokens_generated"],
            "errors_total": self._counters["errors_total"],
            "latency": {
                "count": len(latency_values),
                "mean_ms": sum(latency_values) / len(latency_values) if latency_values else 0,
                "max_ms": max(latency_values) if latency_values else 0,
                "min_ms": min(latency_values) if latency_values else 0,
            },
            "latency_ms": latency_values,
        }
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles.
        
        Returns:
            Dictionary with p50, p95, p99 latency values.
        """
        latency_values = sorted(self._histograms.get("latency_ms", []))
        
        if not latency_values:
            return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
        
        def percentile(values: List[float], p: float) -> float:
            """Calculate percentile."""
            k = (len(values) - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < len(values) else f
            return values[f] + (values[c] - values[f]) * (k - f)
        
        return {
            "p50_ms": round(percentile(latency_values, 0.50), 2),
            "p95_ms": round(percentile(latency_values, 0.95), 2),
            "p99_ms": round(percentile(latency_values, 0.99), 2),
        }
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text exposition format.
        """
        lines = []
        
        # Requests total
        lines.append(f"# HELP {REQUESTS_TOTAL} Total number of requests")
        lines.append(f"# TYPE {REQUESTS_TOTAL} counter")
        lines.append(f'{REQUESTS_TOTAL} {self._counters["requests_total"]}')
        
        # Tokens generated
        lines.append(f"# HELP {TOKENS_GENERATED} Total tokens generated")
        lines.append(f"# TYPE {TOKENS_GENERATED} counter")
        lines.append(f'{TOKENS_GENERATED} {self._counters["tokens_generated"]}')
        
        # Errors total
        lines.append(f"# HELP {ERRORS_TOTAL} Total number of errors")
        lines.append(f"# TYPE {ERRORS_TOTAL} counter")
        lines.append(f'{ERRORS_TOTAL} {self._counters["errors_total"]}')
        
        # Latency histogram
        percentiles = self.get_latency_percentiles()
        lines.append(f"# HELP {LATENCY_HISTOGRAM} Request latency in seconds")
        lines.append(f"# TYPE {LATENCY_HISTOGRAM} summary")
        lines.append(f'{LATENCY_HISTOGRAM}{{quantile="0.5"}} {percentiles["p50_ms"] / 1000}')
        lines.append(f'{LATENCY_HISTOGRAM}{{quantile="0.95"}} {percentiles["p95_ms"] / 1000}')
        lines.append(f'{LATENCY_HISTOGRAM}{{quantile="0.99"}} {percentiles["p99_ms"] / 1000}')
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "request_count": [],
            "latency": [],
            "tokens": [],
            "errors": [],
        }
        self._counters = {
            "requests_total": 0,
            "tokens_generated": 0,
            "errors_total": 0,
        }
        self._histograms = {
            "latency_ms": [],
        }


# Global metrics collector instance
_global_collector: Optional[ServingMetricsCollector] = None


def get_metrics_collector() -> ServingMetricsCollector:
    """Get or create the global metrics collector.
    
    Returns:
        The global ServingMetricsCollector instance.
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = ServingMetricsCollector()
    return _global_collector
