"""
Core benchmarking infrastructure for MedAI Compass.

Provides base classes and utilities for performance benchmarking.
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class SLATargets:
    """
    SLA targets for MedAI Compass.
    
    Based on implementation_plan.md quality gates:
    - Latency p95 ≤ 500ms
    """
    
    latency_p95_ms: float = 500.0
    latency_p99_ms: float = 1000.0
    latency_mean_ms: float = 300.0
    min_throughput_rps: float = 10.0
    min_success_rate: float = 0.99
    max_error_rate: float = 0.01
    
    def __post_init__(self):
        """Validate SLA targets."""
        if self.latency_p95_ms <= 0:
            raise ValueError("latency_p95_ms must be positive")
        if self.min_success_rate < 0 or self.min_success_rate > 1:
            raise ValueError("min_success_rate must be between 0 and 1")


def get_model_benchmark_targets(model_name: str) -> SLATargets:
    """
    Get benchmark targets for a specific model.
    
    Args:
        model_name: Model name (medgemma-4b-it, medgemma-27b-it)
        
    Returns:
        SLATargets for the model
    """
    if "4b" in model_name.lower():
        # 4B model should be faster
        return SLATargets(
            latency_p95_ms=400.0,
            latency_p99_ms=800.0,
            latency_mean_ms=200.0,
            min_throughput_rps=20.0,
        )
    elif "27b" in model_name.lower():
        # 27B model has relaxed targets
        return SLATargets(
            latency_p95_ms=1000.0,
            latency_p99_ms=2000.0,
            latency_mean_ms=500.0,
            min_throughput_rps=5.0,
        )
    else:
        # Default targets
        return SLATargets()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    
    # General
    warmup_iterations: int = 5
    benchmark_iterations: int = 100
    timeout_seconds: int = 300
    
    # Model
    model_name: Optional[str] = None
    
    # Inclusions
    include_inference: bool = True
    include_throughput: bool = True
    include_pipeline: bool = True
    include_resources: bool = True
    
    @classmethod
    def smoke_test(cls) -> "BenchmarkConfig":
        """Quick smoke test configuration."""
        return cls(
            warmup_iterations=1,
            benchmark_iterations=10,
            timeout_seconds=30,
        )
    
    @classmethod
    def ci_config(cls) -> "BenchmarkConfig":
        """CI-friendly configuration."""
        return cls(
            warmup_iterations=2,
            benchmark_iterations=20,
            timeout_seconds=60,
        )
    
    @classmethod
    def full_test(cls) -> "BenchmarkConfig":
        """Full benchmark configuration."""
        return cls(
            warmup_iterations=10,
            benchmark_iterations=500,
            timeout_seconds=600,
        )


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    name: str
    iterations: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    success_rate: float
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_latencies(
        cls,
        name: str,
        latencies: List[float],
        errors: int = 0,
        duration_seconds: float = 0,
    ) -> "BenchmarkResult":
        """
        Create result from raw latency measurements.
        
        Args:
            name: Benchmark name
            latencies: List of latency measurements in ms
            errors: Number of errors
            duration_seconds: Total duration
            
        Returns:
            BenchmarkResult
        """
        if not latencies:
            return cls(
                name=name,
                iterations=0,
                mean_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                throughput_rps=0,
                success_rate=0,
            )
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        # Calculate percentiles
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        total_requests = n + errors
        success_rate = n / total_requests if total_requests > 0 else 0
        throughput = n / duration_seconds if duration_seconds > 0 else 0
        
        return cls(
            name=name,
            iterations=n,
            mean_latency_ms=statistics.mean(latencies),
            p50_latency_ms=sorted_latencies[min(p50_idx, n - 1)],
            p95_latency_ms=sorted_latencies[min(p95_idx, n - 1)],
            p99_latency_ms=sorted_latencies[min(p99_idx, n - 1)],
            min_latency_ms=sorted_latencies[0],
            max_latency_ms=sorted_latencies[-1],
            throughput_rps=throughput,
            success_rate=success_rate,
        )
    
    def passes_sla(self, targets: SLATargets) -> bool:
        """
        Check if result passes SLA targets.
        
        Args:
            targets: SLA targets to check against
            
        Returns:
            True if all targets are met
        """
        return (
            self.p95_latency_ms <= targets.latency_p95_ms and
            self.success_rate >= targets.min_success_rate
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "throughput_rps": self.throughput_rps,
            "success_rate": self.success_rate,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkReport:
    """Collection of benchmark results."""
    
    results: List[BenchmarkResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = ["Benchmark Report", "=" * 50]
        
        for result in self.results:
            lines.append(f"\n{result.name}:")
            lines.append(f"  Iterations: {result.iterations}")
            lines.append(f"  Mean Latency: {result.mean_latency_ms:.2f}ms")
            lines.append(f"  P95 Latency: {result.p95_latency_ms:.2f}ms")
            lines.append(f"  Throughput: {result.throughput_rps:.2f} req/s")
            lines.append(f"  Success Rate: {result.success_rate * 100:.1f}%")
        
        return "\n".join(lines)
    
    def get_sla_summary(self, targets: SLATargets) -> str:
        """Get SLA compliance summary."""
        lines = ["SLA Compliance Summary", "=" * 50]
        
        passing = []
        failing = []
        
        for result in self.results:
            if result.passes_sla(targets):
                passing.append(result.name)
            else:
                failing.append(result.name)
        
        lines.append(f"\nPassing: {len(passing)}")
        for name in passing:
            lines.append(f"  ✓ {name}")
        
        lines.append(f"\nFailing: {len(failing)}")
        for name in failing:
            lines.append(f"  ✗ {name}")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps({
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }, indent=2)


class BenchmarkRunner:
    """
    Orchestrates benchmark execution.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            BenchmarkResult
        """
        latencies: List[float] = []
        errors = 0
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(self.config.benchmark_iterations):
            iter_start = time.perf_counter()
            
            try:
                func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - iter_start) * 1000
                latencies.append(elapsed_ms)
            except Exception as e:
                errors += 1
                logger.debug(f"Benchmark iteration error: {e}")
        
        duration = time.perf_counter() - start_time
        
        result = BenchmarkResult.from_latencies(
            name=name,
            latencies=latencies,
            errors=errors,
            duration_seconds=duration,
        )
        
        self.results.append(result)
        return result
    
    def run_all(self) -> BenchmarkReport:
        """
        Run all configured benchmarks.
        
        Returns:
            BenchmarkReport with all results
        """
        self.results = []
        
        if self.config.include_inference:
            self._run_inference_benchmarks()
        
        if self.config.include_throughput:
            self._run_throughput_benchmarks()
        
        if self.config.include_pipeline:
            self._run_pipeline_benchmarks()
        
        if self.config.include_resources:
            self._run_resource_benchmarks()
        
        return BenchmarkReport(results=self.results)
    
    def _run_inference_benchmarks(self) -> None:
        """Run inference benchmarks."""
        # Placeholder - actual implementation would use real models
        def mock_inference():
            time.sleep(0.01)  # 10ms simulated inference
            return "response"
        
        self.run_benchmark("inference", mock_inference)
    
    def _run_throughput_benchmarks(self) -> None:
        """Run throughput benchmarks."""
        # Placeholder
        pass
    
    def _run_pipeline_benchmarks(self) -> None:
        """Run pipeline benchmarks."""
        # Placeholder
        pass
    
    def _run_resource_benchmarks(self) -> None:
        """Run resource benchmarks."""
        # Placeholder
        pass
