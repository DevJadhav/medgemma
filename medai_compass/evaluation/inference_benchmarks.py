"""
Inference Optimization Benchmarks for MedGemma.

Provides benchmarking tools for inference optimizations:
- Latency benchmarks (p50, p90, p99)
- Throughput benchmarks (requests/sec)
- Batch efficiency benchmarks
- Memory benchmarks (KV cache size)
"""

import time
import torch
import torch.nn as nn
import statistics
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field


@dataclass
class LatencyMetrics:
    """Metrics for inference latency."""
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)


@dataclass
class InferenceThroughputMetrics:
    """Metrics for inference throughput."""
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0


@dataclass
class BatchEfficiencyMetrics:
    """Metrics for batch processing efficiency."""
    throughput_by_batch_size: Dict[int, float] = field(default_factory=dict)
    latency_by_batch_size: Dict[int, float] = field(default_factory=dict)
    optimal_batch_size: int = 1
    efficiency_curve: List[tuple] = field(default_factory=list)


@dataclass
class InferenceMemoryMetrics:
    """Metrics for inference memory usage."""
    model_memory_gb: float = 0.0
    kv_cache_memory_gb: float = 0.0
    peak_memory_gb: float = 0.0
    memory_per_token_kb: float = 0.0
    max_batch_size: int = 0


class LatencyBenchmark:
    """
    Benchmark for inference latency measurement.

    Measures p50, p90, p99 latencies across multiple
    inference requests.

    Example:
        >>> benchmark = LatencyBenchmark()
        >>> metrics = benchmark.run(model, input_text, num_requests=100)
        >>> print(f"P99 latency: {metrics.p99_ms:.2f}ms")
    """

    def __init__(
        self,
        warmup_requests: int = 10,
    ):
        """
        Initialize LatencyBenchmark.

        Args:
            warmup_requests: Number of warmup requests
        """
        self.warmup_requests = warmup_requests

    def run(
        self,
        model: nn.Module,
        inputs: Any,
        num_requests: int = 100,
        max_new_tokens: int = 100,
    ) -> LatencyMetrics:
        """
        Run latency benchmark.

        Args:
            model: Model to benchmark
            inputs: Input data (tensor or dict)
            num_requests: Number of requests to measure
            max_new_tokens: Max tokens to generate per request

        Returns:
            LatencyMetrics with benchmark results
        """
        model.eval()
        latencies = []

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_requests):
                self._run_inference(model, inputs, max_new_tokens)

        # Synchronize before measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure latencies
        with torch.no_grad():
            for _ in range(num_requests):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                self._run_inference(model, inputs, max_new_tokens)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return LatencyMetrics(
            p50_ms=sorted_latencies[int(n * 0.5)] if n > 0 else 0.0,
            p90_ms=sorted_latencies[int(n * 0.9)] if n > 0 else 0.0,
            p99_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
            mean_ms=statistics.mean(latencies) if latencies else 0.0,
            min_ms=min(latencies) if latencies else 0.0,
            max_ms=max(latencies) if latencies else 0.0,
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            latency_samples=latencies,
        )

    def _run_inference(
        self,
        model: nn.Module,
        inputs: Any,
        max_new_tokens: int,
    ) -> Any:
        """Run a single inference."""
        if isinstance(inputs, dict):
            return model(**inputs)
        else:
            return model(inputs)


class ThroughputBenchmark:
    """
    Benchmark for inference throughput measurement.

    Measures requests/sec and tokens/sec for
    continuous inference.

    Example:
        >>> benchmark = ThroughputBenchmark()
        >>> metrics = benchmark.run(model, inputs, duration_seconds=60)
        >>> print(f"Throughput: {metrics.requests_per_second:.1f} req/sec")
    """

    def __init__(
        self,
        warmup_seconds: float = 5.0,
    ):
        """
        Initialize ThroughputBenchmark.

        Args:
            warmup_seconds: Warmup duration in seconds
        """
        self.warmup_seconds = warmup_seconds

    def run(
        self,
        model: nn.Module,
        inputs: Any,
        duration_seconds: float = 30.0,
        output_tokens: int = 100,
    ) -> InferenceThroughputMetrics:
        """
        Run throughput benchmark.

        Args:
            model: Model to benchmark
            inputs: Input data
            duration_seconds: How long to measure
            output_tokens: Tokens generated per request

        Returns:
            InferenceThroughputMetrics with results
        """
        model.eval()

        # Warmup
        warmup_start = time.perf_counter()
        with torch.no_grad():
            while time.perf_counter() - warmup_start < self.warmup_seconds:
                self._run_inference(model, inputs)

        # Measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        total_requests = 0

        with torch.no_grad():
            while time.perf_counter() - start_time < duration_seconds:
                self._run_inference(model, inputs)
                total_requests += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_tokens = total_requests * output_tokens

        return InferenceThroughputMetrics(
            requests_per_second=total_requests / total_time,
            tokens_per_second=total_tokens / total_time,
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
        )

    def _run_inference(self, model: nn.Module, inputs: Any) -> Any:
        """Run a single inference."""
        if isinstance(inputs, dict):
            return model(**inputs)
        else:
            return model(inputs)


class BatchEfficiencyBenchmark:
    """
    Benchmark for batch processing efficiency.

    Measures throughput and latency at different batch sizes
    to find optimal batching configuration.

    Example:
        >>> benchmark = BatchEfficiencyBenchmark()
        >>> metrics = benchmark.run(model, input_factory, batch_sizes=[1, 2, 4, 8, 16])
        >>> print(f"Optimal batch size: {metrics.optimal_batch_size}")
    """

    def __init__(self):
        """Initialize BatchEfficiencyBenchmark."""
        self.latency_benchmark = LatencyBenchmark(warmup_requests=5)

    def run(
        self,
        model: nn.Module,
        input_factory: Callable[[int], Any],
        batch_sizes: List[int],
        num_requests: int = 50,
    ) -> BatchEfficiencyMetrics:
        """
        Run batch efficiency benchmark.

        Args:
            model: Model to benchmark
            input_factory: Function that creates input for given batch size
            batch_sizes: List of batch sizes to test
            num_requests: Requests per batch size

        Returns:
            BatchEfficiencyMetrics with results
        """
        throughput_by_batch = {}
        latency_by_batch = {}
        efficiency_curve = []

        for batch_size in batch_sizes:
            inputs = input_factory(batch_size)

            # Measure latency
            latency_metrics = self.latency_benchmark.run(
                model, inputs, num_requests=num_requests
            )

            # Calculate throughput (samples / latency)
            throughput = batch_size / (latency_metrics.mean_ms / 1000) if latency_metrics.mean_ms > 0 else 0.0

            throughput_by_batch[batch_size] = throughput
            latency_by_batch[batch_size] = latency_metrics.mean_ms
            efficiency_curve.append((batch_size, throughput, latency_metrics.mean_ms))

        # Find optimal batch size (highest throughput)
        optimal_batch = max(throughput_by_batch, key=throughput_by_batch.get) if throughput_by_batch else 1

        return BatchEfficiencyMetrics(
            throughput_by_batch_size=throughput_by_batch,
            latency_by_batch_size=latency_by_batch,
            optimal_batch_size=optimal_batch,
            efficiency_curve=efficiency_curve,
        )


class InferenceMemoryBenchmark:
    """
    Benchmark for inference memory usage.

    Measures model memory, KV cache size, and
    memory per token.

    Example:
        >>> benchmark = InferenceMemoryBenchmark()
        >>> metrics = benchmark.run(model, inputs)
        >>> print(f"KV cache: {metrics.kv_cache_memory_gb:.2f} GB")
    """

    def __init__(self):
        """Initialize InferenceMemoryBenchmark."""
        pass

    def run(
        self,
        model: nn.Module,
        inputs: Any,
        sequence_lengths: List[int] = None,
    ) -> InferenceMemoryMetrics:
        """
        Run memory benchmark.

        Args:
            model: Model to benchmark
            inputs: Input data
            sequence_lengths: Sequence lengths to test

        Returns:
            InferenceMemoryMetrics with results
        """
        if sequence_lengths is None:
            sequence_lengths = [128, 256, 512, 1024]

        model.eval()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Measure baseline model memory
        if torch.cuda.is_available():
            model_memory = torch.cuda.memory_allocated() / 1e9
        else:
            model_memory = 0.0

        # Run inference to measure KV cache
        with torch.no_grad():
            if isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            kv_cache_memory = peak_memory - model_memory
        else:
            peak_memory = kv_cache_memory = 0.0

        # Estimate max batch size
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            available = total_memory - model_memory
            max_batch = int(available / kv_cache_memory) if kv_cache_memory > 0 else 1
        else:
            total_memory = 0
            max_batch = 1

        # Calculate memory per token
        input_tokens = self._count_tokens(inputs)
        memory_per_token = (kv_cache_memory * 1e6) / input_tokens if input_tokens > 0 else 0.0

        return InferenceMemoryMetrics(
            model_memory_gb=model_memory,
            kv_cache_memory_gb=kv_cache_memory,
            peak_memory_gb=peak_memory,
            memory_per_token_kb=memory_per_token,
            max_batch_size=max_batch,
        )

    def _count_tokens(self, inputs: Any) -> int:
        """Count tokens in input."""
        if isinstance(inputs, dict):
            if "input_ids" in inputs:
                return inputs["input_ids"].numel()
            for v in inputs.values():
                if isinstance(v, torch.Tensor):
                    return v.numel()
        elif isinstance(inputs, torch.Tensor):
            return inputs.numel()
        return 1


class InferenceBenchmarkSuite:
    """
    Comprehensive inference benchmark suite.

    Runs all inference benchmarks and aggregates results.

    Example:
        >>> suite = InferenceBenchmarkSuite()
        >>> results = suite.run_all(model, inputs)
    """

    def __init__(self):
        """Initialize InferenceBenchmarkSuite."""
        self.latency_benchmark = LatencyBenchmark()
        self.throughput_benchmark = ThroughputBenchmark()
        self.memory_benchmark = InferenceMemoryBenchmark()

    def run_all(
        self,
        model: nn.Module,
        inputs: Any,
        latency_requests: int = 100,
        throughput_duration: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Run all inference benchmarks.

        Args:
            model: Model to benchmark
            inputs: Input data
            latency_requests: Number of requests for latency benchmark
            throughput_duration: Duration for throughput benchmark

        Returns:
            Dictionary with all benchmark results
        """
        results = {}

        # Run latency benchmark
        results["latency"] = self.latency_benchmark.run(
            model, inputs, num_requests=latency_requests
        )

        # Run throughput benchmark
        results["throughput"] = self.throughput_benchmark.run(
            model, inputs, duration_seconds=throughput_duration
        )

        # Run memory benchmark
        results["memory"] = self.memory_benchmark.run(model, inputs)

        return results


def benchmark_inference_optimization(
    baseline_model: nn.Module,
    optimized_model: nn.Module,
    inputs: Any,
) -> Dict[str, Any]:
    """
    Compare baseline and optimized models.

    Args:
        baseline_model: Baseline model
        optimized_model: Optimized model
        inputs: Input data

    Returns:
        Comparison metrics
    """
    suite = InferenceBenchmarkSuite()

    baseline_results = suite.run_all(baseline_model, inputs)
    optimized_results = suite.run_all(optimized_model, inputs)

    # Calculate speedups
    latency_speedup = (
        baseline_results["latency"].mean_ms / optimized_results["latency"].mean_ms
        if optimized_results["latency"].mean_ms > 0 else 0.0
    )

    throughput_speedup = (
        optimized_results["throughput"].requests_per_second /
        baseline_results["throughput"].requests_per_second
        if baseline_results["throughput"].requests_per_second > 0 else 0.0
    )

    memory_reduction = (
        1 - optimized_results["memory"].peak_memory_gb /
        baseline_results["memory"].peak_memory_gb
        if baseline_results["memory"].peak_memory_gb > 0 else 0.0
    )

    return {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "latency_speedup": latency_speedup,
        "throughput_speedup": throughput_speedup,
        "memory_reduction": memory_reduction,
    }
