"""
Training Optimization Benchmarks for MedGemma.

Provides benchmarking tools for training optimizations:
- Throughput benchmarks (tokens/sec, samples/sec)
- Memory efficiency benchmarks
- Scaling efficiency benchmarks
- Convergence benchmarks
"""

import time
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ThroughputMetrics:
    """Metrics for training throughput."""
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    total_tokens: int = 0
    total_samples: int = 0
    total_time_seconds: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class MemoryMetrics:
    """Metrics for memory efficiency."""
    peak_memory_gb: float = 0.0
    allocated_memory_gb: float = 0.0
    reserved_memory_gb: float = 0.0
    memory_efficiency_ratio: float = 0.0
    memory_per_token_kb: float = 0.0


@dataclass
class ScalingMetrics:
    """Metrics for scaling efficiency."""
    scaling_efficiency: float = 0.0
    communication_overhead: float = 0.0
    weak_scaling_efficiency: float = 0.0
    strong_scaling_efficiency: float = 0.0


@dataclass
class ConvergenceMetrics:
    """Metrics for training convergence."""
    final_loss: float = 0.0
    loss_curve: List[float] = field(default_factory=list)
    gradient_norm_mean: float = 0.0
    gradient_norm_std: float = 0.0
    steps_to_convergence: int = 0


class TrainingThroughputBenchmark:
    """
    Benchmark for training throughput measurement.

    Measures tokens/sec, samples/sec, and GPU utilization
    during training.

    Example:
        >>> benchmark = TrainingThroughputBenchmark()
        >>> metrics = benchmark.run(model, dataloader, steps=100)
        >>> print(f"Throughput: {metrics.tokens_per_second:.0f} tokens/sec")
    """

    def __init__(
        self,
        warmup_steps: int = 10,
        measure_steps: int = 100,
    ):
        """
        Initialize TrainingThroughputBenchmark.

        Args:
            warmup_steps: Number of warmup steps before measurement
            measure_steps: Number of steps to measure
        """
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps

    def run(
        self,
        model: nn.Module,
        dataloader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        steps: Optional[int] = None,
        tokens_per_sample: int = 512,
    ) -> ThroughputMetrics:
        """
        Run throughput benchmark.

        Args:
            model: Model to benchmark
            dataloader: Training dataloader
            optimizer: Optional optimizer (creates one if not provided)
            steps: Override number of measurement steps
            tokens_per_sample: Tokens per sample for calculation

        Returns:
            ThroughputMetrics with benchmark results
        """
        if steps is not None:
            self.measure_steps = steps

        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()

        # Warmup
        data_iter = iter(dataloader)
        for _ in range(self.warmup_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            self._training_step(model, batch, optimizer)

        # Synchronize before measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measurement
        total_samples = 0
        start_time = time.perf_counter()

        for step in range(self.measure_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch_size = self._get_batch_size(batch)
            self._training_step(model, batch, optimizer)
            total_samples += batch_size

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        total_tokens = total_samples * tokens_per_sample

        return ThroughputMetrics(
            tokens_per_second=total_tokens / total_time,
            samples_per_second=total_samples / total_time,
            batches_per_second=self.measure_steps / total_time,
            total_tokens=total_tokens,
            total_samples=total_samples,
            total_time_seconds=total_time,
            gpu_utilization=self._get_gpu_utilization(),
        )

    def _training_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Perform a single training step."""
        optimizer.zero_grad()

        if isinstance(batch, dict):
            outputs = model(**batch)
        else:
            outputs = model(batch)

        loss = outputs.loss if hasattr(outputs, "loss") else outputs
        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()

    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from batch."""
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
        elif isinstance(batch, torch.Tensor):
            return batch.shape[0]
        return 1

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if torch.cuda.is_available():
            try:
                # This is a simplified approximation
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                if reserved > 0:
                    return allocated / reserved
            except Exception:
                pass
        return 0.0


class MemoryEfficiencyBenchmark:
    """
    Benchmark for memory efficiency measurement.

    Measures peak memory, memory per token, and
    memory efficiency ratio.

    Example:
        >>> benchmark = MemoryEfficiencyBenchmark()
        >>> metrics = benchmark.run(model, dataloader)
        >>> print(f"Peak memory: {metrics.peak_memory_gb:.2f} GB")
    """

    def __init__(self):
        """Initialize MemoryEfficiencyBenchmark."""
        pass

    def run(
        self,
        model: nn.Module,
        dataloader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        steps: int = 10,
        tokens_per_sample: int = 512,
    ) -> MemoryMetrics:
        """
        Run memory efficiency benchmark.

        Args:
            model: Model to benchmark
            dataloader: Training dataloader
            optimizer: Optional optimizer
            steps: Number of steps to measure
            tokens_per_sample: Tokens per sample

        Returns:
            MemoryMetrics with benchmark results
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        total_tokens = 0

        data_iter = iter(dataloader)
        for _ in range(steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch_size = self._get_batch_size(batch)
            total_tokens += batch_size * tokens_per_sample

            optimizer.zero_grad()
            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                outputs = model(batch)

            loss = outputs.loss if hasattr(outputs, "loss") else outputs
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            efficiency = allocated / reserved if reserved > 0 else 0.0
            memory_per_token = (peak_memory * 1e6) / total_tokens if total_tokens > 0 else 0.0
        else:
            peak_memory = allocated = reserved = efficiency = memory_per_token = 0.0

        return MemoryMetrics(
            peak_memory_gb=peak_memory,
            allocated_memory_gb=allocated,
            reserved_memory_gb=reserved,
            memory_efficiency_ratio=efficiency,
            memory_per_token_kb=memory_per_token,
        )

    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from batch."""
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
        elif isinstance(batch, torch.Tensor):
            return batch.shape[0]
        return 1


class ScalingEfficiencyBenchmark:
    """
    Benchmark for distributed scaling efficiency.

    Measures weak and strong scaling efficiency
    across multiple GPUs.

    Example:
        >>> benchmark = ScalingEfficiencyBenchmark()
        >>> metrics = benchmark.measure_scaling(model, dataloader, num_gpus=[1, 2, 4, 8])
    """

    def __init__(self):
        """Initialize ScalingEfficiencyBenchmark."""
        self.throughput_benchmark = TrainingThroughputBenchmark(
            warmup_steps=5,
            measure_steps=50,
        )

    def measure_scaling(
        self,
        model_factory: Callable[[], nn.Module],
        dataloader_factory: Callable[[int], Any],
        num_gpus: List[int],
        baseline_throughput: Optional[float] = None,
    ) -> Dict[int, ScalingMetrics]:
        """
        Measure scaling efficiency across different GPU counts.

        Args:
            model_factory: Function that creates the model
            dataloader_factory: Function that creates dataloader given batch size
            num_gpus: List of GPU counts to test
            baseline_throughput: Optional baseline for efficiency calculation

        Returns:
            Dictionary mapping GPU count to ScalingMetrics
        """
        results = {}

        for n_gpus in num_gpus:
            # This would be run in a distributed context in production
            # Here we provide a placeholder implementation
            model = model_factory()
            dataloader = dataloader_factory(n_gpus)

            throughput = self.throughput_benchmark.run(
                model, dataloader, steps=50
            )

            if baseline_throughput is None:
                baseline_throughput = throughput.tokens_per_second

            # Calculate scaling efficiency
            expected_throughput = baseline_throughput * n_gpus
            actual_throughput = throughput.tokens_per_second
            scaling_efficiency = actual_throughput / expected_throughput if expected_throughput > 0 else 0.0

            results[n_gpus] = ScalingMetrics(
                scaling_efficiency=scaling_efficiency,
                communication_overhead=1 - scaling_efficiency,
                weak_scaling_efficiency=scaling_efficiency,
                strong_scaling_efficiency=scaling_efficiency,
            )

        return results


class ConvergenceBenchmark:
    """
    Benchmark for training convergence.

    Tracks loss curves, gradient norms, and
    convergence speed.

    Example:
        >>> benchmark = ConvergenceBenchmark()
        >>> metrics = benchmark.run(model, dataloader, steps=1000)
        >>> print(f"Final loss: {metrics.final_loss:.4f}")
    """

    def __init__(
        self,
        log_interval: int = 10,
    ):
        """
        Initialize ConvergenceBenchmark.

        Args:
            log_interval: How often to log metrics
        """
        self.log_interval = log_interval

    def run(
        self,
        model: nn.Module,
        dataloader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        steps: int = 1000,
        convergence_threshold: float = 0.01,
    ) -> ConvergenceMetrics:
        """
        Run convergence benchmark.

        Args:
            model: Model to benchmark
            dataloader: Training dataloader
            optimizer: Optional optimizer
            steps: Number of training steps
            convergence_threshold: Loss change threshold for convergence

        Returns:
            ConvergenceMetrics with benchmark results
        """
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        loss_curve = []
        gradient_norms = []
        steps_to_convergence = steps
        converged = False

        data_iter = iter(dataloader)
        prev_loss = float("inf")

        for step in range(steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            optimizer.zero_grad()

            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                outputs = model(batch)

            loss = outputs.loss if hasattr(outputs, "loss") else outputs

            if isinstance(loss, torch.Tensor):
                loss.backward()

                # Calculate gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)

                optimizer.step()

                loss_value = loss.item()
                loss_curve.append(loss_value)

                # Check convergence
                if not converged and abs(prev_loss - loss_value) < convergence_threshold:
                    steps_to_convergence = step + 1
                    converged = True

                prev_loss = loss_value

        import statistics
        grad_mean = statistics.mean(gradient_norms) if gradient_norms else 0.0
        grad_std = statistics.stdev(gradient_norms) if len(gradient_norms) > 1 else 0.0

        return ConvergenceMetrics(
            final_loss=loss_curve[-1] if loss_curve else 0.0,
            loss_curve=loss_curve,
            gradient_norm_mean=grad_mean,
            gradient_norm_std=grad_std,
            steps_to_convergence=steps_to_convergence,
        )


class TrainingBenchmarkSuite:
    """
    Comprehensive training benchmark suite.

    Runs all training benchmarks and aggregates results.

    Example:
        >>> suite = TrainingBenchmarkSuite()
        >>> results = suite.run_all(model, dataloader)
    """

    def __init__(self):
        """Initialize TrainingBenchmarkSuite."""
        self.throughput_benchmark = TrainingThroughputBenchmark()
        self.memory_benchmark = MemoryEfficiencyBenchmark()
        self.convergence_benchmark = ConvergenceBenchmark()

    def run_all(
        self,
        model: nn.Module,
        dataloader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        throughput_steps: int = 100,
        memory_steps: int = 10,
        convergence_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Run all benchmarks.

        Args:
            model: Model to benchmark
            dataloader: Training dataloader
            optimizer: Optional optimizer
            throughput_steps: Steps for throughput benchmark
            memory_steps: Steps for memory benchmark
            convergence_steps: Steps for convergence benchmark

        Returns:
            Dictionary with all benchmark results
        """
        results = {}

        # Run throughput benchmark
        results["throughput"] = self.throughput_benchmark.run(
            model, dataloader, optimizer, steps=throughput_steps
        )

        # Run memory benchmark
        results["memory"] = self.memory_benchmark.run(
            model, dataloader, optimizer, steps=memory_steps
        )

        # Run convergence benchmark
        results["convergence"] = self.convergence_benchmark.run(
            model, dataloader, optimizer, steps=convergence_steps
        )

        return results
