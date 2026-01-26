"""
Quality Gates for Training and Inference Optimizations.

Defines quality thresholds that optimizations must meet
to be considered production-ready.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class QualityGateStatus(Enum):
    """Status of a quality gate check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    status: QualityGateStatus
    expected: float
    actual: float
    message: str = ""


@dataclass
class TrainingQualityGates:
    """
    Quality gates for training optimizations.

    Defines minimum thresholds that training optimizations
    must meet for production deployment.
    """

    # Throughput Gates
    min_tokens_per_second: float = 10000.0
    """Minimum tokens/second throughput"""

    min_samples_per_second: float = 10.0
    """Minimum samples/second throughput"""

    # Memory Gates
    max_memory_gb: float = 70.0
    """Maximum GPU memory usage (for 80GB H100)"""

    min_memory_efficiency: float = 0.7
    """Minimum memory efficiency ratio (allocated/reserved)"""

    max_memory_per_token_kb: float = 50.0
    """Maximum memory per token in KB"""

    # Scaling Gates
    min_scaling_efficiency: float = 0.85
    """Minimum scaling efficiency (85% linear)"""

    max_communication_overhead: float = 0.15
    """Maximum communication overhead"""

    # Convergence Gates
    max_gradient_norm: float = 10.0
    """Maximum gradient norm"""

    max_gradient_norm_variance: float = 0.5
    """Maximum gradient norm variance"""

    max_loss_spike_ratio: float = 2.0
    """Maximum loss spike ratio"""

    def check_throughput(
        self,
        tokens_per_second: float,
        samples_per_second: float,
    ) -> List[QualityGateResult]:
        """Check throughput quality gates."""
        results = []

        # Tokens/sec gate
        status = QualityGateStatus.PASSED if tokens_per_second >= self.min_tokens_per_second else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="min_tokens_per_second",
            status=status,
            expected=self.min_tokens_per_second,
            actual=tokens_per_second,
            message=f"Tokens/sec: {tokens_per_second:.0f} (min: {self.min_tokens_per_second:.0f})",
        ))

        # Samples/sec gate
        status = QualityGateStatus.PASSED if samples_per_second >= self.min_samples_per_second else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="min_samples_per_second",
            status=status,
            expected=self.min_samples_per_second,
            actual=samples_per_second,
            message=f"Samples/sec: {samples_per_second:.1f} (min: {self.min_samples_per_second:.1f})",
        ))

        return results

    def check_memory(
        self,
        peak_memory_gb: float,
        memory_efficiency: float,
        memory_per_token_kb: float,
    ) -> List[QualityGateResult]:
        """Check memory quality gates."""
        results = []

        # Peak memory gate
        status = QualityGateStatus.PASSED if peak_memory_gb <= self.max_memory_gb else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="max_memory_gb",
            status=status,
            expected=self.max_memory_gb,
            actual=peak_memory_gb,
            message=f"Peak memory: {peak_memory_gb:.1f}GB (max: {self.max_memory_gb:.1f}GB)",
        ))

        # Memory efficiency gate
        status = QualityGateStatus.PASSED if memory_efficiency >= self.min_memory_efficiency else QualityGateStatus.WARNING
        results.append(QualityGateResult(
            name="min_memory_efficiency",
            status=status,
            expected=self.min_memory_efficiency,
            actual=memory_efficiency,
            message=f"Memory efficiency: {memory_efficiency:.2f} (min: {self.min_memory_efficiency:.2f})",
        ))

        # Memory per token gate
        status = QualityGateStatus.PASSED if memory_per_token_kb <= self.max_memory_per_token_kb else QualityGateStatus.WARNING
        results.append(QualityGateResult(
            name="max_memory_per_token_kb",
            status=status,
            expected=self.max_memory_per_token_kb,
            actual=memory_per_token_kb,
            message=f"Memory/token: {memory_per_token_kb:.1f}KB (max: {self.max_memory_per_token_kb:.1f}KB)",
        ))

        return results

    def check_scaling(
        self,
        scaling_efficiency: float,
        communication_overhead: float,
    ) -> List[QualityGateResult]:
        """Check scaling quality gates."""
        results = []

        # Scaling efficiency gate
        status = QualityGateStatus.PASSED if scaling_efficiency >= self.min_scaling_efficiency else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="min_scaling_efficiency",
            status=status,
            expected=self.min_scaling_efficiency,
            actual=scaling_efficiency,
            message=f"Scaling efficiency: {scaling_efficiency:.2f} (min: {self.min_scaling_efficiency:.2f})",
        ))

        # Communication overhead gate
        status = QualityGateStatus.PASSED if communication_overhead <= self.max_communication_overhead else QualityGateStatus.WARNING
        results.append(QualityGateResult(
            name="max_communication_overhead",
            status=status,
            expected=self.max_communication_overhead,
            actual=communication_overhead,
            message=f"Comm overhead: {communication_overhead:.2f} (max: {self.max_communication_overhead:.2f})",
        ))

        return results

    def check_convergence(
        self,
        gradient_norm: float,
        gradient_norm_std: float,
        loss_values: List[float],
    ) -> List[QualityGateResult]:
        """Check convergence quality gates."""
        results = []

        # Gradient norm gate
        status = QualityGateStatus.PASSED if gradient_norm <= self.max_gradient_norm else QualityGateStatus.WARNING
        results.append(QualityGateResult(
            name="max_gradient_norm",
            status=status,
            expected=self.max_gradient_norm,
            actual=gradient_norm,
            message=f"Gradient norm: {gradient_norm:.2f} (max: {self.max_gradient_norm:.2f})",
        ))

        # Gradient variance gate
        status = QualityGateStatus.PASSED if gradient_norm_std <= self.max_gradient_norm_variance else QualityGateStatus.WARNING
        results.append(QualityGateResult(
            name="max_gradient_norm_variance",
            status=status,
            expected=self.max_gradient_norm_variance,
            actual=gradient_norm_std,
            message=f"Gradient norm std: {gradient_norm_std:.2f} (max: {self.max_gradient_norm_variance:.2f})",
        ))

        # Loss spike check
        if len(loss_values) >= 2:
            max_loss = max(loss_values)
            min_loss = min(loss_values)
            spike_ratio = max_loss / min_loss if min_loss > 0 else 0
            status = QualityGateStatus.PASSED if spike_ratio <= self.max_loss_spike_ratio else QualityGateStatus.FAILED
            results.append(QualityGateResult(
                name="max_loss_spike_ratio",
                status=status,
                expected=self.max_loss_spike_ratio,
                actual=spike_ratio,
                message=f"Loss spike ratio: {spike_ratio:.2f} (max: {self.max_loss_spike_ratio:.2f})",
            ))

        return results


@dataclass
class InferenceQualityGates:
    """
    Quality gates for inference optimizations.

    Defines minimum thresholds for production inference.
    """

    # Latency Gates
    max_p50_latency_ms: float = 50.0
    """Maximum p50 latency in milliseconds"""

    max_p90_latency_ms: float = 100.0
    """Maximum p90 latency in milliseconds"""

    max_p99_latency_ms: float = 200.0
    """Maximum p99 latency in milliseconds"""

    # Throughput Gates
    min_requests_per_second: float = 100.0
    """Minimum requests per second"""

    min_tokens_per_second: float = 5000.0
    """Minimum tokens per second"""

    # Memory Gates
    max_memory_gb: float = 60.0
    """Maximum inference memory usage"""

    max_memory_per_token_kb: float = 10.0
    """Maximum memory per token"""

    # Quality Gates
    max_latency_variance_ms: float = 50.0
    """Maximum latency variance"""

    def check_latency(
        self,
        p50_ms: float,
        p90_ms: float,
        p99_ms: float,
        std_ms: float = 0.0,
    ) -> List[QualityGateResult]:
        """Check latency quality gates."""
        results = []

        # P50 gate
        status = QualityGateStatus.PASSED if p50_ms <= self.max_p50_latency_ms else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="max_p50_latency_ms",
            status=status,
            expected=self.max_p50_latency_ms,
            actual=p50_ms,
            message=f"P50 latency: {p50_ms:.1f}ms (max: {self.max_p50_latency_ms:.1f}ms)",
        ))

        # P90 gate
        status = QualityGateStatus.PASSED if p90_ms <= self.max_p90_latency_ms else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="max_p90_latency_ms",
            status=status,
            expected=self.max_p90_latency_ms,
            actual=p90_ms,
            message=f"P90 latency: {p90_ms:.1f}ms (max: {self.max_p90_latency_ms:.1f}ms)",
        ))

        # P99 gate
        status = QualityGateStatus.PASSED if p99_ms <= self.max_p99_latency_ms else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="max_p99_latency_ms",
            status=status,
            expected=self.max_p99_latency_ms,
            actual=p99_ms,
            message=f"P99 latency: {p99_ms:.1f}ms (max: {self.max_p99_latency_ms:.1f}ms)",
        ))

        # Variance gate
        if std_ms > 0:
            status = QualityGateStatus.PASSED if std_ms <= self.max_latency_variance_ms else QualityGateStatus.WARNING
            results.append(QualityGateResult(
                name="max_latency_variance_ms",
                status=status,
                expected=self.max_latency_variance_ms,
                actual=std_ms,
                message=f"Latency std: {std_ms:.1f}ms (max: {self.max_latency_variance_ms:.1f}ms)",
            ))

        return results

    def check_throughput(
        self,
        requests_per_second: float,
        tokens_per_second: float,
    ) -> List[QualityGateResult]:
        """Check throughput quality gates."""
        results = []

        # Requests/sec gate
        status = QualityGateStatus.PASSED if requests_per_second >= self.min_requests_per_second else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="min_requests_per_second",
            status=status,
            expected=self.min_requests_per_second,
            actual=requests_per_second,
            message=f"Requests/sec: {requests_per_second:.1f} (min: {self.min_requests_per_second:.1f})",
        ))

        # Tokens/sec gate
        status = QualityGateStatus.PASSED if tokens_per_second >= self.min_tokens_per_second else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="min_tokens_per_second",
            status=status,
            expected=self.min_tokens_per_second,
            actual=tokens_per_second,
            message=f"Tokens/sec: {tokens_per_second:.0f} (min: {self.min_tokens_per_second:.0f})",
        ))

        return results

    def check_memory(
        self,
        peak_memory_gb: float,
        memory_per_token_kb: float,
    ) -> List[QualityGateResult]:
        """Check memory quality gates."""
        results = []

        # Peak memory gate
        status = QualityGateStatus.PASSED if peak_memory_gb <= self.max_memory_gb else QualityGateStatus.FAILED
        results.append(QualityGateResult(
            name="max_memory_gb",
            status=status,
            expected=self.max_memory_gb,
            actual=peak_memory_gb,
            message=f"Peak memory: {peak_memory_gb:.1f}GB (max: {self.max_memory_gb:.1f}GB)",
        ))

        # Memory per token gate
        status = QualityGateStatus.PASSED if memory_per_token_kb <= self.max_memory_per_token_kb else QualityGateStatus.WARNING
        results.append(QualityGateResult(
            name="max_memory_per_token_kb",
            status=status,
            expected=self.max_memory_per_token_kb,
            actual=memory_per_token_kb,
            message=f"Memory/token: {memory_per_token_kb:.1f}KB (max: {self.max_memory_per_token_kb:.1f}KB)",
        ))

        return results


class QualityGateChecker:
    """
    Comprehensive quality gate checker.

    Validates training and inference optimizations against
    quality thresholds.

    Example:
        >>> checker = QualityGateChecker()
        >>> results = checker.check_training(throughput_metrics, memory_metrics)
        >>> if checker.all_passed(results):
        ...     print("All quality gates passed!")
    """

    def __init__(
        self,
        training_gates: Optional[TrainingQualityGates] = None,
        inference_gates: Optional[InferenceQualityGates] = None,
    ):
        """
        Initialize QualityGateChecker.

        Args:
            training_gates: Custom training quality gates
            inference_gates: Custom inference quality gates
        """
        self.training_gates = training_gates or TrainingQualityGates()
        self.inference_gates = inference_gates or InferenceQualityGates()

    def check_training(
        self,
        throughput_metrics: Optional[Dict[str, float]] = None,
        memory_metrics: Optional[Dict[str, float]] = None,
        scaling_metrics: Optional[Dict[str, float]] = None,
        convergence_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[QualityGateResult]:
        """
        Check all training quality gates.

        Args:
            throughput_metrics: Throughput benchmark results
            memory_metrics: Memory benchmark results
            scaling_metrics: Scaling benchmark results
            convergence_metrics: Convergence benchmark results

        Returns:
            List of QualityGateResult
        """
        results = []

        if throughput_metrics:
            results.extend(self.training_gates.check_throughput(
                tokens_per_second=throughput_metrics.get("tokens_per_second", 0),
                samples_per_second=throughput_metrics.get("samples_per_second", 0),
            ))

        if memory_metrics:
            results.extend(self.training_gates.check_memory(
                peak_memory_gb=memory_metrics.get("peak_memory_gb", 0),
                memory_efficiency=memory_metrics.get("memory_efficiency_ratio", 0),
                memory_per_token_kb=memory_metrics.get("memory_per_token_kb", 0),
            ))

        if scaling_metrics:
            results.extend(self.training_gates.check_scaling(
                scaling_efficiency=scaling_metrics.get("scaling_efficiency", 0),
                communication_overhead=scaling_metrics.get("communication_overhead", 0),
            ))

        if convergence_metrics:
            results.extend(self.training_gates.check_convergence(
                gradient_norm=convergence_metrics.get("gradient_norm_mean", 0),
                gradient_norm_std=convergence_metrics.get("gradient_norm_std", 0),
                loss_values=convergence_metrics.get("loss_curve", []),
            ))

        return results

    def check_inference(
        self,
        latency_metrics: Optional[Dict[str, float]] = None,
        throughput_metrics: Optional[Dict[str, float]] = None,
        memory_metrics: Optional[Dict[str, float]] = None,
    ) -> List[QualityGateResult]:
        """
        Check all inference quality gates.

        Args:
            latency_metrics: Latency benchmark results
            throughput_metrics: Throughput benchmark results
            memory_metrics: Memory benchmark results

        Returns:
            List of QualityGateResult
        """
        results = []

        if latency_metrics:
            results.extend(self.inference_gates.check_latency(
                p50_ms=latency_metrics.get("p50_ms", 0),
                p90_ms=latency_metrics.get("p90_ms", 0),
                p99_ms=latency_metrics.get("p99_ms", 0),
                std_ms=latency_metrics.get("std_ms", 0),
            ))

        if throughput_metrics:
            results.extend(self.inference_gates.check_throughput(
                requests_per_second=throughput_metrics.get("requests_per_second", 0),
                tokens_per_second=throughput_metrics.get("tokens_per_second", 0),
            ))

        if memory_metrics:
            results.extend(self.inference_gates.check_memory(
                peak_memory_gb=memory_metrics.get("peak_memory_gb", 0),
                memory_per_token_kb=memory_metrics.get("memory_per_token_kb", 0),
            ))

        return results

    def all_passed(self, results: List[QualityGateResult]) -> bool:
        """Check if all quality gates passed."""
        return all(r.status == QualityGateStatus.PASSED for r in results)

    def any_failed(self, results: List[QualityGateResult]) -> bool:
        """Check if any quality gates failed."""
        return any(r.status == QualityGateStatus.FAILED for r in results)

    def get_summary(self, results: List[QualityGateResult]) -> Dict[str, int]:
        """Get summary of quality gate results."""
        summary = {
            "passed": 0,
            "failed": 0,
            "warning": 0,
            "skipped": 0,
        }
        for result in results:
            summary[result.status.value] += 1
        return summary

    def format_report(self, results: List[QualityGateResult]) -> str:
        """Format quality gate results as a report."""
        lines = ["Quality Gate Report", "=" * 50]

        for result in results:
            status_icon = {
                QualityGateStatus.PASSED: "✓",
                QualityGateStatus.FAILED: "✗",
                QualityGateStatus.WARNING: "⚠",
                QualityGateStatus.SKIPPED: "-",
            }.get(result.status, "?")

            lines.append(f"{status_icon} {result.name}: {result.message}")

        summary = self.get_summary(results)
        lines.append("=" * 50)
        lines.append(f"Summary: {summary['passed']} passed, {summary['failed']} failed, {summary['warning']} warnings")

        return "\n".join(lines)


# Pre-configured quality gates for different environments
PRODUCTION_TRAINING_GATES = TrainingQualityGates(
    min_tokens_per_second=10000,
    min_samples_per_second=10,
    max_memory_gb=70,
    min_memory_efficiency=0.8,
    min_scaling_efficiency=0.85,
)

PRODUCTION_INFERENCE_GATES = InferenceQualityGates(
    max_p50_latency_ms=50,
    max_p90_latency_ms=100,
    max_p99_latency_ms=200,
    min_requests_per_second=100,
    min_tokens_per_second=5000,
)

DEVELOPMENT_TRAINING_GATES = TrainingQualityGates(
    min_tokens_per_second=1000,
    min_samples_per_second=1,
    max_memory_gb=40,
    min_memory_efficiency=0.5,
    min_scaling_efficiency=0.7,
)

DEVELOPMENT_INFERENCE_GATES = InferenceQualityGates(
    max_p50_latency_ms=200,
    max_p90_latency_ms=500,
    max_p99_latency_ms=1000,
    min_requests_per_second=10,
    min_tokens_per_second=500,
)
