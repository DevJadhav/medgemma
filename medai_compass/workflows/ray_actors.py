"""
Ray Actors for Production ML Pipeline.

Provides long-running Ray actors for:
- Continuous model evaluation
- Distributed metrics aggregation
- Checkpoint lifecycle management
- Model health monitoring

These actors are designed for production use with:
- Fault tolerance via actor restart
- State persistence
- Async operations
- Metrics export
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationResult:
    """Result from model evaluation."""

    checkpoint_path: str
    metrics: Dict[str, float]
    benchmarks: List[str]
    timestamp: str
    duration_seconds: float
    passed_quality_gate: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "metrics": self.metrics,
            "benchmarks": self.benchmarks,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "passed_quality_gate": self.passed_quality_gate,
            "error": self.error,
        }


@dataclass
class MetricEntry:
    """Single metric entry."""

    step: int
    timestamp: str
    values: Dict[str, float]
    worker_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            **self.values,
            "worker_id": self.worker_id,
        }


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    path: str
    step: int
    timestamp: str
    metrics: Dict[str, float]
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "step": self.step,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "is_best": self.is_best,
        }


# =============================================================================
# Ray Actor Factory
# =============================================================================

def create_ray_evaluation_actor(
    checkpoint_dir: str,
    eval_dataset_path: str,
    benchmarks: Optional[List[str]] = None,
    num_gpus: float = 0.5,
):
    """
    Create a Ray actor for model evaluation.

    Args:
        checkpoint_dir: Directory to watch for checkpoints
        eval_dataset_path: Path to evaluation dataset
        benchmarks: List of benchmarks to run
        num_gpus: GPU resources per actor

    Returns:
        Ray actor handle
    """
    try:
        import ray
    except ImportError:
        raise ImportError("ray required for Ray actors")

    @ray.remote(num_gpus=num_gpus, max_restarts=3)
    class RayEvaluationActor:
        """
        Ray actor for continuous model evaluation.

        Evaluates checkpoints as they become available and tracks
        the best performing model.
        """

        def __init__(
            self,
            checkpoint_dir: str,
            eval_dataset_path: str,
            benchmarks: Optional[List[str]] = None,
        ):
            self.checkpoint_dir = checkpoint_dir
            self.eval_dataset_path = eval_dataset_path
            self.benchmarks = benchmarks or ["medqa", "pubmedqa"]
            self.results: Dict[str, EvaluationResult] = {}
            self.best_checkpoint: Optional[str] = None
            self.best_metric_value: float = 0.0
            self._running = True
            self._model = None
            self._tokenizer = None

            logger.info(f"EvaluationActor initialized for {checkpoint_dir}")

        async def evaluate_checkpoint(
            self,
            checkpoint_path: str,
            quality_thresholds: Optional[Dict[str, float]] = None,
        ) -> EvaluationResult:
            """
            Evaluate a single checkpoint.

            Args:
                checkpoint_path: Path to checkpoint
                quality_thresholds: Optional thresholds for quality gate

            Returns:
                Evaluation result
            """
            start_time = time.time()
            quality_thresholds = quality_thresholds or {
                "accuracy": 0.70,
                "safety_score": 0.95,
            }

            try:
                # Run evaluation
                metrics = await self._run_evaluation(checkpoint_path)

                # Check quality gate
                passed = all(
                    metrics.get(k, 0) >= v
                    for k, v in quality_thresholds.items()
                )

                result = EvaluationResult(
                    checkpoint_path=checkpoint_path,
                    metrics=metrics,
                    benchmarks=self.benchmarks,
                    timestamp=datetime.now().isoformat(),
                    duration_seconds=time.time() - start_time,
                    passed_quality_gate=passed,
                )

                # Track results
                self.results[checkpoint_path] = result

                # Update best checkpoint
                primary_metric = metrics.get("accuracy", 0)
                if primary_metric > self.best_metric_value:
                    self.best_metric_value = primary_metric
                    self.best_checkpoint = checkpoint_path

                return result

            except Exception as e:
                logger.error(f"Evaluation failed for {checkpoint_path}: {e}")
                return EvaluationResult(
                    checkpoint_path=checkpoint_path,
                    metrics={},
                    benchmarks=self.benchmarks,
                    timestamp=datetime.now().isoformat(),
                    duration_seconds=time.time() - start_time,
                    passed_quality_gate=False,
                    error=str(e),
                )

        async def _run_evaluation(self, checkpoint_path: str) -> Dict[str, float]:
            """Run actual evaluation (mock for testing)."""
            # In production, this would load model and run benchmarks
            await asyncio.sleep(0.1)  # Simulate evaluation time

            return {
                "accuracy": 0.75,
                "safety_score": 0.98,
                "latency_ms": 150.0,
            }

        def get_results(self) -> Dict[str, Dict[str, Any]]:
            """Get all evaluation results."""
            return {k: v.to_dict() for k, v in self.results.items()}

        def get_best_checkpoint(self) -> Optional[str]:
            """Get path to best checkpoint."""
            return self.best_checkpoint

        def get_result(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
            """Get result for specific checkpoint."""
            result = self.results.get(checkpoint_path)
            return result.to_dict() if result else None

        def stop(self) -> None:
            """Stop the actor."""
            self._running = False
            logger.info("EvaluationActor stopped")

    # Create and return the actor
    return RayEvaluationActor.remote(
        checkpoint_dir=checkpoint_dir,
        eval_dataset_path=eval_dataset_path,
        benchmarks=benchmarks,
    )


def create_ray_metrics_aggregator(
    export_interval_seconds: float = 30.0,
    max_entries: int = 10000,
):
    """
    Create a Ray actor for metrics aggregation.

    Args:
        export_interval_seconds: Interval for metrics export
        max_entries: Maximum metric entries to retain

    Returns:
        Ray actor handle
    """
    try:
        import ray
    except ImportError:
        raise ImportError("ray required for Ray actors")

    @ray.remote(num_cpus=1)
    class RayMetricsAggregator:
        """
        Ray actor for aggregating metrics from distributed workers.

        Collects metrics from training workers, computes statistics,
        and exports for monitoring.
        """

        def __init__(
            self,
            export_interval_seconds: float = 30.0,
            max_entries: int = 10000,
        ):
            self.export_interval = export_interval_seconds
            self.max_entries = max_entries
            self.entries: List[MetricEntry] = []
            self.current_step = 0
            self._running = True
            self._start_time = time.time()

            logger.info("MetricsAggregator initialized")

        def add_metrics(
            self,
            metrics: Dict[str, float],
            step: int,
            worker_id: Optional[str] = None,
        ) -> None:
            """
            Add metrics from a worker.

            Args:
                metrics: Metric values
                step: Training step
                worker_id: ID of reporting worker
            """
            entry = MetricEntry(
                step=step,
                timestamp=datetime.now().isoformat(),
                values=metrics,
                worker_id=worker_id,
            )

            self.entries.append(entry)
            self.current_step = max(self.current_step, step)

            # Trim old entries
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]

        def get_metrics(
            self,
            last_n: Optional[int] = None,
            from_step: Optional[int] = None,
        ) -> List[Dict[str, Any]]:
            """
            Get collected metrics.

            Args:
                last_n: Return only last N entries
                from_step: Return entries from this step onwards

            Returns:
                List of metric entries
            """
            entries = self.entries

            if from_step is not None:
                entries = [e for e in entries if e.step >= from_step]

            if last_n is not None:
                entries = entries[-last_n:]

            return [e.to_dict() for e in entries]

        def get_current_step(self) -> int:
            """Get current training step."""
            return self.current_step

        def get_statistics(
            self,
            window: int = 100,
            metric_names: Optional[List[str]] = None,
        ) -> Dict[str, Dict[str, float]]:
            """
            Compute statistics over recent metrics.

            Args:
                window: Number of entries to consider
                metric_names: Specific metrics to analyze

            Returns:
                Statistics per metric
            """
            recent = self.entries[-window:] if self.entries else []

            if not recent:
                return {}

            # Get all metric names if not specified
            if metric_names is None:
                metric_names = set()
                for entry in recent:
                    metric_names.update(entry.values.keys())
                metric_names = list(metric_names)

            stats = {}
            for name in metric_names:
                values = [
                    e.values[name] for e in recent
                    if name in e.values
                ]

                if not values:
                    continue

                stats[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                    "count": len(values),
                }

            return stats

        def get_summary(self) -> Dict[str, Any]:
            """Get aggregator summary."""
            uptime = time.time() - self._start_time

            return {
                "current_step": self.current_step,
                "total_entries": len(self.entries),
                "uptime_seconds": uptime,
                "entries_per_second": len(self.entries) / max(uptime, 1),
                "latest_entry": self.entries[-1].to_dict() if self.entries else None,
                "statistics": self.get_statistics(),
            }

        def to_prometheus(self) -> str:
            """Export metrics in Prometheus format."""
            lines = []
            stats = self.get_statistics()

            for metric_name, metric_stats in stats.items():
                safe_name = metric_name.replace("/", "_").replace("-", "_")

                lines.extend([
                    f"# HELP training_{safe_name} {metric_name}",
                    f"# TYPE training_{safe_name} gauge",
                    f'training_{safe_name}{{stat="mean"}} {metric_stats["mean"]:.6f}',
                    f'training_{safe_name}{{stat="latest"}} {metric_stats["latest"]:.6f}',
                ])

            return "\n".join(lines)

        def stop(self) -> None:
            """Stop the aggregator."""
            self._running = False
            logger.info("MetricsAggregator stopped")

    return RayMetricsAggregator.remote(
        export_interval_seconds=export_interval_seconds,
        max_entries=max_entries,
    )


def create_ray_checkpoint_manager(
    checkpoint_dir: str,
    max_checkpoints: int = 3,
    num_cpus: int = 1,
):
    """
    Create a Ray actor for checkpoint management.

    Args:
        checkpoint_dir: Directory for checkpoints
        max_checkpoints: Maximum checkpoints to keep
        num_cpus: CPU resources for actor

    Returns:
        Ray actor handle
    """
    try:
        import ray
    except ImportError:
        raise ImportError("ray required for Ray actors")

    @ray.remote(num_cpus=num_cpus)
    class RayCheckpointManager:
        """
        Ray actor for checkpoint lifecycle management.

        Handles checkpoint registration, cleanup, and best model tracking.
        """

        def __init__(
            self,
            checkpoint_dir: str,
            max_checkpoints: int = 3,
        ):
            self.checkpoint_dir = Path(checkpoint_dir)
            self.max_checkpoints = max_checkpoints
            self.checkpoints: List[CheckpointInfo] = []
            self.best_checkpoint: Optional[str] = None
            self._primary_metric = "eval_loss"
            self._metric_mode = "min"  # "min" or "max"

            # Create checkpoint directory
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"CheckpointManager initialized at {checkpoint_dir}")

        def register_checkpoint(
            self,
            path: str,
            step: int,
            metrics: Dict[str, float],
        ) -> CheckpointInfo:
            """
            Register a saved checkpoint.

            Args:
                path: Checkpoint path
                step: Training step
                metrics: Current metrics

            Returns:
                Checkpoint info
            """
            # Check if this is the best checkpoint
            is_best = self._is_better_checkpoint(metrics)

            info = CheckpointInfo(
                path=path,
                step=step,
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
                is_best=is_best,
            )

            self.checkpoints.append(info)

            if is_best:
                self.best_checkpoint = path

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            return info

        def _is_better_checkpoint(self, metrics: Dict[str, float]) -> bool:
            """Check if checkpoint is better than current best."""
            if not self.best_checkpoint or not self.checkpoints:
                return True

            current_value = metrics.get(self._primary_metric)
            if current_value is None:
                return False

            # Find best checkpoint's metric
            best_info = next(
                (c for c in self.checkpoints if c.path == self.best_checkpoint),
                None,
            )

            if best_info is None:
                return True

            best_value = best_info.metrics.get(self._primary_metric)
            if best_value is None:
                return True

            if self._metric_mode == "min":
                return current_value < best_value
            else:
                return current_value > best_value

        def _cleanup_old_checkpoints(self) -> None:
            """Remove old checkpoints exceeding limit."""
            if len(self.checkpoints) <= self.max_checkpoints:
                return

            # Sort by metric (keep best) and step (keep recent)
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda c: (
                    c.metrics.get(self._primary_metric, float("inf"))
                    if self._metric_mode == "min"
                    else -c.metrics.get(self._primary_metric, float("-inf")),
                    -c.step,
                ),
            )

            # Keep best and most recent
            to_keep = sorted_checkpoints[:self.max_checkpoints]
            to_remove = [c for c in self.checkpoints if c not in to_keep]

            for checkpoint in to_remove:
                self._remove_checkpoint_files(checkpoint.path)
                self.checkpoints.remove(checkpoint)

            logger.info(f"Cleaned up {len(to_remove)} old checkpoints")

        def _remove_checkpoint_files(self, path: str) -> None:
            """Remove checkpoint files from disk."""
            import shutil

            checkpoint_path = Path(path)
            if checkpoint_path.exists():
                if checkpoint_path.is_dir():
                    shutil.rmtree(checkpoint_path)
                else:
                    checkpoint_path.unlink()
                logger.info(f"Removed checkpoint: {path}")

        def get_best_checkpoint(self) -> Optional[str]:
            """Get path to best checkpoint."""
            return self.best_checkpoint

        def get_latest_checkpoint(self) -> Optional[str]:
            """Get path to most recent checkpoint."""
            if not self.checkpoints:
                return None

            latest = max(self.checkpoints, key=lambda c: c.step)
            return latest.path

        def list_checkpoints(self) -> List[Dict[str, Any]]:
            """List all checkpoints."""
            return [c.to_dict() for c in self.checkpoints]

        def get_checkpoint_info(self, path: str) -> Optional[Dict[str, Any]]:
            """Get info for specific checkpoint."""
            info = next((c for c in self.checkpoints if c.path == path), None)
            return info.to_dict() if info else None

        def set_primary_metric(
            self,
            metric_name: str,
            mode: str = "min",
        ) -> None:
            """
            Set primary metric for best checkpoint selection.

            Args:
                metric_name: Name of the metric
                mode: "min" for lower is better, "max" for higher is better
            """
            self._primary_metric = metric_name
            self._metric_mode = mode

            # Re-evaluate best checkpoint
            self._update_best_checkpoint()

        def _update_best_checkpoint(self) -> None:
            """Re-evaluate which checkpoint is best."""
            if not self.checkpoints:
                self.best_checkpoint = None
                return

            if self._metric_mode == "min":
                best = min(
                    self.checkpoints,
                    key=lambda c: c.metrics.get(
                        self._primary_metric, float("inf")
                    ),
                )
            else:
                best = max(
                    self.checkpoints,
                    key=lambda c: c.metrics.get(
                        self._primary_metric, float("-inf")
                    ),
                )

            self.best_checkpoint = best.path

            # Update is_best flags
            for c in self.checkpoints:
                c.is_best = (c.path == self.best_checkpoint)

        def get_summary(self) -> Dict[str, Any]:
            """Get checkpoint manager summary."""
            return {
                "checkpoint_dir": str(self.checkpoint_dir),
                "total_checkpoints": len(self.checkpoints),
                "max_checkpoints": self.max_checkpoints,
                "best_checkpoint": self.best_checkpoint,
                "latest_checkpoint": self.get_latest_checkpoint(),
                "primary_metric": self._primary_metric,
                "metric_mode": self._metric_mode,
            }

    return RayCheckpointManager.remote(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
    )


# =============================================================================
# Health Monitor Actor
# =============================================================================

def create_ray_health_monitor(
    check_interval_seconds: float = 30.0,
    failure_threshold: int = 3,
):
    """
    Create a Ray actor for monitoring training health.

    Args:
        check_interval_seconds: Interval between health checks
        failure_threshold: Consecutive failures before alert

    Returns:
        Ray actor handle
    """
    try:
        import ray
    except ImportError:
        raise ImportError("ray required for Ray actors")

    @ray.remote(num_cpus=0.5)
    class RayHealthMonitor:
        """
        Ray actor for monitoring training pipeline health.

        Tracks worker health, detects stalls, and manages alerts.
        """

        def __init__(
            self,
            check_interval_seconds: float = 30.0,
            failure_threshold: int = 3,
        ):
            self.check_interval = check_interval_seconds
            self.failure_threshold = failure_threshold
            self.worker_status: Dict[str, Dict[str, Any]] = {}
            self.alerts: List[Dict[str, Any]] = []
            self._running = True
            self._start_time = time.time()

            logger.info("HealthMonitor initialized")

        def report_worker_health(
            self,
            worker_id: str,
            status: str,
            metrics: Optional[Dict[str, Any]] = None,
        ) -> None:
            """
            Report worker health status.

            Args:
                worker_id: Worker identifier
                status: "healthy", "degraded", or "unhealthy"
                metrics: Optional health metrics
            """
            self.worker_status[worker_id] = {
                "status": status,
                "last_report": datetime.now().isoformat(),
                "metrics": metrics or {},
                "consecutive_failures": (
                    0 if status == "healthy"
                    else self.worker_status.get(worker_id, {}).get(
                        "consecutive_failures", 0
                    ) + 1
                ),
            }

            # Check for alert condition
            failures = self.worker_status[worker_id]["consecutive_failures"]
            if failures >= self.failure_threshold:
                self._create_alert(
                    worker_id=worker_id,
                    alert_type="worker_unhealthy",
                    message=f"Worker {worker_id} has {failures} consecutive failures",
                )

        def _create_alert(
            self,
            worker_id: str,
            alert_type: str,
            message: str,
        ) -> None:
            """Create an alert."""
            alert = {
                "worker_id": worker_id,
                "type": alert_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "resolved": False,
            }

            self.alerts.append(alert)
            logger.warning(f"ALERT: {message}")

        def get_worker_status(self) -> Dict[str, Dict[str, Any]]:
            """Get all worker status."""
            return self.worker_status

        def get_alerts(
            self,
            unresolved_only: bool = True,
        ) -> List[Dict[str, Any]]:
            """Get alerts."""
            if unresolved_only:
                return [a for a in self.alerts if not a["resolved"]]
            return self.alerts

        def resolve_alert(self, index: int) -> bool:
            """Resolve an alert by index."""
            if 0 <= index < len(self.alerts):
                self.alerts[index]["resolved"] = True
                return True
            return False

        def get_summary(self) -> Dict[str, Any]:
            """Get health monitor summary."""
            healthy = sum(
                1 for s in self.worker_status.values()
                if s["status"] == "healthy"
            )
            unhealthy = sum(
                1 for s in self.worker_status.values()
                if s["status"] == "unhealthy"
            )

            return {
                "total_workers": len(self.worker_status),
                "healthy_workers": healthy,
                "unhealthy_workers": unhealthy,
                "unresolved_alerts": len(self.get_alerts()),
                "uptime_seconds": time.time() - self._start_time,
            }

        def stop(self) -> None:
            """Stop the monitor."""
            self._running = False

    return RayHealthMonitor.remote(
        check_interval_seconds=check_interval_seconds,
        failure_threshold=failure_threshold,
    )


# =============================================================================
# Actor Manager
# =============================================================================

class RayActorManager:
    """
    Manager for Ray actors in training pipeline.

    Provides unified interface for creating and managing actors.
    """

    def __init__(self):
        self.actors: Dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize Ray if needed."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self._initialized = True
        except ImportError:
            logger.warning("Ray not available")

    def create_evaluation_actor(
        self,
        checkpoint_dir: str,
        eval_dataset_path: str,
        **kwargs,
    ) -> Any:
        """Create and register evaluation actor."""
        self.initialize()
        actor = create_ray_evaluation_actor(
            checkpoint_dir=checkpoint_dir,
            eval_dataset_path=eval_dataset_path,
            **kwargs,
        )
        self.actors["evaluation"] = actor
        return actor

    def create_metrics_aggregator(self, **kwargs) -> Any:
        """Create and register metrics aggregator."""
        self.initialize()
        actor = create_ray_metrics_aggregator(**kwargs)
        self.actors["metrics"] = actor
        return actor

    def create_checkpoint_manager(
        self,
        checkpoint_dir: str,
        **kwargs,
    ) -> Any:
        """Create and register checkpoint manager."""
        self.initialize()
        actor = create_ray_checkpoint_manager(
            checkpoint_dir=checkpoint_dir,
            **kwargs,
        )
        self.actors["checkpoint"] = actor
        return actor

    def create_health_monitor(self, **kwargs) -> Any:
        """Create and register health monitor."""
        self.initialize()
        actor = create_ray_health_monitor(**kwargs)
        self.actors["health"] = actor
        return actor

    def get_actor(self, name: str) -> Optional[Any]:
        """Get actor by name."""
        return self.actors.get(name)

    def shutdown_all(self) -> None:
        """Shutdown all actors."""
        import ray

        for name, actor in self.actors.items():
            try:
                ray.kill(actor)
                logger.info(f"Killed actor: {name}")
            except Exception as e:
                logger.warning(f"Error killing actor {name}: {e}")

        self.actors.clear()
