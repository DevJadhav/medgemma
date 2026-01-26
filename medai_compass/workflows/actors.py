"""
Ray Actors for background processing.

Provides long-running actors for:
- Continuous evaluation
- Metrics aggregation
- Checkpoint management
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluationActor:
    """
    Actor for continuous model evaluation.

    Runs in background, evaluating checkpoints as they're saved.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        eval_dataset_path: str,
        benchmarks: Optional[List[str]] = None,
    ):
        """
        Initialize evaluation actor.

        Args:
            checkpoint_dir: Directory to watch for checkpoints
            eval_dataset_path: Path to evaluation dataset
            benchmarks: List of benchmarks to run
        """
        self.checkpoint_dir = checkpoint_dir
        self.eval_dataset_path = eval_dataset_path
        self.benchmarks = benchmarks or ["medqa", "pubmedqa"]
        self.results: Dict[str, Dict[str, Any]] = {}
        self.running = True

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Evaluate a single checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Evaluation results
        """
        try:
            from medai_compass.evaluation import run_evaluation

            result = run_evaluation(
                model_path=checkpoint_path,
                dataset_path=self.eval_dataset_path,
                benchmarks=self.benchmarks,
            )

            self.results[checkpoint_path] = {
                "timestamp": datetime.now().isoformat(),
                "metrics": result,
            }

            return result

        except Exception as e:
            logger.error(f"Evaluation failed for {checkpoint_path}: {e}")
            return {"error": str(e)}

    def get_results(self) -> Dict[str, Any]:
        """Get all evaluation results."""
        return self.results

    def get_best_checkpoint(self, metric: str = "accuracy") -> Optional[str]:
        """
        Get best checkpoint based on metric.

        Args:
            metric: Metric to optimize

        Returns:
            Path to best checkpoint
        """
        if not self.results:
            return None

        best_path = max(
            self.results.keys(),
            key=lambda p: self.results[p]["metrics"].get(metric, 0),
        )
        return best_path

    def stop(self) -> None:
        """Stop the actor."""
        self.running = False


class MetricsAggregator:
    """
    Actor for aggregating training metrics.

    Collects metrics from distributed workers.
    """

    def __init__(self):
        """Initialize metrics aggregator."""
        self.metrics: List[Dict[str, Any]] = []
        self.current_step = 0

    def add_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Add metrics from a worker.

        Args:
            metrics: Metrics dictionary
            step: Training step
        """
        self.metrics.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        })
        self.current_step = max(self.current_step, step)

    def get_metrics(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get collected metrics.

        Args:
            last_n: Return only last N metrics

        Returns:
            List of metrics
        """
        if last_n:
            return self.metrics[-last_n:]
        return self.metrics

    def get_current_step(self) -> int:
        """Get current training step."""
        return self.current_step

    def get_average_metrics(self, window: int = 100) -> Dict[str, float]:
        """
        Get average metrics over window.

        Args:
            window: Number of steps to average

        Returns:
            Averaged metrics
        """
        recent = self.metrics[-window:] if len(self.metrics) >= window else self.metrics

        if not recent:
            return {}

        avg_metrics = {}
        metric_keys = [k for k in recent[0].keys() if k not in ["step", "timestamp"]]

        for key in metric_keys:
            values = [
                m[key] for m in recent
                if key in m and isinstance(m[key], (int, float))
            ]
            if values:
                avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_steps": self.current_step,
            "num_entries": len(self.metrics),
            "latest_metrics": self.metrics[-1] if self.metrics else None,
            "averages": self.get_average_metrics(),
        }


class CheckpointManager:
    """
    Actor for checkpoint management.

    Handles checkpoint saving, loading, and cleanup.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Dict[str, Any]] = []

    def save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, float],
        model_path: str,
    ) -> str:
        """
        Record a saved checkpoint.

        Args:
            step: Training step
            metrics: Current metrics
            model_path: Path where checkpoint was saved

        Returns:
            Checkpoint path
        """
        checkpoint_info = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "path": model_path,
        }

        self.checkpoints.append(checkpoint_info)

        # Cleanup old checkpoints if needed
        self._cleanup_old_checkpoints()

        return model_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max limit."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by metric (keep best) and step (keep recent)
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda c: (c["metrics"].get("eval_loss", float("inf")), -c["step"]),
            )

            # Keep best and most recent
            to_keep = sorted_checkpoints[:self.max_checkpoints]
            to_remove = [c for c in self.checkpoints if c not in to_keep]

            for checkpoint in to_remove:
                self._remove_checkpoint(checkpoint["path"])
                self.checkpoints.remove(checkpoint)

    def _remove_checkpoint(self, path: str) -> None:
        """Remove checkpoint files."""
        import shutil
        from pathlib import Path

        checkpoint_path = Path(path)
        if checkpoint_path.exists():
            if checkpoint_path.is_dir():
                shutil.rmtree(checkpoint_path)
            else:
                checkpoint_path.unlink()

    def get_best_checkpoint(self, metric: str = "eval_loss") -> Optional[str]:
        """
        Get path to best checkpoint.

        Args:
            metric: Metric to optimize (lower is better for loss)

        Returns:
            Path to best checkpoint
        """
        if not self.checkpoints:
            return None

        best = min(
            self.checkpoints,
            key=lambda c: c["metrics"].get(metric, float("inf")),
        )
        return best["path"]

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None

        latest = max(self.checkpoints, key=lambda c: c["step"])
        return latest["path"]

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return self.checkpoints
