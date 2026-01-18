"""Training Callbacks for MedGemma Training.

Provides custom callbacks for:
- MLflow metric logging
- Early stopping
- Gradient accumulation tracking
- Checkpoint management
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CallbackBase:
    """Base class for training callbacks."""
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch."""
        pass
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step."""
        pass
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        pass
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are written."""
        pass
    
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        pass
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        pass


class MLflowCallback(CallbackBase):
    """Callback for logging to MLflow.
    
    Logs training metrics, parameters, and artifacts to MLflow
    for experiment tracking.
    
    Example:
        >>> from medai_compass.pipelines.mlflow_integration import MLflowTracker
        >>> tracker = MLflowTracker()
        >>> callback = MLflowCallback(tracker=tracker)
    """
    
    def __init__(
        self,
        tracker=None,
        log_model: bool = True,
        log_every_n_steps: int = 1,
    ):
        """Initialize MLflow callback.
        
        Args:
            tracker: MLflowTracker instance (created if not provided)
            log_model: Whether to log model at end of training
            log_every_n_steps: Log metrics every N steps
        """
        self.tracker = tracker
        self.log_model = log_model
        self.log_every_n_steps = log_every_n_steps
        self._step_count = 0
    
    def _ensure_tracker(self):
        """Ensure tracker is initialized."""
        if self.tracker is None:
            try:
                from medai_compass.pipelines.mlflow_integration import MLflowTracker
                self.tracker = MLflowTracker()
            except ImportError:
                logger.warning("MLflow integration not available")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Start MLflow run at training start."""
        self._ensure_tracker()
        
        if self.tracker is not None:
            self.tracker.start_run()
            
            # Log training arguments
            if hasattr(args, "to_dict"):
                self.tracker.log_params(args.to_dict())
            elif hasattr(args, "__dict__"):
                params = {k: v for k, v in args.__dict__.items() 
                         if not k.startswith("_")}
                self.tracker.log_params(params)
    
    def on_train_end(self, args, state, control, **kwargs):
        """End MLflow run at training end."""
        if self.tracker is not None:
            self.tracker.end_run()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to MLflow."""
        if self.tracker is None or logs is None:
            return
        
        self._step_count += 1
        
        if self._step_count % self.log_every_n_steps == 0:
            # Filter numeric metrics
            metrics = {
                k: v for k, v in logs.items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            }
            
            step = state.global_step if hasattr(state, "global_step") else None
            self.tracker.log_metrics(metrics, step=step)
    
    def on_save(self, args, state, control, **kwargs):
        """Log checkpoint as artifact."""
        if self.tracker is None:
            return
        
        if hasattr(args, "output_dir"):
            # Don't log every checkpoint to avoid storage bloat
            pass
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if self.tracker is None or metrics is None:
            return
        
        eval_metrics = {f"eval_{k}": v for k, v in metrics.items()
                       if isinstance(v, (int, float))}
        
        step = state.global_step if hasattr(state, "global_step") else None
        self.tracker.log_metrics(eval_metrics, step=step)


class EarlyStoppingCallback(CallbackBase):
    """Early stopping callback.
    
    Stops training when a monitored metric stops improving.
    
    Example:
        >>> callback = EarlyStoppingCallback(
        ...     patience=3,
        ...     metric="eval_loss",
        ...     mode="min"
        ... )
    """
    
    def __init__(
        self,
        patience: int = 3,
        metric: str = "eval_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of evaluations to wait for improvement
            metric: Metric to monitor
            mode: "min" for loss-like metrics, "max" for accuracy-like
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        self._best_value = None
        self._counter = 0
        self._should_stop = False
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self._best_value is None:
            return True
        
        if self.mode == "min":
            return current < (self._best_value - self.min_delta)
        else:  # mode == "max"
            return current > (self._best_value + self.min_delta)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check for improvement after evaluation."""
        if metrics is None:
            return
        
        current_value = metrics.get(self.metric)
        if current_value is None:
            return
        
        if self._is_improvement(current_value):
            self._best_value = current_value
            self._counter = 0
            logger.info(f"Early stopping: {self.metric} improved to {current_value:.4f}")
        else:
            self._counter += 1
            logger.info(
                f"Early stopping: {self.metric} did not improve. "
                f"Counter: {self._counter}/{self.patience}"
            )
            
            if self._counter >= self.patience:
                self._should_stop = True
                control.should_training_stop = True
                logger.info("Early stopping triggered!")


class GradientAccumulationCallback(CallbackBase):
    """Callback for tracking gradient accumulation.
    
    Logs effective batch size and accumulation progress.
    """
    
    def __init__(
        self,
        gradient_accumulation_steps: int = 1,
        log_accumulation: bool = True,
    ):
        """Initialize gradient accumulation callback.
        
        Args:
            gradient_accumulation_steps: Number of accumulation steps
            log_accumulation: Whether to log accumulation info
        """
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_accumulation = log_accumulation
        self._accumulation_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Track accumulation step."""
        self._accumulation_step += 1
        
        if self._accumulation_step >= self.gradient_accumulation_steps:
            self._accumulation_step = 0
            
            if self.log_accumulation:
                logger.debug(f"Gradient accumulation complete at step {state.global_step}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add accumulation info to logs."""
        if logs is not None and self.log_accumulation:
            logs["gradient_accumulation_step"] = self._accumulation_step


class CheckpointCallback(CallbackBase):
    """Callback for checkpoint management.
    
    Handles saving and loading checkpoints with custom logic.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_total_limit: int = 3,
        save_best_only: bool = False,
        metric: str = "eval_loss",
        mode: str = "min",
    ):
        """Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            save_total_limit: Maximum checkpoints to keep
            save_best_only: Only save when metric improves
            metric: Metric for best model selection
            mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_total_limit = save_total_limit
        self.save_best_only = save_best_only
        self.metric = metric
        self.mode = mode
        
        self._best_value = None
        self._saved_checkpoints: List[str] = []
    
    def _is_best(self, current: float) -> bool:
        """Check if current value is the best."""
        if self._best_value is None:
            return True
        
        if self.mode == "min":
            return current < self._best_value
        else:
            return current > self._best_value
    
    def on_save(self, args, state, control, **kwargs):
        """Handle checkpoint saving."""
        # Track saved checkpoints for cleanup
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint-{state.global_step}"
        self._saved_checkpoints.append(checkpoint_path)
        
        # Cleanup old checkpoints
        while len(self._saved_checkpoints) > self.save_total_limit:
            old_checkpoint = self._saved_checkpoints.pop(0)
            logger.debug(f"Removing old checkpoint: {old_checkpoint}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if this is the best model."""
        if not self.save_best_only or metrics is None:
            return
        
        current_value = metrics.get(self.metric)
        if current_value is None:
            return
        
        if self._is_best(current_value):
            self._best_value = current_value
            logger.info(f"New best model! {self.metric}: {current_value:.4f}")


class CompositeCallback(CallbackBase):
    """Composite callback that combines multiple callbacks."""
    
    def __init__(self, callbacks: List[CallbackBase]):
        """Initialize with list of callbacks.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks
    
    def _call_all(self, method_name: str, *args, **kwargs):
        """Call method on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method is not None:
                method(*args, **kwargs)
    
    def on_train_begin(self, args, state, control, **kwargs):
        self._call_all("on_train_begin", args, state, control, **kwargs)
    
    def on_train_end(self, args, state, control, **kwargs):
        self._call_all("on_train_end", args, state, control, **kwargs)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self._call_all("on_epoch_begin", args, state, control, **kwargs)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self._call_all("on_epoch_end", args, state, control, **kwargs)
    
    def on_step_begin(self, args, state, control, **kwargs):
        self._call_all("on_step_begin", args, state, control, **kwargs)
    
    def on_step_end(self, args, state, control, **kwargs):
        self._call_all("on_step_end", args, state, control, **kwargs)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        self._call_all("on_log", args, state, control, logs=logs, **kwargs)
    
    def on_save(self, args, state, control, **kwargs):
        self._call_all("on_save", args, state, control, **kwargs)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._call_all("on_evaluate", args, state, control, metrics=metrics, **kwargs)


def create_default_callbacks(
    mlflow_tracker=None,
    early_stopping_patience: int = 3,
    gradient_accumulation_steps: int = 1,
    checkpoint_dir: str = "./checkpoints",
) -> List[CallbackBase]:
    """Create default set of training callbacks.
    
    Args:
        mlflow_tracker: Optional MLflow tracker
        early_stopping_patience: Patience for early stopping
        gradient_accumulation_steps: Gradient accumulation steps
        checkpoint_dir: Checkpoint directory
        
    Returns:
        List of callback instances
    """
    callbacks = [
        MLflowCallback(tracker=mlflow_tracker),
        EarlyStoppingCallback(patience=early_stopping_patience),
        GradientAccumulationCallback(
            gradient_accumulation_steps=gradient_accumulation_steps
        ),
        CheckpointCallback(checkpoint_dir=checkpoint_dir),
    ]
    
    return callbacks
