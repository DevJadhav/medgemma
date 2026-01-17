"""Training metrics collection and logging.

Provides utilities for:
- Logging training metrics to MLflow
- Collecting GPU utilization metrics
- Tracking training progress
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of training metrics at a point in time."""
    
    step: int
    timestamp: datetime
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    throughput: Optional[float] = None  # samples/sec
    gpu_memory_used_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    """
    Tracker for training metrics with MLflow integration.
    
    Collects and logs:
    - Training loss and learning rate
    - GPU memory and utilization
    - Throughput (samples/second)
    - Custom metrics
    """
    
    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize metrics tracker.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
            run_name: MLflow run name
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.experiment_name = experiment_name
        self.run_name = run_name
        
        self._mlflow_run = None
        self._history: List[MetricSnapshot] = []
        self._start_time = None
        self._samples_processed = 0
    
    def start(self, params: Optional[Dict[str, Any]] = None):
        """
        Start metrics tracking.
        
        Args:
            params: Training parameters to log
        """
        self._start_time = time.time()
        self._samples_processed = 0
        
        try:
            import mlflow
            
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
            
            self._mlflow_run = mlflow.start_run(run_name=self.run_name)
            
            if params:
                mlflow.log_params(params)
                
            logger.info(f"Started MLflow run: {self._mlflow_run.info.run_id}")
            
        except ImportError:
            logger.warning("MLflow not available, metrics will only be stored locally")
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
    
    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        batch_size: int = 1,
        **additional_metrics,
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Current training step
            loss: Training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            batch_size: Batch size (for throughput calculation)
            **additional_metrics: Any additional metrics to log
        """
        # Update samples processed
        self._samples_processed += batch_size
        
        # Calculate throughput
        elapsed = time.time() - self._start_time if self._start_time else 0
        throughput = self._samples_processed / elapsed if elapsed > 0 else 0
        
        # Get GPU metrics
        gpu_metrics = get_gpu_metrics()
        
        # Create snapshot
        snapshot = MetricSnapshot(
            step=step,
            timestamp=datetime.now(),
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            throughput=throughput,
            gpu_memory_used_gb=gpu_metrics.get("gpu_memory_used_gb"),
            gpu_utilization=gpu_metrics.get("gpu_utilization"),
            additional_metrics=additional_metrics,
        )
        
        self._history.append(snapshot)
        
        # Log to MLflow
        self._log_to_mlflow(snapshot)
    
    def _log_to_mlflow(self, snapshot: MetricSnapshot):
        """Log metrics to MLflow."""
        try:
            import mlflow
            
            metrics = {
                "loss": snapshot.loss,
                "learning_rate": snapshot.learning_rate,
            }
            
            if snapshot.grad_norm is not None:
                metrics["grad_norm"] = snapshot.grad_norm
            
            if snapshot.throughput is not None:
                metrics["throughput_samples_sec"] = snapshot.throughput
            
            if snapshot.gpu_memory_used_gb is not None:
                metrics["gpu_memory_used_gb"] = snapshot.gpu_memory_used_gb
            
            if snapshot.gpu_utilization is not None:
                metrics["gpu_utilization"] = snapshot.gpu_utilization
            
            metrics.update(snapshot.additional_metrics)
            
            mlflow.log_metrics(metrics, step=snapshot.step)
            
        except Exception as e:
            logger.debug(f"Failed to log to MLflow: {e}")
    
    def finish(self, final_metrics: Optional[Dict[str, Any]] = None):
        """
        Finish metrics tracking.
        
        Args:
            final_metrics: Final metrics to log
        """
        try:
            import mlflow
            
            if final_metrics:
                mlflow.log_metrics(final_metrics)
            
            if self._mlflow_run:
                mlflow.end_run()
                logger.info(f"Finished MLflow run: {self._mlflow_run.info.run_id}")
                
        except Exception as e:
            logger.warning(f"Failed to finish MLflow run: {e}")
    
    def get_history(self) -> List[MetricSnapshot]:
        """Get full metrics history."""
        return self._history.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from history."""
        if not self._history:
            return {}
        
        losses = [s.loss for s in self._history]
        
        return {
            "total_steps": len(self._history),
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "avg_loss": sum(losses) / len(losses),
            "total_samples": self._samples_processed,
            "avg_throughput": self._samples_processed / (time.time() - self._start_time)
            if self._start_time else 0,
        }


def log_training_step(
    metrics: Dict[str, Any],
    step: int,
    mlflow_tracking_uri: Optional[str] = None,
):
    """
    Log training step metrics to MLflow.
    
    Convenience function for simple metric logging.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        mlflow_tracking_uri: Optional MLflow tracking URI
    """
    try:
        import mlflow
        
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        mlflow.log_metrics(metrics, step=step)
        
    except ImportError:
        logger.debug("MLflow not available")
    except Exception as e:
        logger.debug(f"Failed to log metrics: {e}")


def get_gpu_metrics() -> Dict[str, Any]:
    """
    Get current GPU utilization metrics.
    
    Returns:
        Dictionary with:
        - gpu_memory_used_gb: Memory used in GB
        - gpu_memory_total_gb: Total memory in GB
        - gpu_utilization: GPU utilization percentage
        - gpu_temperature: GPU temperature in Celsius
    """
    metrics = {}
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Memory metrics
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            
            metrics["gpu_memory_used_gb"] = memory_allocated / (1024 ** 3)
            metrics["gpu_memory_reserved_gb"] = memory_reserved / (1024 ** 3)
            
            # Device info
            device_props = torch.cuda.get_device_properties(0)
            metrics["gpu_memory_total_gb"] = device_props.total_memory / (1024 ** 3)
            metrics["gpu_name"] = device_props.name
            
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to get PyTorch GPU metrics: {e}")
    
    # Try to get utilization from nvidia-smi via pynvml
    try:
        import pynvml
        
        pynvml.nvmlInit()
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics["gpu_utilization"] = util.gpu
        metrics["gpu_memory_utilization"] = util.memory
        
        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        metrics["gpu_temperature_celsius"] = temp
        
        pynvml.nvmlShutdown()
        
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to get NVML metrics: {e}")
    
    return metrics


def get_system_metrics() -> Dict[str, Any]:
    """
    Get system resource metrics.
    
    Returns:
        Dictionary with CPU, memory, and disk metrics
    """
    metrics = {}
    
    try:
        import psutil
        
        # CPU
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        metrics["cpu_count"] = psutil.cpu_count()
        
        # Memory
        memory = psutil.virtual_memory()
        metrics["memory_used_gb"] = memory.used / (1024 ** 3)
        metrics["memory_total_gb"] = memory.total / (1024 ** 3)
        metrics["memory_percent"] = memory.percent
        
        # Disk
        disk = psutil.disk_usage("/")
        metrics["disk_used_gb"] = disk.used / (1024 ** 3)
        metrics["disk_total_gb"] = disk.total / (1024 ** 3)
        metrics["disk_percent"] = disk.percent
        
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to get system metrics: {e}")
    
    return metrics
