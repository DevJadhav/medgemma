"""MLflow Integration Module for MedGemma Training.

Provides experiment tracking, model registry, and artifact management
using MLflow for MedGemma fine-tuning pipelines.

Phase 3 Deliverable: medai_compass/pipelines/mlflow_integration.py
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiment.
    
    Attributes:
        name: Experiment name
        tags: Experiment tags
        description: Experiment description
    """
    
    name: str
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    def get_default_tags(self) -> Dict[str, str]:
        """Get default tags for experiment.
        
        Returns:
            Dictionary of default tags
        """
        return {
            "framework": "transformers",
            "library": "peft",
            "task": "instruction-tuning",
            "project": "medai-compass",
        }
    
    def get_all_tags(self) -> Dict[str, str]:
        """Get all tags including defaults.
        
        Returns:
            Merged dictionary of tags
        """
        all_tags = self.get_default_tags()
        all_tags.update(self.tags)
        return all_tags


class MLflowTracker:
    """MLflow experiment tracker for MedGemma training.
    
    Provides:
    - Experiment management
    - Metric and parameter logging
    - Artifact management
    - Model registry integration
    
    Example:
        >>> tracker = MLflowTracker(
        ...     tracking_uri="http://localhost:5000",
        ...     experiment_name="medgemma-finetuning"
        ... )
        >>> tracker.start_run(run_name="training-v1")
        >>> tracker.log_params({"learning_rate": 2e-4})
        >>> tracker.log_metrics({"loss": 0.5}, step=100)
        >>> tracker.end_run()
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "medgemma-finetuning",
        artifact_location: Optional[str] = None,
    ):
        """Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: Optional artifact storage location
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        
        self._run_id: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._mlflow = None
        
        # Initialize MLflow
        self._init_mlflow()
    
    def _init_mlflow(self):
        """Initialize MLflow connection."""
        try:
            import mlflow
            self._mlflow = mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location,
                )
            else:
                self._experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            
            logger.info(f"MLflow initialized: {self.tracking_uri}")
            
        except ImportError:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            self._mlflow = None
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self._mlflow = None
    
    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._run_id
    
    @property
    def is_available(self) -> bool:
        """Check if MLflow is available."""
        return self._mlflow is not None
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Optional[str]:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            nested: Whether this is a nested run
            
        Returns:
            Run ID or None if MLflow not available
        """
        if not self.is_available:
            logger.warning("MLflow not available, skipping run start")
            return None
        
        # Generate run name if not provided
        if run_name is None:
            run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            run = self._mlflow.start_run(
                run_name=run_name,
                nested=nested,
            )
            self._run_id = run.info.run_id
            
            # Set tags
            if tags:
                self._mlflow.set_tags(tags)
            
            logger.info(f"Started MLflow run: {self._run_id}")
            return self._run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.is_available or self._run_id is None:
            return
        
        try:
            self._mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self._run_id}")
            self._run_id = None
            
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        if not self.is_available:
            return
        
        try:
            # Convert non-string values
            clean_params = {}
            for key, value in params.items():
                if isinstance(value, (list, dict)):
                    clean_params[key] = str(value)
                else:
                    clean_params[key] = value
            
            self._mlflow.log_params(clean_params)
            
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
        """
        if not self.is_available:
            return
        
        try:
            self._mlflow.log_metrics(metrics, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ):
        """Log a single metric to MLflow.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.is_available:
            return
        
        try:
            self._mlflow.log_metric(key, value, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log metric {key}: {e}")
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ):
        """Log an artifact to MLflow.
        
        Args:
            local_path: Path to local file or directory
            artifact_path: Optional path within artifact store
        """
        if not self.is_available:
            return
        
        try:
            path = Path(local_path)
            
            if path.is_dir():
                self._mlflow.log_artifacts(str(path), artifact_path)
            else:
                self._mlflow.log_artifact(str(path), artifact_path)
            
            logger.info(f"Logged artifact: {local_path}")
            
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """Log a model to MLflow.
        
        Args:
            model: Model to log
            artifact_path: Path within artifact store
            registered_model_name: Optional name for model registry
        """
        if not self.is_available:
            return
        
        try:
            # Use transformers flavor if available
            try:
                import mlflow.transformers
                mlflow.transformers.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                )
            except (ImportError, AttributeError):
                # Fallback to pyfunc
                self._mlflow.pyfunc.log_model(
                    artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name,
                )
            
            logger.info(f"Logged model: {artifact_path}")
            
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def register_model(
        self,
        model_path: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Register a model in the Model Registry.
        
        Args:
            model_path: Path to model artifacts
            name: Name for the registered model
            tags: Optional tags for the model
            
        Returns:
            Model URI or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            # Register model
            model_uri = f"runs:/{self._run_id}/{model_path}" if self._run_id else model_path
            
            result = self._mlflow.register_model(
                model_uri=model_uri,
                name=name,
                tags=tags,
            )
            
            logger.info(f"Registered model: {name} (version {result.version})")
            return f"models:/{name}/{result.version}"
            
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
            return None
    
    def transition_model_stage(
        self,
        name: str,
        stage: str,
        version: Optional[int] = None,
        archive_existing: bool = True,
    ):
        """Transition model to a new stage.
        
        Args:
            name: Registered model name
            stage: Target stage (Staging, Production, Archived)
            version: Model version (latest if not specified)
            archive_existing: Whether to archive existing models in stage
        """
        if not self.is_available:
            return
        
        try:
            client = self._mlflow.tracking.MlflowClient()
            
            # Get latest version if not specified
            if version is None:
                versions = client.get_latest_versions(name, stages=["None"])
                if versions:
                    version = versions[0].version
                else:
                    logger.warning(f"No versions found for model: {name}")
                    return
            
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            
            logger.info(f"Transitioned {name} v{version} to {stage}")
            
        except Exception as e:
            logger.warning(f"Failed to transition model stage: {e}")
    
    def get_run_metrics(self, run_id: Optional[str] = None) -> Dict[str, float]:
        """Get metrics for a run.
        
        Args:
            run_id: Run ID (current run if not specified)
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_available:
            return {}
        
        try:
            run_id = run_id or self._run_id
            if run_id is None:
                return {}
            
            run = self._mlflow.get_run(run_id)
            return run.data.metrics
            
        except Exception as e:
            logger.warning(f"Failed to get run metrics: {e}")
            return {}
    
    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for runs in the experiment.
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum results to return
            order_by: List of columns to order by
            
        Returns:
            List of run dictionaries
        """
        if not self.is_available:
            return []
        
        try:
            runs = self._mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
            )
            
            return runs.to_dict(orient="records")
            
        except Exception as e:
            logger.warning(f"Failed to search runs: {e}")
            return []


class MLflowCallbackMixin:
    """Mixin for adding MLflow logging to training callbacks."""
    
    def __init__(self, tracker: Optional[MLflowTracker] = None, **kwargs):
        """Initialize with optional MLflow tracker.
        
        Args:
            tracker: MLflowTracker instance
        """
        super().__init__(**kwargs)
        self.tracker = tracker
    
    def _log_to_mlflow(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics to MLflow if tracker available.
        
        Args:
            metrics: Metrics to log
            step: Step number
        """
        if self.tracker is not None:
            self.tracker.log_metrics(metrics, step=step)
