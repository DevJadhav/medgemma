"""Checkpoint manager for MedGemma training.

Provides checkpoint saving, loading, and management with support for:
- Local filesystem storage
- MinIO/S3 cloud storage
- Checkpoint versioning and pruning
"""

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    
    checkpoint_id: str
    model_name: str
    step: int
    loss: Optional[float]
    created_at: datetime
    path: str
    is_best: bool = False
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class CheckpointManager:
    """
    Manager for training checkpoints with MinIO/S3 support.
    
    Handles:
    - Saving checkpoints locally and to S3
    - Loading checkpoints from local or S3
    - Pruning old checkpoints based on retention policy
    - Tracking best checkpoints by metric
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "/checkpoints",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "checkpoints",
        max_checkpoints: int = 3,
        use_s3: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Local directory for checkpoints
            s3_bucket: S3/MinIO bucket name
            s3_prefix: Prefix for S3 objects
            max_checkpoints: Maximum checkpoints to keep
            use_s3: Whether to use S3 storage
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket or os.environ.get("CHECKPOINT_BUCKET", "medai-checkpoints")
        self.s3_prefix = s3_prefix
        self.max_checkpoints = max_checkpoints
        self.use_s3 = use_s3
        
        self._minio_client = None
        
    def _get_minio_client(self):
        """Get or create MinIO client."""
        if self._minio_client is None and self.use_s3:
            try:
                from minio import Minio
                
                endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
                access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
                secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
                
                self._minio_client = Minio(
                    endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=False,
                )
                
                # Ensure bucket exists
                if not self._minio_client.bucket_exists(self.s3_bucket):
                    self._minio_client.make_bucket(self.s3_bucket)
                    
            except ImportError:
                logger.warning("MinIO client not available, using local storage only")
                self.use_s3 = False
            except Exception as e:
                logger.warning(f"Failed to connect to MinIO: {e}, using local storage")
                self.use_s3 = False
                
        return self._minio_client
    
    def save(
        self,
        model,
        tokenizer,
        step: int,
        model_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> CheckpointMetadata:
        """
        Save a training checkpoint.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            step: Current training step
            model_name: Name of the model
            metrics: Optional training metrics
            is_best: Whether this is the best checkpoint
            
        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        metrics = metrics or {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{model_name}_{step}_{timestamp}"
        
        # Create local checkpoint directory
        local_path = self.checkpoint_dir / checkpoint_id
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model and tokenizer
            model.save_pretrained(str(local_path))
            tokenizer.save_pretrained(str(local_path))
            
            # Save metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                model_name=model_name,
                step=step,
                loss=metrics.get("loss"),
                created_at=datetime.now(),
                path=str(local_path),
                is_best=is_best,
                metrics=metrics,
            )
            
            self._save_metadata(local_path, metadata)
            
            # Upload to S3 if enabled
            if self.use_s3:
                self._upload_to_s3(local_path, checkpoint_id)
            
            # Prune old checkpoints
            self._prune_checkpoints(model_name)
            
            logger.info(f"Saved checkpoint: {checkpoint_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load(
        self,
        checkpoint_id: Optional[str] = None,
        model_name: Optional[str] = None,
        step: Optional[int] = None,
        load_best: bool = False,
    ) -> Path:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint ID to load
            model_name: Model name to filter checkpoints
            step: Specific step to load
            load_best: Load the best checkpoint
            
        Returns:
            Path to the loaded checkpoint directory
        """
        if checkpoint_id:
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            
            # Try to download from S3 if not local
            if not checkpoint_path.exists() and self.use_s3:
                self._download_from_s3(checkpoint_id, checkpoint_path)
            
            if checkpoint_path.exists():
                return checkpoint_path
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        if load_best:
            return self._get_best_checkpoint(model_name)
        
        return self._get_latest_checkpoint(model_name, step)
    
    def _get_latest_checkpoint(
        self, 
        model_name: Optional[str] = None,
        step: Optional[int] = None
    ) -> Path:
        """Get the latest checkpoint matching criteria."""
        checkpoints = self._list_checkpoints(model_name)
        
        if step:
            checkpoints = [c for c in checkpoints if c.step == step]
        
        if not checkpoints:
            raise FileNotFoundError("No matching checkpoints found")
        
        # Sort by step (descending)
        checkpoints.sort(key=lambda c: c.step, reverse=True)
        return Path(checkpoints[0].path)
    
    def _get_best_checkpoint(self, model_name: Optional[str] = None) -> Path:
        """Get the best checkpoint by loss."""
        checkpoints = self._list_checkpoints(model_name)
        
        # Filter to those with loss metric
        checkpoints = [c for c in checkpoints if c.loss is not None]
        
        if not checkpoints:
            raise FileNotFoundError("No checkpoints with loss metric found")
        
        # Sort by loss (ascending)
        checkpoints.sort(key=lambda c: c.loss)
        return Path(checkpoints[0].path)
    
    def _list_checkpoints(
        self, 
        model_name: Optional[str] = None
    ) -> List[CheckpointMetadata]:
        """List all checkpoints, optionally filtered by model name."""
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            metadata_path = checkpoint_dir / "metadata.yaml"
            if metadata_path.exists():
                metadata = self._load_metadata(checkpoint_dir)
                
                if model_name is None or metadata.model_name == model_name:
                    checkpoints.append(metadata)
        
        return checkpoints
    
    def _save_metadata(self, checkpoint_path: Path, metadata: CheckpointMetadata):
        """Save checkpoint metadata to YAML file."""
        import yaml
        
        metadata_path = checkpoint_path / "metadata.yaml"
        data = {
            "checkpoint_id": metadata.checkpoint_id,
            "model_name": metadata.model_name,
            "step": metadata.step,
            "loss": metadata.loss,
            "created_at": metadata.created_at.isoformat(),
            "is_best": metadata.is_best,
            "metrics": metadata.metrics,
        }
        
        with open(metadata_path, "w") as f:
            yaml.dump(data, f)
    
    def _load_metadata(self, checkpoint_path: Path) -> CheckpointMetadata:
        """Load checkpoint metadata from YAML file."""
        import yaml
        
        metadata_path = checkpoint_path / "metadata.yaml"
        
        with open(metadata_path) as f:
            data = yaml.safe_load(f)
        
        return CheckpointMetadata(
            checkpoint_id=data["checkpoint_id"],
            model_name=data["model_name"],
            step=data["step"],
            loss=data.get("loss"),
            created_at=datetime.fromisoformat(data["created_at"]),
            path=str(checkpoint_path),
            is_best=data.get("is_best", False),
            metrics=data.get("metrics", {}),
        )
    
    def _upload_to_s3(self, local_path: Path, checkpoint_id: str):
        """Upload checkpoint to S3/MinIO."""
        client = self._get_minio_client()
        if not client:
            return
        
        try:
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    object_name = f"{self.s3_prefix}/{checkpoint_id}/{relative_path}"
                    
                    client.fput_object(
                        self.s3_bucket,
                        object_name,
                        str(file_path),
                    )
            
            logger.info(f"Uploaded checkpoint to s3://{self.s3_bucket}/{self.s3_prefix}/{checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to S3: {e}")
    
    def _download_from_s3(self, checkpoint_id: str, local_path: Path):
        """Download checkpoint from S3/MinIO."""
        client = self._get_minio_client()
        if not client:
            return
        
        try:
            local_path.mkdir(parents=True, exist_ok=True)
            prefix = f"{self.s3_prefix}/{checkpoint_id}/"
            
            objects = client.list_objects(self.s3_bucket, prefix=prefix, recursive=True)
            
            for obj in objects:
                relative_path = obj.object_name[len(prefix):]
                file_path = local_path / relative_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                client.fget_object(self.s3_bucket, obj.object_name, str(file_path))
            
            logger.info(f"Downloaded checkpoint from S3: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to download checkpoint from S3: {e}")
    
    def _prune_checkpoints(self, model_name: str):
        """Prune old checkpoints, keeping only max_checkpoints."""
        checkpoints = self._list_checkpoints(model_name)
        
        # Sort by step (descending)
        checkpoints.sort(key=lambda c: c.step, reverse=True)
        
        # Keep best checkpoint regardless of step
        best_checkpoint = min(
            (c for c in checkpoints if c.loss is not None),
            key=lambda c: c.loss,
            default=None
        )
        
        to_delete = []
        kept = 0
        
        for checkpoint in checkpoints:
            is_best = best_checkpoint and checkpoint.checkpoint_id == best_checkpoint.checkpoint_id
            
            if kept < self.max_checkpoints or is_best:
                kept += 1
            else:
                to_delete.append(checkpoint)
        
        for checkpoint in to_delete:
            self._delete_checkpoint(checkpoint)
    
    def _delete_checkpoint(self, metadata: CheckpointMetadata):
        """Delete a checkpoint from local and S3 storage."""
        # Delete local
        checkpoint_path = Path(metadata.path)
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            logger.info(f"Deleted local checkpoint: {metadata.checkpoint_id}")
        
        # Delete from S3
        if self.use_s3:
            client = self._get_minio_client()
            if client:
                try:
                    prefix = f"{self.s3_prefix}/{metadata.checkpoint_id}/"
                    objects = client.list_objects(self.s3_bucket, prefix=prefix, recursive=True)
                    
                    for obj in objects:
                        client.remove_object(self.s3_bucket, obj.object_name)
                    
                    logger.info(f"Deleted S3 checkpoint: {metadata.checkpoint_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to delete S3 checkpoint: {e}")


# Convenience functions
def save_checkpoint(
    model,
    tokenizer,
    step: int,
    model_name: str,
    checkpoint_dir: str = "/checkpoints",
    metrics: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> CheckpointMetadata:
    """
    Save a training checkpoint.
    
    Convenience function for CheckpointManager.save().
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    return manager.save(model, tokenizer, step, model_name, metrics, is_best)


def load_checkpoint(
    checkpoint_id: Optional[str] = None,
    model_name: Optional[str] = None,
    checkpoint_dir: str = "/checkpoints",
    load_best: bool = False,
) -> Path:
    """
    Load a checkpoint.
    
    Convenience function for CheckpointManager.load().
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    return manager.load(checkpoint_id, model_name, load_best=load_best)


def get_latest_checkpoint(
    model_name: Optional[str] = None,
    checkpoint_dir: str = "/checkpoints",
) -> Path:
    """
    Get the latest checkpoint for a model.
    
    Convenience function for getting the most recent checkpoint.
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    return manager.load(model_name=model_name)
