"""Base Synthetic Data Generator.

Provides foundation classes for all synthetic data generators with:
- Batch generation with configurable batch_size
- Progress tracking using tqdm
- Checkpoint save/resume functionality
- DVC integration for versioned checkpoints

Default configuration:
- target_count: 2500
- batch_size: 50
- checkpoint_interval: 100
"""

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    
    target_count: int = 2500
    batch_size: int = 50
    checkpoint_interval: int = 100
    model_name: str = "google/medgemma-27b-text-it"
    device: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: Optional[int] = None


@dataclass
class CheckpointData:
    """Checkpoint data structure."""
    
    step: int
    records: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "records": self.records,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointData":
        """Create from dictionary."""
        return cls(
            step=data["step"],
            records=data["records"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", ""),
        )


class BaseSyntheticGenerator(ABC):
    """
    Base class for synthetic data generators.
    
    Provides common functionality:
    - Batch generation with progress tracking (tqdm)
    - Checkpoint save/resume (JSON format)
    - DVC integration for versioned checkpoints
    
    Attributes:
        target_count: Target number of samples to generate (default: 2500)
        batch_size: Number of samples per batch (default: 50)
        checkpoint_interval: Save checkpoint every N samples (default: 100)
        checkpoint_dir: Directory for checkpoint files
        use_dvc: Whether to track checkpoints with DVC
    """
    
    def __init__(
        self,
        target_count: int = 2500,
        batch_size: int = 50,
        checkpoint_interval: int = 100,
        checkpoint_dir: Optional[str] = None,
        use_dvc: bool = True,
        mock_mode: bool = False,
        mock_dvc: bool = False,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the base generator.
        
        Args:
            target_count: Target number of samples to generate
            batch_size: Number of samples per batch
            checkpoint_interval: Save checkpoint every N samples
            checkpoint_dir: Directory for checkpoint files
            use_dvc: Whether to track checkpoints with DVC
            mock_mode: Enable mock mode for testing
            mock_dvc: Mock DVC operations for testing
            config: Optional full configuration object
        """
        self.config = config or GenerationConfig(
            target_count=target_count,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
        )
        
        self.target_count = target_count
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_dvc = use_dvc
        self.mock_mode = mock_mode
        self.mock_dvc = mock_dvc
        
        # Track DVC operations for testing
        self.dvc_tracked_checkpoints = 0
        
        # Ensure checkpoint directory exists
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_progress_bar(
        self,
        total: int,
        desc: str = "Generating",
        unit: str = "samples",
    ) -> tqdm:
        """
        Create a tqdm progress bar.
        
        Args:
            total: Total number of items
            desc: Description for the progress bar
            unit: Unit name for items
            
        Returns:
            tqdm progress bar instance
        """
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            ncols=100,
            leave=True,
        )
    
    def save_checkpoint(
        self,
        data: Union[Dict[str, Any], CheckpointData],
        step: int,
        track_with_dvc: bool = False,
    ) -> Path:
        """
        Save a checkpoint to disk.
        
        Args:
            data: Checkpoint data to save
            step: Current step/count
            track_with_dvc: Whether to track with DVC
            
        Returns:
            Path to saved checkpoint file
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set")
        
        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert CheckpointData to dict if needed
        if isinstance(data, CheckpointData):
            data = data.to_dict()
        
        # Add step if not present
        if "step" not in data:
            data["step"] = step
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.json"
        
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")
        
        # Track with DVC if enabled
        if (track_with_dvc or self.use_dvc) and not self.mock_dvc:
            self._dvc_track(checkpoint_path)
        elif self.mock_dvc and (track_with_dvc or self.use_dvc):
            self.dvc_tracked_checkpoints += 1
            logger.debug(f"Mock DVC tracking for checkpoint {step}")
        
        return checkpoint_path
    
    def load_checkpoint(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint from disk.
        
        Args:
            step: Step number to load
            
        Returns:
            Checkpoint data or None if not found
        """
        if self.checkpoint_dir is None:
            return None
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.json"
        
        if not checkpoint_path.exists():
            return None
        
        with open(checkpoint_path) as f:
            return json.load(f)
    
    def find_latest_checkpoint(self) -> Optional[int]:
        """
        Find the latest checkpoint step.
        
        Returns:
            Latest checkpoint step number or None
        """
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return None
        
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        
        if not checkpoint_files:
            return None
        
        # Extract step numbers from filenames
        steps = []
        for f in checkpoint_files:
            try:
                step = int(f.stem.replace("checkpoint_", ""))
                steps.append(step)
            except ValueError:
                continue
        
        return max(steps) if steps else None
    
    def _dvc_track(self, path: Path):
        """Track a file with DVC."""
        try:
            result = subprocess.run(
                ["dvc", "add", str(path)],
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                self.dvc_tracked_checkpoints += 1
                logger.info(f"Tracked {path} with DVC")
            else:
                logger.warning(f"DVC tracking warning: {result.stderr}")
                
        except FileNotFoundError:
            logger.debug("DVC not available, skipping tracking")
    
    def iter_batches(self) -> Iterator[List[int]]:
        """
        Iterate over batch indices.
        
        Yields:
            List of indices for each batch
        """
        for i in range(0, self.target_count, self.batch_size):
            end = min(i + self.batch_size, self.target_count)
            yield list(range(i, end))
    
    @abstractmethod
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a single synthetic sample.
        
        Must be implemented by subclasses.
        
        Returns:
            Generated sample as dictionary
        """
        pass
    
    def generate_batch(
        self,
        count: Optional[int] = None,
        show_progress: bool = True,
        save_checkpoints: bool = False,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of synthetic samples.
        
        Args:
            count: Number of samples to generate (default: target_count)
            show_progress: Show tqdm progress bar
            save_checkpoints: Save periodic checkpoints
            checkpoint_dir: Override checkpoint directory
            **kwargs: Additional arguments for generate_single
            
        Returns:
            List of generated samples
        """
        count = count or self.target_count
        
        # Update checkpoint dir if provided
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Create progress bar
        pbar = None
        if show_progress:
            pbar = self.create_progress_bar(count, desc="Generating")
        
        try:
            for i in range(count):
                # Generate single sample
                sample = self.generate_single(**kwargs)
                results.append(sample)
                
                # Update progress
                if pbar:
                    pbar.update(1)
                
                # Save checkpoint if needed
                if save_checkpoints and self.checkpoint_dir:
                    if (i + 1) % self.checkpoint_interval == 0:
                        checkpoint_data = {
                            "step": i + 1,
                            "records": results.copy(),
                            "metadata": {
                                "kwargs": {k: str(v) for k, v in kwargs.items()},
                            },
                        }
                        self.save_checkpoint(checkpoint_data, step=i + 1)
        
        finally:
            if pbar:
                pbar.close()
        
        return results
    
    def resume_and_complete(
        self,
        target_count: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Resume from checkpoint and complete generation.
        
        Args:
            target_count: Total target count
            **kwargs: Arguments for generate_single
            
        Returns:
            Complete list of generated samples
        """
        # Find latest checkpoint
        latest_step = self.find_latest_checkpoint()
        
        if latest_step is None:
            # No checkpoint, start fresh
            return self.generate_batch(count=target_count, **kwargs)
        
        # Load checkpoint
        checkpoint_data = self.load_checkpoint(latest_step)
        
        if checkpoint_data is None:
            return self.generate_batch(count=target_count, **kwargs)
        
        # Get existing records
        existing_records = checkpoint_data.get("records", [])
        remaining = target_count - len(existing_records)
        
        if remaining <= 0:
            return existing_records[:target_count]
        
        logger.info(f"Resuming from checkpoint {latest_step}, generating {remaining} more samples")
        
        # Generate remaining samples
        additional = self.generate_batch(
            count=remaining,
            save_checkpoints=True,
            **kwargs,
        )
        
        return existing_records + additional
