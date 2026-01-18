"""Synthetic Data Versioning (Task 5.6).

DVC-integrated versioning for synthetic data:
- Version synthetic datasets with metadata
- Track generation checkpoints
- Push/pull to remote storage
- Generation metadata tracking

Extends existing versioning from medai_compass.pipelines.versioning.
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataVersion:
    """Represents a versioned synthetic dataset."""
    
    name: str
    hash: str
    path: str
    created_at: str
    size_bytes: int = 0
    record_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    generator_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "hash": self.hash,
            "path": self.path,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
            "metadata": self.metadata,
            "generator_info": self.generator_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticDataVersion":
        """Create from dictionary."""
        return cls(**data)


class SyntheticDataVersionManager:
    """
    Version manager for synthetic data with DVC integration.
    
    Provides:
    - Versioning of synthetic datasets
    - Checkpoint versioning during generation
    - Remote storage integration via DVC
    - Generation metadata tracking
    
    Attributes:
        repo_path: Path to the DVC repository
        remote_url: URL for remote storage
        mock_dvc: Mock DVC operations for testing
    """
    
    def __init__(
        self,
        repo_path: Optional[str] = None,
        remote_url: Optional[str] = None,
        auto_init: bool = True,
        mock_dvc: bool = False,
    ):
        """
        Initialize the version manager.
        
        Args:
            repo_path: Path to DVC repository
            remote_url: URL for remote storage (S3, GCS, etc.)
            auto_init: Auto-initialize DVC if not present
            mock_dvc: Mock DVC operations for testing
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.remote_url = remote_url
        self.mock_dvc = mock_dvc
        
        # Version registry
        self._registry_path = self.repo_path / ".synthetic_versions.json"
        self._versions: Dict[str, SyntheticDataVersion] = {}
        
        # Load existing versions
        self._load_registry()
        
        # Initialize DVC if needed
        if auto_init and not mock_dvc:
            self._ensure_dvc_init()
        
        logger.info(f"Initialized SyntheticDataVersionManager at {self.repo_path}")
    
    def _ensure_dvc_init(self):
        """Ensure DVC is initialized in the repository."""
        dvc_dir = self.repo_path / ".dvc"
        
        if not dvc_dir.exists():
            try:
                result = subprocess.run(
                    ["dvc", "init"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if result.returncode == 0:
                    logger.info(f"Initialized DVC in {self.repo_path}")
                else:
                    logger.warning(f"DVC init warning: {result.stderr}")
                    
            except FileNotFoundError:
                logger.warning("DVC not installed, using local versioning only")
        
        # Configure remote if specified
        if self.remote_url:
            self._configure_remote()
    
    def _configure_remote(self):
        """Configure DVC remote storage."""
        try:
            subprocess.run(
                ["dvc", "remote", "add", "-d", "synthetic", self.remote_url],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            logger.info(f"Configured DVC remote: {self.remote_url}")
        except FileNotFoundError:
            logger.warning("DVC not available for remote configuration")
    
    def _load_registry(self):
        """Load version registry from file."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path) as f:
                    data = json.load(f)
                
                self._versions = {
                    name: SyntheticDataVersion.from_dict(v)
                    for name, v in data.items()
                }
                
            except Exception as e:
                logger.warning(f"Failed to load version registry: {e}")
                self._versions = {}
    
    def _save_registry(self):
        """Save version registry to file."""
        try:
            data = {
                name: version.to_dict()
                for name, version in self._versions.items()
            }
            
            with open(self._registry_path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save version registry: {e}")
    
    def version_dataset(
        self,
        data_path: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SyntheticDataVersion:
        """
        Version a synthetic dataset.
        
        Args:
            data_path: Path to the dataset
            name: Name for this version
            metadata: Optional metadata (generator, params, etc.)
            
        Returns:
            SyntheticDataVersion object
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Compute hash
        data_hash = self._compute_hash(data_path)
        
        # Get size and record count
        size_bytes = self._get_size(data_path)
        record_count = self._count_records(data_path)
        
        # Extract generator info from metadata
        generator_info = {}
        if metadata:
            generator_info = {
                k: v for k, v in metadata.items()
                if k in ["model", "generator", "target_count", "batch_size"]
            }
        
        # Create version
        version = SyntheticDataVersion(
            name=name,
            hash=data_hash,
            path=str(data_path),
            created_at=datetime.now(timezone.utc).isoformat(),
            size_bytes=size_bytes,
            record_count=record_count,
            metadata=metadata or {},
            generator_info=generator_info,
        )
        
        # Track with DVC
        if not self.mock_dvc:
            self._dvc_track(data_path)
        
        # Store in registry
        self._versions[name] = version
        self._save_registry()
        
        logger.info(f"Versioned dataset '{name}' with hash {data_hash[:8]}...")
        
        return version
    
    def version_checkpoint(
        self,
        checkpoint_path: str,
        step: int,
        generator_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SyntheticDataVersion:
        """
        Version a generation checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Generation step number
            generator_name: Name of the generator
            metadata: Optional additional metadata
            
        Returns:
            SyntheticDataVersion object
        """
        version_name = f"{generator_name}_checkpoint_{step}"
        
        checkpoint_metadata = metadata or {}
        checkpoint_metadata["step"] = step
        checkpoint_metadata["generator"] = generator_name
        checkpoint_metadata["type"] = "checkpoint"
        
        return self.version_dataset(
            data_path=checkpoint_path,
            name=version_name,
            metadata=checkpoint_metadata,
        )
    
    def get_version(self, name: str) -> Optional[SyntheticDataVersion]:
        """Get version by name."""
        return self._versions.get(name)
    
    def get_version_by_hash(self, hash_prefix: str) -> Optional[SyntheticDataVersion]:
        """
        Get version by hash prefix.
        
        Args:
            hash_prefix: First N characters of the hash
            
        Returns:
            Matching version or None
        """
        for version in self._versions.values():
            if version.hash.startswith(hash_prefix):
                return version
        return None
    
    def list_versions(self) -> List[SyntheticDataVersion]:
        """List all versioned datasets."""
        return list(self._versions.values())
    
    def list_checkpoints(self, generator_name: Optional[str] = None) -> List[SyntheticDataVersion]:
        """
        List versioned checkpoints.
        
        Args:
            generator_name: Optional filter by generator
            
        Returns:
            List of checkpoint versions
        """
        checkpoints = []
        
        for version in self._versions.values():
            if version.metadata.get("type") == "checkpoint":
                if generator_name is None or version.metadata.get("generator") == generator_name:
                    checkpoints.append(version)
        
        return sorted(checkpoints, key=lambda v: v.metadata.get("step", 0))
    
    def push(self, name: Optional[str] = None) -> bool:
        """
        Push dataset(s) to remote storage.
        
        Args:
            name: Optional specific dataset to push
            
        Returns:
            True if successful
        """
        if self.mock_dvc:
            logger.info(f"Mock push for {name or 'all'}")
            return True
        
        if not self.remote_url:
            logger.error("No remote URL configured")
            return False
        
        try:
            cmd = ["dvc", "push"]
            
            if name:
                version = self.get_version(name)
                if version:
                    dvc_file = Path(version.path).with_suffix(".dvc")
                    if dvc_file.exists():
                        cmd.append(str(dvc_file))
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                logger.info(f"Pushed {name or 'all datasets'} to remote")
                return True
            else:
                logger.error(f"DVC push failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.warning("DVC not available for push")
            return False
    
    def pull(self, name: Optional[str] = None) -> bool:
        """
        Pull dataset(s) from remote storage.
        
        Args:
            name: Optional specific dataset to pull
            
        Returns:
            True if successful
        """
        if self.mock_dvc:
            logger.info(f"Mock pull for {name or 'all'}")
            return True
        
        try:
            cmd = ["dvc", "pull"]
            
            if name:
                version = self.get_version(name)
                if version:
                    dvc_file = Path(version.path).with_suffix(".dvc")
                    if dvc_file.exists():
                        cmd.append(str(dvc_file))
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            return result.returncode == 0
            
        except FileNotFoundError:
            logger.warning("DVC not available for pull")
            return False
    
    def _dvc_track(self, path: Path):
        """Track a path with DVC."""
        try:
            result = subprocess.run(
                ["dvc", "add", str(path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                logger.warning(f"DVC tracking warning: {result.stderr}")
                
        except FileNotFoundError:
            logger.debug("DVC not available, skipping tracking")
    
    def _compute_hash(self, data_path: Path) -> str:
        """Compute hash of dataset."""
        hasher = hashlib.sha256()
        
        if data_path.is_file():
            with open(data_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        else:
            for file_path in sorted(data_path.rglob("*")):
                if file_path.is_file():
                    hasher.update(str(file_path.relative_to(data_path)).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_size(self, data_path: Path) -> int:
        """Get total size in bytes."""
        if data_path.is_file():
            return data_path.stat().st_size
        
        total = 0
        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        
        return total
    
    def _count_records(self, data_path: Path) -> int:
        """Count records in dataset."""
        total = 0
        
        json_files = []
        if data_path.is_file() and data_path.suffix == ".json":
            json_files = [data_path]
        elif data_path.is_dir():
            json_files = list(data_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    total += len(data)
                elif isinstance(data, dict):
                    if "records" in data:
                        total += len(data["records"])
                    else:
                        total += 1
            except:
                pass
        
        return total
