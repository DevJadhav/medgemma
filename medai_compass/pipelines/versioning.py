"""Data Versioning with DVC Integration.

Provides data versioning for medical datasets to ensure reproducible
training runs and data lineage tracking.
"""

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    """Represents a versioned dataset."""
    
    name: str
    hash: str
    path: str
    created_at: str
    size_bytes: int = 0
    record_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVersion":
        """Create from dictionary."""
        return cls(**data)


class DataVersionManager:
    """
    Manager for data versioning using DVC (Data Version Control).
    
    Provides:
    - Dataset tracking and versioning
    - Remote storage integration (S3, GCS, etc.)
    - Version listing and retrieval
    - Metadata management
    
    Attributes:
        repo_path: Path to the DVC repository
        remote_url: URL for remote storage
    """
    
    def __init__(
        self,
        repo_path: Optional[str] = None,
        remote_url: Optional[str] = None,
        auto_init: bool = True,
    ):
        """
        Initialize the version manager.
        
        Args:
            repo_path: Path to DVC repository (defaults to current directory)
            remote_url: URL for remote storage
            auto_init: Automatically initialize DVC if not present
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.remote_url = remote_url
        
        # Version registry file
        self._registry_path = self.repo_path / ".dvc_versions.json"
        self._versions: Dict[str, DatasetVersion] = {}
        
        # Load existing versions
        self._load_registry()
        
        # Initialize DVC if needed
        if auto_init:
            self._ensure_dvc_init()
    
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
    
    def _load_registry(self):
        """Load version registry from file."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path) as f:
                    data = json.load(f)
                
                self._versions = {
                    name: DatasetVersion.from_dict(v)
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
    
    def track(
        self,
        data_path: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        quality_report: Optional[Any] = None,
    ) -> DatasetVersion:
        """
        Track a dataset directory for versioning.
        
        Args:
            data_path: Path to the dataset directory
            name: Name for the dataset version
            metadata: Optional metadata to store with version
            quality_report: Optional quality report from DataQualityMonitor
            
        Returns:
            DatasetVersion object
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Compute hash of the data
        data_hash = self._compute_hash(data_path)
        
        # Get size and record count
        size_bytes = self._get_size(data_path)
        record_count = self._count_records(data_path)
        
        # Build metadata
        version_metadata = metadata or {}
        
        if quality_report:
            version_metadata["quality_passed"] = getattr(quality_report, "passed", None)
            version_metadata["quality_checks"] = getattr(quality_report, "total_checks", None)
        
        # Create version
        version = DatasetVersion(
            name=name,
            hash=data_hash,
            path=str(data_path),
            created_at=datetime.now(timezone.utc).isoformat(),
            size_bytes=size_bytes,
            record_count=record_count,
            metadata=version_metadata,
        )
        
        # Track with DVC if available
        self._dvc_track(data_path)
        
        # Store in registry
        self._versions[name] = version
        self._save_registry()
        
        logger.info(f"Tracked dataset '{name}' with hash {data_hash[:8]}...")
        
        return version
    
    def _compute_hash(self, data_path: Path) -> str:
        """Compute hash of dataset."""
        hasher = hashlib.sha256()
        
        if data_path.is_file():
            with open(data_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        else:
            # Hash all files in directory
            for file_path in sorted(data_path.rglob("*")):
                if file_path.is_file():
                    hasher.update(str(file_path.relative_to(data_path)).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_size(self, data_path: Path) -> int:
        """Get total size of dataset in bytes."""
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
                    # FHIR Bundle
                    if data.get("resourceType") == "Bundle":
                        total += len(data.get("entry", []))
                    else:
                        total += 1
            except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
                logger.debug(f"Failed to count records in {json_file}: {e}")

        return total
    
    def _dvc_track(self, data_path: Path):
        """Track dataset with DVC."""
        try:
            result = subprocess.run(
                ["dvc", "add", str(data_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                logger.warning(f"DVC tracking warning: {result.stderr}")
        except FileNotFoundError:
            logger.debug("DVC not available, skipping DVC tracking")
    
    def get_version(self, name: str) -> Optional[DatasetVersion]:
        """
        Get version information by name.
        
        Args:
            name: Dataset name
            
        Returns:
            DatasetVersion or None if not found
        """
        return self._versions.get(name)
    
    def list_versions(self) -> List[DatasetVersion]:
        """
        List all tracked dataset versions.
        
        Returns:
            List of DatasetVersion objects
        """
        return list(self._versions.values())
    
    def checkout(self, name: str) -> bool:
        """
        Checkout a specific dataset version.
        
        Args:
            name: Dataset name
            
        Returns:
            True if successful
        """
        version = self.get_version(name)
        
        if version is None:
            logger.error(f"Version '{name}' not found")
            return False
        
        try:
            # Use DVC checkout
            dvc_file = Path(version.path).with_suffix(".dvc")
            
            if dvc_file.exists():
                result = subprocess.run(
                    ["dvc", "checkout", str(dvc_file)],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if result.returncode == 0:
                    logger.info(f"Checked out dataset '{name}'")
                    return True
                else:
                    logger.error(f"DVC checkout failed: {result.stderr}")
                    return False
            else:
                logger.warning(f"DVC file not found for '{name}'")
                return False
                
        except FileNotFoundError:
            logger.warning("DVC not available for checkout")
            return False
    
    def push(self, name: Optional[str] = None) -> bool:
        """
        Push dataset(s) to remote storage.
        
        Args:
            name: Optional specific dataset to push
            
        Returns:
            True if successful
        """
        if not self.remote_url:
            logger.error("No remote URL configured")
            return False
        
        try:
            cmd = ["dvc", "push"]
            if name:
                version = self.get_version(name)
                if version:
                    cmd.append(version.path)
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            return result.returncode == 0
            
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
        try:
            cmd = ["dvc", "pull"]
            if name:
                version = self.get_version(name)
                if version:
                    dvc_file = Path(version.path).with_suffix(".dvc")
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
