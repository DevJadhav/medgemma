"""Dataset download Celery tasks.

Provides background tasks for:
- Async dataset downloads
- Progress tracking
- Download verification
"""

import logging
from pathlib import Path
from typing import Optional

from medai_compass.workers.celery import app
from medai_compass.utils.dataset_downloader import DatasetDownloader, DATASETS

logger = logging.getLogger(__name__)


@app.task(bind=True, name="medai_compass.workers.download_tasks.download_dataset")
def download_dataset(
    self,
    dataset_id: str,
    output_dir: str,
    force: bool = False,
    physionet_username: Optional[str] = None,
    physionet_password: Optional[str] = None,
) -> dict:
    """
    Download a dataset asynchronously.
    
    Args:
        dataset_id: Dataset identifier
        output_dir: Output directory
        force: Overwrite existing
        physionet_username: PhysioNet credentials
        physionet_password: PhysioNet credentials
        
    Returns:
        Download results
    """
    if dataset_id not in DATASETS:
        return {
            "status": "error",
            "error": f"Unknown dataset: {dataset_id}",
            "available": list(DATASETS.keys()),
        }
    
    dataset_info = DATASETS[dataset_id]
    
    def progress_callback(message: str, percent: float):
        self.update_state(
            state="PROGRESS",
            meta={
                "status": message,
                "percent": percent,
                "dataset": dataset_id,
            }
        )
    
    try:
        downloader = DatasetDownloader(
            output_dir=Path(output_dir),
            physionet_username=physionet_username,
            physionet_password=physionet_password,
        )
        
        downloader.set_progress_callback(progress_callback)
        
        path = downloader.download(dataset_id, force=force)
        
        return {
            "status": "completed",
            "dataset_id": dataset_id,
            "dataset_name": dataset_info.name,
            "path": str(path),
            "size_gb": dataset_info.size_gb,
        }
        
    except NotImplementedError as e:
        return {
            "status": "not_implemented",
            "dataset_id": dataset_id,
            "message": str(e),
            "instructions": dataset_info.download_instructions,
        }
        
    except Exception as e:
        logger.error(f"Download failed for {dataset_id}: {e}")
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "error": str(e),
        }


@app.task(name="medai_compass.workers.download_tasks.download_recommended")
def download_recommended(output_dir: str) -> dict:
    """
    Download recommended starter datasets.
    
    Downloads in order:
    1. MedQuAD (small, QA)
    2. MedDialog (conversations)
    
    Args:
        output_dir: Output directory
        
    Returns:
        Download results
    """
    recommended = ["medquad", "meddialog"]
    results = {}
    
    downloader = DatasetDownloader(output_dir=Path(output_dir))
    
    for dataset_id in recommended:
        try:
            path = downloader.download(dataset_id)
            results[dataset_id] = {
                "status": "completed",
                "path": str(path),
            }
        except Exception as e:
            results[dataset_id] = {
                "status": "error",
                "error": str(e),
            }
    
    return {
        "status": "completed",
        "datasets": results,
    }


@app.task(name="medai_compass.workers.download_tasks.list_available_datasets")
def list_available_datasets() -> dict:
    """List all available datasets."""
    return {
        "datasets": [
            {
                "id": key,
                "name": info.name,
                "url": info.url,
                "access": info.access.value,
                "size_gb": info.size_gb,
                "description": info.description,
                "requires_credentials": info.requires_credentials,
            }
            for key, info in DATASETS.items()
        ],
        "recommended": ["medquad", "meddialog", "chestxray14"],
        "requires_credentials": [
            key for key, info in DATASETS.items() 
            if info.requires_credentials
        ],
    }


@app.task(name="medai_compass.workers.download_tasks.verify_download")
def verify_download(dataset_path: str, dataset_id: str) -> dict:
    """
    Verify a downloaded dataset.
    
    Args:
        dataset_path: Path to downloaded dataset
        dataset_id: Expected dataset ID
        
    Returns:
        Verification results
    """
    path = Path(dataset_path)
    
    if not path.exists():
        return {
            "status": "error",
            "error": "Path does not exist",
            "path": dataset_path,
        }
    
    # Count files
    total_files = sum(1 for _ in path.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    size_gb = total_size / (1024 ** 3)
    
    # Check for expected structure
    expected_files = {
        "medquad": ["README.md"],
        "meddialog": ["README.md"],
        "chestxray14": ["README.md"],  # Labels file if downloaded
        "synthea": ["synthea.jar"],
    }
    
    found_expected = []
    missing_expected = []
    
    for expected in expected_files.get(dataset_id, []):
        if (path / expected).exists():
            found_expected.append(expected)
        else:
            missing_expected.append(expected)
    
    return {
        "status": "ok" if not missing_expected else "incomplete",
        "dataset_id": dataset_id,
        "path": dataset_path,
        "total_files": total_files,
        "size_gb": round(size_gb, 2),
        "found_expected": found_expected,
        "missing_expected": missing_expected,
    }
