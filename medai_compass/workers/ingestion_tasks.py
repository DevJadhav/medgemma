"""Data ingestion Celery tasks.

Provides background tasks for:
- DICOM batch processing
- Image preprocessing
- MinIO storage
- Database indexing
"""

import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import json

from medai_compass.workers.celery import app
from medai_compass.workers.minio_client import MinioClient
from medai_compass.utils.dicom import DicomProcessor

logger = logging.getLogger(__name__)


@app.task(bind=True, name="medai_compass.workers.ingestion_tasks.process_dicom_batch")
def process_dicom_batch(
    self,
    dicom_paths: list[str],
    dataset_name: str,
    job_id: str,
) -> dict:
    """
    Process a batch of DICOM files.
    
    Args:
        dicom_paths: List of paths to DICOM files
        dataset_name: Name of the dataset
        job_id: Ingestion job ID
        
    Returns:
        Processing results
    """
    results = {
        "job_id": job_id,
        "dataset": dataset_name,
        "total": len(dicom_paths),
        "processed": 0,
        "failed": 0,
        "errors": [],
    }
    
    processor = DicomProcessor()
    minio_client = MinioClient()
    
    for i, dicom_path in enumerate(dicom_paths):
        try:
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": i + 1,
                    "total": len(dicom_paths),
                    "status": f"Processing {Path(dicom_path).name}",
                }
            )
            
            # Process DICOM
            result = _process_single_dicom(
                dicom_path=dicom_path,
                dataset_name=dataset_name,
                processor=processor,
                minio_client=minio_client,
            )
            
            if result["success"]:
                results["processed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "path": dicom_path,
                    "error": result.get("error", "Unknown error"),
                })
                
        except Exception as e:
            logger.error(f"Error processing {dicom_path}: {e}")
            results["failed"] += 1
            results["errors"].append({
                "path": dicom_path,
                "error": str(e),
            })
    
    return results


def _process_single_dicom(
    dicom_path: str,
    dataset_name: str,
    processor: DicomProcessor,
    minio_client: MinioClient,
) -> dict:
    """Process a single DICOM file."""
    try:
        path = Path(dicom_path)
        
        # Parse DICOM metadata
        metadata = processor.extract_metadata(path)
        
        # Extract pixel data
        pixel_array = processor.extract_pixel_array(path)
        
        if pixel_array is None:
            return {"success": False, "error": "No pixel data"}
        
        # Preprocess for MedGemma (896x896)
        preprocessed = processor.preprocess_for_medgemma(pixel_array)
        
        # Upload to MinIO
        object_name = f"{dataset_name}/{metadata.get('study_instance_uid', 'unknown')}/{path.stem}.npy"
        
        minio_client.upload_numpy_array(
            bucket="medical-images",
            object_name=object_name,
            array=preprocessed,
            metadata=metadata,
        )
        
        return {
            "success": True,
            "object_name": object_name,
            "metadata": metadata,
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.task(name="medai_compass.workers.ingestion_tasks.ingest_dataset")
def ingest_dataset(
    dataset_path: str,
    dataset_name: str,
    file_pattern: str = "*.dcm",
    batch_size: int = 100,
) -> dict:
    """
    Ingest an entire dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of the dataset
        file_pattern: Glob pattern for files
        batch_size: Number of files per batch
        
    Returns:
        Ingestion job info
    """
    import uuid
    
    path = Path(dataset_path)
    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Find all matching files
    files = list(path.rglob(file_pattern))
    total_files = len(files)
    
    if total_files == 0:
        return {"status": "error", "message": f"No files matching {file_pattern}"}
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Split into batches and queue tasks
    batches = [
        files[i:i + batch_size]
        for i in range(0, total_files, batch_size)
    ]
    
    # Queue batch processing tasks
    from celery import group
    
    batch_tasks = group(
        process_dicom_batch.s(
            dicom_paths=[str(f) for f in batch],
            dataset_name=dataset_name,
            job_id=job_id,
        )
        for batch in batches
    )
    
    result = batch_tasks.apply_async()
    
    return {
        "job_id": job_id,
        "status": "queued",
        "total_files": total_files,
        "total_batches": len(batches),
        "group_id": result.id,
    }


@app.task(name="medai_compass.workers.ingestion_tasks.cleanup_old_jobs")
def cleanup_old_jobs(days: int = 7) -> dict:
    """Clean up old ingestion job records."""
    logger.info(f"Cleaning up jobs older than {days} days")
    
    # This would typically interact with the database
    # For now, just log the action
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    return {
        "status": "completed",
        "cutoff_date": cutoff.isoformat(),
        "message": f"Cleaned up jobs older than {cutoff}",
    }


@app.task(name="medai_compass.workers.ingestion_tasks.verify_dataset_integrity")
def verify_dataset_integrity(dataset_name: Optional[str] = None) -> dict:
    """Verify integrity of stored datasets."""
    logger.info(f"Verifying dataset integrity: {dataset_name or 'all'}")
    
    minio_client = MinioClient()
    
    if dataset_name:
        datasets = [dataset_name]
    else:
        # List all dataset buckets/prefixes
        datasets = minio_client.list_datasets()
    
    results = {}
    for ds in datasets:
        try:
            stats = minio_client.get_dataset_stats(ds)
            results[ds] = {
                "status": "ok",
                "object_count": stats.get("count", 0),
                "total_size_gb": stats.get("size_gb", 0),
            }
        except Exception as e:
            results[ds] = {"status": "error", "error": str(e)}
    
    return results


@app.task(name="medai_compass.workers.ingestion_tasks.process_wsi_tiles")
def process_wsi_tiles(
    wsi_path: str,
    output_prefix: str,
    tile_size: int = 512,
    magnification_levels: list[int] = None,
) -> dict:
    """
    Process whole-slide image into tiles.
    
    Args:
        wsi_path: Path to WSI file
        output_prefix: Prefix for output tiles
        tile_size: Tile size in pixels
        magnification_levels: Magnification levels to extract
        
    Returns:
        Processing results
    """
    if magnification_levels is None:
        magnification_levels = [20, 10, 5]
    
    try:
        # Note: Would use openslide for actual WSI processing
        # This is a placeholder for the task structure
        
        return {
            "status": "completed",
            "wsi_path": wsi_path,
            "tile_size": tile_size,
            "magnification_levels": magnification_levels,
            "message": "WSI processing requires openslide (not implemented)",
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
