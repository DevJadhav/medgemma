"""MinIO client for medical image storage.

Provides:
- Image upload/download
- Numpy array storage
- Dataset organization
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class MinioClient:
    """Client for MinIO object storage."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False,
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO endpoint (default from env)
            access_key: Access key (default from env)
            secret_key: Secret key (default from env)
            secure: Use HTTPS
        """
        self.endpoint = endpoint or os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self.secure = secure
        
        self._client = None
    
    @property
    def client(self):
        """Lazy-load MinIO client."""
        if self._client is None:
            try:
                from minio import Minio
                self._client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure,
                )
            except ImportError:
                raise RuntimeError("minio package not installed. Run: pip install minio")
        return self._client
    
    def ensure_bucket(self, bucket_name: str) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Error ensuring bucket {bucket_name}: {e}")
            raise
    
    def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_path: Path,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload a file to MinIO.
        
        Args:
            bucket: Bucket name
            object_name: Object name/path
            file_path: Local file path
            content_type: MIME type
            metadata: Additional metadata
            
        Returns:
            Object name
        """
        self.ensure_bucket(bucket)
        
        self.client.fput_object(
            bucket,
            object_name,
            str(file_path),
            content_type=content_type,
            metadata=metadata,
        )
        
        logger.debug(f"Uploaded {file_path} to {bucket}/{object_name}")
        return object_name
    
    def upload_numpy_array(
        self,
        bucket: str,
        object_name: str,
        array: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload a numpy array to MinIO.
        
        Args:
            bucket: Bucket name
            object_name: Object name (should end with .npy)
            array: Numpy array
            metadata: Additional metadata
            
        Returns:
            Object name
        """
        self.ensure_bucket(bucket)
        
        # Serialize array
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        data = buffer.getvalue()
        
        # Add array shape to metadata
        if metadata is None:
            metadata = {}
        metadata["array_shape"] = str(array.shape)
        metadata["array_dtype"] = str(array.dtype)
        
        # Convert metadata values to strings
        str_metadata = {k: str(v) for k, v in metadata.items()}
        
        self.client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="application/octet-stream",
            metadata=str_metadata,
        )
        
        logger.debug(f"Uploaded array {array.shape} to {bucket}/{object_name}")
        return object_name
    
    def download_numpy_array(
        self,
        bucket: str,
        object_name: str,
    ) -> np.ndarray:
        """
        Download a numpy array from MinIO.
        
        Args:
            bucket: Bucket name
            object_name: Object name
            
        Returns:
            Numpy array
        """
        response = self.client.get_object(bucket, object_name)
        buffer = io.BytesIO(response.read())
        response.close()
        response.release_conn()
        
        buffer.seek(0)
        return np.load(buffer)
    
    def download_file(
        self,
        bucket: str,
        object_name: str,
        file_path: Path,
    ) -> Path:
        """
        Download a file from MinIO.
        
        Args:
            bucket: Bucket name
            object_name: Object name
            file_path: Local destination path
            
        Returns:
            Local file path
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client.fget_object(bucket, object_name, str(file_path))
        
        logger.debug(f"Downloaded {bucket}/{object_name} to {file_path}")
        return file_path
    
    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[dict]:
        """
        List objects in a bucket.
        
        Args:
            bucket: Bucket name
            prefix: Object prefix filter
            recursive: Include nested objects
            
        Returns:
            List of object info dicts
        """
        objects = []
        
        for obj in self.client.list_objects(bucket, prefix=prefix, recursive=recursive):
            objects.append({
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "is_dir": obj.is_dir,
            })
        
        return objects
    
    def list_datasets(self) -> list[str]:
        """
        List all dataset prefixes in medical-images bucket.
        
        Returns:
            List of dataset names
        """
        bucket = "medical-images"
        
        try:
            if not self.client.bucket_exists(bucket):
                return []
            
            # Get unique top-level prefixes
            datasets = set()
            for obj in self.client.list_objects(bucket, recursive=False):
                if obj.is_dir:
                    datasets.add(obj.object_name.rstrip("/"))
            
            return list(datasets)
            
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []
    
    def get_dataset_stats(self, dataset_name: str) -> dict:
        """
        Get statistics for a dataset.
        
        Args:
            dataset_name: Dataset name/prefix
            
        Returns:
            Stats dict with count, size_gb
        """
        bucket = "medical-images"
        prefix = f"{dataset_name}/"
        
        total_count = 0
        total_size = 0
        
        for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True):
            if not obj.is_dir:
                total_count += 1
                total_size += obj.size
        
        return {
            "count": total_count,
            "size_gb": total_size / (1024 ** 3),
        }
    
    def delete_object(self, bucket: str, object_name: str) -> None:
        """Delete an object."""
        self.client.remove_object(bucket, object_name)
        logger.debug(f"Deleted {bucket}/{object_name}")
    
    def get_presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires_hours: int = 1,
    ) -> str:
        """
        Get a presigned URL for downloading.
        
        Args:
            bucket: Bucket name
            object_name: Object name
            expires_hours: URL expiration in hours
            
        Returns:
            Presigned URL
        """
        from datetime import timedelta
        
        return self.client.presigned_get_object(
            bucket,
            object_name,
            expires=timedelta(hours=expires_hours),
        )
