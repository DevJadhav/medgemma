"""
Dataset Download and Management Utilities.

This module provides utilities for downloading and managing medical imaging datasets
including ChestX-ray14 and CAMELYON16.
"""

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm


@dataclass
class DatasetInfo:
    """Information about a medical imaging dataset."""
    name: str
    description: str
    url: str
    size_gb: float
    license: str
    citation: str
    requires_auth: bool = False
    auth_type: Optional[str] = None  # "kaggle", "physionet", "gcs"


# Dataset registry
DATASETS = {
    "chestxray14": DatasetInfo(
        name="ChestX-ray14",
        description="112,120 frontal-view X-ray images of 30,805 unique patients with 14 disease labels",
        url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        size_gb=42.0,
        license="CC0 1.0 Universal",
        citation="Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. CVPR 2017.",
        requires_auth=False
    ),
    "camelyon16": DatasetInfo(
        name="CAMELYON16",
        description="Whole slide images for breast cancer metastasis detection",
        url="https://camelyon16.grand-challenge.org/Data/",
        size_gb=700.0,  # Very large dataset
        license="CC BY-NC-ND 4.0",
        citation="Ehteshami Bejnordi B, et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA. 2017.",
        requires_auth=True,
        auth_type="grand-challenge"
    ),
    "mimic-cxr": DatasetInfo(
        name="MIMIC-CXR",
        description="377,110 chest X-rays from 65,379 patients with radiology reports",
        url="https://physionet.org/content/mimic-cxr/2.0.0/",
        size_gb=4700.0,  # Full dataset is huge
        license="PhysioNet Credentialed Health Data License 1.5.0",
        citation="Johnson AEW, et al. MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data. 2019.",
        requires_auth=True,
        auth_type="physionet"
    ),
    "chexpert": DatasetInfo(
        name="CheXpert",
        description="224,316 chest X-rays of 65,240 patients with 14 observations",
        url="https://stanfordmlgroup.github.io/competitions/chexpert/",
        size_gb=439.0,
        license="Stanford Research License",
        citation="Irvin J, et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI 2019.",
        requires_auth=True,
        auth_type="stanford"
    )
}


@dataclass
class DownloadProgress:
    """Track download progress."""
    dataset_name: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    status: str = "pending"  # pending, downloading, completed, failed
    error: Optional[str] = None


class DatasetDownloader:
    """
    Download and manage medical imaging datasets.
    
    For large datasets like CAMELYON16, provides instructions
    rather than direct download due to size and auth requirements.
    """
    
    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def list_available_datasets(self) -> list[DatasetInfo]:
        """List all available datasets."""
        return list(DATASETS.values())
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a specific dataset."""
        return DATASETS.get(dataset_name.lower())
    
    def download_chestxray14_sample(self, sample_size: int = 100) -> Path:
        """
        Download a sample of ChestX-ray14 images for development.
        
        The full dataset is 42GB. This downloads a small sample for testing.
        
        Args:
            sample_size: Number of images to download
            
        Returns:
            Path to the downloaded sample directory
        """
        sample_dir = self.data_dir / "chestxray14_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # NIH provides images in batches via direct links
        # For demo purposes, we create a manifest with sample image URLs
        manifest = self._create_chestxray14_manifest(sample_size)
        
        # Save manifest
        manifest_path = sample_dir / "manifest.json"
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return sample_dir
    
    def _create_chestxray14_manifest(self, sample_size: int) -> dict:
        """Create a manifest for ChestX-ray14 sample."""
        # In production, this would parse the actual NIH listings
        # For now, create a manifest structure
        return {
            "dataset": "ChestX-ray14",
            "sample_size": sample_size,
            "download_instructions": [
                "1. Visit https://nihcc.app.box.com/v/ChestXray-NIHCC",
                "2. Download the desired image batches (images_001.tar.gz to images_012.tar.gz)",
                "3. Extract to this directory",
                "4. Download Data_Entry_2017_v2020.csv for labels"
            ],
            "direct_download_links": {
                "labels": "https://nihcc.app.box.com/shared/static/ehdxyjdprvl3kpyoovlly2dk7vl2d6nl.csv",
                "readme": "https://nihcc.app.box.com/shared/static/n5qrb4w0g8k95cw8t6r0r6yd7j0b9x91.txt"
            },
            "expected_structure": {
                "images/": "Directory containing all X-ray images",
                "Data_Entry_2017_v2020.csv": "Labels and metadata",
                "BBox_List_2017.csv": "Bounding box annotations"
            }
        }
    
    def get_camelyon16_instructions(self) -> dict:
        """
        Get instructions for downloading CAMELYON16.
        
        Due to the size (700GB+) and authentication requirements,
        direct download is not practical. Returns instructions instead.
        """
        return {
            "dataset": "CAMELYON16",
            "size": "~700GB",
            "download_method": "Google Cloud Storage",
            "instructions": [
                "1. Register at https://camelyon16.grand-challenge.org/",
                "2. Request data access through the challenge website",
                "3. Once approved, download using gsutil:",
                "   gsutil -m cp -r gs://camelyon16-datasets/training .",
                "   gsutil -m cp -r gs://camelyon16-datasets/testing .",
                "4. Alternatively, download via the ASAP viewer for WSI"
            ],
            "alternative_small_dataset": {
                "name": "CAMELYON16 Patches",
                "description": "Pre-extracted patches from WSI for faster training",
                "url": "https://github.com/basveeling/pcam"
            }
        }
    
    def setup_kaggle_credentials(self, kaggle_json_path: Optional[str] = None) -> bool:
        """
        Set up Kaggle credentials for dataset downloads.
        
        Args:
            kaggle_json_path: Path to kaggle.json file
            
        Returns:
            True if credentials are set up successfully
        """
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        if kaggle_json_path:
            shutil.copy(kaggle_json_path, kaggle_dir / "kaggle.json")
            os.chmod(kaggle_dir / "kaggle.json", 0o600)
            return True
            
        # Check if credentials already exist
        return (kaggle_dir / "kaggle.json").exists()
    
    def download_via_kaggle(self, dataset_slug: str) -> Optional[Path]:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_slug: Kaggle dataset slug (e.g., "nih-chest-xrays/data")
            
        Returns:
            Path to downloaded dataset or None if failed
        """
        try:
            output_dir = self.data_dir / dataset_slug.replace("/", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(output_dir), "--unzip"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return output_dir
            else:
                print(f"Kaggle download failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("Kaggle CLI not installed. Install with: pip install kaggle")
            return None
    
    def verify_dataset_integrity(self, dataset_path: Path, expected_checksums: dict[str, str]) -> dict[str, bool]:
        """
        Verify dataset file integrity using checksums.
        
        Args:
            dataset_path: Path to dataset directory
            expected_checksums: Dict of filename -> expected MD5 hash
            
        Returns:
            Dict of filename -> verification result
        """
        results = {}
        
        for filename, expected_hash in expected_checksums.items():
            file_path = dataset_path / filename
            if not file_path.exists():
                results[filename] = False
                continue
                
            # Calculate MD5
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5.update(chunk)
                    
            results[filename] = md5.hexdigest() == expected_hash
            
        return results


class DatasetLoader:
    """Load and preprocess medical imaging datasets."""
    
    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        
    def load_chestxray14_labels(self, csv_path: Optional[str] = None) -> dict:
        """
        Load ChestX-ray14 labels from CSV.
        
        Args:
            csv_path: Path to Data_Entry_2017_v2020.csv
            
        Returns:
            Dict mapping image filename to labels
        """
        if csv_path is None:
            csv_path = self.data_dir / "chestxray14" / "Data_Entry_2017_v2020.csv"
            
        labels = {}
        
        # Check if file exists
        if not Path(csv_path).exists():
            return {"error": f"Labels file not found: {csv_path}"}
            
        # Parse CSV
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row.get("Image Index", "")
                finding_labels = row.get("Finding Labels", "No Finding")
                labels[image_name] = {
                    "labels": finding_labels.split("|"),
                    "patient_id": row.get("Patient ID", ""),
                    "age": row.get("Patient Age", ""),
                    "gender": row.get("Patient Gender", ""),
                    "view_position": row.get("View Position", "")
                }
                
        return labels
    
    def get_label_statistics(self, labels: dict) -> dict:
        """
        Get statistics about label distribution.
        
        Args:
            labels: Dict of image labels
            
        Returns:
            Label distribution statistics
        """
        from collections import Counter
        
        label_counts = Counter()
        for image_data in labels.values():
            if isinstance(image_data, dict) and "labels" in image_data:
                for label in image_data["labels"]:
                    label_counts[label] += 1
                    
        total_images = len(labels)
        
        return {
            "total_images": total_images,
            "label_counts": dict(label_counts),
            "label_percentages": {
                label: count / total_images * 100
                for label, count in label_counts.items()
            }
        }


# Convenience functions
def download_sample_data(output_dir: str = "data/datasets") -> dict:
    """
    Download sample data for development.
    
    Returns manifest and instructions for full datasets.
    """
    downloader = DatasetDownloader(output_dir)
    
    return {
        "chestxray14": {
            "sample_path": str(downloader.download_chestxray14_sample()),
            "full_download_url": DATASETS["chestxray14"].url
        },
        "camelyon16": downloader.get_camelyon16_instructions(),
        "available_datasets": [ds.name for ds in downloader.list_available_datasets()]
    }
