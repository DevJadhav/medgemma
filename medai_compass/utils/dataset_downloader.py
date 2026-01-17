"""Dataset downloader for MedAI Compass.

Downloads medical imaging datasets from Section 12.3 of the implementation plan:
- MIMIC-IV (PhysioNet credentialed)
- MIMIC-CXR (PhysioNet credentialed)
- ChestX-ray14 (NIH - open)
- CAMELYON16 (Grand Challenge)
- LIDC-IDRI (TCIA - open)
- n2c2 (Harvard credentialed)
- Synthea (open - synthetic)
- MedQuAD (GitHub - open)
- MedDialog (GitHub - open)

Priority order (recommended):
1. Synthea - Instant, synthetic, no credentials
2. MedQuAD - Small, QA pairs, GitHub
3. MedDialog - Conversations, GitHub
4. ChestX-ray14 - Open, 42GB
5. LIDC-IDRI - Open, CT scans
6. MIMIC-CXR - Credentialed, 4.7TB
7. MIMIC-IV - Credentialed, EHR data
8. CAMELYON16 - Pathology WSI
9. n2c2 - NLP datasets, credentialed
"""

import os
import logging
import subprocess
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
import urllib.request
import json

logger = logging.getLogger(__name__)


class DatasetAccess(Enum):
    """Dataset access levels."""
    OPEN = "open"                    # Freely available
    CREDENTIALED = "credentialed"    # Requires account/approval
    APPLICATION = "application"       # Requires formal application


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    url: str
    access: DatasetAccess
    size_gb: float
    description: str
    download_instructions: str
    requires_credentials: bool = False


# Dataset registry from Section 12.3
DATASETS = {
    "synthea": DatasetInfo(
        name="Synthea",
        url="https://synthetichealth.github.io/synthea/",
        access=DatasetAccess.OPEN,
        size_gb=0.5,
        description="Synthetic patient data generator for realistic EHR data",
        download_instructions="Install Synthea JAR and generate data locally",
        requires_credentials=False,
    ),
    "medquad": DatasetInfo(
        name="MedQuAD",
        url="https://github.com/abachaa/MedQuAD",
        access=DatasetAccess.OPEN,
        size_gb=0.1,
        description="Medical Question Answering Dataset - 47K QA pairs from NIH",
        download_instructions="Clone from GitHub",
        requires_credentials=False,
    ),
    "meddialog": DatasetInfo(
        name="MedDialog",
        url="https://github.com/UCSD-AI4H/Medical-Dialogue-System",
        access=DatasetAccess.OPEN,
        size_gb=0.3,
        description="Medical dialogue dataset for conversational AI",
        download_instructions="Clone from GitHub and download data files",
        requires_credentials=False,
    ),
    "chestxray14": DatasetInfo(
        name="ChestX-ray14",
        url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        access=DatasetAccess.OPEN,
        size_gb=42.0,
        description="112,120 chest X-ray images with 14 disease labels from NIH",
        download_instructions="Download from NIH Box (requires Box account)",
        requires_credentials=False,
    ),
    "lidc_idri": DatasetInfo(
        name="LIDC-IDRI",
        url="https://www.cancerimagingarchive.net/collection/lidc-idri/",
        access=DatasetAccess.OPEN,
        size_gb=125.0,
        description="1,018 CT scans with lung nodule annotations",
        download_instructions="Download via TCIA NBIA Data Retriever",
        requires_credentials=False,
    ),
    "mimic_cxr": DatasetInfo(
        name="MIMIC-CXR",
        url="https://physionet.org/content/mimic-cxr/",
        access=DatasetAccess.CREDENTIALED,
        size_gb=4700.0,
        description="377,110 chest X-rays with radiology reports from BIDMC",
        download_instructions="Requires PhysioNet credentialed access (1-2 weeks)",
        requires_credentials=True,
    ),
    "mimic_iv": DatasetInfo(
        name="MIMIC-IV",
        url="https://physionet.org/content/mimiciv/3.1/",
        access=DatasetAccess.CREDENTIALED,
        size_gb=7.0,
        description="Comprehensive EHR data from BIDMC ICU patients",
        download_instructions="Requires PhysioNet credentialed access",
        requires_credentials=True,
    ),
    "camelyon16": DatasetInfo(
        name="CAMELYON16",
        url="https://camelyon16.grand-challenge.org/",
        access=DatasetAccess.OPEN,
        size_gb=700.0,
        description="400 whole-slide images for breast cancer metastasis detection",
        download_instructions="Register at Grand Challenge and download",
        requires_credentials=False,
    ),
    "n2c2": DatasetInfo(
        name="n2c2",
        url="https://n2c2.dbmi.hms.harvard.edu/data-sets",
        access=DatasetAccess.CREDENTIALED,
        size_gb=0.5,
        description="NLP datasets for clinical text processing challenges",
        download_instructions="Requires Harvard DBMI data use agreement",
        requires_credentials=True,
    ),
}


class DatasetDownloader:
    """Downloads and prepares medical datasets."""
    
    def __init__(
        self,
        output_dir: Path,
        physionet_username: Optional[str] = None,
        physionet_password: Optional[str] = None,
    ):
        """
        Initialize downloader.
        
        Args:
            output_dir: Base directory for downloaded datasets
            physionet_username: PhysioNet credentials (for MIMIC)
            physionet_password: PhysioNet credentials
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.physionet_username = physionet_username or os.environ.get("PHYSIONET_USERNAME")
        self.physionet_password = physionet_password or os.environ.get("PHYSIONET_PASSWORD")
        
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for progress updates: callback(message, percent)."""
        self._progress_callback = callback
    
    def _report_progress(self, message: str, percent: float = 0) -> None:
        """Report progress."""
        logger.info(f"{message} ({percent:.1f}%)")
        if self._progress_callback:
            self._progress_callback(message, percent)
    
    def list_datasets(self) -> list[dict]:
        """List all available datasets with their status."""
        result = []
        for key, info in DATASETS.items():
            dataset_path = self.output_dir / key
            result.append({
                "id": key,
                "name": info.name,
                "url": info.url,
                "access": info.access.value,
                "size_gb": info.size_gb,
                "description": info.description,
                "requires_credentials": info.requires_credentials,
                "downloaded": dataset_path.exists(),
                "path": str(dataset_path) if dataset_path.exists() else None,
            })
        return result
    
    def download(
        self,
        dataset_id: str,
        force: bool = False,
    ) -> Path:
        """
        Download a dataset.
        
        Args:
            dataset_id: Dataset identifier (e.g., "synthea", "medquad")
            force: Overwrite if already exists
            
        Returns:
            Path to downloaded dataset
        """
        if dataset_id not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(DATASETS.keys())}")
        
        info = DATASETS[dataset_id]
        dataset_path = self.output_dir / dataset_id
        
        if dataset_path.exists() and not force:
            logger.info(f"Dataset {info.name} already exists at {dataset_path}")
            return dataset_path
        
        if dataset_path.exists() and force:
            logger.info(f"Removing existing dataset at {dataset_path}")
            shutil.rmtree(dataset_path)
        
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Dispatch to specific downloader
        download_method = getattr(self, f"_download_{dataset_id}", None)
        if download_method:
            download_method(dataset_path)
        else:
            raise NotImplementedError(f"Downloader for {dataset_id} not implemented")
        
        return dataset_path
    
    def _download_synthea(self, output_path: Path) -> None:
        """Download and run Synthea synthetic data generator."""
        self._report_progress("Downloading Synthea", 0)
        
        # Check if Java is available
        try:
            subprocess.run(["java", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Java is required to run Synthea. Please install Java 11+.")
        
        # Download Synthea JAR
        synthea_version = "3.2.0"
        jar_url = f"https://github.com/synthetichealth/synthea/releases/download/v{synthea_version}/synthea-with-dependencies.jar"
        jar_path = output_path / "synthea.jar"
        
        self._report_progress("Downloading Synthea JAR", 10)
        urllib.request.urlretrieve(jar_url, jar_path)
        
        # Generate synthetic patients
        self._report_progress("Generating synthetic patients", 50)
        subprocess.run(
            [
                "java", "-jar", str(jar_path),
                "-p", "1000",  # Generate 1000 patients
                "-c", "synthea.properties",
                "--exporter.fhir.export", "true",
                "--exporter.csv.export", "true",
            ],
            cwd=output_path,
            check=True,
        )
        
        self._report_progress("Synthea data generation complete", 100)
    
    def _download_medquad(self, output_path: Path) -> None:
        """Clone MedQuAD from GitHub."""
        self._report_progress("Cloning MedQuAD repository", 0)
        
        subprocess.run(
            ["git", "clone", "--depth", "1", 
             "https://github.com/abachaa/MedQuAD.git",
             str(output_path)],
            check=True,
        )
        
        self._report_progress("MedQuAD download complete", 100)
    
    def _download_meddialog(self, output_path: Path) -> None:
        """Clone MedDialog from GitHub."""
        self._report_progress("Cloning MedDialog repository", 0)
        
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/UCSD-AI4H/Medical-Dialogue-System.git",
             str(output_path)],
            check=True,
        )
        
        self._report_progress("MedDialog download complete", 100)
    
    def _download_chestxray14(self, output_path: Path) -> None:
        """
        Download ChestX-ray14 from NIH.
        
        Note: Full dataset requires manual download from NIH Box.
        This method downloads the labels and provides instructions.
        """
        self._report_progress("Setting up ChestX-ray14", 0)
        
        # Download labels CSV
        labels_url = "https://nihcc.app.box.com/shared/static/azcjsdtmwj7q08r2jqv7x3gplm0xlbxj.csv"
        labels_path = output_path / "Data_Entry_2017.csv"
        
        self._report_progress("Downloading labels file", 20)
        try:
            urllib.request.urlretrieve(labels_url, labels_path)
        except Exception as e:
            logger.warning(f"Could not download labels automatically: {e}")
        
        # Create instructions file
        instructions = """# ChestX-ray14 Dataset Download Instructions

The full ChestX-ray14 dataset must be downloaded manually from NIH Box:
https://nihcc.app.box.com/v/ChestXray-NIHCC

## Steps:
1. Go to the URL above
2. Create a Box account if you don't have one
3. Download the following files:
   - images_001.tar.gz through images_012.tar.gz (12 files, ~3.5GB each)
   - Data_Entry_2017.csv (labels)
   - BBox_List_2017.csv (bounding boxes)

## File Structure:
After extraction, organize as:
```
chestxray14/
├── images/
│   └── *.png (112,120 images)
├── Data_Entry_2017.csv
└── BBox_List_2017.csv
```

## Citation:
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. 
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.
CVPR 2017.
"""
        (output_path / "README.md").write_text(instructions)
        
        self._report_progress("ChestX-ray14 setup complete - manual download required", 100)
        logger.warning(
            "ChestX-ray14 images must be downloaded manually from NIH Box. "
            f"See instructions at: {output_path / 'README.md'}"
        )
    
    def _download_lidc_idri(self, output_path: Path) -> None:
        """
        Download LIDC-IDRI from TCIA.
        
        Note: Requires NBIA Data Retriever for full download.
        """
        self._report_progress("Setting up LIDC-IDRI", 0)
        
        instructions = """# LIDC-IDRI Dataset Download Instructions

The LIDC-IDRI dataset must be downloaded using TCIA NBIA Data Retriever:
https://www.cancerimagingarchive.net/collection/lidc-idri/

## Steps:
1. Go to the collection page above
2. Click "Download" to get the TCIA manifest file
3. Download and install NBIA Data Retriever:
   https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
4. Open the manifest file with NBIA Data Retriever
5. Select download location and start download

## Dataset Info:
- 1,018 CT scans
- 244,527 images total
- ~125 GB compressed

## Citation:
Armato III, S. G., et al. The Lung Image Database Consortium (LIDC) 
and Image Database Resource Initiative (IDRI). Medical Physics 38(2), 2011.
"""
        (output_path / "README.md").write_text(instructions)
        
        self._report_progress("LIDC-IDRI setup complete - manual download required", 100)
        logger.warning(
            "LIDC-IDRI requires NBIA Data Retriever for download. "
            f"See instructions at: {output_path / 'README.md'}"
        )
    
    def _download_mimic_cxr(self, output_path: Path) -> None:
        """Download MIMIC-CXR from PhysioNet (requires credentials)."""
        if not self.physionet_username or not self.physionet_password:
            self._create_physionet_instructions(output_path, "MIMIC-CXR", 
                                                "mimic-cxr-jpg/2.0.0/")
            return
        
        self._report_progress("Downloading MIMIC-CXR from PhysioNet", 0)
        
        # Use wget for PhysioNet download
        subprocess.run(
            [
                "wget", "-r", "-N", "-c", "-np",
                f"--user={self.physionet_username}",
                f"--password={self.physionet_password}",
                "https://physionet.org/files/mimic-cxr-jpg/2.0.0/",
            ],
            cwd=output_path,
            check=True,
        )
        
        self._report_progress("MIMIC-CXR download complete", 100)
    
    def _download_mimic_iv(self, output_path: Path) -> None:
        """Download MIMIC-IV from PhysioNet (requires credentials)."""
        if not self.physionet_username or not self.physionet_password:
            self._create_physionet_instructions(output_path, "MIMIC-IV", 
                                                "mimiciv/3.1/")
            return
        
        self._report_progress("Downloading MIMIC-IV from PhysioNet", 0)
        
        subprocess.run(
            [
                "wget", "-r", "-N", "-c", "-np",
                f"--user={self.physionet_username}",
                f"--password={self.physionet_password}",
                "https://physionet.org/files/mimiciv/3.1/",
            ],
            cwd=output_path,
            check=True,
        )
        
        self._report_progress("MIMIC-IV download complete", 100)
    
    def _download_camelyon16(self, output_path: Path) -> None:
        """Setup CAMELYON16 download instructions."""
        self._report_progress("Setting up CAMELYON16", 0)
        
        instructions = """# CAMELYON16 Dataset Download Instructions

The CAMELYON16 dataset must be downloaded from Grand Challenge:
https://camelyon16.grand-challenge.org/

## Steps:
1. Create an account at Grand Challenge
2. Go to the CAMELYON16 challenge page
3. Navigate to "Data" section
4. Download training and test sets

## Dataset Info:
- 400 whole-slide images (WSI)
- Training: 270 (110 with metastases)
- Test: 130
- ~700 GB total

## Citation:
Bejnordi, B.E., et al. Diagnostic Assessment of Deep Learning Algorithms 
for Detection of Lymph Node Metastases in Women With Breast Cancer. 
JAMA 318(22), 2017.
"""
        (output_path / "README.md").write_text(instructions)
        
        self._report_progress("CAMELYON16 setup complete - manual download required", 100)
    
    def _download_n2c2(self, output_path: Path) -> None:
        """Setup n2c2 download instructions."""
        self._report_progress("Setting up n2c2", 0)
        
        instructions = """# n2c2 NLP Datasets Download Instructions

The n2c2 datasets require a data use agreement with Harvard DBMI:
https://n2c2.dbmi.hms.harvard.edu/data-sets

## Available Datasets:
- 2006: Smoking status
- 2008: Obesity challenge
- 2009: Medication extraction
- 2010: Concept, assertion, relation
- 2011: Coreference resolution
- 2012: Temporal relations
- 2014: De-identification, heart disease
- 2018: Adverse drug events, cohort selection
- 2019: Clinical trial matching
- 2022: Social determinants of health

## Steps:
1. Go to the URL above
2. Register for an account
3. Complete the data use agreement
4. Download the datasets you need

## Citation:
Refer to individual challenge papers for citations.
"""
        (output_path / "README.md").write_text(instructions)
        
        self._report_progress("n2c2 setup complete - DUA required", 100)
    
    def _create_physionet_instructions(
        self,
        output_path: Path,
        dataset_name: str,
        dataset_path: str,
    ) -> None:
        """Create instructions for PhysioNet datasets."""
        instructions = f"""# {dataset_name} Dataset Download Instructions

This dataset requires PhysioNet credentialed access.

## Steps to Get Access:
1. Create account at https://physionet.org/
2. Complete CITI training (human subjects research)
3. Apply for credentialed access
4. Wait for approval (1-2 weeks)

## Download Command:
Once approved, set environment variables:
```bash
export PHYSIONET_USERNAME="your_username"
export PHYSIONET_PASSWORD="your_password"
```

Then run:
```bash
wget -r -N -c -np \\
    --user=$PHYSIONET_USERNAME \\
    --password=$PHYSIONET_PASSWORD \\
    https://physionet.org/files/{dataset_path}
```

Or use the dataset downloader:
```python
from medai_compass.utils.dataset_downloader import DatasetDownloader

downloader = DatasetDownloader(
    output_dir="./data",
    physionet_username="your_username",
    physionet_password="your_password",
)
downloader.download("{dataset_name.lower().replace('-', '_')}")
```
"""
        (output_path / "README.md").write_text(instructions)
        
        logger.warning(
            f"{dataset_name} requires PhysioNet credentials. "
            f"See instructions at: {output_path / 'README.md'}"
        )


def download_recommended_datasets(output_dir: Path) -> list[Path]:
    """
    Download recommended starter datasets.
    
    Priority order:
    1. Synthea (synthetic, instant)
    2. MedQuAD (QA, small)
    3. MedDialog (conversations)
    
    Returns:
        List of paths to downloaded datasets
    """
    downloader = DatasetDownloader(output_dir)
    
    recommended = ["medquad", "meddialog"]  # Synthea requires Java
    downloaded = []
    
    for dataset_id in recommended:
        try:
            path = downloader.download(dataset_id)
            downloaded.append(path)
            logger.info(f"Downloaded {dataset_id} to {path}")
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {e}")
    
    return downloaded
