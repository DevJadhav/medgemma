"""DICOM file handling utilities for MedAI Compass.

Provides functions for:
- Parsing DICOM metadata (patient ID, study info, modality)
- Extracting and normalizing pixel data
- Applying CT window/level adjustments
- Preprocessing images for model input
- Processing 3D CT/MRI volumes
"""

from pathlib import Path
from typing import Any

import numpy as np


class DicomParseError(Exception):
    """Raised when DICOM parsing fails."""
    pass


def parse_dicom_metadata(dicom_path: str | Path) -> dict[str, Any]:
    """
    Parse DICOM file and extract metadata.
    
    Args:
        dicom_path: Path to DICOM file
        
    Returns:
        Dictionary containing:
        - patient_id: Patient identifier
        - study_date: Study date
        - study_description: Study description
        - modality: Imaging modality (CT, MRI, CR, etc.)
        
    Raises:
        DicomParseError: If file is not a valid DICOM
    """
    try:
        import pydicom
        
        ds = pydicom.dcmread(str(dicom_path), force=True)
        
        # Validate this is actually a DICOM file by checking for required elements
        # A valid DICOM should have transfer syntax or recognizable DICOM attributes
        has_valid_meta = (
            hasattr(ds, 'file_meta') 
            and ds.file_meta is not None 
            and len(ds.file_meta) > 0
        )
        has_dicom_attrs = hasattr(ds, 'PatientID') or hasattr(ds, 'Modality')
        
        if not has_valid_meta and not has_dicom_attrs:
            raise DicomParseError(f"File does not appear to be a valid DICOM: {dicom_path}")
        
        return {
            "patient_id": str(getattr(ds, "PatientID", "UNKNOWN")),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "study_description": str(getattr(ds, "StudyDescription", "")),
            "modality": str(getattr(ds, "Modality", "UNKNOWN")),
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "institution_name": str(getattr(ds, "InstitutionName", "")),
        }
    except DicomParseError:
        raise
    except Exception as e:
        raise DicomParseError(f"Failed to parse DICOM file: {dicom_path}. Error: {e}")


def extract_pixel_data(
    dicom_path: str | Path, 
    normalize: bool = False
) -> np.ndarray:
    """
    Extract pixel data from DICOM file.
    
    Args:
        dicom_path: Path to DICOM file
        normalize: If True, normalize pixel values to 0-1 range
        
    Returns:
        Numpy array of pixel data
    """
    try:
        import pydicom
        
        ds = pydicom.dcmread(str(dicom_path), force=True)
        pixel_array = ds.pixel_array.astype(np.float32)
        
        if normalize:
            min_val = pixel_array.min()
            max_val = pixel_array.max()
            if max_val > min_val:
                pixel_array = (pixel_array - min_val) / (max_val - min_val)
            else:
                pixel_array = np.zeros_like(pixel_array)
                
        return pixel_array
    except Exception as e:
        raise DicomParseError(f"Failed to extract pixel data: {e}")


def apply_windowing(
    ct_data: np.ndarray,
    window_center: float,
    window_width: float
) -> np.ndarray:
    """
    Apply CT window/level adjustments.
    
    Common presets:
    - Lung: center=-600, width=1500
    - Bone: center=400, width=1800
    - Soft tissue: center=40, width=400
    
    Args:
        ct_data: CT image data in Hounsfield units
        window_center: Window center value
        window_width: Window width value
        
    Returns:
        Windowed image normalized to 0-1
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    # Clip values to window range
    windowed = np.clip(ct_data, min_val, max_val)
    
    # Normalize to 0-1
    windowed = (windowed - min_val) / (max_val - min_val)
    
    return windowed.astype(np.float32)


def resize_for_model(
    image: np.ndarray,
    target_size: tuple[int, int] = (896, 896)
) -> np.ndarray:
    """
    Resize image for MedGemma model input (896x896).
    
    Args:
        image: Input image array (H, W) or (H, W, C)
        target_size: Target dimensions (height, width)
        
    Returns:
        Resized image array
    """
    from PIL import Image
    
    # Handle grayscale vs color
    if len(image.shape) == 2:
        mode = "L"
    elif image.shape[2] == 3:
        mode = "RGB"
    else:
        mode = "L"
        
    pil_img = Image.fromarray(image.astype(np.uint8), mode=mode)
    resized = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    
    return np.array(resized)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to RGB.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        
    Returns:
        RGB image (H, W, 3)
    """
    if len(image.shape) == 2:
        # Grayscale to RGB
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Already RGB
        return image
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Single channel to RGB
        return np.concatenate([image, image, image], axis=-1)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")


def extract_slices(
    volume: np.ndarray,
    num_slices: int = 8,
    axis: int = 0
) -> list[np.ndarray]:
    """
    Extract evenly spaced 2D slices from 3D volume.
    
    Args:
        volume: 3D volume array (D, H, W)
        num_slices: Number of slices to extract
        axis: Axis to slice along (0 = axial for CT)
        
    Returns:
        List of 2D slice arrays
    """
    depth = volume.shape[axis]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    slices = []
    for idx in indices:
        if axis == 0:
            slices.append(volume[idx, :, :])
        elif axis == 1:
            slices.append(volume[:, idx, :])
        else:
            slices.append(volume[:, :, idx])
            
    return slices


def create_mip(
    volume: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Create Maximum Intensity Projection from 3D volume.
    
    Args:
        volume: 3D volume array (D, H, W)
        axis: Axis to project along
        
    Returns:
        2D MIP image
    """
    return np.max(volume, axis=axis)
