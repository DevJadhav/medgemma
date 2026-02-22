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


def create_minip(
    volume: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Create Minimum Intensity Projection from 3D volume.

    Useful for visualizing airways and low-density structures.

    Args:
        volume: 3D volume array (D, H, W)
        axis: Axis to project along

    Returns:
        2D MinIP image
    """
    return np.min(volume, axis=axis)


def create_average_projection(
    volume: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Create average intensity projection from 3D volume.

    Args:
        volume: 3D volume array (D, H, W)
        axis: Axis to project along

    Returns:
        2D average projection image
    """
    return np.mean(volume, axis=axis).astype(volume.dtype)


def load_volume_from_series(
    dicom_dir: str | Path,
    sort_by: str = "InstanceNumber"
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load 3D volume from a directory of DICOM slices.

    Args:
        dicom_dir: Directory containing DICOM series
        sort_by: DICOM attribute to sort slices by

    Returns:
        Tuple of (volume array, metadata dict)

    Raises:
        DicomParseError: If directory contains no valid DICOM files
    """
    import pydicom

    dicom_dir = Path(dicom_dir)
    dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))

    if not dicom_files:
        # Try loading all files (some DICOM files have no extension)
        dicom_files = [f for f in dicom_dir.iterdir() if f.is_file()]

    if not dicom_files:
        raise DicomParseError(f"No DICOM files found in {dicom_dir}")

    # Load and sort slices
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise DicomParseError(f"No valid DICOM files with pixel data in {dicom_dir}")

    # Sort by specified attribute
    slices.sort(key=lambda x: float(getattr(x, sort_by, 0)))

    # Extract metadata from first slice
    first_slice = slices[0]
    metadata = {
        "patient_id": str(getattr(first_slice, "PatientID", "UNKNOWN")),
        "modality": str(getattr(first_slice, "Modality", "UNKNOWN")),
        "study_description": str(getattr(first_slice, "StudyDescription", "")),
        "series_description": str(getattr(first_slice, "SeriesDescription", "")),
        "num_slices": len(slices),
        "slice_thickness": float(getattr(first_slice, "SliceThickness", 1.0)),
        "pixel_spacing": list(getattr(first_slice, "PixelSpacing", [1.0, 1.0])),
    }

    # Stack into volume
    volume = np.stack([s.pixel_array for s in slices], axis=0)

    return volume.astype(np.float32), metadata


def multiplanar_reconstruction(
    volume: np.ndarray,
    plane: str = "axial",
    slice_idx: int = None
) -> np.ndarray:
    """
    Extract a slice in the specified anatomical plane (MPR).

    Args:
        volume: 3D volume (D, H, W) in standard orientation
        plane: 'axial', 'coronal', or 'sagittal'
        slice_idx: Slice index (uses middle if None)

    Returns:
        2D slice in requested plane
    """
    if plane.lower() == "axial":
        axis = 0
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2
        return volume[slice_idx, :, :]

    elif plane.lower() == "coronal":
        axis = 1
        if slice_idx is None:
            slice_idx = volume.shape[1] // 2
        return volume[:, slice_idx, :]

    elif plane.lower() == "sagittal":
        axis = 2
        if slice_idx is None:
            slice_idx = volume.shape[2] // 2
        return volume[:, :, slice_idx]

    else:
        raise ValueError(f"Unknown plane: {plane}. Use 'axial', 'coronal', or 'sagittal'")


def prepare_3d_for_medgemma(
    volume: np.ndarray,
    num_key_slices: int = 8,
    target_size: tuple[int, int] = (896, 896),
    window_center: float = None,
    window_width: float = None
) -> list[np.ndarray]:
    """
    Prepare 3D volume for MedGemma 27B's 3D imaging capabilities.

    MedGemma 27B supports 3D imaging by processing multiple key slices
    from different anatomical planes.

    Args:
        volume: 3D volume (D, H, W)
        num_key_slices: Number of key slices per plane
        target_size: Target size for each slice
        window_center: Optional CT window center
        window_width: Optional CT window width

    Returns:
        List of preprocessed 2D slices ready for model input
    """
    processed_slices = []

    # Apply windowing if specified
    if window_center is not None and window_width is not None:
        volume = apply_windowing(volume, window_center, window_width)
    else:
        # Normalize to 0-1
        min_val = volume.min()
        max_val = volume.max()
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)

    # Extract key slices from each anatomical plane
    for plane in ["axial", "coronal", "sagittal"]:
        if plane == "axial":
            depth = volume.shape[0]
        elif plane == "coronal":
            depth = volume.shape[1]
        else:
            depth = volume.shape[2]

        # Get evenly spaced slice indices
        indices = np.linspace(
            depth * 0.1,  # Skip edges
            depth * 0.9,
            num_key_slices,
            dtype=int
        )

        for idx in indices:
            slice_2d = multiplanar_reconstruction(volume, plane, idx)

            # Convert to uint8
            slice_uint8 = (slice_2d * 255).astype(np.uint8)

            # Resize
            resized = resize_for_model(slice_uint8, target_size)

            # Ensure RGB
            rgb = ensure_rgb(resized)

            processed_slices.append(rgb)

    return processed_slices


def calculate_volume_statistics(volume: np.ndarray) -> dict[str, float]:
    """
    Calculate statistics for a 3D volume.

    Args:
        volume: 3D volume array

    Returns:
        Dictionary with volume statistics
    """
    return {
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "std": float(volume.std()),
        "shape": list(volume.shape),
        "total_voxels": int(np.prod(volume.shape)),
    }


def create_3d_montage(
    volume: np.ndarray,
    rows: int = 4,
    cols: int = 4,
    axis: int = 0
) -> np.ndarray:
    """
    Create a montage of evenly spaced slices from a 3D volume.

    Useful for creating overview images of CT/MRI scans.

    Args:
        volume: 3D volume (D, H, W)
        rows: Number of rows in montage
        cols: Number of columns in montage
        axis: Axis to slice along

    Returns:
        2D montage image
    """
    num_slices = rows * cols
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

    # Create montage
    slice_h, slice_w = slices[0].shape
    montage = np.zeros((rows * slice_h, cols * slice_w), dtype=slices[0].dtype)

    for i, s in enumerate(slices):
        row = i // cols
        col = i % cols
        montage[row * slice_h:(row + 1) * slice_h, col * slice_w:(col + 1) * slice_w] = s

    return montage


class DicomProcessor:
    """
    High-level DICOM processing class for batch operations.
    
    Provides a unified interface for:
    - Extracting metadata from DICOM files
    - Extracting and normalizing pixel arrays
    - Preprocessing images for MedGemma model input
    """
    
    def __init__(self, target_size: tuple[int, int] = (896, 896)):
        """
        Initialize the DICOM processor.
        
        Args:
            target_size: Target image size for model input (default 896x896 for MedGemma)
        """
        self.target_size = target_size
    
    def extract_metadata(self, dicom_path: str | Path) -> dict[str, Any]:
        """
        Extract metadata from a DICOM file.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Dictionary containing DICOM metadata
        """
        try:
            import pydicom
            
            ds = pydicom.dcmread(str(dicom_path), force=True)
            
            return {
                "patient_id": str(getattr(ds, "PatientID", "UNKNOWN")),
                "study_date": str(getattr(ds, "StudyDate", "")),
                "study_description": str(getattr(ds, "StudyDescription", "")),
                "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
                "modality": str(getattr(ds, "Modality", "UNKNOWN")),
                "series_description": str(getattr(ds, "SeriesDescription", "")),
                "institution_name": str(getattr(ds, "InstitutionName", "")),
                "manufacturer": str(getattr(ds, "Manufacturer", "")),
                "rows": int(getattr(ds, "Rows", 0)),
                "columns": int(getattr(ds, "Columns", 0)),
            }
        except Exception as e:
            raise DicomParseError(f"Failed to extract metadata from {dicom_path}: {e}")
    
    def extract_pixel_array(self, dicom_path: str | Path, normalize: bool = True) -> np.ndarray | None:
        """
        Extract pixel array from a DICOM file.
        
        Args:
            dicom_path: Path to DICOM file
            normalize: Whether to normalize pixel values to 0-1 range
            
        Returns:
            Pixel array as numpy array, or None if no pixel data
        """
        try:
            import pydicom
            
            ds = pydicom.dcmread(str(dicom_path), force=True)
            
            if not hasattr(ds, 'PixelData'):
                return None
            
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply rescale slope/intercept for CT images
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            if normalize:
                # Normalize to 0-1 range
                min_val = pixel_array.min()
                max_val = pixel_array.max()
                if max_val > min_val:
                    pixel_array = (pixel_array - min_val) / (max_val - min_val)
                else:
                    pixel_array = np.zeros_like(pixel_array)
            
            return pixel_array
            
        except Exception as e:
            raise DicomParseError(f"Failed to extract pixel array from {dicom_path}: {e}")
    
    def preprocess_for_medgemma(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Preprocess pixel array for MedGemma model input.
        
        Args:
            pixel_array: Input pixel array (can be 2D grayscale or 3D RGB)
            
        Returns:
            Preprocessed array of shape (target_size, target_size, 3)
        """
        from PIL import Image
        
        # Handle 2D grayscale images
        if pixel_array.ndim == 2:
            # Convert to RGB by repeating channels
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 1:
            pixel_array = np.repeat(pixel_array, 3, axis=-1)
        
        # Normalize to 0-255 for PIL
        if pixel_array.max() <= 1.0:
            pixel_array = (pixel_array * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)
        
        # Resize to target size
        img = Image.fromarray(pixel_array)
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convert back to numpy and normalize to 0-1
        result = np.array(img).astype(np.float32) / 255.0
        
        return result
    
    def process_file(self, dicom_path: str | Path) -> dict[str, Any]:
        """
        Process a single DICOM file and return all relevant data.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Dictionary with metadata, pixel_array, and preprocessed_array
        """
        metadata = self.extract_metadata(dicom_path)
        pixel_array = self.extract_pixel_array(dicom_path)
        
        preprocessed = None
        if pixel_array is not None:
            preprocessed = self.preprocess_for_medgemma(pixel_array)
        
        return {
            "metadata": metadata,
            "pixel_array": pixel_array,
            "preprocessed": preprocessed,
        }

