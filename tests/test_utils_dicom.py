"""Tests for DICOM utilities - Written FIRST (TDD)."""

import pytest
import numpy as np
from pathlib import Path


class TestDicomMetadataExtraction:
    """Test DICOM metadata parsing."""

    def test_parse_dicom_metadata_returns_patient_id(self, sample_dicom_path):
        """Test extraction of patient ID from DICOM headers."""
        from medai_compass.utils.dicom import parse_dicom_metadata
        
        metadata = parse_dicom_metadata(sample_dicom_path)
        
        assert "patient_id" in metadata
        assert isinstance(metadata["patient_id"], str)

    def test_parse_dicom_metadata_returns_study_info(self, sample_dicom_path):
        """Test extraction of study date and description."""
        from medai_compass.utils.dicom import parse_dicom_metadata
        
        metadata = parse_dicom_metadata(sample_dicom_path)
        
        assert "study_date" in metadata
        assert "study_description" in metadata

    def test_parse_dicom_metadata_returns_modality(self, sample_dicom_path):
        """Test extraction of imaging modality (CT, MRI, CR, etc)."""
        from medai_compass.utils.dicom import parse_dicom_metadata
        
        metadata = parse_dicom_metadata(sample_dicom_path)
        
        assert "modality" in metadata
        assert metadata["modality"] in ["CT", "MRI", "CR", "DX", "MG", "US", "PT", "NM", "UNKNOWN"]

    def test_parse_invalid_file_raises_error(self, tmp_path):
        """Test that invalid DICOM file raises appropriate error."""
        from medai_compass.utils.dicom import parse_dicom_metadata, DicomParseError
        
        invalid_file = tmp_path / "not_dicom.txt"
        invalid_file.write_text("This is not a DICOM file")
        
        with pytest.raises(DicomParseError):
            parse_dicom_metadata(str(invalid_file))


class TestDicomPixelDataExtraction:
    """Test DICOM pixel data extraction and normalization."""

    def test_extract_pixel_data_returns_numpy_array(self, sample_dicom_path):
        """Test pixel data extraction returns numpy array."""
        from medai_compass.utils.dicom import extract_pixel_data
        
        pixel_data = extract_pixel_data(sample_dicom_path)
        
        assert isinstance(pixel_data, np.ndarray)

    def test_extract_pixel_data_normalized(self, sample_dicom_path):
        """Test pixel data is normalized to 0-1 range after extraction."""
        from medai_compass.utils.dicom import extract_pixel_data
        
        pixel_data = extract_pixel_data(sample_dicom_path, normalize=True)
        
        assert pixel_data.min() >= 0.0
        assert pixel_data.max() <= 1.0

    def test_extract_pixel_data_preserves_shape(self, sample_dicom_path):
        """Test 2D images return (H, W) shape."""
        from medai_compass.utils.dicom import extract_pixel_data
        
        pixel_data = extract_pixel_data(sample_dicom_path)
        
        assert len(pixel_data.shape) >= 2


class TestCTWindowing:
    """Test CT window/level adjustments."""

    def test_apply_lung_window(self):
        """Test lung window settings: center=-600, width=1500."""
        from medai_compass.utils.dicom import apply_windowing
        
        # Create sample CT data in Hounsfield units
        ct_data = np.array([-1000, -600, 0, 100, 500], dtype=np.float32)
        
        windowed = apply_windowing(ct_data, window_center=-600, window_width=1500)
        
        # Values should be scaled to 0-1 within window
        assert windowed.min() >= 0.0
        assert windowed.max() <= 1.0

    def test_apply_bone_window(self):
        """Test bone window settings: center=400, width=1800."""
        from medai_compass.utils.dicom import apply_windowing
        
        ct_data = np.array([-500, 0, 400, 800, 1500], dtype=np.float32)
        
        windowed = apply_windowing(ct_data, window_center=400, window_width=1800)
        
        assert windowed.min() >= 0.0
        assert windowed.max() <= 1.0

    def test_apply_soft_tissue_window(self):
        """Test soft tissue window settings: center=40, width=400."""
        from medai_compass.utils.dicom import apply_windowing
        
        ct_data = np.array([-100, 0, 40, 100, 200], dtype=np.float32)
        
        windowed = apply_windowing(ct_data, window_center=40, window_width=400)
        
        assert windowed.min() >= 0.0
        assert windowed.max() <= 1.0


class TestImagePreprocessing:
    """Test image preprocessing for model input."""

    def test_resize_to_model_input(self, sample_image_array):
        """Test resizing image to 896x896 for MedGemma."""
        from medai_compass.utils.dicom import resize_for_model
        
        resized = resize_for_model(sample_image_array, target_size=(896, 896))
        
        assert resized.shape[:2] == (896, 896)

    def test_convert_to_rgb(self):
        """Test grayscale to RGB conversion for model input."""
        from medai_compass.utils.dicom import ensure_rgb
        
        grayscale = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        rgb = ensure_rgb(grayscale)
        
        assert rgb.shape == (512, 512, 3)

    def test_rgb_passthrough(self):
        """Test RGB image passes through unchanged."""
        from medai_compass.utils.dicom import ensure_rgb
        
        rgb_input = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        rgb_output = ensure_rgb(rgb_input)
        
        np.testing.assert_array_equal(rgb_input, rgb_output)


class Test3DVolumeProcesing:
    """Test 3D CT/MRI volume processing."""

    def test_extract_slices_from_volume(self, sample_ct_volume):
        """Test extraction of 2D slices from 3D volume."""
        from medai_compass.utils.dicom import extract_slices

        slices = extract_slices(sample_ct_volume, num_slices=8)

        assert len(slices) == 8
        assert all(s.shape == sample_ct_volume.shape[1:] for s in slices)

    def test_create_mip_projection(self, sample_ct_volume):
        """Test Maximum Intensity Projection creation."""
        from medai_compass.utils.dicom import create_mip

        mip = create_mip(sample_ct_volume, axis=0)

        assert mip.shape == sample_ct_volume.shape[1:]

    def test_create_minip_projection(self, sample_ct_volume):
        """Test Minimum Intensity Projection creation."""
        from medai_compass.utils.dicom import create_minip

        minip = create_minip(sample_ct_volume, axis=0)

        assert minip.shape == sample_ct_volume.shape[1:]
        # MinIP should be less than or equal to MIP at every pixel
        from medai_compass.utils.dicom import create_mip
        mip = create_mip(sample_ct_volume, axis=0)
        assert np.all(minip <= mip)

    def test_create_average_projection(self, sample_ct_volume):
        """Test average intensity projection creation."""
        from medai_compass.utils.dicom import create_average_projection

        avg = create_average_projection(sample_ct_volume, axis=0)

        assert avg.shape == sample_ct_volume.shape[1:]

    def test_multiplanar_reconstruction_axial(self, sample_ct_volume):
        """Test MPR extraction in axial plane."""
        from medai_compass.utils.dicom import multiplanar_reconstruction

        axial = multiplanar_reconstruction(sample_ct_volume, plane="axial")

        # Axial slice should have H x W dimensions
        assert axial.shape == (sample_ct_volume.shape[1], sample_ct_volume.shape[2])

    def test_multiplanar_reconstruction_coronal(self, sample_ct_volume):
        """Test MPR extraction in coronal plane."""
        from medai_compass.utils.dicom import multiplanar_reconstruction

        coronal = multiplanar_reconstruction(sample_ct_volume, plane="coronal")

        # Coronal slice should have D x W dimensions
        assert coronal.shape == (sample_ct_volume.shape[0], sample_ct_volume.shape[2])

    def test_multiplanar_reconstruction_sagittal(self, sample_ct_volume):
        """Test MPR extraction in sagittal plane."""
        from medai_compass.utils.dicom import multiplanar_reconstruction

        sagittal = multiplanar_reconstruction(sample_ct_volume, plane="sagittal")

        # Sagittal slice should have D x H dimensions
        assert sagittal.shape == (sample_ct_volume.shape[0], sample_ct_volume.shape[1])

    def test_prepare_3d_for_medgemma(self, sample_ct_volume):
        """Test preparing 3D volume for MedGemma 1.5 input."""
        from medai_compass.utils.dicom import prepare_3d_for_medgemma

        # Request 4 slices per plane (3 planes = 12 total slices)
        slices = prepare_3d_for_medgemma(
            sample_ct_volume,
            num_key_slices=4,
            target_size=(896, 896)
        )

        # 3 planes x 4 slices = 12 total
        assert len(slices) == 12
        # Each slice should be RGB with target size
        for s in slices:
            assert s.shape == (896, 896, 3)

    def test_calculate_volume_statistics(self, sample_ct_volume):
        """Test volume statistics calculation."""
        from medai_compass.utils.dicom import calculate_volume_statistics

        stats = calculate_volume_statistics(sample_ct_volume)

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "shape" in stats
        assert "total_voxels" in stats

    def test_create_3d_montage(self, sample_ct_volume):
        """Test 3D montage creation."""
        from medai_compass.utils.dicom import create_3d_montage

        montage = create_3d_montage(sample_ct_volume, rows=4, cols=4, axis=0)

        expected_h = 4 * sample_ct_volume.shape[1]
        expected_w = 4 * sample_ct_volume.shape[2]
        assert montage.shape == (expected_h, expected_w)
