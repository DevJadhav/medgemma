"""
Tests for the data pipeline module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from medai_compass.utils.data_pipeline import (
    DataLoadingPipeline,
    SyntheaDataGenerator,
    SyntheticPatient,
    generate_sample_data,
)


class TestSyntheticPatient:
    """Tests for SyntheticPatient dataclass."""
    
    def test_create_patient(self):
        """Test creating a synthetic patient."""
        patient = SyntheticPatient(
            patient_id="test-001",
            name="Test Patient",
            birth_date="1990-01-15",
            gender="female"
        )
        
        assert patient.patient_id == "test-001"
        assert patient.name == "Test Patient"
        assert patient.birth_date == "1990-01-15"
        assert patient.gender == "female"
        assert patient.conditions is None
        assert patient.medications is None
        
    def test_patient_with_conditions(self):
        """Test patient with medical conditions."""
        patient = SyntheticPatient(
            patient_id="test-002",
            name="Test Patient Two",
            birth_date="1985-06-20",
            gender="male",
            conditions=[
                {"code": "38341003", "display": "Hypertension"}
            ]
        )
        
        assert len(patient.conditions) == 1
        assert patient.conditions[0]["display"] == "Hypertension"


class TestSyntheaDataGenerator:
    """Tests for SyntheaDataGenerator class."""
    
    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates the data directory."""
        data_dir = tmp_path / "synthetic_data"
        generator = SyntheaDataGenerator(str(data_dir))
        
        assert data_dir.exists()
        
    def test_generate_mock_patients(self, tmp_path):
        """Test generating mock patients."""
        generator = SyntheaDataGenerator(str(tmp_path))
        patients = generator.generate_mock_patients(count=5)
        
        assert len(patients) == 5
        assert all(isinstance(p, SyntheticPatient) for p in patients)
        
        # Check patient IDs are unique
        patient_ids = [p.patient_id for p in patients]
        assert len(set(patient_ids)) == 5
        
    def test_generate_mock_patients_have_conditions(self, tmp_path):
        """Test that generated patients have conditions."""
        generator = SyntheaDataGenerator(str(tmp_path))
        patients = generator.generate_mock_patients(count=3)
        
        for patient in patients:
            assert patient.conditions is not None
            assert len(patient.conditions) > 0
            
    def test_generate_mock_patients_have_observations(self, tmp_path):
        """Test that generated patients have observations."""
        generator = SyntheaDataGenerator(str(tmp_path))
        patients = generator.generate_mock_patients(count=3)
        
        for patient in patients:
            assert patient.observations is not None
            assert len(patient.observations) == 3  # Height, Weight, BP
            
    def test_save_as_fhir_bundle(self, tmp_path):
        """Test saving patients as FHIR bundle."""
        generator = SyntheaDataGenerator(str(tmp_path))
        patients = generator.generate_mock_patients(count=2)
        
        bundle_path = generator.save_as_fhir_bundle(patients, "test_bundle.json")
        
        assert bundle_path.exists()
        
        with open(bundle_path) as f:
            bundle = json.load(f)
            
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert len(bundle["entry"]) > 0
        
    def test_fhir_bundle_contains_patient_resources(self, tmp_path):
        """Test that FHIR bundle contains Patient resources."""
        generator = SyntheaDataGenerator(str(tmp_path))
        patients = generator.generate_mock_patients(count=2)
        
        bundle_path = generator.save_as_fhir_bundle(patients)
        
        with open(bundle_path) as f:
            bundle = json.load(f)
            
        patient_resources = [
            e for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Patient"
        ]
        
        assert len(patient_resources) == 2
        
    def test_load_fhir_bundle(self, tmp_path):
        """Test loading a FHIR bundle from file."""
        generator = SyntheaDataGenerator(str(tmp_path))
        patients = generator.generate_mock_patients(count=3)
        bundle_path = generator.save_as_fhir_bundle(patients)
        
        loaded_bundle = generator.load_fhir_bundle(str(bundle_path))
        
        assert loaded_bundle["resourceType"] == "Bundle"
        assert len(loaded_bundle["entry"]) > 0
        
    def test_extract_patients_from_bundle(self, tmp_path):
        """Test extracting patients from FHIR bundle."""
        generator = SyntheaDataGenerator(str(tmp_path))
        patients = generator.generate_mock_patients(count=3)
        bundle_path = generator.save_as_fhir_bundle(patients)
        
        bundle = generator.load_fhir_bundle(str(bundle_path))
        extracted = generator.extract_patients_from_bundle(bundle)
        
        assert len(extracted) == 3
        assert all(p["resourceType"] == "Patient" for p in extracted)


class TestDataLoadingPipeline:
    """Tests for DataLoadingPipeline class."""
    
    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates the data directory."""
        data_dir = tmp_path / "pipeline_data"
        pipeline = DataLoadingPipeline(str(data_dir))
        
        assert data_dir.exists()
        
    def test_load_dicom_dataset_empty_dir(self, tmp_path):
        """Test loading DICOM from empty directory."""
        pipeline = DataLoadingPipeline(str(tmp_path))
        
        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()
        
        files = pipeline.load_dicom_dataset(str(dicom_dir))
        
        assert files == []
        
    def test_load_dicom_dataset_with_files(self, tmp_path):
        """Test loading DICOM files from directory."""
        pipeline = DataLoadingPipeline(str(tmp_path))
        
        dicom_dir = tmp_path / "dicoms"
        dicom_dir.mkdir()
        
        # Create mock DICOM files
        (dicom_dir / "image1.dcm").write_text("mock dicom")
        (dicom_dir / "image2.DCM").write_text("mock dicom")
        (dicom_dir / "not_dicom.txt").write_text("not dicom")
        
        files = pipeline.load_dicom_dataset(str(dicom_dir))
        
        assert len(files) == 2
        assert all(f.suffix.lower() == ".dcm" for f in files)
        
    def test_load_dicom_dataset_nonexistent(self, tmp_path):
        """Test loading from non-existent directory."""
        pipeline = DataLoadingPipeline(str(tmp_path))
        
        files = pipeline.load_dicom_dataset(str(tmp_path / "nonexistent"))
        
        assert files == []
        
    def test_create_dataset_manifest(self, tmp_path):
        """Test creating a dataset manifest."""
        pipeline = DataLoadingPipeline(str(tmp_path))
        
        files = [Path("/data/file1.dcm"), Path("/data/file2.dcm")]
        manifest = pipeline.create_dataset_manifest("test_dataset", files)
        
        assert manifest["dataset_name"] == "test_dataset"
        assert manifest["file_count"] == 2
        assert len(manifest["files"]) == 2
        
        # Check manifest file was created
        manifest_path = tmp_path / "test_dataset_manifest.json"
        assert manifest_path.exists()


class TestGenerateSampleData:
    """Tests for the convenience function."""
    
    def test_generate_sample_data(self, tmp_path):
        """Test generating sample data."""
        output_path = generate_sample_data(str(tmp_path), patient_count=5)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            bundle = json.load(f)
            
        patients = [
            e for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Patient"
        ]
        
        assert len(patients) == 5
