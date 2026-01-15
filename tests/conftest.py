"""Pytest configuration and shared fixtures for MedAI Compass tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Generator, Any
import numpy as np


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def sample_dicom_path(tmp_path) -> str:
    """Create a valid mock DICOM file for testing."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
    import datetime
    
    # Create file meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    # Create the FileDataset
    dicom_file = tmp_path / "sample.dcm"
    ds = FileDataset(str(dicom_file), {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Add required elements
    ds.PatientID = "TEST001"
    ds.PatientName = "Test^Patient"
    ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
    ds.StudyDescription = "Test Study"
    ds.Modality = "CT"
    ds.SeriesDescription = "Test Series"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 64
    ds.Columns = 64
    ds.PixelRepresentation = 1
    
    # Create pixel data (64x64 grayscale image)
    pixel_array = np.random.randint(-1000, 1000, (64, 64), dtype=np.int16)
    ds.PixelData = pixel_array.tobytes()
    
    # Save
    ds.save_as(str(dicom_file))
    
    return str(dicom_file)


@pytest.fixture
def sample_image_array() -> np.ndarray:
    """Create a sample image array for testing."""
    return np.random.randint(0, 255, (896, 896, 3), dtype=np.uint8)


@pytest.fixture
def sample_ct_volume() -> np.ndarray:
    """Create a sample 3D CT volume for testing."""
    return np.random.randint(-1000, 1000, (64, 512, 512), dtype=np.int16)


# =============================================================================
# Model Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_medgemma_model() -> MagicMock:
    """Mock MedGemma model to avoid loading actual weights in CI."""
    model = MagicMock()
    model.generate.return_value = MagicMock(
        sequences=[[1, 2, 3, 4, 5]],
    )
    return model


@pytest.fixture
def mock_medgemma_tokenizer() -> MagicMock:
    """Mock MedGemma tokenizer."""
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "Sample AI-generated medical finding."
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.return_tensors = "pt"
    return tokenizer


@pytest.fixture
def mock_path_foundation() -> MagicMock:
    """Mock Path Foundation model for histopathology."""
    model = MagicMock()
    # Returns 384-dim embeddings
    model.return_value = np.random.randn(1, 384).astype(np.float32)
    return model


@pytest.fixture
def mock_cxr_foundation() -> MagicMock:
    """Mock CXR Foundation model for chest X-rays."""
    model = MagicMock()
    # Returns language-aligned vectors
    model.return_value = np.random.randn(1, 768).astype(np.float32)
    return model


# =============================================================================
# FHIR Mock Fixtures
# =============================================================================

@pytest.fixture
def sample_fhir_patient() -> dict[str, Any]:
    """Sample FHIR Patient resource."""
    return {
        "resourceType": "Patient",
        "id": "patient-123",
        "name": [{"family": "Doe", "given": ["John"]}],
        "birthDate": "1980-01-15",
        "gender": "male",
    }


@pytest.fixture
def sample_fhir_condition() -> dict[str, Any]:
    """Sample FHIR Condition resource."""
    return {
        "resourceType": "Condition",
        "id": "condition-456",
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "38341003",
                    "display": "Hypertension",
                }
            ]
        },
    }


@pytest.fixture
def sample_fhir_medication() -> dict[str, Any]:
    """Sample FHIR MedicationRequest resource."""
    return {
        "resourceType": "MedicationRequest",
        "id": "med-789",
        "status": "active",
        "medicationCodeableConcept": {
            "coding": [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "197361",
                    "display": "Lisinopril 10 MG Oral Tablet",
                }
            ]
        },
    }


# =============================================================================
# Guardrail Test Fixtures
# =============================================================================

@pytest.fixture
def sample_text_with_phi() -> str:
    """Sample text containing PHI for testing detection."""
    return """
    Patient John Doe, MRN: 12345678, DOB: 01/15/1980.
    Contact: 555-123-4567, email: john.doe@email.com
    SSN: 123-45-6789
    Address: 123 Main Street, Boston, MA 02101
    """


@pytest.fixture
def sample_text_without_phi() -> str:
    """Sample medical text without PHI."""
    return """
    Chest X-ray demonstrates bilateral infiltrates consistent with pneumonia.
    No pleural effusion identified. Heart size is within normal limits.
    Recommend follow-up imaging in 2 weeks.
    """


@pytest.fixture
def sample_critical_finding() -> str:
    """Sample text containing critical finding."""
    return """
    CRITICAL FINDING: Large left-sided tension pneumothorax identified.
    Immediate clinical correlation and intervention recommended.
    """


@pytest.fixture
def sample_safety_concern_text() -> str:
    """Sample patient message with safety concern."""
    return "I've been feeling hopeless lately and want to end my life."


# =============================================================================
# Agent Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_diagnostic_state() -> dict[str, Any]:
    """Mock diagnostic state for LangGraph tests."""
    return {
        "patient_id": "patient-123",
        "session_id": "session-abc",
        "images": ["/path/to/image.dcm"],
        "findings": [],
        "confidence_scores": [],
        "requires_review": False,
        "audit_trail": [],
        "fhir_context": {},
    }


@pytest.fixture
def mock_workflow_inputs() -> dict[str, str]:
    """Mock inputs for workflow agent tests."""
    return {
        "notes": "Patient admitted for chest pain. Troponin elevated.",
        "diagnosis": "Acute coronary syndrome",
    }


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
def mock_async_client() -> AsyncMock:
    """Mock async HTTP client for API tests."""
    client = AsyncMock()
    client.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"status": "ok"},
    )
    client.post.return_value = MagicMock(
        status_code=201,
        json=lambda: {"id": "new-resource-id"},
    )
    return client


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def mock_db_session() -> MagicMock:
    """Mock database session for testing."""
    session = MagicMock()
    session.execute.return_value = MagicMock(scalars=lambda: MagicMock(all=lambda: []))
    session.commit = MagicMock()
    session.rollback = MagicMock()
    return session


# =============================================================================
# Security Fixtures
# =============================================================================

@pytest.fixture
def sample_encryption_key() -> bytes:
    """Sample Fernet encryption key for testing."""
    from cryptography.fernet import Fernet
    return Fernet.generate_key()


@pytest.fixture
def sample_jwt_secret() -> str:
    """Sample JWT secret for testing."""
    return "test-jwt-secret-key-for-testing-only"
