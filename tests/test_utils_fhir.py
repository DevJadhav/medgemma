"""Tests for FHIR client utilities - Written FIRST (TDD)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFHIRPatientContext:
    """Test FHIR patient context retrieval."""

    def test_get_patient_returns_basic_info(self, sample_fhir_patient):
        """Test retrieval of basic patient information."""
        from medai_compass.utils.fhir import FHIRClient
        
        client = FHIRClient(base_url="http://localhost:8080/fhir")
        
        with patch.object(client, '_get_resource', return_value=sample_fhir_patient):
            patient = client.get_patient("patient-123")
        
        assert patient["id"] == "patient-123"
        assert "name" in patient

    def test_get_patient_conditions(self, sample_fhir_condition):
        """Test retrieval of patient conditions/diagnoses."""
        from medai_compass.utils.fhir import FHIRClient
        
        client = FHIRClient(base_url="http://localhost:8080/fhir")
        
        with patch.object(client, '_search_resources', return_value=[sample_fhir_condition]):
            conditions = client.get_patient_conditions("patient-123")
        
        assert len(conditions) >= 1
        assert conditions[0]["code"]["coding"][0]["display"] == "Hypertension"

    def test_get_patient_medications(self, sample_fhir_medication):
        """Test retrieval of active medications."""
        from medai_compass.utils.fhir import FHIRClient
        
        client = FHIRClient(base_url="http://localhost:8080/fhir")
        
        with patch.object(client, '_search_resources', return_value=[sample_fhir_medication]):
            medications = client.get_patient_medications("patient-123")
        
        assert len(medications) >= 1
        assert medications[0]["status"] == "active"

    def test_get_patient_context_aggregates_all(
        self, sample_fhir_patient, sample_fhir_condition, sample_fhir_medication
    ):
        """Test get_patient_context returns aggregated patient data."""
        from medai_compass.utils.fhir import FHIRClient
        
        client = FHIRClient(base_url="http://localhost:8080/fhir")
        
        with patch.object(client, '_get_resource', return_value=sample_fhir_patient):
            with patch.object(
                client, '_search_resources', 
                side_effect=[[sample_fhir_condition], [sample_fhir_medication], []]
            ):
                context = client.get_patient_context("patient-123")
        
        assert "patient" in context
        assert "conditions" in context
        assert "medications" in context
        assert "allergies" in context


class TestFHIRDiagnosticReport:
    """Test FHIR DiagnosticReport creation."""

    def test_create_diagnostic_report_structure(self):
        """Test creating a DiagnosticReport resource."""
        from medai_compass.utils.fhir import create_diagnostic_report
        
        report = create_diagnostic_report(
            patient_id="patient-123",
            study_id="study-456",
            findings=["No acute abnormality", "Heart size normal"],
            impression="Negative chest X-ray",
            performer_id="ai-system"
        )
        
        assert report["resourceType"] == "DiagnosticReport"
        assert report["subject"]["reference"] == "Patient/patient-123"
        assert report["status"] in ["preliminary", "final"]

    def test_diagnostic_report_includes_conclusion(self):
        """Test report includes conclusion text."""
        from medai_compass.utils.fhir import create_diagnostic_report
        
        report = create_diagnostic_report(
            patient_id="patient-123",
            study_id="study-456",
            findings=["Finding 1"],
            impression="Test impression",
            performer_id="ai-system"
        )
        
        assert "conclusion" in report
        assert report["conclusion"] == "Test impression"

    def test_diagnostic_report_includes_ai_system_flag(self):
        """Test report is flagged as AI-generated."""
        from medai_compass.utils.fhir import create_diagnostic_report
        
        report = create_diagnostic_report(
            patient_id="patient-123",
            study_id="study-456",
            findings=["Finding 1"],
            impression="Test impression",
            performer_id="ai-system"
        )
        
        # Should have extension or identifier indicating AI generation
        assert "extension" in report or "performer" in report


class TestFHIRConnection:
    """Test FHIR server connection handling."""

    def test_client_initialization(self):
        """Test FHIR client initializes with base URL."""
        from medai_compass.utils.fhir import FHIRClient
        
        client = FHIRClient(base_url="http://localhost:8080/fhir")
        
        assert client.base_url == "http://localhost:8080/fhir"

    def test_client_with_auth_token(self):
        """Test FHIR client accepts auth token."""
        from medai_compass.utils.fhir import FHIRClient
        
        client = FHIRClient(
            base_url="http://localhost:8080/fhir",
            auth_token="Bearer test-token"
        )
        
        assert client.auth_token == "Bearer test-token"

    def test_connection_error_handling(self):
        """Test graceful handling of connection errors."""
        from medai_compass.utils.fhir import FHIRClient, FHIRConnectionError
        
        client = FHIRClient(base_url="http://invalid-server:9999/fhir")
        
        with pytest.raises(FHIRConnectionError):
            client.get_patient("patient-123")
