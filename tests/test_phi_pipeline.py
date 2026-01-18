"""TDD Tests for PHI Detection Pipeline.

Tests for PHI detection and strict filtering in data pipeline.
Reuses existing PHI detector from medai_compass.guardrails.phi_detection.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml


# =============================================================================
# PHI Pipeline Module Structure Tests
# =============================================================================

class TestPHIPipelineModuleStructure:
    """Test PHI pipeline module file structure."""
    
    @pytest.fixture
    def phi_module_path(self) -> Path:
        """Path to PHI pipeline module."""
        return Path(__file__).parent.parent / "medai_compass" / "pipelines" / "phi_detection.py"
    
    def test_phi_module_exists(self, phi_module_path: Path):
        """PHI pipeline module must exist."""
        assert phi_module_path.exists(), f"Module not found at {phi_module_path}"
    
    def test_phi_module_has_required_classes(self, phi_module_path: Path):
        """Module should have required classes."""
        content = phi_module_path.read_text()
        
        required_classes = [
            "PHIPipelineFilter",
            "PHIFilterConfig",
            "PHIFilterResult",
        ]
        
        for cls in required_classes:
            assert f"class {cls}" in content, f"Missing class: {cls}"
    
    def test_phi_module_reuses_existing_detector(self, phi_module_path: Path):
        """Module should reuse existing PHI detector."""
        content = phi_module_path.read_text()
        
        # Should import from guardrails
        assert "guardrails" in content or "phi_detection" in content, \
            "Should reuse existing PHI detector from guardrails"


# =============================================================================
# PHIPipelineFilter Class Tests
# =============================================================================

class TestPHIPipelineFilterCreation:
    """Test PHIPipelineFilter instantiation."""
    
    def test_create_filter(self):
        """Should create PHI filter instance."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        assert phi_filter is not None
    
    def test_create_filter_with_strict_mode(self):
        """Should support strict filtering mode (default)."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        
        assert phi_filter.strict is True
    
    def test_strict_mode_is_default(self):
        """Strict mode should be the default."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        # Per user requirement: strictly filter PHI records
        assert phi_filter.strict is True


# =============================================================================
# PHI Detection Tests
# =============================================================================

class TestPHIDetection:
    """Test PHI detection functionality."""
    
    def test_detect_ssn(self):
        """Should detect Social Security Numbers."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        text = "Patient SSN is 123-45-6789."
        result = phi_filter.scan(text)
        
        assert result.contains_phi is True
        assert "ssn" in result.phi_types
    
    def test_detect_mrn(self):
        """Should detect Medical Record Numbers."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        text = "MRN: 12345678"
        result = phi_filter.scan(text)
        
        assert result.contains_phi is True
        assert "mrn" in result.phi_types
    
    def test_detect_phone_number(self):
        """Should detect phone numbers."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        text = "Contact the patient at 555-123-4567."
        result = phi_filter.scan(text)
        
        assert result.contains_phi is True
        assert "phone" in result.phi_types
    
    def test_detect_email(self):
        """Should detect email addresses."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        text = "Email: patient@hospital.com"
        result = phi_filter.scan(text)
        
        assert result.contains_phi is True
        assert "email" in result.phi_types
    
    def test_detect_date_of_birth(self):
        """Should detect dates of birth."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        text = "DOB: 01/15/1990"
        result = phi_filter.scan(text)
        
        assert result.contains_phi is True
        assert "dob" in result.phi_types
    
    def test_detect_address(self):
        """Should detect street addresses."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        text = "Patient lives at 123 Main Street, Boston"
        result = phi_filter.scan(text)
        
        assert result.contains_phi is True
        assert "address" in result.phi_types
    
    def test_clean_text_has_no_phi(self):
        """Should correctly identify clean text."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter()
        
        text = "Hypertension is a condition where blood pressure is elevated."
        result = phi_filter.scan(text)
        
        assert result.contains_phi is False
        assert len(result.phi_types) == 0


# =============================================================================
# Strict Filter Tests (Per User Requirement)
# =============================================================================

class TestStrictPHIFiltering:
    """Test strict PHI filtering - records with PHI are completely removed."""
    
    def test_filter_record_with_phi_returns_none(self):
        """Strict filter should return None for records with PHI."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        
        record = {
            "question": "What medications does patient 123-45-6789 take?",
            "answer": "The patient takes metformin."
        }
        
        filtered = phi_filter.filter_record(record)
        
        # Strict mode: completely remove record
        assert filtered is None
    
    def test_filter_clean_record_returns_record(self):
        """Should return clean records unchanged."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        
        record = {
            "question": "What is hypertension?",
            "answer": "Hypertension is high blood pressure."
        }
        
        filtered = phi_filter.filter_record(record)
        
        assert filtered is not None
        assert filtered["question"] == record["question"]
    
    def test_filter_checks_all_text_fields(self):
        """Should check all text fields in record."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        
        # PHI in answer only
        record = {
            "question": "What is the treatment?",
            "answer": "Call the patient at 555-123-4567 to discuss."
        }
        
        filtered = phi_filter.filter_record(record)
        
        assert filtered is None  # Should still filter
    
    def test_filter_nested_fields(self):
        """Should check nested fields for PHI."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        
        record = {
            "patient": {
                "name": "John Doe",
                "ssn": "123-45-6789"  # PHI in nested field
            },
            "diagnosis": "Hypertension"
        }
        
        filtered = phi_filter.filter_record(record)
        
        assert filtered is None


# =============================================================================
# Batch Filtering Tests
# =============================================================================

class TestBatchPHIFiltering:
    """Test batch PHI filtering."""
    
    @pytest.fixture
    def mixed_records(self) -> List[Dict]:
        """Records with and without PHI."""
        return [
            {"question": "What is diabetes?", "answer": "A metabolic disease."},
            {"question": "Patient SSN 123-45-6789 history?", "answer": "N/A"},
            {"question": "How to treat pneumonia?", "answer": "Antibiotics."},
            {"question": "Call 555-123-4567 for results", "answer": "Pending"},
        ]
    
    def test_filter_batch_strict(self, mixed_records):
        """Strict filtering should remove all PHI records."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        filtered = phi_filter.filter_batch(mixed_records)
        
        # Should only keep clean records
        assert len(filtered) == 2
    
    def test_filter_batch_returns_statistics(self, mixed_records):
        """Batch filter should return filtering statistics."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        filtered, stats = phi_filter.filter_batch(mixed_records, return_stats=True)
        
        assert stats["total"] == 4
        assert stats["filtered"] == 2
        assert stats["kept"] == 2
    
    def test_filter_logs_phi_detections(self, mixed_records):
        """Should log PHI detections for audit."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True)
        filtered, stats = phi_filter.filter_batch(mixed_records, return_stats=True)
        
        # Should have audit log of detected PHI types
        assert "phi_types_found" in stats
        assert "ssn" in stats["phi_types_found"]
        assert "phone" in stats["phi_types_found"]


# =============================================================================
# PHIFilterConfig Tests
# =============================================================================

class TestPHIFilterConfig:
    """Test PHI filter configuration."""
    
    def test_create_config(self):
        """Should create filter config."""
        from medai_compass.pipelines.phi_detection import PHIFilterConfig
        
        config = PHIFilterConfig(
            strict=True,
            log_detections=True,
            scan_nested=True
        )
        
        assert config.strict is True
        assert config.log_detections is True
    
    def test_config_defaults(self):
        """Should have secure defaults."""
        from medai_compass.pipelines.phi_detection import PHIFilterConfig
        
        config = PHIFilterConfig()
        
        # Secure defaults
        assert config.strict is True  # Per user requirement
        assert config.scan_nested is True  # Check all fields


# =============================================================================
# PHIFilterResult Tests
# =============================================================================

class TestPHIFilterResult:
    """Test PHI filter result dataclass."""
    
    def test_create_result(self):
        """Should create filter result."""
        from medai_compass.pipelines.phi_detection import PHIFilterResult
        
        result = PHIFilterResult(
            contains_phi=True,
            phi_types=["ssn", "phone"],
            phi_locations={"ssn": [10, 21], "phone": [50, 62]},
            risk_level="high"
        )
        
        assert result.contains_phi is True
        assert len(result.phi_types) == 2
    
    def test_result_risk_level(self):
        """Result should include risk assessment."""
        from medai_compass.pipelines.phi_detection import PHIFilterResult
        
        result = PHIFilterResult(
            contains_phi=True,
            phi_types=["ssn"],
            phi_locations={},
            risk_level="critical"  # SSN is critical
        )
        
        assert result.risk_level == "critical"


# =============================================================================
# Integration with Pipeline Tests
# =============================================================================

class TestPHIPipelineIntegration:
    """Test PHI filter integration with data pipeline."""
    
    @pytest.fixture
    def sample_data_with_phi(self, tmp_path) -> Path:
        """Create sample data with PHI."""
        records = [
            {"question": "What is diabetes?", "answer": "A metabolic disease."},
            {"question": "Patient MRN: 12345678 diagnosis?", "answer": "Hypertension"},
            {"question": "Treatment options?", "answer": "Diet and exercise."},
        ]
        
        data_path = tmp_path / "phi_test.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        return tmp_path
    
    def test_pipeline_filters_phi_by_default(self, sample_data_with_phi):
        """Pipeline should filter PHI by default."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        
        # PHI filtering should be enabled
        assert pipeline.config.phi_filter_enabled is True
        assert pipeline.config.phi_filter_strict is True
    
    def test_pipeline_preprocessing_removes_phi(self, sample_data_with_phi):
        """Preprocessing should remove PHI records."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(sample_data_with_phi),
            dataset_type="generic"
        )
        
        # Preprocess with PHI filtering
        processed = pipeline.preprocess(dataset)
        
        # Should have fewer records (PHI removed)
        original_count = dataset.count()
        processed_count = processed.count()
        
        assert processed_count < original_count
        assert processed_count == 2  # Only clean records


# =============================================================================
# Audit Logging Tests
# =============================================================================

class TestPHIAuditLogging:
    """Test PHI detection audit logging."""
    
    def test_audit_log_created(self):
        """Should create audit log for PHI detections."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True, log_detections=True)
        
        record = {"question": "SSN 123-45-6789", "answer": "Test"}
        phi_filter.filter_record(record)
        
        # Should have logged the detection
        assert len(phi_filter.audit_log) > 0
    
    def test_audit_log_contains_phi_type(self):
        """Audit log should contain PHI type."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True, log_detections=True)
        
        record = {"question": "Patient email: test@hospital.com", "answer": "N/A"}
        phi_filter.filter_record(record)
        
        log_entry = phi_filter.audit_log[-1]
        assert "email" in log_entry["phi_types"]
    
    def test_audit_log_has_timestamp(self):
        """Audit log should have timestamp."""
        from medai_compass.pipelines.phi_detection import PHIPipelineFilter
        
        phi_filter = PHIPipelineFilter(strict=True, log_detections=True)
        
        record = {"question": "MRN: 12345678", "answer": "Test"}
        phi_filter.filter_record(record)
        
        log_entry = phi_filter.audit_log[-1]
        assert "timestamp" in log_entry
