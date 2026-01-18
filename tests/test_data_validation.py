"""TDD Tests for Data Validation Module.

Tests for schema validation and data quality checks.
Uses Synthea and MedQuAD sample data.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml


# =============================================================================
# Validation Module Structure Tests
# =============================================================================

class TestValidationModuleStructure:
    """Test validation module file structure."""
    
    @pytest.fixture
    def validation_module_path(self) -> Path:
        """Path to validation module."""
        return Path(__file__).parent.parent / "medai_compass" / "pipelines" / "validation.py"
    
    def test_validation_module_exists(self, validation_module_path: Path):
        """Validation module must exist."""
        assert validation_module_path.exists(), f"Module not found at {validation_module_path}"
    
    def test_validation_has_required_classes(self, validation_module_path: Path):
        """Module should have required classes."""
        content = validation_module_path.read_text()
        
        required_classes = [
            "DataValidator",
            "ValidationResult",
            "MedicalRecordSchema",
        ]
        
        for cls in required_classes:
            assert f"class {cls}" in content, f"Missing class: {cls}"
    
    def test_validation_has_schema_definitions(self, validation_module_path: Path):
        """Module should define schemas for different data types."""
        content = validation_module_path.read_text()
        
        # Should have schema definitions
        assert "schema" in content.lower(), "Should define data schemas"


# =============================================================================
# DataValidator Class Tests
# =============================================================================

class TestDataValidatorCreation:
    """Test DataValidator instantiation."""
    
    def test_create_validator(self):
        """Should create validator instance."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        assert validator is not None
    
    def test_create_validator_with_strict_mode(self):
        """Should support strict validation mode."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator(strict=True)
        assert validator.strict is True
    
    def test_validator_has_validate_method(self):
        """Validator should have validate method."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        assert hasattr(validator, "validate")
        assert callable(validator.validate)


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Test schema-based validation."""
    
    def test_validate_valid_qa_record(self):
        """Should validate a valid QA record."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {
            "question": "What is hypertension?",
            "answer": "High blood pressure condition.",
            "source": "MedQuAD"
        }
        
        result = validator.validate(record, schema_type="qa")
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_reject_invalid_qa_record_missing_answer(self):
        """Should reject QA record without answer."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {
            "question": "What is hypertension?",
            # Missing answer
        }
        
        result = validator.validate(record, schema_type="qa")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("answer" in str(e).lower() for e in result.errors)
    
    def test_validate_valid_fhir_patient(self):
        """Should validate a valid FHIR Patient resource."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        patient = {
            "resourceType": "Patient",
            "id": "test-001",
            "name": [{"text": "Test Patient"}],
            "birthDate": "1990-01-15",
            "gender": "female"
        }
        
        result = validator.validate(patient, schema_type="fhir_patient")
        
        assert result.is_valid is True
    
    def test_reject_invalid_fhir_patient(self):
        """Should reject invalid FHIR Patient."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        patient = {
            "resourceType": "Patient",
            # Missing required id
            "name": [{"text": "Test Patient"}],
        }
        
        result = validator.validate(patient, schema_type="fhir_patient")
        
        assert result.is_valid is False


# =============================================================================
# Medical Record Schema Tests
# =============================================================================

class TestMedicalRecordSchema:
    """Test MedicalRecordSchema definitions."""
    
    def test_schema_for_synthea(self):
        """Should have schema for Synthea data."""
        from medai_compass.pipelines.validation import MedicalRecordSchema
        
        schema = MedicalRecordSchema.get_schema("synthea")
        
        assert schema is not None
        assert "patient" in schema.required_fields or "resourceType" in schema.required_fields
    
    def test_schema_for_medquad(self):
        """Should have schema for MedQuAD data."""
        from medai_compass.pipelines.validation import MedicalRecordSchema
        
        schema = MedicalRecordSchema.get_schema("medquad")
        
        assert schema is not None
        assert "question" in schema.required_fields
        assert "answer" in schema.required_fields
    
    def test_schema_for_instruction_format(self):
        """Should have schema for instruction-tuning format."""
        from medai_compass.pipelines.validation import MedicalRecordSchema
        
        schema = MedicalRecordSchema.get_schema("instruction")
        
        assert schema is not None
        # Instruction format: instruction, input (optional), output
        assert "instruction" in schema.required_fields or "prompt" in schema.required_fields


# =============================================================================
# Validation Result Tests
# =============================================================================

class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_create_valid_result(self):
        """Should create valid result."""
        from medai_compass.pipelines.validation import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            record_id="test-001"
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_create_invalid_result_with_errors(self):
        """Should create invalid result with errors."""
        from medai_compass.pipelines.validation import ValidationResult
        
        result = ValidationResult(
            is_valid=False,
            errors=["Missing required field: answer"],
            warnings=[],
            record_id="test-002"
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 1
    
    def test_result_has_record_id(self):
        """Result should track record ID for debugging."""
        from medai_compass.pipelines.validation import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            record_id="patient-12345"
        )
        
        assert result.record_id == "patient-12345"


# =============================================================================
# Batch Validation Tests
# =============================================================================

class TestBatchValidation:
    """Test validating batches of records."""
    
    @pytest.fixture
    def sample_records(self) -> List[Dict]:
        """Create sample records for batch validation."""
        return [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3"},  # Invalid - missing answer
            {"question": "Q4", "answer": "A4"},
        ]
    
    def test_validate_batch(self, sample_records):
        """Should validate a batch of records."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        results = validator.validate_batch(sample_records, schema_type="qa")
        
        assert len(results) == 4
    
    def test_batch_validation_returns_summary(self, sample_records):
        """Batch validation should return summary."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        results = validator.validate_batch(sample_records, schema_type="qa")
        
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = sum(1 for r in results if not r.is_valid)
        
        assert valid_count == 3
        assert invalid_count == 1
    
    def test_batch_strict_mode_fails_on_any_invalid(self, sample_records):
        """Strict mode should raise on any invalid record."""
        from medai_compass.pipelines.validation import DataValidator, ValidationError
        
        validator = DataValidator(strict=True)
        
        with pytest.raises(ValidationError):
            validator.validate_batch(sample_records, schema_type="qa")


# =============================================================================
# Field Type Validation Tests
# =============================================================================

class TestFieldTypeValidation:
    """Test validation of field types."""
    
    def test_validate_string_field(self):
        """Should validate string fields."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {"question": "Valid question?", "answer": "Valid answer."}
        
        result = validator.validate(record, schema_type="qa")
        
        assert result.is_valid is True
    
    def test_reject_non_string_question(self):
        """Should reject non-string question."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {"question": 12345, "answer": "Valid answer."}  # question is int
        
        result = validator.validate(record, schema_type="qa")
        
        assert result.is_valid is False
    
    def test_validate_optional_fields(self):
        """Should accept missing optional fields."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {
            "question": "What is diabetes?",
            "answer": "A metabolic disease.",
            # "source" is optional
        }
        
        result = validator.validate(record, schema_type="qa")
        
        assert result.is_valid is True
    
    def test_warn_on_empty_fields(self):
        """Should warn on empty but present fields."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {
            "question": "What is diabetes?",
            "answer": "",  # Empty answer
        }
        
        result = validator.validate(record, schema_type="qa")
        
        # Empty string might be invalid or warning depending on schema
        assert len(result.warnings) > 0 or result.is_valid is False


# =============================================================================
# Data Quality Checks Tests
# =============================================================================

class TestDataQualityChecks:
    """Test data quality validation checks."""
    
    def test_check_text_length_minimum(self):
        """Should check minimum text length."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {
            "question": "Q?",  # Too short
            "answer": "A very detailed answer about medical conditions."
        }
        
        result = validator.validate(record, schema_type="qa")
        
        # Should warn or error about short question
        assert len(result.warnings) > 0 or result.is_valid is False
    
    def test_check_text_length_maximum(self):
        """Should check maximum text length."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {
            "question": "What is hypertension?",
            "answer": "A" * 100000  # Extremely long
        }
        
        result = validator.validate(record, schema_type="qa")
        
        # Should warn about very long answer
        assert len(result.warnings) > 0
    
    def test_check_language_is_english(self):
        """Should validate language (English expected for MedGemma)."""
        from medai_compass.pipelines.validation import DataValidator
        
        validator = DataValidator()
        record = {
            "question": "What is hypertension?",
            "answer": "Hypertension is high blood pressure."
        }
        
        result = validator.validate(record, schema_type="qa")
        
        # English text should pass
        assert result.is_valid is True


# =============================================================================
# Integration with Pipeline Tests
# =============================================================================

class TestValidationPipelineIntegration:
    """Test validation integration with data pipeline."""
    
    @pytest.fixture
    def sample_data_path(self, tmp_path) -> Path:
        """Create sample data file."""
        records = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        
        data_path = tmp_path / "validation_test.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        return tmp_path
    
    def test_validator_works_with_ray_pipeline(self, sample_data_path):
        """Validator should integrate with Ray pipeline."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        from medai_compass.pipelines.validation import DataValidator
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        validator = DataValidator()
        
        # Load data
        dataset = pipeline.load_dataset(str(sample_data_path), dataset_type="generic")
        
        # Validate sample
        sample = dataset.take(1)[0]
        result = validator.validate(sample, schema_type="qa")
        
        assert result.is_valid is True
