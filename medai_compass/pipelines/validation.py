"""Data Validation Module for Medical Data Pipeline.

Provides schema validation and data quality checks for medical datasets
including Synthea, MedQuAD, and instruction-tuning formats.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

import yaml

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""
    pass


@dataclass
class ValidationResult:
    """Result of validating a single record."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    record_id: Optional[str] = None
    
    def __post_init__(self):
        # Ensure is_valid is consistent with errors
        if self.errors and self.is_valid:
            self.is_valid = False


@dataclass
class SchemaDefinition:
    """Definition of a data schema."""
    
    name: str
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    min_lengths: Dict[str, int] = field(default_factory=dict)
    max_lengths: Dict[str, int] = field(default_factory=dict)


class MedicalRecordSchema:
    """Schema definitions for medical record types."""
    
    # Schema registry
    _schemas: Dict[str, SchemaDefinition] = {}
    
    @classmethod
    def _init_schemas(cls):
        """Initialize built-in schemas."""
        if cls._schemas:
            return
        
        # QA schema (MedQuAD format)
        cls._schemas["qa"] = SchemaDefinition(
            name="qa",
            required_fields=["question", "answer"],
            optional_fields=["source", "focus", "url"],
            field_types={"question": str, "answer": str},
            min_lengths={"question": 5, "answer": 5},
            max_lengths={"question": 10000, "answer": 50000},
        )
        
        # MedQuAD schema (alias for QA)
        cls._schemas["medquad"] = SchemaDefinition(
            name="medquad",
            required_fields=["question", "answer"],
            optional_fields=["source", "focus", "url", "category"],
            field_types={"question": str, "answer": str},
            min_lengths={"question": 5, "answer": 5},
            max_lengths={"question": 10000, "answer": 50000},
        )
        
        # Instruction schema (instruction-tuning format)
        cls._schemas["instruction"] = SchemaDefinition(
            name="instruction",
            required_fields=["instruction", "output"],
            optional_fields=["input", "source", "category"],
            field_types={"instruction": str, "output": str, "input": str},
            min_lengths={"instruction": 5, "output": 1},
            max_lengths={"instruction": 5000, "output": 50000},
        )
        
        # Synthea FHIR Patient schema
        cls._schemas["synthea"] = SchemaDefinition(
            name="synthea",
            required_fields=["resourceType"],
            optional_fields=["id", "name", "birthDate", "gender", "conditions"],
            field_types={"resourceType": str, "id": str},
        )
        
        # FHIR Patient schema
        cls._schemas["fhir_patient"] = SchemaDefinition(
            name="fhir_patient",
            required_fields=["resourceType", "id"],
            optional_fields=["name", "birthDate", "gender", "address"],
            field_types={"resourceType": str, "id": str},
        )
        
        # Generic schema (minimal requirements)
        cls._schemas["generic"] = SchemaDefinition(
            name="generic",
            required_fields=[],  # No required fields
            optional_fields=["text", "prompt", "completion"],
            field_types={},
        )
    
    @classmethod
    def get_schema(cls, schema_type: str) -> SchemaDefinition:
        """
        Get schema definition by type.
        
        Args:
            schema_type: Schema type name
            
        Returns:
            SchemaDefinition for the type
        """
        cls._init_schemas()
        
        if schema_type not in cls._schemas:
            raise ValueError(f"Unknown schema type: {schema_type}. "
                           f"Available: {list(cls._schemas.keys())}")
        
        return cls._schemas[schema_type]
    
    @classmethod
    def register_schema(cls, schema: SchemaDefinition):
        """Register a custom schema."""
        cls._init_schemas()
        cls._schemas[schema.name] = schema


class DataValidator:
    """
    Validator for medical data records.
    
    Supports schema validation, field type checking, and data quality checks.
    
    Attributes:
        strict: If True, raise ValidationError on invalid records in batch
    """
    
    def __init__(
        self,
        strict: bool = False,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the validator.
        
        Args:
            strict: Raise ValidationError on invalid records
            config_path: Optional path to validation config
        """
        self.strict = strict
        self._config = self._load_config(config_path)
        
        # Initialize schemas
        MedicalRecordSchema._init_schemas()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load validation configuration."""
        if config_path:
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Load from default location
        default_path = Path(__file__).parent.parent.parent / "config" / "pipeline.yaml"
        if default_path.exists():
            with open(default_path) as f:
                config = yaml.safe_load(f)
                return config.get("validation", {})
        
        return {}
    
    def validate(
        self,
        record: Dict[str, Any],
        schema_type: str = "generic",
        record_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a single record against a schema.
        
        Args:
            record: Record to validate
            schema_type: Schema type to validate against
            record_id: Optional record ID for debugging
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Auto-detect record ID
        if record_id is None:
            record_id = record.get("id") or record.get("patient_id") or "unknown"
        
        try:
            schema = MedicalRecordSchema.get_schema(schema_type)
        except ValueError as e:
            errors.append(str(e))
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                record_id=record_id,
            )
        
        # Check required fields
        for field_name in schema.required_fields:
            if field_name not in record:
                errors.append(f"Missing required field: {field_name}")
            elif record[field_name] is None:
                errors.append(f"Required field is None: {field_name}")
        
        # Check field types
        for field_name, expected_type in schema.field_types.items():
            if field_name in record and record[field_name] is not None:
                if not isinstance(record[field_name], expected_type):
                    errors.append(
                        f"Invalid type for {field_name}: expected {expected_type.__name__}, "
                        f"got {type(record[field_name]).__name__}"
                    )
        
        # Check minimum lengths
        for field_name, min_len in schema.min_lengths.items():
            if field_name in record and record[field_name] is not None:
                value = record[field_name]
                if isinstance(value, str) and len(value) < min_len:
                    if len(value) == 0:
                        errors.append(f"Field {field_name} is empty")
                    else:
                        warnings.append(
                            f"Field {field_name} is short ({len(value)} chars, min {min_len})"
                        )
        
        # Check maximum lengths
        for field_name, max_len in schema.max_lengths.items():
            if field_name in record and record[field_name] is not None:
                value = record[field_name]
                if isinstance(value, str) and len(value) > max_len:
                    warnings.append(
                        f"Field {field_name} is long ({len(value)} chars, max {max_len})"
                    )
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            record_id=record_id,
        )
    
    def validate_batch(
        self,
        records: List[Dict[str, Any]],
        schema_type: str = "generic",
    ) -> List[ValidationResult]:
        """
        Validate a batch of records.
        
        Args:
            records: List of records to validate
            schema_type: Schema type to validate against
            
        Returns:
            List of ValidationResults
            
        Raises:
            ValidationError: If strict mode and any record is invalid
        """
        results = []
        
        for i, record in enumerate(records):
            record_id = record.get("id") or f"record_{i}"
            result = self.validate(record, schema_type, record_id)
            results.append(result)
            
            if self.strict and not result.is_valid:
                raise ValidationError(
                    f"Validation failed for {record_id}: {result.errors}"
                )
        
        return results
    
    def validate_dataset_file(
        self,
        file_path: str,
        schema_type: str = "generic",
    ) -> Dict[str, Any]:
        """
        Validate a dataset file.
        
        Args:
            file_path: Path to JSON dataset file
            schema_type: Schema type to validate against
            
        Returns:
            Summary of validation results
        """
        import json
        
        with open(file_path) as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            records = [data]
        else:
            records = data
        
        results = self.validate_batch(records, schema_type)
        
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = sum(1 for r in results if not r.is_valid)
        warning_count = sum(len(r.warnings) for r in results)
        
        return {
            "total": len(results),
            "valid": valid_count,
            "invalid": invalid_count,
            "warnings": warning_count,
            "results": results,
        }
