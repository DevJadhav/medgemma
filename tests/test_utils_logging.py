"""Tests for logging utilities - Written FIRST (TDD)."""

import pytest
from unittest.mock import MagicMock, patch
import json


class TestMedicalAuditLogger:
    """Test medical audit logging for HIPAA compliance."""

    def test_log_creates_structured_entry(self):
        """Test log entry has all required fields."""
        from medai_compass.utils.logging import MedicalAuditLogger
        
        logger = MedicalAuditLogger()
        
        entry = logger.log_event(
            event_type="inference",
            user_id="user-123",
            action="analyze_image",
            resource_type="DiagnosticReport"
        )
        
        assert "timestamp" in entry
        assert "event_id" in entry
        assert entry["event_type"] == "inference"

    def test_log_hashes_user_id(self):
        """Test user IDs are hashed for privacy."""
        from medai_compass.utils.logging import MedicalAuditLogger
        
        logger = MedicalAuditLogger()
        
        entry = logger.log_event(
            event_type="access",
            user_id="user-123",
            action="view_patient",
            resource_type="Patient"
        )
        
        # User ID should be hashed, not plain text
        assert entry["user_id_hash"] != "user-123"
        assert len(entry["user_id_hash"]) == 64  # SHA-256 hex

    def test_log_ai_inference_captures_metrics(self):
        """Test AI inference logging includes model metrics."""
        from medai_compass.utils.logging import MedicalAuditLogger
        
        logger = MedicalAuditLogger()
        
        entry = logger.log_ai_inference(
            user_id="user-123",
            model_name="medgemma-4b",
            confidence=0.92,
            uncertainty=0.08,
            processing_time_ms=1250.5,
            escalated=False
        )
        
        assert entry["model_name"] == "medgemma-4b"
        assert entry["confidence"] == 0.92
        assert entry["processing_time_ms"] == 1250.5


class TestLogFormatter:
    """Test log formatting utilities."""

    def test_format_as_json(self):
        """Test logs can be formatted as JSON."""
        from medai_compass.utils.logging import format_log_entry
        
        entry = {"event_type": "test", "message": "Test message"}
        
        formatted = format_log_entry(entry, format="json")
        
        parsed = json.loads(formatted)
        assert parsed["event_type"] == "test"

    def test_format_redacts_phi(self):
        """Test PHI is automatically redacted in logs."""
        from medai_compass.utils.logging import format_log_entry
        
        entry = {"message": "Patient SSN: 123-45-6789"}
        
        formatted = format_log_entry(entry, format="json")
        
        assert "123-45-6789" not in formatted
        assert "REDACTED" in formatted
