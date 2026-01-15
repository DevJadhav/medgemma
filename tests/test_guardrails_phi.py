"""Tests for PHI detection and masking - Written FIRST (TDD)."""

import pytest


class TestPHIPatternDetection:
    """Test PHI pattern detection."""

    def test_detect_ssn(self):
        """Test SSN pattern detection: 123-45-6789."""
        from medai_compass.guardrails.phi_detection import detect_phi
        
        text = "Patient SSN is 123-45-6789 for records."
        
        result = detect_phi(text)
        
        assert "ssn" in result
        assert "123-45-6789" in result["ssn"]

    def test_detect_mrn(self):
        """Test MRN pattern detection: MRN: 12345678."""
        from medai_compass.guardrails.phi_detection import detect_phi
        
        text = "MRN: 12345678 for patient records."
        
        result = detect_phi(text)
        
        assert "mrn" in result
        assert any("12345678" in m for m in result["mrn"])

    def test_detect_phone_number(self):
        """Test phone number detection."""
        from medai_compass.guardrails.phi_detection import detect_phi
        
        text = "Contact the patient at 555-123-4567."
        
        result = detect_phi(text)
        
        assert "phone" in result
        assert "555-123-4567" in result["phone"]

    def test_detect_email(self):
        """Test email pattern detection."""
        from medai_compass.guardrails.phi_detection import detect_phi
        
        text = "Send results to john.doe@hospital.com"
        
        result = detect_phi(text)
        
        assert "email" in result
        assert "john.doe@hospital.com" in result["email"]

    def test_detect_date_of_birth(self):
        """Test DOB pattern detection: MM/DD/YYYY."""
        from medai_compass.guardrails.phi_detection import detect_phi
        
        text = "Patient DOB: 01/15/1980"
        
        result = detect_phi(text)
        
        assert "dob" in result
        assert "01/15/1980" in result["dob"]

    def test_no_false_positives_medical_text(self, sample_text_without_phi):
        """Test normal medical text doesn't trigger false positives."""
        from medai_compass.guardrails.phi_detection import detect_phi
        
        result = detect_phi(sample_text_without_phi)
        
        # Should have no PHI detected
        total_phi = sum(len(v) for v in result.values())
        assert total_phi == 0


class TestPHIMasking:
    """Test PHI masking functionality."""

    def test_mask_ssn(self):
        """Test SSN is replaced with [SSN_REDACTED]."""
        from medai_compass.guardrails.phi_detection import mask_phi
        
        text = "Patient SSN is 123-45-6789 for records."
        
        masked = mask_phi(text)
        
        assert "123-45-6789" not in masked
        assert "[SSN_REDACTED]" in masked

    def test_mask_multiple_phi(self, sample_text_with_phi):
        """Test masking multiple PHI instances."""
        from medai_compass.guardrails.phi_detection import mask_phi
        
        masked = mask_phi(sample_text_with_phi)
        
        # Original PHI should not be present
        assert "123-45-6789" not in masked
        assert "12345678" not in masked
        assert "555-123-4567" not in masked
        
        # Redaction markers should be present
        assert "[SSN_REDACTED]" in masked
        assert "[PHONE_REDACTED]" in masked

    def test_mask_preserves_medical_content(self):
        """Test masking preserves non-PHI medical content."""
        from medai_compass.guardrails.phi_detection import mask_phi
        
        text = "Patient (SSN: 123-45-6789) has bilateral infiltrates."
        
        masked = mask_phi(text)
        
        assert "bilateral infiltrates" in masked
        assert "[SSN_REDACTED]" in masked

    def test_mask_returns_detection_list(self):
        """Test mask_phi returns list of detected PHI."""
        from medai_compass.guardrails.phi_detection import mask_phi
        
        text = "SSN: 123-45-6789, Phone: 555-123-4567"
        
        masked, detected = mask_phi(text, return_detected=True)
        
        assert len(detected) >= 2
        assert any("ssn" in d.lower() for d in detected)


class TestPHIValidator:
    """Test PHI validation for safe output."""

    def test_validate_clean_text_passes(self, sample_text_without_phi):
        """Test clean text passes validation."""
        from medai_compass.guardrails.phi_detection import validate_no_phi
        
        is_safe, issues = validate_no_phi(sample_text_without_phi)
        
        assert is_safe is True
        assert len(issues) == 0

    def test_validate_phi_text_fails(self, sample_text_with_phi):
        """Test text with PHI fails validation."""
        from medai_compass.guardrails.phi_detection import validate_no_phi
        
        is_safe, issues = validate_no_phi(sample_text_with_phi)
        
        assert is_safe is False
        assert len(issues) > 0
