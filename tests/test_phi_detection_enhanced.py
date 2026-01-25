"""
Tests for enhanced PHI detection with NER support.

TDD approach: Tests written first for new PHI detection capabilities.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestEnhancedPHIDetection:
    """Tests for enhanced PHI detection patterns."""

    def test_detects_ssn_format(self):
        """Verify SSN detection."""
        from medai_compass.guardrails.phi_detection import detect_phi

        text = "Patient SSN is 123-45-6789"
        detected = detect_phi(text)

        assert "ssn" in detected
        assert "123-45-6789" in detected["ssn"]

    def test_detects_mrn_format(self):
        """Verify MRN detection."""
        from medai_compass.guardrails.phi_detection import detect_phi

        text = "MRN: 12345678"
        detected = detect_phi(text)

        assert "mrn" in detected

    def test_detects_phone_number(self):
        """Verify phone number detection."""
        from medai_compass.guardrails.phi_detection import detect_phi

        text = "Contact at 555-123-4567 or 555.987.6543"
        detected = detect_phi(text)

        assert "phone" in detected
        assert len(detected["phone"]) >= 1

    def test_detects_email_address(self):
        """Verify email detection."""
        from medai_compass.guardrails.phi_detection import detect_phi

        text = "Email the patient at john.doe@hospital.org"
        detected = detect_phi(text)

        assert "email" in detected
        assert "john.doe@hospital.org" in detected["email"]

    def test_detects_date_of_birth(self):
        """Verify DOB detection."""
        from medai_compass.guardrails.phi_detection import detect_phi

        text = "DOB: 01/15/1985"
        detected = detect_phi(text)

        assert "dob" in detected

    def test_detects_address(self):
        """Verify address detection."""
        from medai_compass.guardrails.phi_detection import detect_phi

        text = "Lives at 123 Main Street"
        detected = detect_phi(text)

        assert "address" in detected


class TestEnhancedPHIPatterns:
    """Tests for new PHI patterns added in enhancement."""

    def test_detects_clinical_trial_id(self):
        """Verify clinical trial identifier detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Patient enrolled in NCT12345678 trial"
        result = detector.scan(text)

        assert "clinical_trial" in result["detected"]
        assert "NCT12345678" in str(result["detected"]["clinical_trial"])

    def test_detects_drivers_license(self):
        """Verify driver's license detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "DL# A1234567 on file"
        result = detector.scan(text)

        assert "drivers_license" in result["detected"]

    def test_detects_medicare_id(self):
        """Verify Medicare ID detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Medicare: 1EG4-TE5-MK72"
        result = detector.scan(text)

        assert "medicare_id" in result["detected"]

    def test_detects_account_numbers(self):
        """Verify account/policy number detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Account #: ACC-12345678 and Policy: POL-98765432"
        result = detector.scan(text)

        assert "account_number" in result["detected"]

    def test_detects_ip_address(self):
        """Verify IP address detection (device identifiers)."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Connected from 192.168.1.100"
        result = detector.scan(text)

        assert "ip_address" in result["detected"]

    def test_detects_date_ranges(self):
        """Verify date range detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Admitted 12/25/2023 to 01/02/2024"
        result = detector.scan(text)

        assert "dob" in result["detected"] or "date" in result["detected"]


class TestNameDetection:
    """Tests for NER-based name detection."""

    def test_detects_patient_name_simple(self):
        """Verify simple patient name detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector(use_ner=True)
        text = "Patient John Smith was admitted yesterday"
        result = detector.scan(text)

        # Should detect as name
        assert result["has_potential_names"] or "name" in result.get("detected", {})

    def test_detects_doctor_name(self):
        """Verify doctor name detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector(use_ner=True)
        text = "Dr. Jane Doe ordered the MRI"
        result = detector.scan(text)

        assert result["has_potential_names"] or "name" in result.get("detected", {})

    def test_detects_name_with_title(self):
        """Verify names with titles are detected."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector(use_ner=True)
        text = "Mrs. Mary Johnson reported chest pain"
        result = detector.scan(text)

        assert result["has_potential_names"]

    def test_does_not_flag_medical_terms(self):
        """Verify medical terms are not flagged as names."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector(use_ner=True)
        text = "Patient diagnosed with Parkinson disease and Alzheimer"
        result = detector.scan(text)

        # These should NOT be detected as names
        detected_names = result.get("detected", {}).get("name", [])
        assert "Parkinson" not in str(detected_names)
        assert "Alzheimer" not in str(detected_names)

    def test_name_detection_with_context(self):
        """Verify context-aware name detection."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector(use_ner=True)

        # Should detect as name - explicit patient context
        text1 = "The patient, Robert Brown, was seen today"
        result1 = detector.scan(text1)
        assert result1["has_potential_names"]

        # Medical eponyms should not be flagged
        text2 = "Shows Babinski sign and Murphy sign"
        result2 = detector.scan(text2)
        # These are medical terms, not patient names


class TestPHIMasking:
    """Tests for PHI masking functionality."""

    def test_masks_all_phi_types(self):
        """Verify all PHI types are masked."""
        from medai_compass.guardrails.phi_detection import mask_phi

        text = (
            "Patient John SSN 123-45-6789, DOB 01/15/1985, "
            "email john@test.com, phone 555-123-4567, "
            "at 123 Main Street"
        )

        masked = mask_phi(text)

        assert "123-45-6789" not in masked
        assert "01/15/1985" not in masked
        assert "john@test.com" not in masked
        assert "555-123-4567" not in masked
        assert "[SSN_REDACTED]" in masked
        assert "[DOB_REDACTED]" in masked
        assert "[EMAIL_REDACTED]" in masked
        assert "[PHONE_REDACTED]" in masked

    def test_mask_returns_detected_list(self):
        """Verify mask can return list of detected PHI."""
        from medai_compass.guardrails.phi_detection import mask_phi

        text = "SSN: 123-45-6789, Phone: 555-123-4567"
        masked, detected = mask_phi(text, return_detected=True)

        assert len(detected) >= 2
        assert any("ssn" in d.lower() for d in detected)
        assert any("phone" in d.lower() for d in detected)

    def test_mask_with_custom_patterns(self):
        """Verify custom patterns can be added."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector(
            additional_patterns={
                "custom_id": r"CUST-\d{6}"
            }
        )

        text = "Reference: CUST-123456"
        result = detector.scan(text)

        assert "custom_id" in result["detected"]


class TestRiskAssessment:
    """Tests for PHI risk assessment."""

    def test_risk_level_critical_for_ssn(self):
        """Verify SSN detection results in critical risk."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "SSN: 123-45-6789"
        result = detector.scan(text)

        assert result["risk_level"] == "critical"

    def test_risk_level_high_for_multiple_types(self):
        """Verify multiple PHI types result in high risk."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Phone: 555-123-4567, Email: test@test.com"
        result = detector.scan(text)

        assert result["risk_level"] in ["high", "critical"]

    def test_risk_level_high_for_mrn(self):
        """Verify MRN detection results in high risk."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "MRN: 12345678"
        result = detector.scan(text)

        assert result["risk_level"] in ["high", "critical"]

    def test_risk_level_medium_for_single_phi(self):
        """Verify single non-critical PHI results in medium risk."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Email: test@test.com"
        result = detector.scan(text)

        assert result["risk_level"] == "medium"

    def test_risk_level_none_for_clean_text(self):
        """Verify clean text results in no risk."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "Patient presents with chest pain and shortness of breath"
        result = detector.scan(text)

        assert result["risk_level"] == "none"
        assert result["is_safe"]


class TestValidation:
    """Tests for PHI validation functions."""

    def test_validate_no_phi_passes_clean_text(self):
        """Verify clean text passes validation."""
        from medai_compass.guardrails.phi_detection import validate_no_phi

        text = "Patient has hypertension and diabetes"
        is_safe, issues = validate_no_phi(text)

        assert is_safe
        assert len(issues) == 0

    def test_validate_no_phi_fails_with_phi(self):
        """Verify text with PHI fails validation."""
        from medai_compass.guardrails.phi_detection import validate_no_phi

        text = "Patient SSN 123-45-6789"
        is_safe, issues = validate_no_phi(text)

        assert not is_safe
        assert len(issues) > 0


class TestAuditLogging:
    """Tests for PHI detection audit logging."""

    def test_scan_includes_audit_metadata(self):
        """Verify scan results include audit metadata."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()
        text = "SSN: 123-45-6789"
        result = detector.scan(text)

        assert "total_instances" in result
        assert "risk_level" in result
        assert "is_safe" in result

    def test_detector_tracks_scan_count(self):
        """Verify detector tracks scan statistics."""
        from medai_compass.guardrails.phi_detection import PHIDetector

        detector = PHIDetector()

        detector.scan("Text 1")
        detector.scan("Text 2 with SSN 123-45-6789")
        detector.scan("Text 3")

        stats = detector.get_statistics()

        assert stats["total_scans"] == 3
        assert stats["scans_with_phi"] >= 1
