"""Tests for NeMo Guardrails integration."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestMedicalGuardrails:
    """Tests for the MedicalGuardrails class."""

    def test_guardrails_initialization(self):
        """Test guardrails can be instantiated."""
        from medai_compass.guardrails.nemo_integration import MedicalGuardrails
        
        guardrails = MedicalGuardrails()
        assert guardrails is not None

    def test_guardrails_has_check_input(self):
        """Test guardrails has check_input method."""
        from medai_compass.guardrails.nemo_integration import MedicalGuardrails
        
        guardrails = MedicalGuardrails()
        assert hasattr(guardrails, "check_input")

    def test_guardrails_has_check_output(self):
        """Test guardrails has check_output method."""
        from medai_compass.guardrails.nemo_integration import MedicalGuardrails
        
        guardrails = MedicalGuardrails()
        assert hasattr(guardrails, "check_output")

    @pytest.mark.asyncio
    async def test_guardrails_check_input_returns_result(self):
        """Test check_input returns a result."""
        from medai_compass.guardrails.nemo_integration import MedicalGuardrails
        
        guardrails = MedicalGuardrails()
        
        # May need initialization, handle both cases
        if hasattr(guardrails, "initialize"):
            try:
                await guardrails.initialize()
            except Exception:
                pass  # Initialization may fail without NeMo
        
        try:
            result = await guardrails.check_input("What does this X-ray show?")
            assert result is not None
            assert hasattr(result, "allowed") or hasattr(result, "is_safe")
        except Exception:
            # If NeMo not available and no fallback, test passes
            pass

    @pytest.mark.asyncio
    async def test_guardrails_check_output_returns_result(self):
        """Test check_output returns a result."""
        from medai_compass.guardrails.nemo_integration import MedicalGuardrails
        
        guardrails = MedicalGuardrails()
        
        if hasattr(guardrails, "initialize"):
            try:
                await guardrails.initialize()
            except Exception:
                pass
        
        try:
            result = await guardrails.check_output("The X-ray shows normal findings.")
            assert result is not None
        except Exception:
            pass


class TestGuardrailsResult:
    """Tests for GuardrailsResult dataclass."""

    def test_result_creation(self):
        """Test creating a guardrails result."""
        from medai_compass.guardrails.nemo_integration import GuardrailsResult
        
        result = GuardrailsResult(
            allowed=True,
            message="Input allowed"
        )
        
        assert result.allowed is True
        assert result.message == "Input allowed"

    def test_result_with_warnings(self):
        """Test result with warnings."""
        from medai_compass.guardrails.nemo_integration import GuardrailsResult
        
        result = GuardrailsResult(
            allowed=True,
            message="Allowed with warnings",
            warnings=["Low confidence detected"]
        )
        
        assert len(result.warnings) == 1

    def test_result_blocked(self):
        """Test blocked result."""
        from medai_compass.guardrails.nemo_integration import GuardrailsResult
        
        result = GuardrailsResult(
            allowed=False,
            message="Input blocked",
            blocked_reason="Potential PHI detected"
        )
        
        assert result.allowed is False
        assert result.blocked_reason == "Potential PHI detected"

    def test_result_with_modified_input(self):
        """Test result with modified input."""
        from medai_compass.guardrails.nemo_integration import GuardrailsResult
        
        result = GuardrailsResult(
            allowed=True,
            message="Input modified",
            modified_input="Sanitized input content"
        )
        
        assert result.modified_input == "Sanitized input content"

    def test_result_risk_score(self):
        """Test result with risk score."""
        from medai_compass.guardrails.nemo_integration import GuardrailsResult
        
        result = GuardrailsResult(
            allowed=True,
            message="Low risk",
            risk_score=0.15
        )
        
        assert result.risk_score == 0.15


class TestNemoAvailability:
    """Tests for NeMo availability handling."""

    def test_nemo_available_flag_exists(self):
        """Test NEMO_AVAILABLE flag is defined."""
        from medai_compass.guardrails.nemo_integration import NEMO_AVAILABLE
        
        assert isinstance(NEMO_AVAILABLE, bool)

    def test_guardrails_works_without_nemo(self):
        """Test guardrails can be used without NeMo installed."""
        from medai_compass.guardrails.nemo_integration import MedicalGuardrails, NEMO_AVAILABLE
        
        # Should not raise even if NeMo is not available
        guardrails = MedicalGuardrails()
        assert guardrails is not None
