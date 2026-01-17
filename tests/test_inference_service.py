"""Tests for unified inference service."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestMedGemmaInferenceService:
    """Tests for the unified MedGemma inference service."""

    def test_service_initialization(self):
        """Test service initialization."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService()
        assert service.model_name == "google/medgemma-4b-it"
        assert service._initialized is False

    def test_service_custom_model(self):
        """Test service with custom model."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService(model_name="google/medgemma-27b-it")
        assert service.model_name == "google/medgemma-27b-it"

    def test_service_force_local(self):
        """Test service with force_local flag."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService(force_local=True)
        assert service.force_local is True

    def test_service_prefer_modal(self):
        """Test service with prefer_modal flag."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService(prefer_modal=True)
        assert service.prefer_modal is True

    @pytest.mark.asyncio
    async def test_service_has_initialize_method(self):
        """Test service has initialize method."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService()
        assert hasattr(service, "initialize")
        assert callable(service.initialize)

    @pytest.mark.asyncio
    async def test_service_has_generate_method(self):
        """Test service has generate method."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService()
        assert hasattr(service, "generate")

    @pytest.mark.asyncio
    async def test_service_has_analyze_image_method(self):
        """Test service has analyze_image method."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService()
        assert hasattr(service, "analyze_image")


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_inference_result_creation(self):
        """Test creating an inference result."""
        from medai_compass.models.inference_service import InferenceResult
        
        result = InferenceResult(
            response="Test response",
            confidence=0.85,
            model="google/medgemma-4b-it",
            backend="local",
            device="cuda",
            processing_time_ms=150.5
        )
        
        assert result.response == "Test response"
        assert result.confidence == 0.85
        assert result.backend == "local"
        assert result.processing_time_ms == 150.5
        assert result.error is None

    def test_inference_result_with_error(self):
        """Test inference result with error."""
        from medai_compass.models.inference_service import InferenceResult
        
        result = InferenceResult(
            response="",
            confidence=0.0,
            model="google/medgemma-4b-it",
            backend="local",
            device="cpu",
            error="Model loading failed"
        )
        
        assert result.error == "Model loading failed"
        assert result.response == ""

    def test_inference_result_default_values(self):
        """Test inference result default values."""
        from medai_compass.models.inference_service import InferenceResult
        
        result = InferenceResult(
            response="Test",
            confidence=0.9,
            model="google/medgemma-4b-it",
            backend="modal",
            device="h100"
        )
        
        assert result.tokens_generated == 0
        assert result.processing_time_ms == 0.0
        assert result.error is None
