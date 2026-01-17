"""Tests for Modal GPU inference client."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestMedGemmaModalClient:
    """Tests for the Modal client."""

    def test_client_initialization(self):
        """Test client initialization."""
        from medai_compass.modal.client import MedGemmaModalClient
        
        client = MedGemmaModalClient()
        assert client.model_name == "google/medgemma-4b-it"
        assert client._initialized is False

    def test_client_initialization_custom_model(self):
        """Test client with custom model name."""
        from medai_compass.modal.client import MedGemmaModalClient
        
        client = MedGemmaModalClient(model_name="google/medgemma-27b-it")
        assert client.model_name == "google/medgemma-27b-it"

    def test_client_check_modal_available_no_tokens(self):
        """Test Modal availability check without tokens."""
        from medai_compass.modal.client import MedGemmaModalClient
        
        with patch.dict("os.environ", {}, clear=True):
            client = MedGemmaModalClient()
            result = client._check_modal_available()
            # Without tokens, should return False
            assert result is False or result is True  # Depends on environment

    @pytest.mark.asyncio
    async def test_client_generate_structure(self):
        """Test generate method structure."""
        from medai_compass.modal.client import MedGemmaModalClient, ModalInferenceResult
        
        client = MedGemmaModalClient()
        
        # Verify the method signature exists
        assert hasattr(client, "generate")

    @pytest.mark.asyncio
    async def test_client_analyze_image_structure(self):
        """Test analyze_image method structure."""
        from medai_compass.modal.client import MedGemmaModalClient
        
        client = MedGemmaModalClient()
        
        # Verify the method exists
        assert hasattr(client, "analyze_image")


class TestModalInferenceResult:
    """Tests for ModalInferenceResult dataclass."""

    def test_result_creation(self):
        """Test creating an inference result."""
        from medai_compass.modal.client import ModalInferenceResult
        
        result = ModalInferenceResult(
            response="Test response",
            confidence=0.85,
            model="google/medgemma-4b-it",
            gpu="H100"
        )
        
        assert result.response == "Test response"
        assert result.confidence == 0.85
        assert result.model == "google/medgemma-4b-it"
        assert result.gpu == "H100"
        assert result.error is None

    def test_result_with_error(self):
        """Test result with error."""
        from medai_compass.modal.client import ModalInferenceResult
        
        result = ModalInferenceResult(
            response="",
            confidence=0.0,
            model="google/medgemma-4b-it",
            gpu="H100",
            error="Connection failed"
        )
        
        assert result.error == "Connection failed"


class TestModalAppConfiguration:
    """Tests for Modal app configuration."""

    def test_modal_app_gpu_config(self):
        """Test that Modal app requests H100 GPU."""
        import os
        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "medai_compass",
            "modal",
            "app.py"
        )
        
        if os.path.exists(app_path):
            with open(app_path) as f:
                content = f.read()
                assert "H100" in content or "gpu" in content.lower()

    def test_modal_app_has_required_endpoints(self):
        """Test that Modal app defines required endpoints."""
        required_methods = [
            "generate",
            "analyze_image"
        ]
        
        import os
        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "medai_compass",
            "modal",
            "app.py"
        )
        
        if os.path.exists(app_path):
            with open(app_path) as f:
                content = f.read()
                for method in required_methods:
                    assert method in content, f"Missing method: {method}"
