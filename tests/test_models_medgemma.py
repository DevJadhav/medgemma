"""Tests for MedGemma model wrapper - Written FIRST (TDD)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys


@pytest.fixture
def mock_transformers():
    """Mock transformers module completely."""
    mock_model = MagicMock()
    mock_model.device = "cpu"
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "AI generated response"
    
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    
    mock_auto_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    mock_auto_processor = MagicMock()
    mock_bnb = MagicMock()
    
    mock_transformers = MagicMock()
    mock_transformers.AutoModelForCausalLM = mock_auto_model
    mock_transformers.AutoTokenizer = mock_auto_tokenizer
    mock_transformers.AutoProcessor = mock_auto_processor
    mock_transformers.BitsAndBytesConfig = mock_bnb
    
    return mock_transformers, mock_model, mock_tokenizer


class TestMedGemmaModelLoading:
    """Test MedGemma model loading."""

    def test_load_4b_model_with_quantization(self, mock_transformers):
        """Test loading 4B model with 4-bit quantization."""
        mock_tf, mock_model, mock_tokenizer = mock_transformers
        
        with patch.dict(sys.modules, {'transformers': mock_tf}):
            # Clear cached import
            if 'medai_compass.models.medgemma' in sys.modules:
                del sys.modules['medai_compass.models.medgemma']
            
            from medai_compass.models.medgemma import MedGemmaWrapper
            wrapper = MedGemmaWrapper(
                model_name="google/medgemma-4b-it",
                quantization="4bit"
            )
        
        assert wrapper.model is not None
        assert wrapper.quantization == "4bit"

    def test_load_27b_model(self, mock_transformers):
        """Test loading 27B model."""
        mock_tf, mock_model, mock_tokenizer = mock_transformers
        
        with patch.dict(sys.modules, {'transformers': mock_tf}):
            if 'medai_compass.models.medgemma' in sys.modules:
                del sys.modules['medai_compass.models.medgemma']
            
            from medai_compass.models.medgemma import MedGemmaWrapper
            wrapper = MedGemmaWrapper(model_name="google/medgemma-27b-it")
        
        assert wrapper.model is not None


class TestMedGemmaInference:
    """Test MedGemma inference capabilities."""

    def test_confidence_extraction(self):
        """Test extraction of confidence scores from output."""
        from medai_compass.models.medgemma import extract_confidence
        
        # Response with explicit confidence
        assert extract_confidence("Findings show pneumonia. Confidence: 0.92") == 0.92
        assert extract_confidence("I am 85% confident") == 0.85
        assert extract_confidence("[0.95] certainty") == 0.95
        assert extract_confidence("No confidence mentioned") == 0.5  # Default

    def test_text_inference_mocked(self, mock_transformers):
        """Test text-only inference with mocked model."""
        mock_tf, mock_model, mock_tokenizer = mock_transformers
        
        with patch.dict(sys.modules, {'transformers': mock_tf}):
            if 'medai_compass.models.medgemma' in sys.modules:
                del sys.modules['medai_compass.models.medgemma']
                
            from medai_compass.models.medgemma import MedGemmaWrapper
            wrapper = MedGemmaWrapper(model_name="google/medgemma-4b-it")
            
            # Directly test that wrapper has expected structure
            assert hasattr(wrapper, 'generate')
            assert hasattr(wrapper, 'model')
            assert hasattr(wrapper, 'tokenizer')

    def test_multimodal_flag(self, mock_transformers):
        """Test multimodal model loading."""
        mock_tf, mock_model, mock_tokenizer = mock_transformers
        
        with patch.dict(sys.modules, {'transformers': mock_tf}):
            if 'medai_compass.models.medgemma' in sys.modules:
                del sys.modules['medai_compass.models.medgemma']
                
            from medai_compass.models.medgemma import MedGemmaWrapper
            wrapper = MedGemmaWrapper(
                model_name="google/medgemma-4b-it",
                multimodal=True
            )
            
            assert wrapper.multimodal is True
            assert wrapper.processor is not None


class TestMedGemmaBatching:
    """Test batch inference."""

    def test_batch_method_exists(self, mock_transformers):
        """Test batched inference method exists."""
        mock_tf, mock_model, mock_tokenizer = mock_transformers
        
        with patch.dict(sys.modules, {'transformers': mock_tf}):
            if 'medai_compass.models.medgemma' in sys.modules:
                del sys.modules['medai_compass.models.medgemma']
                
            from medai_compass.models.medgemma import MedGemmaWrapper
            wrapper = MedGemmaWrapper(model_name="google/medgemma-4b-it")
            
            assert hasattr(wrapper, 'generate_batch')
