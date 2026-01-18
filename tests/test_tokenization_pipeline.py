"""TDD Tests for Tokenization Pipeline.

Tests for model-aware tokenization supporting MedGemma 4B/27B.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml


# =============================================================================
# Tokenization Module Structure Tests
# =============================================================================

class TestTokenizationModuleStructure:
    """Test tokenization module file structure."""
    
    @pytest.fixture
    def tokenization_module_path(self) -> Path:
        """Path to tokenization module."""
        return Path(__file__).parent.parent / "medai_compass" / "pipelines" / "tokenization.py"
    
    def test_tokenization_module_exists(self, tokenization_module_path: Path):
        """Tokenization module must exist."""
        assert tokenization_module_path.exists(), f"Module not found at {tokenization_module_path}"
    
    def test_tokenization_has_required_classes(self, tokenization_module_path: Path):
        """Module should have required classes."""
        content = tokenization_module_path.read_text()
        
        required_classes = [
            "MedicalTokenizer",
            "TokenizationConfig",
        ]
        
        for cls in required_classes:
            assert f"class {cls}" in content, f"Missing class: {cls}"
    
    def test_tokenization_has_model_selection(self, tokenization_module_path: Path):
        """Module should support model selection."""
        content = tokenization_module_path.read_text()
        
        # Should reference model selection
        assert "model" in content.lower(), "Should support model selection"
        assert "medgemma" in content.lower() or "select_model" in content, \
            "Should reference MedGemma models"


# =============================================================================
# MedicalTokenizer Class Tests
# =============================================================================

class TestMedicalTokenizerCreation:
    """Test MedicalTokenizer instantiation."""
    
    def test_create_tokenizer_for_4b(self):
        """Should create tokenizer for MedGemma 4B."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        assert tokenizer is not None
        assert tokenizer.model_name == "medgemma_4b_it"
    
    def test_create_tokenizer_for_27b(self):
        """Should create tokenizer for MedGemma 27B."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-27b")
        
        assert tokenizer is not None
        assert tokenizer.model_name == "medgemma_27b_it"
    
    def test_tokenizer_has_max_length(self):
        """Tokenizer should have max_length from model config."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        # MedGemma supports 128K context
        assert tokenizer.max_length > 0
        assert tokenizer.max_length <= 128000


# =============================================================================
# TokenizationConfig Tests
# =============================================================================

class TestTokenizationConfig:
    """Test TokenizationConfig dataclass."""
    
    def test_create_config(self):
        """Should create tokenization config."""
        from medai_compass.pipelines.tokenization import TokenizationConfig
        
        config = TokenizationConfig(
            model_name="medgemma-4b",
            max_length=4096,
            padding="max_length",
            truncation=True
        )
        
        assert config.max_length == 4096
        assert config.truncation is True
    
    def test_config_defaults_for_4b(self):
        """Should have appropriate defaults for 4B model."""
        from medai_compass.pipelines.tokenization import TokenizationConfig
        
        config = TokenizationConfig.for_model("medgemma-4b")
        
        assert config.max_length > 0
        # 4B can handle larger batches
        assert config.model_name == "medgemma_4b_it"
    
    def test_config_defaults_for_27b(self):
        """Should have appropriate defaults for 27B model."""
        from medai_compass.pipelines.tokenization import TokenizationConfig
        
        config = TokenizationConfig.for_model("medgemma-27b")
        
        assert config.max_length > 0
        # 27B may need different settings
        assert config.model_name == "medgemma_27b_it"


# =============================================================================
# Tokenization Tests
# =============================================================================

class TestTokenization:
    """Test tokenization functionality."""
    
    @pytest.fixture
    def sample_text(self) -> str:
        """Sample medical text for tokenization."""
        return "What are the symptoms of hypertension? Hypertension often has no symptoms."
    
    @pytest.fixture
    def sample_qa_record(self) -> Dict[str, str]:
        """Sample QA record for tokenization."""
        return {
            "question": "What is diabetes?",
            "answer": "Diabetes is a metabolic disease characterized by high blood sugar."
        }
    
    def test_tokenize_text(self, sample_text):
        """Should tokenize plain text."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        result = tokenizer.tokenize(sample_text)
        
        assert "input_ids" in result
        assert len(result["input_ids"]) > 0
    
    def test_tokenize_returns_attention_mask(self, sample_text):
        """Should return attention mask."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        result = tokenizer.tokenize(sample_text)
        
        assert "attention_mask" in result
        assert len(result["attention_mask"]) == len(result["input_ids"])
    
    def test_tokenize_qa_record(self, sample_qa_record):
        """Should tokenize QA record into instruction format."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        result = tokenizer.tokenize_qa(sample_qa_record)
        
        assert "input_ids" in result
        assert "labels" in result  # For training
    
    def test_truncation_works(self):
        """Should truncate long sequences."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b", max_length=50)
        
        long_text = "word " * 1000  # Very long text
        result = tokenizer.tokenize(long_text)
        
        assert len(result["input_ids"]) <= 50


# =============================================================================
# Instruction Format Tests
# =============================================================================

class TestInstructionFormat:
    """Test instruction-tuning format tokenization."""
    
    def test_format_for_instruction_tuning(self):
        """Should format record for instruction tuning."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        record = {
            "instruction": "Answer the medical question.",
            "input": "What causes hypertension?",
            "output": "Hypertension can be caused by genetics, diet, and lifestyle."
        }
        
        formatted = tokenizer.format_instruction(record)
        
        assert isinstance(formatted, str)
        assert "instruction" in formatted.lower() or record["instruction"] in formatted
    
    def test_qa_to_instruction_format(self):
        """Should convert QA to instruction format."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        qa = {
            "question": "What is diabetes?",
            "answer": "Diabetes is a metabolic disease."
        }
        
        instruction = tokenizer.qa_to_instruction(qa)
        
        assert "instruction" in instruction or "prompt" in instruction
        assert qa["question"] in str(instruction.values())
    
    def test_synthea_to_instruction_format(self):
        """Should convert Synthea patient data to instruction format."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        patient = {
            "resourceType": "Patient",
            "id": "synth-001",
            "name": [{"text": "Test Patient"}],
            "conditions": [{"display": "Hypertension"}]
        }
        
        instruction = tokenizer.patient_to_instruction(patient)
        
        assert instruction is not None
        # Should create clinical reasoning prompt
        assert isinstance(instruction, dict)


# =============================================================================
# Batch Tokenization Tests
# =============================================================================

class TestBatchTokenization:
    """Test batch tokenization."""
    
    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for batch tokenization."""
        return [
            "What is hypertension?",
            "Diabetes is a metabolic disease.",
            "How to treat pneumonia?",
        ]
    
    def test_tokenize_batch(self, sample_texts):
        """Should tokenize batch of texts."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        results = tokenizer.tokenize_batch(sample_texts)
        
        assert len(results["input_ids"]) == 3
    
    def test_batch_padding(self, sample_texts):
        """Batch should be padded to same length."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b", padding=True)
        results = tokenizer.tokenize_batch(sample_texts)
        
        # All sequences should have same length when padded
        lengths = [len(ids) for ids in results["input_ids"]]
        assert len(set(lengths)) == 1  # All same length


# =============================================================================
# Special Tokens Tests
# =============================================================================

class TestSpecialTokens:
    """Test special token handling."""
    
    def test_has_special_tokens(self):
        """Tokenizer should have special tokens."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        # Should have standard special tokens
        assert tokenizer.pad_token_id is not None or tokenizer.eos_token_id is not None
    
    def test_instruction_uses_special_format(self):
        """Instruction format should use special delimiters."""
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        record = {
            "instruction": "Test instruction",
            "input": "Test input",
            "output": "Test output"
        }
        
        formatted = tokenizer.format_instruction(record)
        
        # Should have some delimiter structure
        assert "\n" in formatted or "<" in formatted or "###" in formatted


# =============================================================================
# Integration Tests
# =============================================================================

class TestTokenizationIntegration:
    """Test tokenization integration with pipeline."""
    
    @pytest.fixture
    def sample_data_path(self, tmp_path) -> Path:
        """Create sample data file."""
        records = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        
        data_path = tmp_path / "tokenization_test.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        return tmp_path
    
    def test_tokenizer_works_with_pipeline(self, sample_data_path):
        """Tokenizer should integrate with data pipeline."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        from medai_compass.pipelines.tokenization import MedicalTokenizer
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        tokenizer = MedicalTokenizer(model_name="medgemma-4b")
        
        # Load and preprocess should apply tokenization
        dataset = pipeline.load_dataset(str(sample_data_path), dataset_type="generic")
        
        # Get a sample and tokenize
        sample = dataset.take(1)[0]
        
        if "question" in sample:
            result = tokenizer.tokenize_qa(sample)
            assert "input_ids" in result
