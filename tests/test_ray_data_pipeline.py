"""TDD Tests for Ray Data Pipeline.

Tests for distributed medical data processing using Ray Data.
Uses Synthea and MedQuAD datasets for testing.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import yaml


# =============================================================================
# Pipeline Configuration Tests
# =============================================================================

class TestPipelineConfiguration:
    """Test pipeline configuration loading and validation."""
    
    @pytest.fixture
    def config_path(self) -> Path:
        """Path to pipeline configuration."""
        return Path(__file__).parent.parent / "config" / "pipeline.yaml"
    
    def test_pipeline_config_exists(self):
        """Pipeline configuration file should exist."""
        config_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        assert config_path.exists(), f"Pipeline config not found at {config_path}"
    
    def test_pipeline_config_has_required_sections(self):
        """Pipeline config should have all required sections."""
        config_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        required_sections = [
            "ray_data",
            "validation",
            "tokenization",
            "phi_detection",
            "datasets",
        ]
        
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"
    
    def test_pipeline_config_has_model_profiles(self):
        """Config should have profiles for both MedGemma 4B and 27B."""
        config_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        model_profiles = config.get("model_profiles", {})
        assert "medgemma_4b" in model_profiles, "Missing 4B model profile"
        assert "medgemma_27b" in model_profiles, "Missing 27B model profile"


# =============================================================================
# Ray Data Pipeline Module Tests
# =============================================================================

class TestRayPipelineModule:
    """Test Ray pipeline module structure."""
    
    @pytest.fixture
    def pipeline_module_path(self) -> Path:
        """Path to ray pipeline module."""
        return Path(__file__).parent.parent / "medai_compass" / "pipelines" / "ray_pipeline.py"
    
    def test_ray_pipeline_module_exists(self, pipeline_module_path: Path):
        """Ray pipeline module must exist."""
        assert pipeline_module_path.exists(), f"Module not found at {pipeline_module_path}"
    
    def test_ray_pipeline_has_required_classes(self, pipeline_module_path: Path):
        """Module should have required classes."""
        content = pipeline_module_path.read_text()
        
        required_classes = [
            "MedicalDataPipeline",
            "PipelineConfig",
        ]
        
        for cls in required_classes:
            assert f"class {cls}" in content, f"Missing class: {cls}"
    
    def test_ray_pipeline_has_required_methods(self, pipeline_module_path: Path):
        """Pipeline should have required data processing methods."""
        content = pipeline_module_path.read_text()
        
        required_methods = [
            "load_dataset",
            "preprocess",
            "create_splits",
            "get_batch_iterator",
        ]
        
        for method in required_methods:
            assert f"def {method}" in content, f"Missing method: {method}"
    
    def test_ray_pipeline_imports_ray_data(self, pipeline_module_path: Path):
        """Pipeline should import Ray Data."""
        content = pipeline_module_path.read_text()
        assert "ray.data" in content or "from ray import data" in content, \
            "Pipeline should import Ray Data"


# =============================================================================
# MedicalDataPipeline Class Tests
# =============================================================================

class TestMedicalDataPipelineCreation:
    """Test MedicalDataPipeline instantiation."""
    
    def test_create_pipeline_with_4b_model(self):
        """Should create pipeline for MedGemma 4B."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        
        assert pipeline is not None
        assert pipeline.model_name == "medgemma_4b_it"
    
    def test_create_pipeline_with_27b_model(self):
        """Should create pipeline for MedGemma 27B."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-27b")
        
        assert pipeline is not None
        assert pipeline.model_name == "medgemma_27b_it"
    
    def test_pipeline_has_phi_filter_enabled_by_default(self):
        """PHI filtering should be enabled by default (strict mode)."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        
        assert pipeline.config.phi_filter_enabled is True
        assert pipeline.config.phi_filter_strict is True
    
    def test_pipeline_batch_size_differs_by_model(self):
        """Batch size should be model-specific (smaller for 27B)."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline_4b = MedicalDataPipeline(model_name="medgemma-4b")
        pipeline_27b = MedicalDataPipeline(model_name="medgemma-27b")
        
        assert pipeline_4b.config.batch_size >= pipeline_27b.config.batch_size


# =============================================================================
# Synthea Dataset Loading Tests
# =============================================================================

class TestSyntheaDatasetLoading:
    """Test loading Synthea synthetic data."""
    
    @pytest.fixture
    def synthea_sample_data(self, tmp_path) -> Path:
        """Create sample Synthea FHIR bundle."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "synth-0001",
                        "name": [{"text": "Test Patient"}],
                        "birthDate": "1990-01-15",
                        "gender": "female"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "synth-0001-cond-1",
                        "subject": {"reference": "Patient/synth-0001"},
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": "38341003",
                                "display": "Hypertension"
                            }]
                        }
                    }
                }
            ]
        }
        
        bundle_path = tmp_path / "synthea_bundle.json"
        with open(bundle_path, "w") as f:
            json.dump(bundle, f)
        
        return tmp_path
    
    def test_load_synthea_fhir_bundle(self, synthea_sample_data):
        """Should load Synthea FHIR bundle."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(synthea_sample_data),
            dataset_type="synthea"
        )
        
        assert dataset is not None
        assert dataset.count() > 0
    
    def test_synthea_records_have_required_fields(self, synthea_sample_data):
        """Loaded Synthea records should have required fields."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(synthea_sample_data),
            dataset_type="synthea"
        )
        
        sample = dataset.take(1)[0]
        
        # Should have text for training
        assert "text" in sample or "input" in sample
    
    def test_synthea_converts_to_instruction_format(self, synthea_sample_data):
        """Synthea data should convert to instruction format."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(synthea_sample_data),
            dataset_type="synthea"
        )
        
        processed = pipeline.preprocess(dataset)
        sample = processed.take(1)[0]
        
        # Instruction format should have instruction, input, output
        assert "instruction" in sample or "prompt" in sample


# =============================================================================
# MedQuAD Dataset Loading Tests
# =============================================================================

class TestMedQuADDatasetLoading:
    """Test loading MedQuAD QA dataset."""
    
    @pytest.fixture
    def medquad_sample_data(self, tmp_path) -> Path:
        """Create sample MedQuAD data."""
        qa_data = [
            {
                "question": "What is hypertension?",
                "answer": "Hypertension is high blood pressure, a condition where the force of blood against artery walls is too high.",
                "source": "MedQuAD",
                "focus": "hypertension"
            },
            {
                "question": "What are the symptoms of diabetes?",
                "answer": "Common symptoms include increased thirst, frequent urination, extreme hunger, and unexplained weight loss.",
                "source": "MedQuAD",
                "focus": "diabetes"
            }
        ]
        
        data_path = tmp_path / "medquad_sample.json"
        with open(data_path, "w") as f:
            json.dump(qa_data, f)
        
        return tmp_path
    
    def test_load_medquad_dataset(self, medquad_sample_data):
        """Should load MedQuAD dataset."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(medquad_sample_data),
            dataset_type="medquad"
        )
        
        assert dataset is not None
        assert dataset.count() > 0
    
    def test_medquad_has_qa_format(self, medquad_sample_data):
        """MedQuAD records should have question/answer format."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(medquad_sample_data),
            dataset_type="medquad"
        )
        
        sample = dataset.take(1)[0]
        
        assert "question" in sample or "instruction" in sample
        assert "answer" in sample or "output" in sample
    
    def test_medquad_preprocessed_for_training(self, medquad_sample_data):
        """Preprocessed MedQuAD should be ready for training."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(medquad_sample_data),
            dataset_type="medquad"
        )
        
        processed = pipeline.preprocess(dataset)
        sample = processed.take(1)[0]
        
        # Should have fields expected by training
        assert "prompt" in sample or "input_ids" in sample


# =============================================================================
# Train/Val Split Tests
# =============================================================================

class TestDatasetSplitting:
    """Test train/validation splitting."""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path) -> Path:
        """Create sample dataset with multiple records."""
        records = [
            {"question": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(100)
        ]
        
        data_path = tmp_path / "sample_data.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        return tmp_path
    
    def test_create_train_val_split(self, sample_dataset):
        """Should create train/val splits."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(sample_dataset),
            dataset_type="generic"
        )
        
        train_ds, val_ds = pipeline.create_splits(dataset, val_ratio=0.1)
        
        assert train_ds is not None
        assert val_ds is not None
    
    def test_split_ratios_are_correct(self, sample_dataset):
        """Split should respect the ratio parameter."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(sample_dataset),
            dataset_type="generic"
        )
        
        train_ds, val_ds = pipeline.create_splits(dataset, val_ratio=0.2)
        
        total = dataset.count()
        train_count = train_ds.count()
        val_count = val_ds.count()
        
        # Allow small variance due to rounding
        assert abs(val_count / total - 0.2) < 0.05
        assert train_count + val_count == total


# =============================================================================
# Batch Iterator Tests
# =============================================================================

class TestBatchIterator:
    """Test batch iterator for training."""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path) -> Path:
        """Create sample dataset."""
        records = [
            {"question": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(50)
        ]
        
        data_path = tmp_path / "batch_data.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        return tmp_path
    
    def test_get_batch_iterator(self, sample_dataset):
        """Should create batch iterator."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(sample_dataset),
            dataset_type="generic"
        )
        
        batches = pipeline.get_batch_iterator(dataset, batch_size=8)
        
        batch = next(iter(batches))
        assert batch is not None
    
    def test_batch_size_matches_config(self, sample_dataset):
        """Batch size should match configuration."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(
            str(sample_dataset),
            dataset_type="generic"
        )
        
        batch_size = 8
        batches = pipeline.get_batch_iterator(dataset, batch_size=batch_size)
        
        batch = next(iter(batches))
        # Last batch might be smaller
        assert len(batch) <= batch_size


# =============================================================================
# Integration Tests (Require Ray)
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestRayDataIntegration:
    """Integration tests requiring Ray cluster."""
    
    @pytest.fixture
    def ray_context(self):
        """Initialize Ray for testing."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(num_cpus=2, ignore_reinit_error=True)
            yield ray
            # Don't shutdown - let other tests use it
        except ImportError:
            pytest.skip("Ray not installed")
    
    def test_distributed_preprocessing(self, ray_context, tmp_path):
        """Test preprocessing with Ray Data."""
        from medai_compass.pipelines.ray_pipeline import MedicalDataPipeline
        
        # Create sample data
        records = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(100)
        ]
        data_path = tmp_path / "distributed_test.json"
        with open(data_path, "w") as f:
            json.dump(records, f)
        
        pipeline = MedicalDataPipeline(model_name="medgemma-4b")
        dataset = pipeline.load_dataset(str(tmp_path), dataset_type="generic")
        processed = pipeline.preprocess(dataset)
        
        # Should process without errors
        assert processed.count() == 100
