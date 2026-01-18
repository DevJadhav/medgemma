"""Tests for Phase 5: Synthetic Data Pipeline.

Tests synthetic data generation for medical AI training using
MedGemma 27B IT as the primary generator with TDD approach.

Covers Tasks 5.1-5.7:
- 5.1: Clinical text generator
- 5.2: Patient dialogue generator
- 5.3: Medical image synthesizer
- 5.4: Synthea integration / structured data
- 5.5: Data quality assurance
- 5.6: Synthetic data versioning (DVC)
- 5.7: Diversity analysis tools
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# =============================================================================
# Task 5.1: Clinical Text Generator Tests
# =============================================================================

class TestClinicalTextGenerator:
    """Tests for clinical text generation using MedGemma 27B IT."""
    
    def test_generator_initialization(self):
        """Test ClinicalTextGenerator initializes with MedGemma 27B IT."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator(
            model_name="google/medgemma-27b-text-it",
            device="cpu",
        )
        
        assert generator.model_name == "google/medgemma-27b-text-it"
        assert generator.device == "cpu"
    
    def test_generator_default_model_is_27b(self):
        """Test that default model is MedGemma 27B IT."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator()
        
        assert "27b" in generator.model_name.lower()
    
    def test_generate_clinical_note(self):
        """Test generating a single clinical note."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator(mock_mode=True)
        
        note = generator.generate_clinical_note(
            specialty="cardiology",
            note_type="progress_note",
        )
        
        assert isinstance(note, dict)
        assert "text" in note
        assert "specialty" in note
        assert "note_type" in note
        assert note["specialty"] == "cardiology"
    
    def test_generate_radiology_report(self):
        """Test generating a radiology report."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator(mock_mode=True)
        
        report = generator.generate_radiology_report(
            modality="chest_xray",
            findings_type="normal",
        )
        
        assert isinstance(report, dict)
        assert "text" in report
        assert "modality" in report
        assert report["modality"] == "chest_xray"
    
    def test_batch_generation_with_progress(self):
        """Test batch generation with tqdm progress tracking."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator(
            mock_mode=True,
            target_count=100,
            batch_size=25,
        )
        
        # Generate with progress tracking
        results = generator.generate_batch(
            specialty="oncology",
            note_type="consultation",
            count=100,
            show_progress=True,
        )
        
        assert len(results) == 100
        assert all("text" in r for r in results)
    
    def test_checkpoint_save_and_resume(self):
        """Test checkpoint save/resume functionality."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            
            generator = ClinicalTextGenerator(
                mock_mode=True,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_interval=10,
            )
            
            # Generate partial batch
            results = generator.generate_batch(
                specialty="neurology",
                count=25,
                save_checkpoints=True,
            )
            
            assert len(results) == 25
            
            # Verify checkpoint was saved
            checkpoint_files = list(checkpoint_dir.glob("*.json"))
            assert len(checkpoint_files) >= 1
    
    def test_specialty_templates_available(self):
        """Test that specialty templates are available."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator(mock_mode=True)
        
        specialties = generator.list_specialties()
        
        assert "cardiology" in specialties
        assert "oncology" in specialties
        assert "radiology" in specialties
        assert "neurology" in specialties
        assert "emergency" in specialties
    
    def test_note_types_available(self):
        """Test that note types are available."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator(mock_mode=True)
        
        note_types = generator.list_note_types()
        
        assert "progress_note" in note_types
        assert "discharge_summary" in note_types
        assert "consultation" in note_types
        assert "operative_note" in note_types


# =============================================================================
# Task 5.2: Patient Dialogue Generator Tests
# =============================================================================

class TestDialogueGenerator:
    """Tests for patient-provider dialogue generation."""
    
    def test_dialogue_generator_initialization(self):
        """Test DialogueGenerator initializes correctly."""
        from medai_compass.synthetic.dialogue_generator import DialogueGenerator
        
        generator = DialogueGenerator(
            model_name="google/medgemma-27b-text-it",
            mock_mode=True,
        )
        
        assert "27b" in generator.model_name.lower()
    
    def test_generate_single_turn_dialogue(self):
        """Test generating a single dialogue turn."""
        from medai_compass.synthetic.dialogue_generator import DialogueGenerator
        
        generator = DialogueGenerator(mock_mode=True)
        
        turn = generator.generate_turn(
            context="Patient presents with chest pain",
            speaker="patient",
        )
        
        assert isinstance(turn, dict)
        assert "speaker" in turn
        assert "utterance" in turn
    
    def test_generate_multi_turn_conversation(self):
        """Test generating multi-turn conversation."""
        from medai_compass.synthetic.dialogue_generator import DialogueGenerator
        
        generator = DialogueGenerator(mock_mode=True)
        
        conversation = generator.generate_conversation(
            scenario="initial_consultation",
            num_turns=6,
        )
        
        assert isinstance(conversation, dict)
        assert "turns" in conversation
        assert len(conversation["turns"]) == 6
        assert "scenario" in conversation
    
    def test_conversation_with_context(self):
        """Test conversation with medical context."""
        from medai_compass.synthetic.dialogue_generator import DialogueGenerator
        
        generator = DialogueGenerator(mock_mode=True)
        
        conversation = generator.generate_conversation(
            scenario="symptom_assessment",
            patient_context={
                "age": 45,
                "gender": "female",
                "chief_complaint": "persistent headache",
            },
            num_turns=4,
        )
        
        assert len(conversation["turns"]) == 4
        assert "patient_context" in conversation
    
    def test_batch_dialogue_generation(self):
        """Test batch generation of dialogues."""
        from medai_compass.synthetic.dialogue_generator import DialogueGenerator
        
        generator = DialogueGenerator(
            mock_mode=True,
            batch_size=10,
        )
        
        dialogues = generator.generate_batch(
            scenario="follow_up_visit",
            count=50,
            show_progress=True,
        )
        
        assert len(dialogues) == 50
        assert all("turns" in d for d in dialogues)
    
    def test_scenario_types_available(self):
        """Test that dialogue scenarios are available."""
        from medai_compass.synthetic.dialogue_generator import DialogueGenerator
        
        generator = DialogueGenerator(mock_mode=True)
        
        scenarios = generator.list_scenarios()
        
        assert "initial_consultation" in scenarios
        assert "symptom_assessment" in scenarios
        assert "follow_up_visit" in scenarios
        assert "medication_discussion" in scenarios
        assert "discharge_instructions" in scenarios


# =============================================================================
# Task 5.3: Medical Image Synthesizer Tests
# =============================================================================

class TestImageSynthesizer:
    """Tests for medical image synthesis with diffusion models."""
    
    def test_image_synthesizer_initialization(self):
        """Test ImageSynthesizer initializes correctly."""
        from medai_compass.synthetic.image_generator import ImageSynthesizer
        
        synthesizer = ImageSynthesizer(
            base_model="stabilityai/stable-diffusion-2-1",
            device="cpu",
        )
        
        assert synthesizer.base_model == "stabilityai/stable-diffusion-2-1"
    
    def test_available_medical_datasets(self):
        """Test listing available open-source medical imaging datasets."""
        from medai_compass.synthetic.image_generator import ImageSynthesizer
        
        synthesizer = ImageSynthesizer(mock_mode=True)
        
        datasets = synthesizer.list_available_datasets()
        
        # Prioritize open-source datasets
        assert "chexpert" in datasets  # Stanford - open access
        assert "mimic_cxr" in datasets  # PhysioNet - open access
        assert "nih_chestxray" in datasets  # NIH - public domain
        assert "padchest" in datasets  # Open access
    
    def test_generate_chest_xray(self):
        """Test generating synthetic chest X-ray."""
        from medai_compass.synthetic.image_generator import ImageSynthesizer
        
        synthesizer = ImageSynthesizer(mock_mode=True)
        
        result = synthesizer.generate_image(
            modality="chest_xray",
            condition="normal",
        )
        
        assert isinstance(result, dict)
        assert "image" in result or "image_path" in result
        assert "modality" in result
        assert "condition" in result
    
    def test_generate_with_condition(self):
        """Test generating image with specific condition."""
        from medai_compass.synthetic.image_generator import ImageSynthesizer
        
        synthesizer = ImageSynthesizer(mock_mode=True)
        
        result = synthesizer.generate_image(
            modality="chest_xray",
            condition="pneumonia",
            severity="moderate",
        )
        
        assert result["condition"] == "pneumonia"
        assert "severity" in result
    
    def test_batch_image_generation(self):
        """Test batch generation of synthetic images."""
        from medai_compass.synthetic.image_generator import ImageSynthesizer
        
        synthesizer = ImageSynthesizer(
            mock_mode=True,
            batch_size=10,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = synthesizer.generate_batch(
                modality="chest_xray",
                conditions=["normal", "pneumonia", "cardiomegaly"],
                count=30,
                output_dir=tmpdir,
                show_progress=True,
            )
            
            assert len(results) == 30
    
    def test_training_pipeline_config(self):
        """Test training pipeline configuration for fine-tuning."""
        from medai_compass.synthetic.image_generator import ImageTrainingPipeline
        
        pipeline = ImageTrainingPipeline(
            base_model="stabilityai/stable-diffusion-2-1",
            dataset_name="chexpert",
            output_dir="/tmp/model_output",
        )
        
        config = pipeline.get_training_config()
        
        assert "learning_rate" in config
        assert "num_epochs" in config
        assert "batch_size" in config
        assert "gradient_accumulation_steps" in config
    
    def test_training_pipeline_with_accelerate(self):
        """Test training pipeline uses accelerate for distributed training."""
        from medai_compass.synthetic.image_generator import ImageTrainingPipeline
        
        pipeline = ImageTrainingPipeline(
            base_model="stabilityai/stable-diffusion-2-1",
            dataset_name="nih_chestxray",
            use_accelerate=True,
        )
        
        assert pipeline.use_accelerate is True
        assert "accelerate" in pipeline.get_dependencies()
    
    def test_checkpoint_during_training(self):
        """Test checkpoint saving during training."""
        from medai_compass.synthetic.image_generator import ImageTrainingPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ImageTrainingPipeline(
                base_model="stabilityai/stable-diffusion-2-1",
                dataset_name="chexpert",
                output_dir=tmpdir,
                checkpoint_interval=100,
                mock_mode=True,
            )
            
            # Simulate training step with checkpoint
            pipeline.save_checkpoint(step=100)
            
            checkpoint_path = Path(tmpdir) / "checkpoints" / "checkpoint-100"
            assert pipeline.checkpoint_exists(step=100)
    
    def test_resume_training_from_checkpoint(self):
        """Test resuming training from checkpoint."""
        from medai_compass.synthetic.image_generator import ImageTrainingPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ImageTrainingPipeline(
                base_model="stabilityai/stable-diffusion-2-1",
                dataset_name="chexpert",
                output_dir=tmpdir,
                mock_mode=True,
            )
            
            # Save checkpoint
            pipeline.save_checkpoint(step=100)
            
            # Resume from checkpoint
            resumed_step = pipeline.resume_from_checkpoint()
            
            assert resumed_step == 100


# =============================================================================
# Task 5.4: Structured Data Generator (Synthea Integration) Tests
# =============================================================================

class TestStructuredDataGenerator:
    """Tests for structured data generation with Synthea and CTGAN."""
    
    def test_synthea_integration(self):
        """Test Synthea integration for EHR data generation."""
        from medai_compass.synthetic.structured_generator import StructuredDataGenerator
        
        generator = StructuredDataGenerator(mock_mode=True)
        
        # Generate FHIR resources
        fhir_data = generator.generate_fhir_bundle(
            patient_count=10,
            conditions=["diabetes", "hypertension"],
        )
        
        assert isinstance(fhir_data, dict)
        assert fhir_data.get("resourceType") == "Bundle"
        assert len(fhir_data.get("entry", [])) >= 10
    
    def test_ctgan_tabular_generation(self):
        """Test CTGAN for tabular data augmentation."""
        from medai_compass.synthetic.structured_generator import StructuredDataGenerator
        
        generator = StructuredDataGenerator(mock_mode=True)
        
        # Generate tabular data
        tabular_data = generator.generate_tabular(
            schema="patient_demographics",
            count=100,
        )
        
        assert len(tabular_data) == 100
        assert all("age" in row for row in tabular_data)
        assert all("gender" in row for row in tabular_data)
    
    def test_rare_condition_augmentation(self):
        """Test augmentation for rare medical conditions."""
        from medai_compass.synthetic.structured_generator import StructuredDataGenerator
        
        generator = StructuredDataGenerator(mock_mode=True)
        
        # Generate data for rare conditions
        rare_data = generator.augment_rare_conditions(
            conditions=["kawasaki_disease", "huntington_disease"],
            samples_per_condition=50,
        )
        
        assert len(rare_data) == 100  # 50 * 2 conditions
    
    def test_batch_fhir_generation(self):
        """Test batch FHIR resource generation."""
        from medai_compass.synthetic.structured_generator import StructuredDataGenerator
        
        generator = StructuredDataGenerator(
            mock_mode=True,
            batch_size=50,
        )
        
        bundles = generator.generate_fhir_batch(
            patient_count=500,
            show_progress=True,
        )
        
        total_patients = sum(
            len([e for e in b.get("entry", []) if e.get("resource", {}).get("resourceType") == "Patient"])
            for b in bundles
        )
        assert total_patients >= 500
    
    def test_synthea_available_modules(self):
        """Test listing available Synthea modules."""
        from medai_compass.synthetic.structured_generator import StructuredDataGenerator
        
        generator = StructuredDataGenerator(mock_mode=True)
        
        modules = generator.list_synthea_modules()
        
        assert "diabetes" in modules
        assert "hypertension" in modules
        assert "cancer" in modules
        assert "covid19" in modules


# =============================================================================
# Task 5.5: Data Quality Assurance Tests
# =============================================================================

class TestSyntheticDataQA:
    """Tests for synthetic data quality assurance."""
    
    def test_phi_detection_in_synthetic_data(self):
        """Test PHI detection in synthetic data."""
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        
        qa = SyntheticDataQA()
        
        # Test with clean data
        clean_text = "Patient presents with chest pain and shortness of breath."
        result = qa.check_phi(clean_text)
        
        assert result["contains_phi"] is False
        
        # Test with PHI
        phi_text = "John Smith (SSN: 123-45-6789) was admitted on 01/15/2024."
        result = qa.check_phi(phi_text)
        
        assert result["contains_phi"] is True
        assert len(result["phi_types"]) > 0
    
    def test_medical_terminology_validation(self):
        """Test medical terminology validation."""
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        
        qa = SyntheticDataQA()
        
        # Valid medical text
        valid_text = "Patient diagnosed with acute myocardial infarction. Started on aspirin and beta-blockers."
        result = qa.validate_terminology(valid_text)
        
        assert result["is_valid"] is True
        assert len(result["medical_terms_found"]) > 0
    
    def test_clinical_consistency_check(self):
        """Test clinical consistency validation."""
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        
        qa = SyntheticDataQA()
        
        # Check for clinical inconsistencies
        record = {
            "diagnosis": "pregnancy",
            "patient": {"gender": "male", "age": 45},
        }
        
        result = qa.check_clinical_consistency(record)
        
        assert result["is_consistent"] is False
        assert len(result["inconsistencies"]) > 0
    
    def test_quality_score_calculation(self):
        """Test overall quality score calculation."""
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        
        qa = SyntheticDataQA()
        
        # Generate quality score for synthetic record
        record = {
            "text": "Patient presents with hypertension. Blood pressure 150/95.",
            "specialty": "cardiology",
            "patient": {"age": 55, "gender": "male"},
        }
        
        score = qa.calculate_quality_score(record)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_batch_quality_validation(self):
        """Test batch quality validation."""
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        
        qa = SyntheticDataQA()
        
        records = [
            {"text": "Normal chest X-ray.", "modality": "chest_xray"},
            {"text": "Evidence of pneumonia.", "modality": "chest_xray"},
        ] * 50
        
        report = qa.validate_batch(records, show_progress=True)
        
        assert "total_records" in report
        assert "passed_records" in report
        assert "failed_records" in report
        assert "average_quality_score" in report
        assert report["total_records"] == 100
    
    def test_qa_report_generation(self):
        """Test QA report generation."""
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        
        qa = SyntheticDataQA()
        
        records = [
            {"text": "Patient has diabetes mellitus.", "patient": {"age": 60}},
        ] * 10
        
        report = qa.generate_report(records)
        
        assert "summary" in report
        assert "phi_check_results" in report
        assert "quality_scores" in report
        assert "recommendations" in report


# =============================================================================
# Task 5.6: Synthetic Data Versioning Tests
# =============================================================================

class TestSyntheticDataVersioning:
    """Tests for synthetic data versioning with DVC integration."""
    
    def test_version_synthetic_dataset(self):
        """Test versioning a synthetic dataset with DVC."""
        from medai_compass.synthetic.versioning import SyntheticDataVersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SyntheticDataVersionManager(
                repo_path=tmpdir,
                mock_dvc=True,
            )
            
            # Create synthetic data file
            data_path = Path(tmpdir) / "synthetic_data"
            data_path.mkdir()
            (data_path / "data.json").write_text('{"records": []}')
            
            # Version the dataset
            version = manager.version_dataset(
                data_path=str(data_path),
                name="synthetic_v1",
                metadata={"generator": "medgemma-27b", "count": 1000},
            )
            
            assert version is not None
            assert version.name == "synthetic_v1"
            assert "generator" in version.metadata
    
    def test_checkpoint_versioning(self):
        """Test versioning generation checkpoints with DVC."""
        from medai_compass.synthetic.versioning import SyntheticDataVersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SyntheticDataVersionManager(
                repo_path=tmpdir,
                mock_dvc=True,
            )
            
            # Create checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint_100.json"
            checkpoint_path.write_text('{"step": 100, "records": []}')
            
            # Version checkpoint
            version = manager.version_checkpoint(
                checkpoint_path=str(checkpoint_path),
                step=100,
                generator_name="clinical_text",
            )
            
            assert version is not None
            assert "step" in version.metadata
            assert version.metadata["step"] == 100
    
    def test_list_versioned_datasets(self):
        """Test listing versioned datasets."""
        from medai_compass.synthetic.versioning import SyntheticDataVersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SyntheticDataVersionManager(
                repo_path=tmpdir,
                mock_dvc=True,
            )
            
            # Create and version multiple datasets
            for i in range(3):
                data_path = Path(tmpdir) / f"data_{i}"
                data_path.mkdir()
                (data_path / "data.json").write_text(f'{{"id": {i}}}')
                manager.version_dataset(str(data_path), f"dataset_v{i}")
            
            versions = manager.list_versions()
            
            assert len(versions) == 3
    
    def test_retrieve_version_by_hash(self):
        """Test retrieving dataset version by hash."""
        from medai_compass.synthetic.versioning import SyntheticDataVersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SyntheticDataVersionManager(
                repo_path=tmpdir,
                mock_dvc=True,
            )
            
            # Version a dataset
            data_path = Path(tmpdir) / "data"
            data_path.mkdir()
            (data_path / "data.json").write_text('{"test": true}')
            
            version = manager.version_dataset(str(data_path), "test_dataset")
            
            # Retrieve by hash
            retrieved = manager.get_version_by_hash(version.hash[:8])
            
            assert retrieved is not None
            assert retrieved.name == "test_dataset"
    
    def test_dvc_push_synthetic_data(self):
        """Test pushing synthetic data to DVC remote."""
        from medai_compass.synthetic.versioning import SyntheticDataVersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SyntheticDataVersionManager(
                repo_path=tmpdir,
                remote_url="s3://test-bucket/synthetic",
                mock_dvc=True,
            )
            
            # Version data
            data_path = Path(tmpdir) / "data"
            data_path.mkdir()
            (data_path / "data.json").write_text('{"records": [1,2,3]}')
            
            manager.version_dataset(str(data_path), "push_test")
            
            # Push to remote (mocked)
            result = manager.push("push_test")
            
            assert result is True
    
    def test_generation_metadata_tracking(self):
        """Test tracking generation metadata in versions."""
        from medai_compass.synthetic.versioning import SyntheticDataVersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SyntheticDataVersionManager(
                repo_path=tmpdir,
                mock_dvc=True,
            )
            
            data_path = Path(tmpdir) / "generated"
            data_path.mkdir()
            (data_path / "data.json").write_text('{}')
            
            version = manager.version_dataset(
                str(data_path),
                "tracked_generation",
                metadata={
                    "model": "google/medgemma-27b-text-it",
                    "target_count": 2500,
                    "batch_size": 50,
                    "checkpoint_interval": 100,
                    "generation_params": {
                        "specialty": "cardiology",
                        "temperature": 0.7,
                    },
                },
            )
            
            assert version.metadata["model"] == "google/medgemma-27b-text-it"
            assert version.metadata["target_count"] == 2500


# =============================================================================
# Task 5.7: Diversity Analysis Tests
# =============================================================================

class TestDiversityAnalysis:
    """Tests for synthetic data diversity analysis."""
    
    def test_demographic_distribution_analysis(self):
        """Test analyzing demographic distribution."""
        from medai_compass.synthetic.diversity import DiversityAnalyzer
        
        analyzer = DiversityAnalyzer()
        
        # Sample data with demographics
        records = [
            {"patient": {"age": 25, "gender": "male", "ethnicity": "asian"}},
            {"patient": {"age": 45, "gender": "female", "ethnicity": "white"}},
            {"patient": {"age": 65, "gender": "male", "ethnicity": "black"}},
        ] * 100
        
        distribution = analyzer.analyze_demographics(records)
        
        assert "age_distribution" in distribution
        assert "gender_distribution" in distribution
        assert "ethnicity_distribution" in distribution
    
    def test_condition_distribution_analysis(self):
        """Test analyzing medical condition distribution."""
        from medai_compass.synthetic.diversity import DiversityAnalyzer
        
        analyzer = DiversityAnalyzer()
        
        records = [
            {"condition": "diabetes"},
            {"condition": "hypertension"},
            {"condition": "diabetes"},
            {"condition": "asthma"},
        ] * 50
        
        distribution = analyzer.analyze_conditions(records)
        
        assert "diabetes" in distribution
        assert distribution["diabetes"] == 100  # 50 * 2
    
    def test_identify_coverage_gaps(self):
        """Test identifying coverage gaps in synthetic data."""
        from medai_compass.synthetic.diversity import DiversityAnalyzer
        
        analyzer = DiversityAnalyzer()
        
        # Data with gaps
        records = [
            {"patient": {"age": 30, "gender": "male"}},
            {"patient": {"age": 35, "gender": "male"}},
        ] * 100
        
        gaps = analyzer.identify_gaps(
            records,
            target_distribution={
                "gender": {"male": 0.5, "female": 0.5},
                "age_groups": {"18-30": 0.25, "31-50": 0.25, "51-70": 0.25, "70+": 0.25},
            },
        )
        
        assert "gender" in gaps
        assert "female" in gaps["gender"]["missing"]
    
    def test_diversity_score_calculation(self):
        """Test calculating diversity score."""
        from medai_compass.synthetic.diversity import DiversityAnalyzer
        
        analyzer = DiversityAnalyzer()
        
        # Well-distributed data
        diverse_records = [
            {"patient": {"age": age, "gender": gender}}
            for age in [25, 35, 45, 55, 65, 75]
            for gender in ["male", "female"]
        ] * 10
        
        score = analyzer.calculate_diversity_score(diverse_records)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably diverse
    
    def test_generate_diversity_report(self):
        """Test generating comprehensive diversity report."""
        from medai_compass.synthetic.diversity import DiversityAnalyzer
        
        analyzer = DiversityAnalyzer()
        
        records = [
            {
                "patient": {"age": 45, "gender": "male"},
                "condition": "diabetes",
                "specialty": "endocrinology",
            },
        ] * 100
        
        report = analyzer.generate_report(records)
        
        assert "demographics" in report
        assert "conditions" in report
        assert "diversity_score" in report
        assert "recommendations" in report
    
    def test_specialty_coverage_analysis(self):
        """Test analyzing specialty coverage."""
        from medai_compass.synthetic.diversity import DiversityAnalyzer
        
        analyzer = DiversityAnalyzer()
        
        records = [
            {"specialty": "cardiology"},
            {"specialty": "oncology"},
            {"specialty": "cardiology"},
            {"specialty": "neurology"},
        ] * 50
        
        coverage = analyzer.analyze_specialty_coverage(records)
        
        assert "cardiology" in coverage
        assert coverage["cardiology"] == 100  # 50 * 2


# =============================================================================
# Base Generator Tests (Shared Functionality)
# =============================================================================

class _ConcreteGenerator:
    """Concrete generator for testing base functionality."""
    
    def __init__(self, **kwargs):
        from medai_compass.synthetic.base import BaseSyntheticGenerator
        
        # Create a local concrete class
        class ConcreteGenerator(BaseSyntheticGenerator):
            def generate_single(self, **kwargs):
                return {"id": "test", "data": "mock"}
        
        self._generator = ConcreteGenerator(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self._generator, name)


class TestBaseSyntheticGenerator:
    """Tests for base synthetic generator functionality."""
    
    def test_base_generator_config(self):
        """Test base generator configuration."""
        generator = _ConcreteGenerator(
            target_count=2500,
            batch_size=50,
            checkpoint_interval=100,
        )
        
        assert generator.target_count == 2500
        assert generator.batch_size == 50
        assert generator.checkpoint_interval == 100
    
    def test_progress_tracking_with_tqdm(self):
        """Test progress tracking uses tqdm."""
        generator = _ConcreteGenerator(
            target_count=100,
            batch_size=10,
            mock_mode=True,
        )
        
        # Progress tracker should be available
        assert hasattr(generator, "create_progress_bar")
    
    def test_checkpoint_serialization(self):
        """Test checkpoint JSON serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = _ConcreteGenerator(
                checkpoint_dir=tmpdir,
                checkpoint_interval=10,
                mock_mode=True,
            )
            
            # Save checkpoint
            checkpoint_data = {
                "current_count": 50,
                "records": [{"id": i} for i in range(50)],
                "metadata": {"generator": "test"},
            }
            
            generator.save_checkpoint(checkpoint_data, step=50)
            
            # Load checkpoint
            loaded = generator.load_checkpoint(step=50)
            
            assert loaded["current_count"] == 50
            assert len(loaded["records"]) == 50
    
    def test_resume_from_latest_checkpoint(self):
        """Test resuming from latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = _ConcreteGenerator(
                checkpoint_dir=tmpdir,
                checkpoint_interval=100,
                mock_mode=True,
            )
            
            # Save multiple checkpoints
            for step in [100, 200, 300]:
                generator.save_checkpoint({"step": step}, step=step)
            
            # Resume from latest
            latest = generator.find_latest_checkpoint()
            
            assert latest == 300
    
    def test_dvc_checkpoint_integration(self):
        """Test DVC integration for checkpoint versioning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = _ConcreteGenerator(
                checkpoint_dir=tmpdir,
                use_dvc=True,
                mock_dvc=True,
            )
            
            # Save checkpoint with DVC tracking
            generator.save_checkpoint({"step": 100}, step=100, track_with_dvc=True)
            
            # Verify DVC tracking was called
            assert generator.dvc_tracked_checkpoints >= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestSyntheticPipelineIntegration:
    """Integration tests for the complete synthetic data pipeline."""
    
    def test_end_to_end_text_generation_pipeline(self):
        """Test end-to-end text generation with QA and versioning."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        from medai_compass.synthetic.versioning import SyntheticDataVersionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate synthetic data
            generator = ClinicalTextGenerator(
                mock_mode=True,
                checkpoint_dir=f"{tmpdir}/checkpoints",
            )
            
            records = generator.generate_batch(
                specialty="cardiology",
                count=100,
                save_checkpoints=True,
            )
            
            # Run QA
            qa = SyntheticDataQA()
            report = qa.validate_batch(records)
            
            # Version the data
            manager = SyntheticDataVersionManager(
                repo_path=tmpdir,
                mock_dvc=True,
            )
            
            # Save records
            data_path = Path(tmpdir) / "synthetic_clinical_notes"
            data_path.mkdir()
            (data_path / "records.json").write_text(json.dumps(records))
            
            version = manager.version_dataset(
                str(data_path),
                "clinical_notes_v1",
                metadata={
                    "qa_report": report,
                    "generator": "medgemma-27b",
                },
            )
            
            assert len(records) == 100
            assert report["total_records"] == 100
            assert version is not None
    
    def test_full_pipeline_with_diversity_check(self):
        """Test full pipeline including diversity analysis."""
        from medai_compass.synthetic.structured_generator import StructuredDataGenerator
        from medai_compass.synthetic.quality_assurance import SyntheticDataQA
        from medai_compass.synthetic.diversity import DiversityAnalyzer
        
        # Generate structured data
        generator = StructuredDataGenerator(mock_mode=True)
        
        records = generator.generate_tabular(
            schema="patient_demographics",
            count=200,
        )
        
        # QA check
        qa = SyntheticDataQA()
        qa_report = qa.validate_batch(
            [{"text": str(r), "patient": r} for r in records]
        )
        
        # Diversity analysis
        analyzer = DiversityAnalyzer()
        diversity_report = analyzer.generate_report(
            [{"patient": r} for r in records]
        )
        
        assert len(records) == 200
        assert qa_report["passed_records"] > 0
        assert "diversity_score" in diversity_report


# =============================================================================
# Performance and Scale Tests
# =============================================================================

class TestSyntheticGenerationScale:
    """Tests for synthetic generation at scale (2-3K samples)."""
    
    def test_target_volume_2500_samples(self):
        """Test generating target volume of 2500 samples."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        generator = ClinicalTextGenerator(
            mock_mode=True,
            target_count=2500,
            batch_size=50,
            checkpoint_interval=100,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generator.generate_batch(
                specialty="oncology",
                count=2500,
                checkpoint_dir=tmpdir,
                save_checkpoints=True,
                show_progress=True,
            )
            
            assert len(results) == 2500
            
            # Verify checkpoints were created
            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.json"))
            expected_checkpoints = 2500 // 100  # 25 checkpoints
            assert len(checkpoint_files) >= expected_checkpoints
    
    def test_checkpoint_resume_at_scale(self):
        """Test checkpoint resume functionality at scale."""
        from medai_compass.synthetic.text_generator import ClinicalTextGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ClinicalTextGenerator(
                mock_mode=True,
                checkpoint_dir=tmpdir,
                checkpoint_interval=100,
            )
            
            # Simulate partial generation (stopped at 1500)
            partial_data = {"step": 1500, "records": [{"id": i} for i in range(1500)]}
            generator.save_checkpoint(partial_data, step=1500)
            
            # Resume generation
            results = generator.resume_and_complete(
                target_count=2500,
                specialty="radiology",
            )
            
            assert len(results) == 2500
    
    def test_memory_efficient_batch_processing(self):
        """Test memory-efficient batch processing."""
        generator = _ConcreteGenerator(
            target_count=3000,
            batch_size=50,
            mock_mode=True,
        )
        
        # Process in batches
        batches_processed = 0
        for batch in generator.iter_batches():
            batches_processed += 1
            assert len(batch) <= 50
        
        expected_batches = 3000 // 50  # 60 batches
        assert batches_processed == expected_batches
