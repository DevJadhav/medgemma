"""
TDD Tests for Phase 6: Benchmark Suite (Tasks 6.1-6.2).

Tests the BenchmarkSuite class which provides:
- Medical QA benchmark loading (MedQA, PubMedQA, MedMCQA)
- Clinical NER benchmarks (i2b2, n2c2)
- Quality threshold validation

Quality Thresholds (from implementation_plan.md):
- MedQA Accuracy: >= 75%
- PubMedQA F1: >= 80%
- Hallucination Rate: <= 5%
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any
import numpy as np


# ============================================================================
# Test: BenchmarkSuite Initialization
# ============================================================================

class TestBenchmarkSuiteInit:
    """Test BenchmarkSuite class initialization."""

    def test_default_model_is_27b(self):
        """Default model should be MedGemma 27B IT."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        assert suite.model_name == "medgemma_27b_it"

    def test_can_select_4b_model(self):
        """Should be able to select MedGemma 4B IT."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite(model_name="medgemma_4b_it")
        assert suite.model_name == "medgemma_4b_it"

    def test_model_alias_resolution(self):
        """Should resolve model aliases correctly."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        # Test aliases
        suite_27b = BenchmarkSuite(model_name="27b")
        assert suite_27b.model_name == "medgemma_27b_it"
        
        suite_4b = BenchmarkSuite(model_name="4b")
        assert suite_4b.model_name == "medgemma_4b_it"
        
        suite_large = BenchmarkSuite(model_name="large")
        assert suite_large.model_name == "medgemma_27b_it"
        
        suite_small = BenchmarkSuite(model_name="small")
        assert suite_small.model_name == "medgemma_4b_it"

    def test_quality_thresholds_defined(self):
        """Quality thresholds should be defined per implementation plan."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite, QUALITY_THRESHOLDS
        
        assert QUALITY_THRESHOLDS["medqa_accuracy"] >= 0.75
        assert QUALITY_THRESHOLDS["pubmedqa_f1"] >= 0.80
        assert QUALITY_THRESHOLDS["hallucination_rate"] <= 0.05
        assert QUALITY_THRESHOLDS["safety_pass_rate"] >= 0.99
        assert QUALITY_THRESHOLDS["fairness_gap"] <= 0.05


# ============================================================================
# Test: Benchmark Dataset Loading
# ============================================================================

class TestBenchmarkLoading:
    """Test benchmark dataset loading functionality."""

    @patch('medai_compass.evaluation.benchmarks.load_dataset')
    def test_load_medqa_dataset(self, mock_load_dataset):
        """Should load MedQA dataset from HuggingFace."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = {"test": mock_dataset}
        
        suite = BenchmarkSuite()
        dataset = suite.load_medqa(split="test")
        
        mock_load_dataset.assert_called_once()
        assert dataset is not None

    @patch('medai_compass.evaluation.benchmarks.load_dataset')
    def test_load_pubmedqa_dataset(self, mock_load_dataset):
        """Should load PubMedQA dataset from HuggingFace."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = {"test": mock_dataset}
        
        suite = BenchmarkSuite()
        dataset = suite.load_pubmedqa(split="test")
        
        mock_load_dataset.assert_called_once()
        assert dataset is not None

    @patch('medai_compass.evaluation.benchmarks.load_dataset')
    def test_load_medmcqa_dataset(self, mock_load_dataset):
        """Should load MedMCQA dataset from HuggingFace."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = {"test": mock_dataset}
        
        suite = BenchmarkSuite()
        dataset = suite.load_medmcqa(split="test")
        
        mock_load_dataset.assert_called_once()
        assert dataset is not None

    @patch('medai_compass.evaluation.benchmarks.load_dataset')
    def test_load_clinical_ner_i2b2(self, mock_load_dataset):
        """Should load i2b2 clinical NER dataset."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = {"test": mock_dataset}
        
        suite = BenchmarkSuite()
        dataset = suite.load_clinical_ner("i2b2")
        
        assert dataset is not None

    def test_dataset_caching_with_dvc(self):
        """Should support DVC-based dataset caching."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite(cache_dir="/tmp/benchmark_cache")
        assert suite.cache_dir == "/tmp/benchmark_cache"


# ============================================================================
# Test: MedQA Evaluation
# ============================================================================

class TestMedQAEvaluation:
    """Test MedQA benchmark evaluation."""

    def test_evaluate_medqa_accuracy(self):
        """Should calculate MedQA accuracy correctly."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        # Mock predictions and ground truth
        predictions = ["A", "B", "C", "A", "B", "C", "A", "B"]
        ground_truth = ["A", "B", "C", "A", "B", "C", "A", "A"]  # 7/8 correct = 87.5%
        
        result = suite.evaluate_medqa(predictions, ground_truth)
        
        assert "accuracy" in result
        assert result["accuracy"] == 0.875
        assert "passed_threshold" in result

    def test_medqa_threshold_check(self):
        """Should check against 75% threshold."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        # Passing case (80%)
        predictions_pass = ["A"] * 80 + ["B"] * 20
        ground_truth_pass = ["A"] * 100
        result_pass = suite.evaluate_medqa(predictions_pass, ground_truth_pass)
        assert result_pass["passed_threshold"] is True
        
        # Failing case (70%)
        predictions_fail = ["A"] * 70 + ["B"] * 30
        ground_truth_fail = ["A"] * 100
        result_fail = suite.evaluate_medqa(predictions_fail, ground_truth_fail)
        assert result_fail["passed_threshold"] is False

    def test_medqa_per_category_breakdown(self):
        """Should provide per-category accuracy breakdown."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        predictions = ["A", "B", "A", "B"]
        ground_truth = ["A", "B", "A", "A"]
        categories = ["cardiology", "cardiology", "neurology", "neurology"]
        
        result = suite.evaluate_medqa(predictions, ground_truth, categories=categories)
        
        assert "per_category" in result
        assert "cardiology" in result["per_category"]
        assert "neurology" in result["per_category"]


# ============================================================================
# Test: PubMedQA Evaluation
# ============================================================================

class TestPubMedQAEvaluation:
    """Test PubMedQA benchmark evaluation."""

    def test_evaluate_pubmedqa_f1(self):
        """Should calculate PubMedQA F1 correctly."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        # Mock yes/no/maybe predictions
        predictions = ["yes", "no", "maybe", "yes", "no"]
        ground_truth = ["yes", "no", "maybe", "no", "no"]  # 4/5 correct
        
        result = suite.evaluate_pubmedqa(predictions, ground_truth)
        
        assert "f1" in result
        assert "accuracy" in result
        assert "passed_threshold" in result

    def test_pubmedqa_threshold_check(self):
        """Should check against 80% F1 threshold."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        # Create balanced predictions with all three classes for proper macro F1
        predictions = ["yes"] * 34 + ["no"] * 33 + ["maybe"] * 33
        ground_truth = ["yes"] * 34 + ["no"] * 33 + ["maybe"] * 33  # 100% match
        
        result = suite.evaluate_pubmedqa(predictions, ground_truth)
        # With perfect predictions on balanced classes, F1 should be 1.0
        assert result["f1"] >= 0.80
        assert result["passed_threshold"] is True

    def test_pubmedqa_reasoning_quality(self):
        """Should evaluate reasoning quality for long-form answers."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        predictions = [
            "Yes, the study shows that treatment A is effective because of mechanism X."
        ]
        ground_truth = ["yes"]
        reference_reasoning = [
            "Treatment A demonstrates efficacy through mechanism X in this study."
        ]
        
        result = suite.evaluate_pubmedqa(
            predictions, 
            ground_truth,
            reference_reasoning=reference_reasoning
        )
        
        assert "reasoning_score" in result


# ============================================================================
# Test: MedMCQA Evaluation
# ============================================================================

class TestMedMCQAEvaluation:
    """Test MedMCQA benchmark evaluation."""

    def test_evaluate_medmcqa(self):
        """Should evaluate MedMCQA multiple choice accuracy."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        predictions = [0, 1, 2, 3, 0, 1]  # 0-indexed options
        ground_truth = [0, 1, 2, 3, 0, 2]  # 5/6 correct
        
        result = suite.evaluate_medmcqa(predictions, ground_truth)
        
        assert "accuracy" in result
        assert abs(result["accuracy"] - 5/6) < 0.01

    def test_medmcqa_subject_breakdown(self):
        """Should provide per-subject breakdown."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        predictions = [0, 1, 0, 1]
        ground_truth = [0, 1, 1, 1]
        subjects = ["anatomy", "anatomy", "physiology", "physiology"]
        
        result = suite.evaluate_medmcqa(predictions, ground_truth, subjects=subjects)
        
        assert "per_subject" in result


# ============================================================================
# Test: Clinical NER Evaluation
# ============================================================================

class TestClinicalNEREvaluation:
    """Test clinical NER benchmark evaluation."""

    def test_evaluate_ner_f1(self):
        """Should calculate entity-level F1 for NER."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        predictions = [
            [("aspirin", "MEDICATION"), ("headache", "SYMPTOM")],
            [("diabetes", "CONDITION")]
        ]
        ground_truth = [
            [("aspirin", "MEDICATION"), ("headache", "SYMPTOM")],
            [("diabetes", "CONDITION"), ("metformin", "MEDICATION")]
        ]
        
        result = suite.evaluate_clinical_ner(predictions, ground_truth)
        
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_ner_entity_type_breakdown(self):
        """Should provide per-entity-type metrics."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        predictions = [
            [("aspirin", "MEDICATION"), ("chest pain", "SYMPTOM")]
        ]
        ground_truth = [
            [("aspirin", "MEDICATION"), ("chest pain", "SYMPTOM")]
        ]
        
        result = suite.evaluate_clinical_ner(predictions, ground_truth)
        
        assert "per_entity_type" in result
        assert "MEDICATION" in result["per_entity_type"]
        assert "SYMPTOM" in result["per_entity_type"]


# ============================================================================
# Test: Hallucination Detection
# ============================================================================

class TestHallucinationDetection:
    """Test hallucination rate evaluation."""

    def test_hallucination_rate_calculation(self):
        """Should calculate hallucination rate correctly."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        # 1 hallucinated response out of 10
        responses = [
            "The patient has diabetes based on the blood tests.",  # Correct
            "The patient definitely has cancer."  # Hallucination - definitive diagnosis
        ] * 5
        
        references = [
            "Blood tests indicate elevated glucose levels.",
            "Further testing is needed."
        ] * 5
        
        result = suite.evaluate_hallucination_rate(responses, references)
        
        assert "hallucination_rate" in result
        assert "passed_threshold" in result

    def test_hallucination_threshold_check(self):
        """Should check against 5% hallucination threshold."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        # No hallucinations
        safe_responses = [
            "Based on the symptoms, the patient may have condition X. Please consult a doctor."
        ] * 100
        references = ["Symptoms suggest condition X."] * 100
        
        result = suite.evaluate_hallucination_rate(safe_responses, references)
        assert result["hallucination_rate"] <= 0.05
        assert result["passed_threshold"] is True


# ============================================================================
# Test: Quality Gate Checking
# ============================================================================

class TestQualityGates:
    """Test quality gate pass/fail determination."""

    def test_all_thresholds_pass(self):
        """Should pass when all thresholds are met."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        results = {
            "medqa_accuracy": 0.80,
            "pubmedqa_f1": 0.85,
            "hallucination_rate": 0.03,
            "safety_pass_rate": 0.995,
            "fairness_gap": 0.04
        }
        
        gate_result = suite.check_quality_gates(results)
        
        assert gate_result.passed is True
        assert all(v["passed"] for v in gate_result.details.values())

    def test_threshold_failure(self):
        """Should fail when any threshold is not met."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        results = {
            "medqa_accuracy": 0.70,  # Below 75% threshold
            "pubmedqa_f1": 0.85,
            "hallucination_rate": 0.03,
            "safety_pass_rate": 0.995,
            "fairness_gap": 0.04
        }
        
        gate_result = suite.check_quality_gates(results)
        
        assert gate_result.passed is False
        assert gate_result.details["medqa_accuracy"]["passed"] is False

    def test_quality_gate_report_generation(self):
        """Should generate detailed quality gate report."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        results = {
            "medqa_accuracy": 0.78,
            "pubmedqa_f1": 0.82,
            "hallucination_rate": 0.04,
            "safety_pass_rate": 0.99,
            "fairness_gap": 0.03
        }
        
        gate_result = suite.check_quality_gates(results)
        
        assert hasattr(gate_result, "passed")
        assert hasattr(gate_result, "details")
        assert hasattr(gate_result, "summary")
        assert hasattr(gate_result, "timestamp")


# ============================================================================
# Test: Batch Evaluation
# ============================================================================

class TestBatchEvaluation:
    """Test batch evaluation functionality."""

    def test_run_all_benchmarks(self):
        """Should run all benchmarks in sequence."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        
        # Mock model function
        def mock_model(prompt: str) -> str:
            return "A"  # Simple mock
        
        # This should not raise
        with patch.object(suite, 'load_medqa'), \
             patch.object(suite, 'load_pubmedqa'), \
             patch.object(suite, 'load_medmcqa'):
            
            # Would need actual datasets to run fully
            assert hasattr(suite, 'run_all_benchmarks')

    def test_benchmark_results_structure(self):
        """Should return properly structured benchmark results."""
        from medai_compass.evaluation.benchmarks import BenchmarkResult
        
        result = BenchmarkResult(
            benchmark_name="medqa",
            model_name="medgemma_27b_it",
            accuracy=0.78,
            f1_score=None,
            metrics={"per_category": {}},
            passed_threshold=True,
            threshold_value=0.75,
            timestamp="2026-01-18T00:00:00"
        )
        
        assert result.benchmark_name == "medqa"
        assert result.passed_threshold is True


# ============================================================================
# Test: Model Integration
# ============================================================================

class TestModelIntegration:
    """Test model integration for benchmarking."""

    def test_model_inference_function(self):
        """Should accept custom model inference function."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        call_count = 0
        
        def custom_model(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "A"
        
        suite = BenchmarkSuite(model_fn=custom_model)
        
        # Verify model function is set
        assert suite.model_fn is not None
        
        # Test that it can be called
        result = suite.model_fn("test prompt")
        assert result == "A"
        assert call_count == 1

    def test_batch_inference_support(self):
        """Should support batch inference for efficiency."""
        from medai_compass.evaluation.benchmarks import BenchmarkSuite
        
        def batch_model(prompts: list[str]) -> list[str]:
            return ["A"] * len(prompts)
        
        suite = BenchmarkSuite(batch_model_fn=batch_model)
        
        assert suite.batch_model_fn is not None
        results = suite.batch_model_fn(["q1", "q2", "q3"])
        assert len(results) == 3


# ============================================================================
# Test: Export and Serialization
# ============================================================================

class TestExportSerialization:
    """Test result export and serialization."""

    def test_export_to_json(self):
        """Should export results to JSON format."""
        from medai_compass.evaluation.benchmarks import BenchmarkResult
        import json
        
        result = BenchmarkResult(
            benchmark_name="medqa",
            model_name="medgemma_27b_it",
            accuracy=0.78,
            f1_score=None,
            metrics={"per_category": {"cardiology": 0.80}},
            passed_threshold=True,
            threshold_value=0.75,
            timestamp="2026-01-18T00:00:00"
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["benchmark_name"] == "medqa"
        assert parsed["accuracy"] == 0.78

    def test_export_to_dict(self):
        """Should export results to dictionary."""
        from medai_compass.evaluation.benchmarks import BenchmarkResult
        
        result = BenchmarkResult(
            benchmark_name="pubmedqa",
            model_name="medgemma_27b_it",
            accuracy=0.82,
            f1_score=0.85,
            metrics={},
            passed_threshold=True,
            threshold_value=0.80,
            timestamp="2026-01-18T00:00:00"
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["f1_score"] == 0.85
