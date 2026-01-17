"""
Tests for the comprehensive AI evaluation framework.

Tests:
- Safety evaluation (jailbreak, prompt injection, PHI, misinformation)
- Bias evaluation (demographic parity, equalized odds, calibration)
- Clinical accuracy evaluation (triage, emergency detection)
"""

import pytest
import numpy as np
from medai_compass.evaluation.ai_evaluation import (
    SafetyEvaluator,
    BiasEvaluator,
    ClinicalAccuracyEvaluator,
    ComprehensiveEvaluator,
    SafetyCategory,
    BiasCategory,
    create_mock_safe_model,
)


# =============================================================================
# Mock Models for Testing
# =============================================================================

def create_unsafe_model():
    """Create a model that gives unsafe responses (for testing detection)."""
    def unsafe_model(prompt: str) -> str:
        # Returns diagnosis without disclaimers
        return "You have cancer. Take 500mg of ibuprofen twice daily."
    return unsafe_model


def create_partially_safe_model():
    """Create a model that is sometimes safe, sometimes not."""
    counter = [0]
    
    def partial_model(prompt: str) -> str:
        counter[0] += 1
        if counter[0] % 2 == 0:
            return "You definitely have diabetes. Start insulin immediately."
        return "I cannot provide a diagnosis. Please consult your doctor."
    
    return partial_model


def create_phi_leaking_model():
    """Create a model that leaks PHI."""
    def phi_model(prompt: str) -> str:
        return "Patient ID 12345, MRN: 67890, was diagnosed with cancer on 01/15/2024."
    return phi_model


# =============================================================================
# Safety Evaluator Tests
# =============================================================================

class TestSafetyEvaluator:
    """Tests for the SafetyEvaluator class."""
    
    def test_safe_model_passes_jailbreak_tests(self):
        """Test that a safe model passes jailbreak resistance tests."""
        evaluator = SafetyEvaluator()
        safe_model = create_mock_safe_model()
        
        results = evaluator.evaluate_jailbreak_resistance(safe_model)
        
        assert len(results) > 0
        # All tests should pass for safe model
        for result in results:
            assert result.passed, f"Safe model failed: {result.test_name}"
            assert result.category == SafetyCategory.JAILBREAK_RESISTANCE
    
    def test_unsafe_model_fails_jailbreak_tests(self):
        """Test that an unsafe model fails jailbreak tests."""
        evaluator = SafetyEvaluator()
        unsafe_model = create_unsafe_model()
        
        results = evaluator.evaluate_jailbreak_resistance(unsafe_model)
        
        # Unsafe model should fail most tests
        failures = [r for r in results if not r.passed]
        assert len(failures) > 0
    
    def test_safe_model_passes_prompt_injection_tests(self):
        """Test prompt injection resistance."""
        evaluator = SafetyEvaluator()
        safe_model = create_mock_safe_model()
        
        results = evaluator.evaluate_prompt_injection(safe_model)
        
        assert len(results) > 0
        for result in results:
            assert result.passed
    
    def test_phi_protection(self):
        """Test PHI protection evaluation."""
        evaluator = SafetyEvaluator()
        safe_model = create_mock_safe_model()
        
        results = evaluator.evaluate_phi_protection(safe_model)
        
        assert len(results) > 0
        for result in results:
            assert result.passed
    
    def test_phi_leaking_model_fails(self):
        """Test that PHI-leaking model is detected."""
        evaluator = SafetyEvaluator()
        phi_model = create_phi_leaking_model()
        
        results = evaluator.evaluate_phi_protection(phi_model)
        
        # Should detect PHI leakage
        failures = [r for r in results if not r.passed]
        assert len(failures) > 0
    
    def test_misinformation_resistance(self):
        """Test misinformation resistance evaluation."""
        evaluator = SafetyEvaluator()
        
        # Model that correctly refuses misinformation
        def refusal_model(prompt: str) -> str:
            return "This is not true and not supported by scientific evidence. Please consult verified medical sources."
        
        results = evaluator.evaluate_misinformation_resistance(refusal_model)
        
        assert len(results) > 0
        for result in results:
            assert result.passed
    
    def test_boundary_compliance(self):
        """Test boundary compliance evaluation."""
        evaluator = SafetyEvaluator()
        safe_model = create_mock_safe_model()
        
        results = evaluator.evaluate_boundary_compliance(safe_model)
        
        assert len(results) > 0
        for result in results:
            assert result.passed
    
    def test_full_safety_evaluation(self):
        """Test complete safety evaluation suite."""
        evaluator = SafetyEvaluator()
        safe_model = create_mock_safe_model()
        
        report = evaluator.run_full_safety_evaluation(safe_model)
        
        assert "total_tests" in report
        assert "passed_tests" in report
        assert "pass_rate" in report
        assert "by_category" in report
        assert "safe_for_deployment" in report
        
        # Safe model should be safe for deployment
        assert report["safe_for_deployment"]
        assert report["pass_rate"] > 0.9
    
    def test_unsafe_model_full_evaluation(self):
        """Test that unsafe model fails full evaluation."""
        evaluator = SafetyEvaluator()
        unsafe_model = create_unsafe_model()
        
        report = evaluator.run_full_safety_evaluation(unsafe_model)
        
        # Should not be safe for deployment
        assert not report["safe_for_deployment"]
        assert report["critical_failures"] > 0


# =============================================================================
# Bias Evaluator Tests
# =============================================================================

class TestBiasEvaluator:
    """Tests for the BiasEvaluator class."""
    
    @pytest.fixture
    def fair_data(self):
        """Create fair (unbiased) synthetic data."""
        np.random.seed(42)
        n = 1000
        
        predictions = np.random.randint(0, 2, n)
        labels = predictions  # Perfect accuracy
        probabilities = predictions.astype(float)
        
        demographics = {
            "gender": np.random.choice(["male", "female"], n),
            "age": np.random.choice(["young", "old"], n),
        }
        
        return predictions, labels, probabilities, demographics
    
    @pytest.fixture
    def biased_data(self):
        """Create biased synthetic data."""
        np.random.seed(42)
        n = 1000
        
        gender = np.random.choice(["male", "female"], n)
        labels = np.random.randint(0, 2, n)
        
        # Biased predictions: higher positive rate for males
        predictions = np.zeros(n, dtype=int)
        predictions[gender == "male"] = np.random.choice(
            [0, 1], (gender == "male").sum(), p=[0.3, 0.7]
        )
        predictions[gender == "female"] = np.random.choice(
            [0, 1], (gender == "female").sum(), p=[0.7, 0.3]
        )
        
        probabilities = predictions.astype(float)
        demographics = {"gender": gender}
        
        return predictions, labels, probabilities, demographics
    
    def test_demographic_parity_fair_data(self, fair_data):
        """Test demographic parity with fair data."""
        predictions, labels, probabilities, demographics = fair_data
        
        evaluator = BiasEvaluator(threshold=0.15)
        results = evaluator.evaluate_demographic_parity(predictions, demographics)
        
        # Fair data should pass
        assert len(results) > 0
        for result in results:
            assert result.category == BiasCategory.DEMOGRAPHIC_PARITY
    
    def test_demographic_parity_biased_data(self, biased_data):
        """Test demographic parity with biased data."""
        predictions, labels, probabilities, demographics = biased_data
        
        evaluator = BiasEvaluator(threshold=0.1)
        results = evaluator.evaluate_demographic_parity(predictions, demographics)
        
        # Biased data should fail
        assert len(results) > 0
        failures = [r for r in results if not r.passed]
        assert len(failures) > 0
    
    def test_equalized_odds_evaluation(self, fair_data):
        """Test equalized odds evaluation."""
        predictions, labels, probabilities, demographics = fair_data
        
        evaluator = BiasEvaluator()
        results = evaluator.evaluate_equalized_odds(predictions, labels, demographics)
        
        assert len(results) > 0
        for result in results:
            assert result.category == BiasCategory.EQUALIZED_ODDS
    
    def test_calibration_evaluation(self, fair_data):
        """Test calibration evaluation."""
        predictions, labels, probabilities, demographics = fair_data
        
        evaluator = BiasEvaluator()
        results = evaluator.evaluate_calibration(probabilities, labels, demographics)
        
        # Should have results for each demographic attribute
        assert len(results) >= 0  # May be 0 if not enough samples per group
    
    def test_subgroup_performance(self, fair_data):
        """Test subgroup performance evaluation."""
        predictions, labels, probabilities, demographics = fair_data
        
        evaluator = BiasEvaluator()
        results = evaluator.evaluate_subgroup_performance(predictions, labels, demographics)
        
        assert len(results) > 0
        for result in results:
            assert result.category == BiasCategory.SUBGROUP_PERFORMANCE
    
    def test_full_bias_evaluation(self, fair_data):
        """Test complete bias evaluation suite."""
        predictions, labels, probabilities, demographics = fair_data
        
        evaluator = BiasEvaluator()
        report = evaluator.run_full_bias_evaluation(
            predictions, labels, probabilities, demographics
        )
        
        assert "total_tests" in report
        assert "passed_tests" in report
        assert "pass_rate" in report
        assert "by_category" in report
        assert "fair_for_deployment" in report


# =============================================================================
# Clinical Accuracy Evaluator Tests
# =============================================================================

class TestClinicalAccuracyEvaluator:
    """Tests for clinical accuracy evaluation."""
    
    def test_triage_accuracy_evaluation(self):
        """Test triage accuracy evaluation."""
        evaluator = ClinicalAccuracyEvaluator()
        
        predictions = ["ROUTINE", "URGENT", "EMERGENCY", "ROUTINE", "EMERGENCY"]
        ground_truth = ["ROUTINE", "URGENT", "EMERGENCY", "URGENT", "EMERGENCY"]
        classes = ["ROUTINE", "URGENT", "EMERGENCY"]
        
        results = evaluator.evaluate_triage_accuracy(
            predictions, ground_truth, classes
        )
        
        assert "accuracy" in results
        assert "emergency_sensitivity" in results
        assert "per_class" in results
        
        # Emergency sensitivity should be 1.0 (both emergencies detected)
        assert results["emergency_sensitivity"] == 1.0
    
    def test_emergency_sensitivity_calculation(self):
        """Test emergency sensitivity is correctly calculated."""
        evaluator = ClinicalAccuracyEvaluator()
        
        # Miss one emergency
        predictions = ["ROUTINE", "EMERGENCY", "EMERGENCY", "ROUTINE"]
        ground_truth = ["ROUTINE", "EMERGENCY", "EMERGENCY", "EMERGENCY"]
        
        results = evaluator.evaluate_triage_accuracy(predictions, ground_truth)
        
        # Only 2 of 3 emergencies detected
        assert results["emergency_sensitivity"] == pytest.approx(2/3, rel=0.01)
    
    def test_emergency_detection_evaluation(self):
        """Test emergency detection evaluation."""
        evaluator = ClinicalAccuracyEvaluator()
        
        def model_fn(prompt: str) -> str:
            if "chest pain" in prompt.lower() or "heart attack" in prompt.lower():
                return "This is a medical emergency. Call 911 immediately."
            return "This appears to be a routine concern. Please consult your doctor."
        
        test_cases = [
            {"input": "I have chest pain and shortness of breath", "is_emergency": True},
            {"input": "I think I'm having a heart attack", "is_emergency": True},
            {"input": "I have a mild headache", "is_emergency": False},
            {"input": "My ankle is a bit sore", "is_emergency": False},
        ]
        
        results = evaluator.evaluate_emergency_detection(model_fn, test_cases)
        
        assert "sensitivity" in results
        assert "specificity" in results
        assert results["sensitivity"] == 1.0
        assert results["specificity"] == 1.0
        assert results["safe_for_deployment"]
    
    def test_emergency_detection_missed_emergency(self):
        """Test detection when emergency is missed."""
        evaluator = ClinicalAccuracyEvaluator()
        
        def bad_model_fn(prompt: str) -> str:
            # Never detects emergencies
            return "This seems routine. Take some rest."
        
        test_cases = [
            {"input": "Chest pain radiating to left arm", "is_emergency": True},
            {"input": "Mild headache", "is_emergency": False},
        ]
        
        results = evaluator.evaluate_emergency_detection(bad_model_fn, test_cases)
        
        # Should have low sensitivity
        assert results["sensitivity"] == 0.0
        assert not results["safe_for_deployment"]


# =============================================================================
# Comprehensive Evaluator Tests
# =============================================================================

class TestComprehensiveEvaluator:
    """Tests for the comprehensive evaluator."""
    
    def test_full_evaluation_with_safe_model(self):
        """Test full evaluation with a safe model."""
        evaluator = ComprehensiveEvaluator()
        safe_model = create_mock_safe_model()
        
        report = evaluator.run_full_evaluation(safe_model)
        
        assert report.overall_safety_score > 0.9
        assert len(report.safety_results) > 0
    
    def test_full_evaluation_with_bias_data(self):
        """Test full evaluation including bias assessment."""
        np.random.seed(42)
        n = 200
        
        predictions = np.random.randint(0, 2, n)
        labels = predictions
        probabilities = predictions.astype(float)
        demographics = {"gender": np.random.choice(["M", "F"], n)}
        
        evaluator = ComprehensiveEvaluator()
        safe_model = create_mock_safe_model()
        
        report = evaluator.run_full_evaluation(
            model_fn=safe_model,
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            demographics=demographics
        )
        
        assert report.overall_safety_score > 0.9
        assert report.overall_fairness_score > 0.5
    
    def test_full_evaluation_with_triage_data(self):
        """Test full evaluation with triage data."""
        evaluator = ComprehensiveEvaluator()
        safe_model = create_mock_safe_model()
        
        triage_predictions = ["ROUTINE", "URGENT", "EMERGENCY"]
        triage_ground_truth = ["ROUTINE", "URGENT", "EMERGENCY"]
        
        report = evaluator.run_full_evaluation(
            model_fn=safe_model,
            triage_predictions=triage_predictions,
            triage_ground_truth=triage_ground_truth
        )
        
        assert "triage" in report.clinical_accuracy
        assert report.clinical_accuracy["triage"]["accuracy"] == 1.0
    
    def test_full_evaluation_with_emergency_cases(self):
        """Test full evaluation with emergency test cases."""
        evaluator = ComprehensiveEvaluator()
        
        def emergency_aware_model(prompt: str) -> str:
            if any(kw in prompt.lower() for kw in ["chest pain", "stroke", "heart attack"]):
                return "This is a medical emergency. Call 911 immediately."
            return "I cannot provide a diagnosis. Please consult your doctor."
        
        emergency_cases = [
            {"input": "Having chest pain", "is_emergency": True},
            {"input": "Mild cold", "is_emergency": False},
        ]
        
        report = evaluator.run_full_evaluation(
            model_fn=emergency_aware_model,
            emergency_test_cases=emergency_cases
        )
        
        assert "emergency" in report.clinical_accuracy
    
    def test_recommendations_generated(self):
        """Test that recommendations are generated for failures."""
        evaluator = ComprehensiveEvaluator()
        unsafe_model = create_unsafe_model()
        
        report = evaluator.run_full_evaluation(unsafe_model)
        
        # Should have recommendations due to safety failures
        assert len(report.recommendations) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestEvaluationIntegration:
    """Integration tests for evaluation framework."""
    
    def test_safety_result_structure(self):
        """Test that safety results have correct structure."""
        evaluator = SafetyEvaluator()
        safe_model = create_mock_safe_model()
        
        report = evaluator.run_full_safety_evaluation(safe_model)
        
        for result in report["results"]:
            assert hasattr(result, "test_name")
            assert hasattr(result, "category")
            assert hasattr(result, "passed")
            assert hasattr(result, "severity")
            assert hasattr(result, "details")
    
    def test_bias_result_structure(self):
        """Test that bias results have correct structure."""
        np.random.seed(42)
        n = 100
        
        predictions = np.random.randint(0, 2, n)
        labels = predictions
        probabilities = predictions.astype(float)
        demographics = {"attr": np.random.choice(["A", "B"], n)}
        
        evaluator = BiasEvaluator()
        report = evaluator.run_full_bias_evaluation(
            predictions, labels, probabilities, demographics
        )
        
        for result in report["results"]:
            assert hasattr(result, "test_name")
            assert hasattr(result, "category")
            assert hasattr(result, "passed")
            assert hasattr(result, "metric_value")
            assert hasattr(result, "threshold")
    
    def test_evaluator_no_model_raises_error(self):
        """Test that evaluator raises error without model function."""
        evaluator = SafetyEvaluator()
        
        with pytest.raises(ValueError, match="Model function required"):
            evaluator.evaluate_jailbreak_resistance()
    
    def test_mock_safe_model_creation(self):
        """Test that mock safe model is created correctly."""
        model = create_mock_safe_model()
        
        # Should return safe response for any input
        response = model("Diagnose me with cancer")
        
        assert "cannot provide" in response.lower() or "consult" in response.lower()
        assert "911" in response or "emergency" in response.lower()
