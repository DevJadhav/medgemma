"""Tests for the evaluation framework."""

import numpy as np
import pytest


class TestDiagnosticEvaluation:
    """Tests for diagnostic imaging evaluation metrics."""

    def test_auc_roc_calculation(self):
        """Test AUC-ROC calculation for binary classification."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])

        auc = evaluator.calculate_auc_roc(y_true, y_score)
        assert auc == 1.0

    def test_auc_roc_random_predictions(self):
        """Test AUC-ROC with random predictions."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        # Random predictions should give AUC around 0.5
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_score = np.random.rand(100)

        auc = evaluator.calculate_auc_roc(y_true, y_score)
        assert 0.3 <= auc <= 0.7

    def test_sensitivity_at_specificity(self):
        """Test sensitivity at target specificity."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        sensitivity = evaluator.sensitivity_at_specificity(y_true, y_score, target_specificity=0.75)
        assert 0.0 <= sensitivity <= 1.0

    def test_dice_coefficient(self):
        """Test Dice coefficient for segmentation."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        # Perfect overlap
        pred = np.array([1, 1, 0, 0])
        target = np.array([1, 1, 0, 0])

        dice = evaluator.calculate_dice(pred, target)
        assert dice == 1.0

    def test_dice_coefficient_no_overlap(self):
        """Test Dice coefficient with no overlap."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        pred = np.array([1, 1, 0, 0])
        target = np.array([0, 0, 1, 1])

        dice = evaluator.calculate_dice(pred, target)
        assert dice == 0.0

    def test_iou_calculation(self):
        """Test Intersection over Union for localization."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        # Bounding boxes: [x1, y1, x2, y2]
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]

        iou = evaluator.calculate_iou(box1, box2)
        assert iou == 1.0

    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]

        iou = evaluator.calculate_iou(box1, box2)
        # Intersection = 5*5 = 25, Union = 100 + 100 - 25 = 175
        expected_iou = 25 / 175
        assert abs(iou - expected_iou) < 0.01

    def test_evaluate_batch(self):
        """Test batch evaluation of diagnostic predictions."""
        from medai_compass.evaluation.metrics import DiagnosticEvaluator

        evaluator = DiagnosticEvaluator()

        predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        labels = np.array([[1, 0], [0, 1], [1, 0]])
        pathology_names = ["Finding A", "Finding B"]

        results = evaluator.evaluate_batch(predictions, labels, pathology_names)

        assert results.auc_roc is not None
        assert "Finding A" in results.auc_roc
        assert "Finding B" in results.auc_roc


class TestNLPEvaluation:
    """Tests for clinical NLP evaluation metrics."""

    def test_rouge_l_calculation(self):
        """Test ROUGE-L calculation for summarization."""
        from medai_compass.evaluation.metrics import NLPEvaluator

        evaluator = NLPEvaluator()

        reference = "The patient presents with chest pain and shortness of breath."
        generated = "The patient has chest pain and breathing difficulty."

        rouge_l = evaluator.calculate_rouge_l(reference, generated)
        assert 0.0 <= rouge_l <= 1.0
        assert rouge_l > 0.3  # Should have decent overlap

    def test_rouge_l_identical(self):
        """Test ROUGE-L with identical texts."""
        from medai_compass.evaluation.metrics import NLPEvaluator

        evaluator = NLPEvaluator()

        text = "The patient presents with chest pain."
        rouge_l = evaluator.calculate_rouge_l(text, text)
        assert rouge_l == 1.0

    def test_entity_f1_calculation(self):
        """Test entity extraction F1 score."""
        from medai_compass.evaluation.metrics import NLPEvaluator

        evaluator = NLPEvaluator()

        predicted = {"entities": ["hypertension", "diabetes", "chest pain"]}
        reference = {"entities": ["hypertension", "diabetes", "headache"]}

        f1 = evaluator.calculate_entity_f1([predicted], [reference])
        # 2 true positives, 1 false positive, 1 false negative
        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        expected_f1 = 2/3
        assert abs(f1 - expected_f1) < 0.01

    def test_hallucination_detection(self):
        """Test hallucination detection in generated text."""
        from medai_compass.evaluation.metrics import NLPEvaluator

        evaluator = NLPEvaluator()

        # Text with potential hallucination
        text = "According to your medical records from your last visit..."
        score = evaluator.detect_hallucination_indicators(text)
        assert score > 0.0

    def test_no_hallucination(self):
        """Test hallucination detection with clean text."""
        from medai_compass.evaluation.metrics import NLPEvaluator

        evaluator = NLPEvaluator()

        text = "Based on the symptoms described, this could indicate an infection."
        score = evaluator.detect_hallucination_indicators(text)
        assert score == 0.0

    def test_readability_grade(self):
        """Test Flesch-Kincaid readability grade."""
        from medai_compass.evaluation.metrics import NLPEvaluator

        evaluator = NLPEvaluator()

        # Simple text should have lower grade
        simple_text = "You have a cold. Rest and drink water."
        complex_text = "The etiology of the presenting symptomatology suggests a viral upper respiratory infection with concomitant pharyngitis."

        simple_grade = evaluator.calculate_readability(simple_text)
        complex_grade = evaluator.calculate_readability(complex_text)

        assert simple_grade < complex_grade


class TestCommunicationEvaluation:
    """Tests for patient communication evaluation metrics."""

    def test_response_appropriateness_scoring(self):
        """Test response appropriateness scoring."""
        from medai_compass.evaluation.metrics import CommunicationEvaluator

        evaluator = CommunicationEvaluator()

        # Appropriate response
        response = "I understand you're concerned. Please consult with your healthcare provider for a proper evaluation."
        score = evaluator.score_appropriateness(response)
        assert score >= 0.7

    def test_inappropriate_response_detection(self):
        """Test detection of inappropriate medical claims."""
        from medai_compass.evaluation.metrics import CommunicationEvaluator

        evaluator = CommunicationEvaluator()

        # Inappropriate response making diagnosis
        response = "You definitely have cancer. Start this treatment immediately."
        score = evaluator.score_appropriateness(response)
        assert score < 0.5

    def test_harm_potential_scoring(self):
        """Test harm potential scoring."""
        from medai_compass.evaluation.metrics import CommunicationEvaluator

        evaluator = CommunicationEvaluator()

        # Low harm potential
        safe_response = "Please consult your doctor for medical advice."
        harm_score = evaluator.calculate_harm_potential(safe_response)
        assert harm_score < 0.3

    def test_empathy_scoring(self):
        """Test empathy scoring in responses."""
        from medai_compass.evaluation.metrics import CommunicationEvaluator

        evaluator = CommunicationEvaluator()

        empathetic = "I understand this must be worrying for you. It's completely normal to have concerns about these symptoms."
        cold = "See a doctor."

        empathy_high = evaluator.score_empathy(empathetic)
        empathy_low = evaluator.score_empathy(cold)

        assert empathy_high > empathy_low


class TestDriftDetection:
    """Tests for model drift detection."""

    def test_psi_calculation_no_drift(self):
        """Test Population Stability Index with no drift."""
        from medai_compass.evaluation.drift import DriftDetector

        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        detector = DriftDetector(reference)
        psi = detector.calculate_psi(current)

        # PSI < 0.1 indicates no significant drift
        assert psi < 0.1

    def test_psi_calculation_with_drift(self):
        """Test PSI with significant distribution shift."""
        from medai_compass.evaluation.drift import DriftDetector

        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Shifted mean

        detector = DriftDetector(reference)
        psi = detector.calculate_psi(current)

        # PSI > 0.25 indicates significant drift
        assert psi > 0.25

    def test_ks_test_no_drift(self):
        """Test Kolmogorov-Smirnov test with no drift."""
        from medai_compass.evaluation.drift import DriftDetector

        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        detector = DriftDetector(reference)
        statistic, p_value = detector.ks_test(current)

        # p-value > 0.05 indicates no significant difference
        assert p_value > 0.05

    def test_ks_test_with_drift(self):
        """Test KS test with significant distribution shift."""
        from medai_compass.evaluation.drift import DriftDetector

        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Shifted mean

        detector = DriftDetector(reference)
        statistic, p_value = detector.ks_test(current)

        # p-value < 0.05 indicates significant difference
        assert p_value < 0.05

    def test_comprehensive_drift_check(self):
        """Test comprehensive drift detection."""
        from medai_compass.evaluation.drift import DriftDetector

        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(1, 1, 1000)  # Moderate shift

        detector = DriftDetector(reference)
        result = detector.check_drift(current)

        assert "psi" in result
        assert "ks_statistic" in result
        assert "ks_pvalue" in result
        assert "drift_detected" in result
        assert "severity" in result


class TestMetricsCollector:
    """Tests for Prometheus metrics collection."""

    def test_record_inference_latency(self):
        """Test recording inference latency."""
        from medai_compass.evaluation.metrics import MetricsCollector

        collector = MetricsCollector()

        # Should not raise
        collector.record_inference("medgemma_4b", "diagnostic", 0.5, 0.95)

    def test_record_escalation(self):
        """Test recording escalation event."""
        from medai_compass.evaluation.metrics import MetricsCollector

        collector = MetricsCollector()

        # Should not raise
        collector.record_escalation("diagnostic", "low_confidence")

    def test_update_drift_score(self):
        """Test updating drift score."""
        from medai_compass.evaluation.metrics import MetricsCollector

        collector = MetricsCollector()

        # Should not raise
        collector.update_drift_score("medgemma_4b", 0.15)
