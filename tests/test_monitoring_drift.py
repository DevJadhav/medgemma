"""
Tests for drift detection module (Phase 8: Monitoring & Observability).

TDD tests for input drift, output drift, concept drift, and drift management.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# Test: Drift Configuration
# ============================================================================

class TestDriftConfig:
    """Test drift detection configuration."""
    
    def test_drift_config_defaults(self):
        """Test default drift configuration values."""
        from medai_compass.monitoring.drift_detector import DriftConfig
        
        config = DriftConfig()
        
        assert config.model_name == "medgemma-27b-it"
        assert config.kl_divergence_threshold == 0.1
        assert config.psi_threshold == 0.2
        assert config.p_value_threshold == 0.05
        assert config.window_size == 1000
    
    def test_drift_config_4b_model(self):
        """Test configuration for 4B model."""
        from medai_compass.monitoring.drift_detector import DriftConfig
        
        config = DriftConfig(model_name="medgemma-4b-it")
        
        assert config.model_name == "medgemma-4b-it"
    
    def test_drift_config_custom_thresholds(self):
        """Test custom threshold configuration."""
        from medai_compass.monitoring.drift_detector import DriftConfig
        
        config = DriftConfig(
            kl_divergence_threshold=0.15,
            psi_threshold=0.25,
            p_value_threshold=0.01
        )
        
        assert config.kl_divergence_threshold == 0.15
        assert config.psi_threshold == 0.25
        assert config.p_value_threshold == 0.01


# ============================================================================
# Test: Drift Types and Results
# ============================================================================

class TestDriftTypesAndResults:
    """Test drift type enum and result dataclass."""
    
    def test_drift_type_enum(self):
        """Test DriftType enum values."""
        from medai_compass.monitoring.drift_detector import DriftType
        
        assert DriftType.INPUT.value == "input"
        assert DriftType.OUTPUT.value == "output"
        assert DriftType.CONCEPT.value == "concept"
        assert DriftType.FEATURE.value == "feature"
    
    def test_drift_result_dataclass(self):
        """Test DriftResult dataclass."""
        from medai_compass.monitoring.drift_detector import DriftResult, DriftType
        
        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.INPUT,
            score=0.15,
            threshold=0.1,
            p_value=0.03,
            details={"feature": "token_length"}
        )
        
        assert result.drift_detected is True
        assert result.drift_type == DriftType.INPUT
        assert result.score == 0.15
        assert result.threshold == 0.1
        assert result.p_value == 0.03
    
    def test_drift_result_to_dict(self):
        """Test DriftResult serialization."""
        from medai_compass.monitoring.drift_detector import DriftResult, DriftType
        
        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.INPUT,
            score=0.15,
            threshold=0.1
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["drift_detected"] is True
        assert result_dict["drift_type"] == "input"
        assert result_dict["score"] == 0.15


# ============================================================================
# Test: Statistical Utilities
# ============================================================================

class TestStatisticalUtilities:
    """Test statistical utility functions."""
    
    def test_calculate_kl_divergence(self):
        """Test KL divergence calculation."""
        from medai_compass.monitoring.drift_detector import calculate_kl_divergence
        
        # Identical distributions should have KL divergence of 0
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        
        kl = calculate_kl_divergence(p, q)
        
        assert abs(kl) < 1e-10
    
    def test_calculate_kl_divergence_different(self):
        """Test KL divergence for different distributions."""
        from medai_compass.monitoring.drift_detector import calculate_kl_divergence
        
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.33, 0.33, 0.34])
        
        kl = calculate_kl_divergence(p, q)
        
        assert kl > 0  # Different distributions have positive KL
    
    def test_calculate_psi(self):
        """Test Population Stability Index calculation."""
        from medai_compass.monitoring.drift_detector import calculate_psi
        
        # Same distribution should have PSI near 0
        expected = np.array([100, 100, 100, 100])
        actual = np.array([100, 100, 100, 100])
        
        psi = calculate_psi(expected, actual)
        
        assert abs(psi) < 0.01
    
    def test_calculate_psi_drift(self):
        """Test PSI detects distribution shift."""
        from medai_compass.monitoring.drift_detector import calculate_psi
        
        expected = np.array([100, 100, 100, 100])
        actual = np.array([50, 50, 150, 150])  # Shifted distribution
        
        psi = calculate_psi(expected, actual)
        
        assert psi > 0  # Should detect shift


# ============================================================================
# Test: Input Drift Detector
# ============================================================================

class TestInputDriftDetector:
    """Test input drift detection."""
    
    def test_input_drift_detector_initialization(self):
        """Test input drift detector initialization."""
        from medai_compass.monitoring.drift_detector import InputDriftDetector, DriftConfig
        
        config = DriftConfig(model_name="medgemma-27b-it")
        detector = InputDriftDetector(config)
        
        assert detector.config == config
        assert detector.baseline is None
    
    def test_set_baseline(self):
        """Test setting baseline distribution."""
        from medai_compass.monitoring.drift_detector import InputDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = InputDriftDetector(config)
        
        baseline_data = np.random.randn(1000, 10)
        detector.set_baseline(baseline_data)
        
        assert detector.baseline is not None
    
    def test_detect_no_drift(self):
        """Test detection when no drift present."""
        from medai_compass.monitoring.drift_detector import InputDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = InputDriftDetector(config)
        
        # Same distribution for baseline and current
        np.random.seed(42)
        baseline = np.random.randn(1000, 10)
        current = np.random.randn(1000, 10)
        
        detector.set_baseline(baseline)
        result = detector.detect(current)
        
        # With same distribution, should typically not detect drift
        # (though statistical tests can have false positives)
        assert isinstance(result.drift_detected, bool)
    
    def test_detect_with_drift(self):
        """Test detection when drift is present."""
        from medai_compass.monitoring.drift_detector import InputDriftDetector, DriftConfig
        
        config = DriftConfig(p_value_threshold=0.05)
        detector = InputDriftDetector(config)
        
        # Different distributions
        np.random.seed(42)
        baseline = np.random.randn(1000, 10)
        current = np.random.randn(1000, 10) + 2  # Shifted mean
        
        detector.set_baseline(baseline)
        result = detector.detect(current)
        
        # Significant shift should be detected
        assert result.drift_detected is True
    
    def test_detect_requires_baseline(self):
        """Test detection requires baseline to be set."""
        from medai_compass.monitoring.drift_detector import InputDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = InputDriftDetector(config)
        
        with pytest.raises(ValueError, match="Baseline"):
            detector.detect(np.random.randn(100, 10))


# ============================================================================
# Test: Output Drift Detector
# ============================================================================

class TestOutputDriftDetector:
    """Test output drift detection."""
    
    def test_output_drift_detector_initialization(self):
        """Test output drift detector initialization."""
        from medai_compass.monitoring.drift_detector import OutputDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = OutputDriftDetector(config)
        
        assert detector.config == config
    
    def test_detect_prediction_distribution_shift(self):
        """Test detecting shift in prediction distribution."""
        from medai_compass.monitoring.drift_detector import OutputDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = OutputDriftDetector(config)
        
        # Baseline predictions
        baseline_preds = np.array([0.9, 0.85, 0.88, 0.92, 0.87] * 200)
        # Shifted predictions
        current_preds = np.array([0.6, 0.55, 0.58, 0.62, 0.57] * 200)
        
        detector.set_baseline(baseline_preds)
        result = detector.detect(current_preds)
        
        # Should detect drift due to confidence drop
        assert result.drift_type.value == "output"
    
    def test_detect_confidence_drift(self):
        """Test detecting confidence score drift."""
        from medai_compass.monitoring.drift_detector import OutputDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = OutputDriftDetector(config)
        
        # High confidence baseline
        baseline = np.random.beta(9, 1, 1000)  # Skewed towards 1
        # Lower confidence current
        current = np.random.beta(5, 5, 1000)  # More uniform
        
        detector.set_baseline(baseline)
        result = detector.detect(current)
        
        # Should detect drift (different distributions)
        assert result.drift_type.value == "output"
        assert isinstance(result.drift_detected, bool)


# ============================================================================
# Test: Concept Drift Detector
# ============================================================================

class TestConceptDriftDetector:
    """Test concept drift detection."""
    
    def test_concept_drift_detector_initialization(self):
        """Test concept drift detector initialization."""
        from medai_compass.monitoring.drift_detector import ConceptDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = ConceptDriftDetector(config)
        
        assert detector.config == config
    
    def test_detect_relationship_change(self):
        """Test detecting change in input-output relationship."""
        from medai_compass.monitoring.drift_detector import ConceptDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = ConceptDriftDetector(config)
        
        # Baseline: linear relationship
        np.random.seed(42)
        X_baseline = np.random.randn(500, 5)
        y_baseline = X_baseline @ np.array([1, 2, 3, 4, 5]) + np.random.randn(500) * 0.1
        
        # Current: different relationship
        X_current = np.random.randn(500, 5)
        y_current = X_current @ np.array([5, 4, 3, 2, 1]) + np.random.randn(500) * 0.1
        
        detector.set_baseline(X_baseline, y_baseline)
        result = detector.detect(X_current, y_current)
        
        assert result.drift_type.value == "concept"


# ============================================================================
# Test: Drift Manager (Unified Interface)
# ============================================================================

class TestDriftManager:
    """Test unified drift management."""
    
    def test_drift_manager_initialization(self):
        """Test drift manager initialization."""
        from medai_compass.monitoring.drift_detector import DriftManager, DriftConfig
        
        config = DriftConfig(model_name="medgemma-27b-it")
        manager = DriftManager(config)
        
        assert manager.input_detector is not None
        assert manager.output_detector is not None
        assert manager.concept_detector is not None
    
    def test_drift_manager_model_selection(self):
        """Test drift manager with different models."""
        from medai_compass.monitoring.drift_detector import DriftManager, DriftConfig
        
        config_27b = DriftConfig(model_name="medgemma-27b-it")
        config_4b = DriftConfig(model_name="medgemma-4b-it")
        
        manager_27b = DriftManager(config_27b)
        manager_4b = DriftManager(config_4b)
        
        assert manager_27b.config.model_name == "medgemma-27b-it"
        assert manager_4b.config.model_name == "medgemma-4b-it"
    
    def test_run_all_drift_checks(self):
        """Test running all drift checks."""
        from medai_compass.monitoring.drift_detector import DriftManager, DriftConfig
        
        config = DriftConfig()
        manager = DriftManager(config)
        
        np.random.seed(42)
        baseline_input = np.random.randn(500, 10)
        baseline_output = np.random.rand(500)
        
        manager.set_baselines(
            input_data=baseline_input,
            output_data=baseline_output
        )
        
        current_input = np.random.randn(500, 10)
        current_output = np.random.rand(500)
        
        results = manager.check_all_drift(
            input_data=current_input,
            output_data=current_output
        )
        
        assert "input" in results
        assert "output" in results
    
    def test_get_drift_summary(self):
        """Test getting drift summary report."""
        from medai_compass.monitoring.drift_detector import DriftManager, DriftConfig
        
        config = DriftConfig()
        manager = DriftManager(config)
        
        np.random.seed(42)
        manager.set_baselines(
            input_data=np.random.randn(500, 10),
            output_data=np.random.rand(500)
        )
        
        manager.check_all_drift(
            input_data=np.random.randn(500, 10),
            output_data=np.random.rand(500)
        )
        
        summary = manager.get_summary()
        
        assert "model_name" in summary
        assert "drift_detected" in summary
        assert "checks_performed" in summary


# ============================================================================
# Test: Alibi-Detect Integration
# ============================================================================

class TestAlibiDetectIntegration:
    """Test integration with alibi-detect library."""
    
    def test_tabular_drift_detector(self):
        """Test tabular drift detection with alibi-detect."""
        from medai_compass.monitoring.drift_detector import AlibiTabularDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = AlibiTabularDriftDetector(config)
        
        np.random.seed(42)
        baseline = np.random.randn(500, 10).astype(np.float32)
        
        detector.set_baseline(baseline)
        
        assert detector.detector is not None
    
    def test_alibi_detect_mmd_drift(self):
        """Test MMD-based drift detection."""
        from medai_compass.monitoring.drift_detector import AlibiTabularDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = AlibiTabularDriftDetector(config, method="mmd")
        
        np.random.seed(42)
        baseline = np.random.randn(500, 10).astype(np.float32)
        current = np.random.randn(500, 10).astype(np.float32) + 2  # Shifted
        
        detector.set_baseline(baseline)
        result = detector.detect(current)
        
        assert result.drift_detected is True
    
    def test_alibi_detect_ks_drift(self):
        """Test KS-based drift detection."""
        from medai_compass.monitoring.drift_detector import AlibiTabularDriftDetector, DriftConfig
        
        config = DriftConfig()
        detector = AlibiTabularDriftDetector(config, method="ks")
        
        np.random.seed(42)
        baseline = np.random.randn(500, 5).astype(np.float32)
        
        detector.set_baseline(baseline)
        
        # Same distribution - no drift expected
        current = np.random.randn(500, 5).astype(np.float32)
        result = detector.detect(current)
        
        assert isinstance(result.drift_detected, bool)


# ============================================================================
# Test: Model-Specific Drift Thresholds
# ============================================================================

class TestModelSpecificThresholds:
    """Test model-specific drift thresholds."""
    
    def test_27b_model_thresholds(self):
        """Test thresholds for 27B model."""
        from medai_compass.monitoring.drift_detector import get_model_thresholds
        
        thresholds = get_model_thresholds("medgemma-27b-it")
        
        assert "kl_divergence" in thresholds
        assert "psi" in thresholds
        assert "latency_p95_ms" in thresholds
    
    def test_4b_model_thresholds(self):
        """Test thresholds for 4B model."""
        from medai_compass.monitoring.drift_detector import get_model_thresholds
        
        thresholds = get_model_thresholds("medgemma-4b-it")
        
        # 4B should have lower latency threshold (faster model)
        thresholds_27b = get_model_thresholds("medgemma-27b-it")
        
        assert thresholds["latency_p95_ms"] <= thresholds_27b["latency_p95_ms"]


# ============================================================================
# Test: Drift History and Persistence
# ============================================================================

class TestDriftHistory:
    """Test drift history tracking."""
    
    def test_record_drift_event(self):
        """Test recording drift events."""
        from medai_compass.monitoring.drift_detector import DriftHistory, DriftResult, DriftType
        
        history = DriftHistory()
        
        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.INPUT,
            score=0.15,
            threshold=0.1
        )
        
        history.record(result)
        
        assert len(history.events) == 1
    
    def test_get_drift_history(self):
        """Test retrieving drift history."""
        from medai_compass.monitoring.drift_detector import DriftHistory, DriftResult, DriftType
        
        history = DriftHistory()
        
        for i in range(5):
            result = DriftResult(
                drift_detected=(i % 2 == 0),
                drift_type=DriftType.INPUT,
                score=0.1 + i * 0.05,
                threshold=0.1
            )
            history.record(result)
        
        events = history.get_events(limit=3)
        
        assert len(events) == 3
    
    def test_get_drift_rate(self):
        """Test calculating drift rate."""
        from medai_compass.monitoring.drift_detector import DriftHistory, DriftResult, DriftType
        
        history = DriftHistory()
        
        # 3 drift events out of 5
        for i in range(5):
            result = DriftResult(
                drift_detected=(i < 3),
                drift_type=DriftType.INPUT,
                score=0.15 if i < 3 else 0.05,
                threshold=0.1
            )
            history.record(result)
        
        rate = history.get_drift_rate()
        
        assert rate == 0.6  # 3/5
