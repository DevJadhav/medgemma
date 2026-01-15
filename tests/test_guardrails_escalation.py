"""Tests for human escalation gateway - Written FIRST (TDD)."""

import pytest


class TestCriticalFindingDetection:
    """Test detection of critical medical findings."""

    def test_detect_pneumothorax(self, sample_critical_finding):
        """Test pneumothorax triggers immediate escalation."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response=sample_critical_finding,
            confidence=0.95,
            uncertainty=0.05,
            domain="diagnostic"
        )
        
        assert decision.should_escalate is True
        assert decision.priority == "immediate"

    def test_detect_stroke(self):
        """Test stroke/CVA triggers immediate escalation."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="Findings consistent with acute stroke in the left MCA territory.",
            confidence=0.90,
            uncertainty=0.10,
            domain="diagnostic"
        )
        
        assert decision.should_escalate is True
        assert decision.priority == "immediate"

    def test_detect_myocardial_infarction(self):
        """Test MI triggers immediate escalation."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="ECG changes suggestive of myocardial infarction.",
            confidence=0.92,
            uncertainty=0.08,
            domain="diagnostic"
        )
        
        assert decision.should_escalate is True


class TestSafetyConcernDetection:
    """Test detection of patient safety concerns."""

    def test_detect_suicide_mention(self, sample_safety_concern_text):
        """Test suicide mention triggers immediate escalation."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="I understand you're feeling distressed.",
            confidence=0.95,
            uncertainty=0.05,
            domain="communication",
            user_input=sample_safety_concern_text
        )
        
        assert decision.should_escalate is True
        assert decision.priority == "immediate"
        assert decision.reason.value == "safety_concern"

    def test_detect_self_harm(self):
        """Test self-harm mention triggers escalation."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="Please seek help immediately.",
            confidence=0.95,
            uncertainty=0.05,
            domain="communication",
            user_input="I've been thinking about hurting myself."
        )
        
        assert decision.should_escalate is True
        assert decision.priority == "immediate"


class TestConfidenceEscalation:
    """Test confidence-based escalation."""

    def test_low_confidence_escalates(self):
        """Test confidence < 0.80 triggers escalation."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="Possible small nodule in the right lower lobe.",
            confidence=0.65,
            uncertainty=0.15,
            domain="diagnostic"
        )
        
        assert decision.should_escalate is True
        assert decision.reason.value == "low_confidence"

    def test_high_confidence_no_escalation(self):
        """Test high confidence without critical findings passes."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="No acute abnormality identified. Heart size normal.",
            confidence=0.96,
            uncertainty=0.04,
            domain="diagnostic"
        )
        
        assert decision.should_escalate is False

    def test_domain_specific_thresholds(self):
        """Test different domains have different thresholds."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        # Diagnostic has higher threshold (0.90)
        diagnostic_decision = gateway.evaluate(
            response="Lung fields are clear.",
            confidence=0.87,
            uncertainty=0.05,
            domain="diagnostic"
        )
        
        # Communication has lower threshold (0.80)
        comm_decision = gateway.evaluate(
            response="Here's some health information.",
            confidence=0.87,
            uncertainty=0.05,
            domain="communication"
        )
        
        assert diagnostic_decision.should_escalate is True
        assert comm_decision.should_escalate is False


class TestUncertaintyEscalation:
    """Test uncertainty-based escalation."""

    def test_high_uncertainty_escalates(self):
        """Test uncertainty > 0.20 triggers escalation."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="Unclear findings in the mediastinum.",
            confidence=0.92,  # Above diagnostic threshold (0.90)
            uncertainty=0.25,
            domain="diagnostic"
        )
        
        assert decision.should_escalate is True
        assert decision.reason.value == "high_uncertainty"

    def test_low_uncertainty_passes(self):
        """Test low uncertainty passes if confidence is high."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        
        gateway = HumanEscalationGateway()
        
        decision = gateway.evaluate(
            response="Clear lung fields bilaterally.",
            confidence=0.95,
            uncertainty=0.05,
            domain="diagnostic"
        )
        
        assert decision.should_escalate is False
