"""
Tests for the Communication Agent (AutoGen) module.
"""

from datetime import datetime

import pytest

from medai_compass.agents.communication import (
    AgentResponse,
    ClinicalOversightProxy,
    CommunicationOrchestrator,
    ConversationContext,
    FollowUpSchedulingAgent,
    HealthEducatorAgent,
    MessageCategory,
    PatientMessage,
    TriageAgent,
    TriageResult,
    UrgencyLevel,
)


class TestPatientMessage:
    """Tests for PatientMessage dataclass."""
    
    def test_create_message(self):
        """Test creating a patient message."""
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="I have a headache that won't go away"
        )
        
        assert msg.message_id == "msg-001"
        assert msg.patient_id == "pat-001"
        assert "headache" in msg.content
        assert msg.timestamp is not None


class TestTriageAgent:
    """Tests for TriageAgent class."""
    
    def test_init(self):
        """Test triage agent initialization."""
        agent = TriageAgent()
        assert agent.name == "TriageAgent"
        
    def test_triage_emergency_chest_pain(self):
        """Test emergency triage for chest pain."""
        agent = TriageAgent()
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="I'm having severe chest pain and can't breathe"
        )
        
        result = agent.triage_message(msg)
        
        assert result.urgency == UrgencyLevel.EMERGENCY
        assert result.category == MessageCategory.EMERGENCY
        assert result.requires_human_review is True
        
    def test_triage_emergency_mental_health(self):
        """Test emergency triage for mental health crisis."""
        agent = TriageAgent()
        msg = PatientMessage(
            message_id="msg-002",
            patient_id="pat-002",
            content="I've been having thoughts about wanting to kill myself"
        )
        
        result = agent.triage_message(msg)
        
        assert result.urgency == UrgencyLevel.EMERGENCY
        assert "mental_health_crisis" in result.safety_flags
        
    def test_triage_urgent_symptoms(self):
        """Test urgent triage for concerning symptoms."""
        agent = TriageAgent()
        msg = PatientMessage(
            message_id="msg-003",
            patient_id="pat-003",
            content="I have a high fever and severe pain in my side"
        )
        
        result = agent.triage_message(msg)
        
        assert result.urgency == UrgencyLevel.URGENT
        assert result.requires_human_review is True
        
    def test_triage_medication_question(self):
        """Test routine triage for medication question."""
        agent = TriageAgent()
        msg = PatientMessage(
            message_id="msg-004",
            patient_id="pat-004",
            content="Can I take my blood pressure medication with food?"
        )
        
        result = agent.triage_message(msg)
        
        assert result.category == MessageCategory.MEDICATION_QUESTION
        assert result.urgency in [UrgencyLevel.ROUTINE, UrgencyLevel.INFORMATIONAL]
        
    def test_triage_appointment_request(self):
        """Test triage for appointment request."""
        agent = TriageAgent()
        msg = PatientMessage(
            message_id="msg-005",
            patient_id="pat-005",
            content="I need to schedule an appointment to see my doctor"
        )
        
        result = agent.triage_message(msg)
        
        assert result.category == MessageCategory.APPOINTMENT_REQUEST


class TestHealthEducatorAgent:
    """Tests for HealthEducatorAgent class."""
    
    def test_init(self):
        """Test health educator initialization."""
        agent = HealthEducatorAgent()
        assert agent.name == "HealthEducatorAgent"
        assert len(agent.health_topics) > 0
        
    def test_respond_diabetes_query(self):
        """Test response to diabetes question."""
        agent = HealthEducatorAgent()
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="What should I know about managing my diabetes?"
        )
        
        response = agent.respond_to_query(msg)
        
        assert "diabetes" in response.content.lower()
        assert response.confidence > 0.7
        assert "⚠️" in response.content  # Disclaimer present
        
    def test_respond_hypertension_query(self):
        """Test response to hypertension question."""
        agent = HealthEducatorAgent()
        msg = PatientMessage(
            message_id="msg-002",
            patient_id="pat-002",
            content="How can I manage my hypertension better?"
        )
        
        response = agent.respond_to_query(msg)
        
        assert response.content is not None
        assert len(response.content) > 50
        
    def test_response_includes_disclaimer(self):
        """Test that all responses include disclaimer."""
        agent = HealthEducatorAgent()
        msg = PatientMessage(
            message_id="msg-003",
            patient_id="pat-003",
            content="Any random health question"
        )
        
        response = agent.respond_to_query(msg)
        
        assert "not a substitute for professional medical advice" in response.content


class TestFollowUpSchedulingAgent:
    """Tests for FollowUpSchedulingAgent class."""
    
    def test_init(self):
        """Test scheduling agent initialization."""
        agent = FollowUpSchedulingAgent()
        assert agent.name == "FollowUpSchedulingAgent"
        
    def test_detect_cancellation(self):
        """Test detecting cancellation request."""
        agent = FollowUpSchedulingAgent()
        
        assert agent._detect_scheduling_type("I need to cancel my appointment") == "cancellation"
        
    def test_detect_reschedule(self):
        """Test detecting reschedule request."""
        agent = FollowUpSchedulingAgent()
        
        assert agent._detect_scheduling_type("Can I reschedule my visit?") == "reschedule"
        
    def test_process_scheduling_request(self):
        """Test processing a scheduling request."""
        agent = FollowUpSchedulingAgent()
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="I'd like to schedule a follow-up appointment"
        )
        
        response = agent.process_scheduling_request(msg)
        
        assert response.agent_name == "FollowUpSchedulingAgent"
        assert "follow-up" in response.content.lower() or "appointment" in response.content.lower()


class TestClinicalOversightProxy:
    """Tests for ClinicalOversightProxy class."""
    
    def test_init(self):
        """Test oversight proxy initialization."""
        proxy = ClinicalOversightProxy()
        assert proxy.name == "ClinicalOversightProxy"
        assert len(proxy.pending_reviews) == 0
        
    def test_flag_for_review(self):
        """Test flagging a response for review."""
        proxy = ClinicalOversightProxy()
        
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="Test message"
        )
        response = AgentResponse(
            message_id="resp-001",
            agent_name="TestAgent",
            content="Test response"
        )
        
        review_id = proxy.flag_for_review(msg, response, "Test reason")
        
        assert review_id.startswith("review-")
        assert len(proxy.pending_reviews) == 1
        
    def test_get_pending_reviews(self):
        """Test getting pending reviews."""
        proxy = ClinicalOversightProxy()
        
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="Test"
        )
        response = AgentResponse(
            message_id="resp-001",
            agent_name="Test",
            content="Test"
        )
        
        proxy.flag_for_review(msg, response, "Reason 1")
        proxy.flag_for_review(msg, response, "Reason 2")
        
        pending = proxy.get_pending_reviews()
        
        assert len(pending) == 2
        
    def test_complete_review(self):
        """Test completing a review."""
        proxy = ClinicalOversightProxy()
        
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="Test"
        )
        response = AgentResponse(
            message_id="resp-001",
            agent_name="Test",
            content="Test"
        )
        
        review_id = proxy.flag_for_review(msg, response, "Test reason")
        
        result = proxy.complete_review(
            review_id=review_id,
            approved=True,
            reviewer_id="DR-001",
            notes="Looks good"
        )
        
        assert result is True
        assert len(proxy.get_pending_reviews()) == 0


class TestCommunicationOrchestrator:
    """Tests for CommunicationOrchestrator class."""
    
    def test_init(self):
        """Test orchestrator initialization."""
        orch = CommunicationOrchestrator()
        
        assert orch.triage_agent is not None
        assert orch.educator_agent is not None
        assert orch.scheduling_agent is not None
        assert orch.oversight_proxy is not None
        
    def test_process_emergency_message(self):
        """Test processing emergency message."""
        orch = CommunicationOrchestrator()
        
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="I'm having a heart attack, severe chest pain"
        )
        
        response = orch.process_message(msg)
        
        assert response.requires_clinician_review is True
        assert response.triage_result.urgency == UrgencyLevel.EMERGENCY
        assert "911" in response.content
        
    def test_process_scheduling_message(self):
        """Test processing scheduling message."""
        orch = CommunicationOrchestrator()
        
        msg = PatientMessage(
            message_id="msg-002",
            patient_id="pat-002",
            content="I need to schedule an appointment for next week"
        )
        
        response = orch.process_message(msg)
        
        assert response.triage_result.category == MessageCategory.APPOINTMENT_REQUEST
        
    def test_process_general_health_question(self):
        """Test processing general health question."""
        orch = CommunicationOrchestrator()
        
        msg = PatientMessage(
            message_id="msg-003",
            patient_id="pat-003",
            content="What can I do to better manage my diabetes?"
        )
        
        response = orch.process_message(msg)
        
        assert response.content is not None
        assert len(response.content) > 0
        
    def test_conversation_history_tracking(self):
        """Test that conversation history is tracked."""
        orch = CommunicationOrchestrator()
        
        msg1 = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="Hello, I have a question"
        )
        msg2 = PatientMessage(
            message_id="msg-002",
            patient_id="pat-001",
            content="What about my medication?"
        )
        
        orch.process_message(msg1)
        orch.process_message(msg2)
        
        history = orch.get_conversation_history("pat-001")
        
        # Should have 4 messages (2 patient + 2 agent)
        assert len(history) == 4
        
    def test_mental_health_crisis_response(self):
        """Test response to mental health crisis."""
        orch = CommunicationOrchestrator()
        
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="I've been feeling suicidal lately"
        )
        
        response = orch.process_message(msg)
        
        assert "988" in response.content  # Suicide hotline
        assert response.requires_clinician_review is True


class TestUrgencyLevel:
    """Tests for UrgencyLevel enum."""
    
    def test_urgency_values(self):
        """Test urgency level values."""
        assert UrgencyLevel.EMERGENCY.value == "emergency"
        assert UrgencyLevel.URGENT.value == "urgent"
        assert UrgencyLevel.ROUTINE.value == "routine"


class TestMessageCategory:
    """Tests for MessageCategory enum."""
    
    def test_category_values(self):
        """Test message category values."""
        assert MessageCategory.SYMPTOM_REPORT.value == "symptom_report"
        assert MessageCategory.MEDICATION_QUESTION.value == "medication_question"
        assert MessageCategory.EMERGENCY.value == "emergency"
