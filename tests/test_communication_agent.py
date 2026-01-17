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


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_init(self):
        """Test conversation manager initialization."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(max_history_length=50)

        assert manager.max_history_length == 50
        assert len(manager.conversations) == 0

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a new conversation session."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(use_persistence=False)
        context = await manager.create_session(
            patient_id="pat-001",
            patient_context={
                "conditions": ["diabetes", "hypertension"],
                "medications": ["metformin", "lisinopril"]
            }
        )

        assert context.patient_id == "pat-001"
        assert "diabetes" in context.active_conditions
        assert "metformin" in context.current_medications
        assert context.session_id.startswith("conv-pat-001-")

    @pytest.mark.asyncio
    async def test_add_message(self):
        """Test adding messages to conversation history."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(use_persistence=False)
        await manager.create_session("pat-001")

        success = await manager.add_message(
            patient_id="pat-001",
            role="patient",
            content="I have a question about my medication."
        )

        assert success is True

        context = await manager.get_session("pat-001")
        assert len(context.messages) == 1
        assert context.messages[0]["role"] == "patient"

    @pytest.mark.asyncio
    async def test_add_message_no_session(self):
        """Test adding message without a session returns False."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(use_persistence=False)
        success = await manager.add_message(
            patient_id="nonexistent",
            role="patient",
            content="Test"
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_max_history_length_enforcement(self):
        """Test that history is trimmed when max length exceeded."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(max_history_length=5, use_persistence=False)
        await manager.create_session("pat-001")

        # Add 7 messages
        for i in range(7):
            await manager.add_message("pat-001", "patient", f"Message {i}")

        context = await manager.get_session("pat-001")
        assert len(context.messages) == 5  # Should be trimmed
        assert "Message 2" in context.messages[0]["content"]  # First message should be removed

    @pytest.mark.asyncio
    async def test_get_context_for_prompt(self):
        """Test generating context string for prompts."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(use_persistence=False)
        await manager.create_session(
            "pat-001",
            patient_context={
                "conditions": ["diabetes"],
                "medications": ["metformin"]
            }
        )
        await manager.add_message("pat-001", "patient", "I have a question.")
        await manager.add_message("pat-001", "agent", "How can I help?")

        context_str = manager.get_context_for_prompt("pat-001", max_messages=5)

        assert "diabetes" in context_str
        assert "metformin" in context_str
        assert "patient:" in context_str.lower()
        assert "agent:" in context_str.lower()

    @pytest.mark.asyncio
    async def test_end_session(self):
        """Test ending a session."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(use_persistence=False)
        await manager.create_session("pat-001")
        await manager.add_message("pat-001", "patient", "Test")

        final_state = await manager.end_session("pat-001")

        assert final_state is not None
        assert final_state["patient_id"] == "pat-001"
        assert final_state["message_count"] == 1
        assert await manager.get_session("pat-001") is None

    @pytest.mark.asyncio
    async def test_end_session_nonexistent(self):
        """Test ending nonexistent session returns None."""
        from medai_compass.agents.communication.agents import ConversationManager

        manager = ConversationManager(use_persistence=False)
        result = await manager.end_session("nonexistent")

        assert result is None


class TestConversationSummary:
    """Tests for conversation summary functionality."""
    
    def test_get_conversation_summary(self):
        """Test getting conversation summary."""
        orch = CommunicationOrchestrator()
        
        msg1 = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="I need to schedule an appointment"
        )
        msg2 = PatientMessage(
            message_id="msg-002",
            patient_id="pat-001",
            content="Also, question about my medication"
        )
        
        orch.process_message(msg1)
        orch.process_message(msg2)
        
        summary = orch.get_conversation_summary("pat-001")
        
        assert summary["patient_id"] == "pat-001"
        assert summary["total_messages"] == 4
        assert summary["patient_messages"] == 2
        assert "appointment" in summary["topics_discussed"] or "medication" in summary["topics_discussed"]
    
    def test_clear_conversation(self):
        """Test clearing conversation history."""
        orch = CommunicationOrchestrator()
        
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="Hello"
        )
        orch.process_message(msg)
        
        # Verify conversation exists
        assert len(orch.get_conversation_history("pat-001")) > 0
        
        # Clear it
        result = orch.clear_conversation("pat-001")
        
        assert result is True
        assert orch.get_conversation_history("pat-001") == []
    
    def test_export_conversation(self):
        """Test exporting conversation data."""
        orch = CommunicationOrchestrator()
        
        msg = PatientMessage(
            message_id="msg-001",
            patient_id="pat-001",
            content="Test message"
        )
        orch.process_message(msg, patient_context={
            "conditions": ["diabetes"],
            "medications": ["metformin"]
        })
        
        export = orch.export_conversation("pat-001")
        
        assert export["patient_id"] == "pat-001"
        assert "messages" in export
        assert "exported_at" in export
        assert export["active_conditions"] == ["diabetes"]


class TestMultiLanguageSupport:
    """Tests for multi-language support."""
    
    def test_detect_english(self):
        """Test detecting English language."""
        from medai_compass.agents.communication.agents import MultiLanguageSupport
        
        support = MultiLanguageSupport()
        lang = support.detect_language("I have a headache")
        
        assert lang == "en"
    
    def test_detect_spanish(self):
        """Test detecting Spanish language."""
        from medai_compass.agents.communication.agents import MultiLanguageSupport
        
        support = MultiLanguageSupport()
        lang = support.detect_language("Hola, tengo dolor de cabeza")
        
        assert lang == "es"
    
    def test_get_phrase_english(self):
        """Test getting phrase in English."""
        from medai_compass.agents.communication.agents import MultiLanguageSupport
        
        support = MultiLanguageSupport()
        phrase = support.get_phrase("disclaimer", "en")
        
        assert "educational purposes" in phrase
    
    def test_get_phrase_spanish(self):
        """Test getting phrase in Spanish."""
        from medai_compass.agents.communication.agents import MultiLanguageSupport
        
        support = MultiLanguageSupport()
        phrase = support.get_phrase("disclaimer", "es")
        
        assert "educativos" in phrase
    
    def test_is_language_supported(self):
        """Test checking language support."""
        from medai_compass.agents.communication.agents import MultiLanguageSupport
        
        support = MultiLanguageSupport()
        
        assert support.is_language_supported("en") is True
        assert support.is_language_supported("es") is True
        assert support.is_language_supported("xx") is False
