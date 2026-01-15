"""
Tests for the Workflow Agent (CrewAI) module.
"""

from datetime import datetime

import pytest

from medai_compass.agents.workflow import (
    AgentRole,
    AppointmentRequest,
    DocumentationRequest,
    DocumenterAgent,
    PriorAuthAgent,
    PriorAuthRequest,
    SchedulerAgent,
    WorkflowCrew,
    WorkflowResult,
)


class TestAppointmentRequest:
    """Tests for AppointmentRequest dataclass."""
    
    def test_create_basic_request(self):
        """Test creating a basic appointment request."""
        request = AppointmentRequest(
            patient_id="P001",
            appointment_type="follow_up",
            preferred_dates=["2026-02-01", "2026-02-02"]
        )
        
        assert request.patient_id == "P001"
        assert request.appointment_type == "follow_up"
        assert len(request.preferred_dates) == 2
        assert request.urgency == "routine"
        
    def test_urgent_request(self):
        """Test creating an urgent appointment request."""
        request = AppointmentRequest(
            patient_id="P002",
            appointment_type="specialist_consult",
            preferred_dates=["2026-01-20"],
            urgency="urgent",
            notes="Patient experiencing severe symptoms"
        )
        
        assert request.urgency == "urgent"
        assert request.notes is not None


class TestDocumentationRequest:
    """Tests for DocumentationRequest dataclass."""
    
    def test_create_discharge_summary_request(self):
        """Test creating a discharge summary request."""
        request = DocumentationRequest(
            patient_id="P001",
            document_type="discharge_summary",
            encounter_id="ENC-001",
            clinical_notes=["Patient admitted with chest pain", "Echo normal", "Discharged stable"],
            diagnoses=[{"code": "I20.9", "display": "Angina pectoris, unspecified"}],
            medications=[{"code": "12345", "display": "Aspirin 81mg"}]
        )
        
        assert request.document_type == "discharge_summary"
        assert len(request.clinical_notes) == 3
        assert len(request.diagnoses) == 1


class TestPriorAuthRequest:
    """Tests for PriorAuthRequest dataclass."""
    
    def test_create_prior_auth_request(self):
        """Test creating a prior auth request."""
        request = PriorAuthRequest(
            patient_id="P001",
            procedure_code="27447",
            diagnosis_codes=["M17.11", "M17.12"],
            insurance_id="INS-001",
            provider_id="DR-001",
            clinical_justification="Patient has failed conservative treatment for 6 months"
        )
        
        assert request.procedure_code == "27447"
        assert len(request.diagnosis_codes) == 2


class TestSchedulerAgent:
    """Tests for SchedulerAgent class."""
    
    def test_init(self):
        """Test scheduler agent initialization."""
        agent = SchedulerAgent()
        
        assert agent.role == AgentRole.SCHEDULER
        assert agent.model is None
        
    def test_process_routine_request(self):
        """Test processing a routine scheduling request."""
        agent = SchedulerAgent()
        request = AppointmentRequest(
            patient_id="P001",
            appointment_type="follow_up",
            preferred_dates=["2026-02-01"]
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert result.agent_role == AgentRole.SCHEDULER
        assert "scheduled_date" in result.output
        assert "confirmation_number" in result.output
        
    def test_process_urgent_request(self):
        """Test processing an urgent scheduling request."""
        agent = SchedulerAgent()
        request = AppointmentRequest(
            patient_id="P002",
            appointment_type="urgent_care",
            preferred_dates=["2026-01-20"],
            urgency="urgent"
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert result.processing_time_ms >= 0
        
    def test_estimate_duration(self):
        """Test appointment duration estimation."""
        agent = SchedulerAgent()
        
        assert agent._estimate_duration("new_patient") == 60
        assert agent._estimate_duration("follow_up") == 30
        assert agent._estimate_duration("surgery") == 180
        assert agent._estimate_duration("unknown") == 30  # default
        
    def test_get_instructions(self):
        """Test getting pre-appointment instructions."""
        agent = SchedulerAgent()
        
        instructions = agent._get_instructions("new_patient")
        assert "15 minutes early" in instructions
        
        instructions = agent._get_instructions("annual_exam")
        assert "fast" in instructions.lower()


class TestDocumenterAgent:
    """Tests for DocumenterAgent class."""
    
    def test_init(self):
        """Test documenter agent initialization."""
        agent = DocumenterAgent()
        
        assert agent.role == AgentRole.DOCUMENTER
        
    def test_generate_discharge_summary(self):
        """Test generating a discharge summary."""
        agent = DocumenterAgent()
        request = DocumentationRequest(
            patient_id="P001",
            document_type="discharge_summary",
            encounter_id="ENC-001",
            clinical_notes=["Patient admitted with pneumonia", "Treated with antibiotics", "Symptoms resolved"],
            diagnoses=[{"code": "J18.9", "display": "Pneumonia, unspecified organism"}],
            medications=[{"code": "123", "display": "Amoxicillin 500mg"}]
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert result.output["document_type"] == "discharge_summary"
        assert "DISCHARGE SUMMARY" in result.output["content"]
        assert "Pneumonia" in result.output["content"]
        
    def test_generate_progress_note(self):
        """Test generating a progress note."""
        agent = DocumenterAgent()
        request = DocumentationRequest(
            patient_id="P002",
            document_type="progress_note",
            encounter_id="ENC-002",
            clinical_notes=["Patient reports improvement", "Vitals stable", "Continue current treatment"],
            diagnoses=[{"code": "I10", "display": "Essential hypertension"}]
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert "PROGRESS NOTE" in result.output["content"]
        
    def test_generate_referral_letter(self):
        """Test generating a referral letter."""
        agent = DocumenterAgent()
        request = DocumentationRequest(
            patient_id="P003",
            document_type="referral_letter",
            encounter_id="ENC-003",
            clinical_notes=["Needs cardiology evaluation for arrhythmia"],
            diagnoses=[{"code": "I49.9", "display": "Cardiac arrhythmia, unspecified"}]
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert "REFERRAL" in result.output["content"]
        
    def test_summarize_clinical_notes(self):
        """Test clinical notes summarization."""
        agent = DocumenterAgent()
        
        notes = ["First note", "Second note", "Third note"]
        summary = agent.summarize_clinical_notes(notes)
        
        assert "First" in summary
        assert "Second" in summary
        
    def test_summarize_empty_notes(self):
        """Test summarization with empty notes."""
        agent = DocumenterAgent()
        
        summary = agent.summarize_clinical_notes([])
        assert "No clinical notes" in summary


class TestPriorAuthAgent:
    """Tests for PriorAuthAgent class."""
    
    def test_init(self):
        """Test prior auth agent initialization."""
        agent = PriorAuthAgent()
        
        assert agent.role == AgentRole.PRIOR_AUTH
        assert len(agent.auth_required_procedures) > 0
        
    def test_check_auth_required(self):
        """Test checking if auth is required."""
        agent = PriorAuthAgent()
        
        # Known procedures requiring auth
        assert agent._check_auth_required("27447") is True
        assert agent._check_auth_required("70553") is True
        
        # Unknown procedure
        assert agent._check_auth_required("99999") is False
        
    def test_process_auth_not_required(self):
        """Test processing when auth is not required."""
        agent = PriorAuthAgent()
        request = PriorAuthRequest(
            patient_id="P001",
            procedure_code="99213",  # Office visit - no auth needed
            diagnosis_codes=["Z00.00"],
            insurance_id="INS-001",
            provider_id="DR-001",
            clinical_justification="Annual exam"
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert result.output["status"] == "not_required"
        
    def test_process_strong_justification(self):
        """Test processing with strong clinical justification."""
        agent = PriorAuthAgent()
        request = PriorAuthRequest(
            patient_id="P001",
            procedure_code="27447",  # Total knee replacement
            diagnosis_codes=["M17.11"],
            insurance_id="INS-001",
            provider_id="DR-001",
            clinical_justification="Medical necessity established. Patient has failed conservative treatment including physical therapy for 6 months. Previous diagnostic findings show severe osteoarthritis. Treatment failure documented."
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert result.output["justification_score"] >= 0.7
        
    def test_process_weak_justification(self):
        """Test processing with weak clinical justification."""
        agent = PriorAuthAgent()
        request = PriorAuthRequest(
            patient_id="P002",
            procedure_code="27447",
            diagnosis_codes=["M17.11"],
            insurance_id="INS-002",
            provider_id="DR-002",
            clinical_justification="Knee hurts"
        )
        
        result = agent.process_request(request)
        
        assert result.success is True
        assert result.output["justification_score"] < 0.7
        assert result.output["status"] in ["pending_review", "needs_info"]
        
    def test_validate_justification_scoring(self):
        """Test justification validation scoring."""
        agent = PriorAuthAgent()
        
        # Strong justification
        score = agent._validate_justification(
            "Medical necessity confirmed. Patient symptoms include chronic pain. Failed conservative treatment over 12 months.",
            ["M17.11", "M17.12"],
            "27447"
        )
        assert score > 0.6
        
        # Weak justification
        score = agent._validate_justification(
            "Need MRI",
            [],
            "70553"
        )
        assert score <= 0.6


class TestWorkflowCrew:
    """Tests for WorkflowCrew class."""
    
    def test_init(self):
        """Test crew initialization."""
        crew = WorkflowCrew()
        
        assert crew.scheduler is not None
        assert crew.documenter is not None
        assert crew.prior_auth is not None
        
    def test_process_scheduling(self):
        """Test processing scheduling through crew."""
        crew = WorkflowCrew()
        request = AppointmentRequest(
            patient_id="P001",
            appointment_type="follow_up",
            preferred_dates=["2026-02-01"]
        )
        
        result = crew.process_scheduling(request)
        
        assert result.success is True
        assert result.agent_role == AgentRole.SCHEDULER
        
    def test_process_documentation(self):
        """Test processing documentation through crew."""
        crew = WorkflowCrew()
        request = DocumentationRequest(
            patient_id="P001",
            document_type="progress_note",
            encounter_id="ENC-001",
            clinical_notes=["Patient stable"],
            diagnoses=[{"display": "Hypertension"}]
        )
        
        result = crew.process_documentation(request)
        
        assert result.success is True
        assert result.agent_role == AgentRole.DOCUMENTER
        
    def test_process_prior_auth(self):
        """Test processing prior auth through crew."""
        crew = WorkflowCrew()
        request = PriorAuthRequest(
            patient_id="P001",
            procedure_code="27447",
            diagnosis_codes=["M17.11"],
            insurance_id="INS-001",
            provider_id="DR-001",
            clinical_justification="Medical necessity established"
        )
        
        result = crew.process_prior_auth(request)
        
        assert result.success is True
        assert result.agent_role == AgentRole.PRIOR_AUTH
        
    def test_complex_workflow_with_prior_auth(self):
        """Test complex workflow involving multiple agents."""
        crew = WorkflowCrew()
        
        results = crew.process_complex_workflow(
            scheduling_request=AppointmentRequest(
                patient_id="P001",
                appointment_type="procedure",
                preferred_dates=["2026-02-15"]
            ),
            prior_auth_request=PriorAuthRequest(
                patient_id="P001",
                procedure_code="27447",
                diagnosis_codes=["M17.11"],
                insurance_id="INS-001",
                provider_id="DR-001",
                clinical_justification="Medical necessity established with documented treatment failure"
            )
        )
        
        assert "prior_auth" in results
        assert "scheduling" in results
        
    def test_complex_workflow_blocked_by_auth(self):
        """Test that scheduling is blocked when prior auth fails."""
        crew = WorkflowCrew()
        
        results = crew.process_complex_workflow(
            scheduling_request=AppointmentRequest(
                patient_id="P001",
                appointment_type="procedure",
                preferred_dates=["2026-02-15"]
            ),
            prior_auth_request=PriorAuthRequest(
                patient_id="P001",
                procedure_code="27447",
                diagnosis_codes=["M17.11"],
                insurance_id="INS-001",
                provider_id="DR-001",
                clinical_justification="Pain"  # Weak justification
            )
        )
        
        # If auth needs info, scheduling should be blocked
        if results["prior_auth"].output.get("status") == "needs_info":
            assert results["scheduling"].success is False
            assert "blocked" in results["scheduling"].errors[0].lower()


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""
    
    def test_create_success_result(self):
        """Test creating a successful result."""
        result = WorkflowResult(
            success=True,
            agent_role=AgentRole.SCHEDULER,
            task_id="test-001",
            output={"scheduled": True}
        )
        
        assert result.success is True
        assert result.errors == []
        assert result.timestamp is not None
        
    def test_create_failure_result(self):
        """Test creating a failure result."""
        result = WorkflowResult(
            success=False,
            agent_role=AgentRole.DOCUMENTER,
            task_id="test-002",
            output={},
            errors=["Document generation failed"]
        )
        
        assert result.success is False
        assert len(result.errors) == 1
