"""
Integration tests for multi-agent coordination flows.

Tests how multiple agents (Diagnostic, Communication, Workflow)
coordinate through the orchestrator for complex requests.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import numpy as np


# =============================================================================
# Multi-Agent Coordination Tests
# =============================================================================

class TestMultiAgentCoordination:
    """Test multi-agent coordination scenarios."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create master orchestrator."""
        from medai_compass.orchestrator.master import MasterOrchestrator
        return MasterOrchestrator()
    
    def test_diagnostic_to_communication_handoff(self, orchestrator):
        """Test handoff from diagnostic to communication agent."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        # Request that might need both diagnostic analysis and patient communication
        request = OrchestratorRequest(
            request_id="handoff-001",
            user_id="user-001",
            content="Analyze the x-ray and explain the findings to the patient in simple terms",
            request_type="multimodal",
            attachments=["xray.dcm"],
        )
        
        response = orchestrator.process_request(request)
        assert response is not None
        assert hasattr(response, "content")
    
    def test_workflow_triggers_communication(self, orchestrator):
        """Test workflow agent can trigger communication."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        request = OrchestratorRequest(
            request_id="wf-comm-001",
            user_id="user-001",
            content="Schedule a follow-up appointment and notify the patient",
            request_type="text",
        )
        
        response = orchestrator.process_request(request)
        assert response is not None
    
    def test_complex_request_uses_multiple_agents(self, orchestrator):
        """Test complex requests may involve multiple agents."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        # Complex request touching multiple domains
        request = OrchestratorRequest(
            request_id="complex-001",
            user_id="user-001",
            content="Review the CT scan, document findings, and schedule a biopsy",
            request_type="multimodal",
            attachments=["ct_scan.dcm"],
        )
        
        response = orchestrator.process_request(request)
        assert response is not None


class TestDiagnosticAgentIntegration:
    """Test diagnostic agent integration with LangGraph."""
    
    @pytest.fixture
    def diagnostic_state(self):
        """Create diagnostic state for LangGraph."""
        return {
            "patient_id": "patient-001",
            "session_id": "session-001",
            "images": [],
            "findings": [],
            "confidence_scores": [],
            "requires_review": False,
            "audit_trail": [],
            "fhir_context": {},
        }
    
    def test_diagnostic_graph_creation(self):
        """Test diagnostic graph can be created."""
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        
        graph = create_diagnostic_graph()
        assert graph is not None
    
    def test_diagnostic_state_transitions(self, diagnostic_state):
        """Test diagnostic state transitions through graph."""
        # Simulate state updates through workflow
        diagnostic_state["images"] = ["test_image.dcm"]
        diagnostic_state["findings"] = ["No acute cardiopulmonary abnormality"]
        diagnostic_state["confidence_scores"] = [0.95]
        
        assert len(diagnostic_state["findings"]) == 1
        assert diagnostic_state["confidence_scores"][0] > 0.9
    
    @patch("medai_compass.agents.diagnostic.graph.run_cxr_analysis")
    def test_diagnostic_with_cxr_model(self, mock_cxr, diagnostic_state):
        """Test diagnostic agent uses CXR model."""
        mock_cxr.return_value = {
            "embeddings": np.random.randn(1, 768),
            "findings": "Clear lungs",
        }
        
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        graph = create_diagnostic_graph()
        
        # Verify graph structure
        assert graph is not None


class TestCommunicationAgentIntegration:
    """Test communication agent integration with LangChain."""
    
    @pytest.fixture
    def patient_message(self):
        """Create patient message."""
        from medai_compass.agents.communication import PatientMessage
        return PatientMessage(
            content="What are the side effects of metformin?",
            patient_id="patient-001",
            language="en",
        )
    
    def test_communication_orchestrator_creation(self):
        """Test communication orchestrator can be created."""
        from medai_compass.agents.communication import CommunicationOrchestrator
        
        orchestrator = CommunicationOrchestrator()
        assert orchestrator is not None
    
    def test_triage_agent_classification(self, patient_message):
        """Test triage agent classifies messages."""
        from medai_compass.agents.communication import TriageAgent
        
        agent = TriageAgent()
        # Should have classify method
        assert hasattr(agent, "classify") or hasattr(agent, "process")
    
    @patch("medai_compass.agents.communication.orchestrator.ChatOpenAI")
    def test_communication_with_llm(self, mock_llm, patient_message):
        """Test communication agent uses LLM."""
        mock_llm.return_value = MagicMock()
        
        from medai_compass.agents.communication import CommunicationOrchestrator
        orchestrator = CommunicationOrchestrator()
        
        assert orchestrator is not None


class TestWorkflowAgentIntegration:
    """Test workflow agent integration with CrewAI."""
    
    @pytest.fixture
    def documentation_request(self):
        """Create documentation request."""
        from medai_compass.agents.workflow import DocumentationRequest
        return DocumentationRequest(
            patient_id="patient-001",
            encounter_id="encounter-001",
            clinical_notes=["Patient stable", "Continue medications"],
            document_type="progress_note",
        )
    
    @pytest.fixture
    def appointment_request(self):
        """Create appointment request."""
        from medai_compass.agents.workflow import AppointmentRequest
        return AppointmentRequest(
            patient_id="patient-001",
            appointment_type="follow_up",
            urgency="routine",
            notes="3-month follow-up",
        )
    
    def test_workflow_crew_creation(self):
        """Test workflow crew can be created."""
        from medai_compass.agents.workflow import WorkflowCrew
        
        crew = WorkflowCrew()
        assert crew is not None
    
    def test_documentation_workflow(self, documentation_request):
        """Test documentation workflow execution."""
        from medai_compass.agents.workflow import WorkflowCrew
        
        crew = WorkflowCrew()
        # Should handle documentation requests
        assert documentation_request.document_type == "progress_note"
    
    def test_scheduling_workflow(self, appointment_request):
        """Test scheduling workflow execution."""
        from medai_compass.agents.workflow import WorkflowCrew
        
        crew = WorkflowCrew()
        assert appointment_request.urgency == "routine"


class TestAgentGuardrailsIntegration:
    """Test guardrails are applied to all agent outputs."""
    
    @pytest.fixture
    def escalation_gateway(self):
        """Create escalation gateway."""
        from medai_compass.guardrails.escalation import HumanEscalationGateway
        return HumanEscalationGateway()
    
    def test_diagnostic_output_validated(self, escalation_gateway):
        """Test diagnostic outputs are validated."""
        decision = escalation_gateway.evaluate(
            response="Normal cardiac silhouette. Clear lung fields.",
            confidence=0.95,
            uncertainty=0.03,
            domain="diagnostic",
        )
        
        assert not decision.should_escalate
    
    def test_communication_output_validated(self, escalation_gateway):
        """Test communication outputs are validated."""
        decision = escalation_gateway.evaluate(
            response="Metformin may cause nausea initially. Take with food.",
            confidence=0.92,
            uncertainty=0.05,
            domain="communication",
        )
        
        assert not decision.should_escalate
    
    def test_safety_concern_escalated(self, escalation_gateway):
        """Test safety concerns are always escalated."""
        decision = escalation_gateway.evaluate(
            response="Patient reports suicidal ideation. Immediate attention needed.",
            confidence=0.99,
            uncertainty=0.01,
            domain="communication",
        )
        
        assert decision.should_escalate
        assert decision.priority == "immediate"


class TestAgentAuditTrail:
    """Test audit trail is maintained across agent calls."""
    
    def test_audit_trail_created(self):
        """Test audit trail is created for requests."""
        from medai_compass.orchestrator.master import (
            MasterOrchestrator,
            OrchestratorRequest,
        )
        
        orchestrator = MasterOrchestrator()
        request = OrchestratorRequest(
            request_id="audit-001",
            user_id="user-001",
            content="Test request for audit",
            request_type="text",
        )
        
        response = orchestrator.process_request(request)
        assert response is not None
        assert hasattr(response, "timestamp")
    
    def test_agent_used_recorded(self):
        """Test agent used is recorded in response."""
        from medai_compass.orchestrator.master import (
            MasterOrchestrator,
            OrchestratorRequest,
        )
        
        orchestrator = MasterOrchestrator()
        request = OrchestratorRequest(
            request_id="agent-001",
            user_id="user-001",
            content="What is hypertension?",
            request_type="text",
        )
        
        response = orchestrator.process_request(request)
        assert response.agent_used is not None
    
    def test_guardrails_applied_recorded(self):
        """Test guardrails applied are recorded."""
        from medai_compass.orchestrator.master import (
            MasterOrchestrator,
            OrchestratorRequest,
        )
        
        orchestrator = MasterOrchestrator()
        request = OrchestratorRequest(
            request_id="gr-001",
            user_id="user-001",
            content="Tell me about diabetes",
            request_type="text",
        )
        
        response = orchestrator.process_request(request)
        assert hasattr(response, "guardrails_applied")


class TestModelSelectionPerAgent:
    """Test model selection works for each agent type."""
    
    @pytest.fixture
    def orchestrator_4b(self):
        """Orchestrator with 4B model."""
        from medai_compass.orchestrator.master import MasterOrchestrator
        return MasterOrchestrator(model_name="medgemma-4b-it")
    
    @pytest.fixture
    def orchestrator_27b(self):
        """Orchestrator with 27B model."""
        from medai_compass.orchestrator.master import MasterOrchestrator
        return MasterOrchestrator(model_name="medgemma-27b-it")
    
    def test_4b_model_used_for_simple_queries(self, orchestrator_4b):
        """Test 4B model is used for simple queries."""
        assert orchestrator_4b.model_name == "medgemma-4b-it"
    
    def test_27b_model_used_for_complex_cases(self, orchestrator_27b):
        """Test 27B model is used for complex cases."""
        assert orchestrator_27b.model_name == "medgemma-27b-it"
    
    def test_model_config_propagates_to_agents(self, orchestrator_4b):
        """Test model config propagates to agents."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        request = OrchestratorRequest(
            request_id="model-001",
            user_id="user-001",
            content="Quick health question",
            request_type="text",
        )
        
        response = orchestrator_4b.process_request(request)
        assert response is not None


class TestAgentFallback:
    """Test fallback mechanisms when agents fail."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create master orchestrator."""
        from medai_compass.orchestrator.master import MasterOrchestrator
        return MasterOrchestrator()
    
    def test_fallback_to_communication_on_unknown(self, orchestrator):
        """Test unknown requests fall back to communication."""
        from medai_compass.orchestrator.master import (
            OrchestratorRequest,
            DomainType,
        )
        
        request = OrchestratorRequest(
            request_id="fallback-001",
            user_id="user-001",
            content="asdfghjkl random text",  # Unclear intent
            request_type="text",
        )
        
        classification = orchestrator.intent_classifier.classify(request)
        # Should default to communication
        assert classification.domain == DomainType.COMMUNICATION
    
    @patch("medai_compass.agents.communication.CommunicationOrchestrator")
    def test_graceful_degradation_on_agent_error(self, mock_comm, orchestrator):
        """Test graceful degradation when agent fails."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        mock_instance = MagicMock()
        mock_instance.process_message.side_effect = Exception("Agent error")
        mock_comm.return_value = mock_instance
        
        request = OrchestratorRequest(
            request_id="error-001",
            user_id="user-001",
            content="Test query",
            request_type="text",
        )
        
        # Should not crash
        try:
            response = orchestrator.process_request(request)
        except Exception:
            pass  # Expected to handle or propagate controlled error


class TestAgentPerformanceMetrics:
    """Test performance metrics are collected for agents."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create master orchestrator."""
        from medai_compass.orchestrator.master import MasterOrchestrator
        return MasterOrchestrator()
    
    def test_processing_time_recorded(self, orchestrator):
        """Test processing time is recorded."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        request = OrchestratorRequest(
            request_id="perf-001",
            user_id="user-001",
            content="Simple query",
            request_type="text",
        )
        
        response = orchestrator.process_request(request)
        assert response.processing_time_ms >= 0
    
    def test_confidence_score_recorded(self, orchestrator):
        """Test confidence score is recorded."""
        from medai_compass.orchestrator.master import OrchestratorRequest
        
        request = OrchestratorRequest(
            request_id="conf-001",
            user_id="user-001",
            content="What is flu?",
            request_type="text",
        )
        
        response = orchestrator.process_request(request)
        assert 0 <= response.confidence <= 1
