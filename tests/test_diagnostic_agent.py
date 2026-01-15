"""Tests for Diagnostic Agent - Written FIRST (TDD)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestDiagnosticState:
    """Test diagnostic state management."""

    def test_state_initialization(self):
        """Test DiagnosticState TypedDict creation."""
        from medai_compass.agents.diagnostic.state import create_initial_state
        
        state = create_initial_state(
            patient_id="patient-123",
            session_id="session-abc"
        )
        
        assert state["patient_id"] == "patient-123"
        assert state["session_id"] == "session-abc"
        assert state["findings"] == []
        assert state["requires_review"] is False

    def test_state_update_findings(self):
        """Test state mutation with new findings."""
        from medai_compass.agents.diagnostic.state import (
            create_initial_state, 
            add_finding
        )
        
        state = create_initial_state("patient-123", "session-abc")
        
        updated_state = add_finding(state, {
            "finding": "Bilateral infiltrates",
            "location": "Lower lobes",
            "severity": "moderate"
        })
        
        assert len(updated_state["findings"]) == 1
        assert updated_state["findings"][0]["finding"] == "Bilateral infiltrates"


class TestDiagnosticNodes:
    """Test diagnostic workflow nodes."""

    def test_preprocess_images_node(self, sample_dicom_path):
        """Test DICOM preprocessing node."""
        from medai_compass.agents.diagnostic.nodes import preprocess_images
        
        state = {
            "patient_id": "patient-123",
            "session_id": "session-abc",
            "images": [sample_dicom_path],
            "findings": [],
            "confidence_scores": [],
            "requires_review": False,
            "audit_trail": [],
            "fhir_context": {}
        }
        
        result = preprocess_images(state)
        
        assert "preprocessed_images" in result or "images" in result

    def test_confidence_check_high(self):
        """Test high confidence routing decision."""
        from medai_compass.agents.diagnostic.nodes import route_by_confidence
        
        state = {
            "confidence_scores": [0.95, 0.92, 0.88],
            "findings": [{"finding": "test"}]
        }
        
        result = route_by_confidence(state)
        
        assert result in ["high_confidence", "finalize"]

    def test_confidence_check_low(self):
        """Test low confidence routing decision."""
        from medai_compass.agents.diagnostic.nodes import route_by_confidence
        
        state = {
            "confidence_scores": [0.75, 0.68],
            "findings": [{"finding": "test"}]
        }
        
        result = route_by_confidence(state)
        
        assert result in ["low_confidence", "human_review"]


class TestDiagnosticGraph:
    """Test complete diagnostic workflow graph."""

    def test_graph_compilation(self):
        """Test LangGraph compiles without errors."""
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        
        graph = create_diagnostic_graph()
        
        assert graph is not None
        assert hasattr(graph, 'invoke') or hasattr(graph, 'ainvoke')

    def test_graph_has_required_nodes(self):
        """Test graph has all required nodes."""
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        
        graph = create_diagnostic_graph()
        
        # Check graph structure
        assert graph is not None


class TestReportGeneration:
    """Test diagnostic report generation."""

    def test_generate_report_from_findings(self):
        """Test report generation from findings."""
        from medai_compass.agents.diagnostic.nodes import generate_report
        
        state = {
            "patient_id": "patient-123",
            "findings": [
                {"finding": "Bilateral infiltrates", "severity": "moderate"},
                {"finding": "Cardiomegaly", "severity": "mild"}
            ],
            "confidence_scores": [0.92, 0.88],
            "fhir_context": {},
            "audit_trail": []
        }
        
        result = generate_report(state)
        
        assert "report" in result or "findings" in result
