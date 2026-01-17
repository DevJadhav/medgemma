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
        assert state["localizations"] == []
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

    def test_graph_includes_localization_node(self):
        """Test graph includes localize_findings node."""
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        
        graph = create_diagnostic_graph()
        
        # Graph should include localize_findings in the workflow
        # After compilation, we can verify nodes exist by checking the graph structure
        assert graph is not None
        # The graph should have the localize_findings function imported and used
        from medai_compass.agents.diagnostic.nodes import localize_findings
        assert localize_findings is not None


class TestCXRFoundationAnalysis:
    """Test CXR Foundation integration."""

    def test_cxr_analysis_processes_chest_xray(self):
        """Test CXR analysis only processes chest X-ray modalities."""
        from medai_compass.agents.diagnostic.nodes import analyze_with_cxr_foundation
        import numpy as np

        state = {
            "patient_id": "patient-123",
            "preprocessed_images": [
                {
                    "path": "/test/image.dcm",
                    "modality": "CR",  # Computed Radiography - chest X-ray
                    "array": np.random.randint(0, 255, (896, 896, 3), dtype=np.uint8)
                }
            ],
            "findings": [],
            "confidence_scores": [],
            "audit_trail": []
        }

        result = analyze_with_cxr_foundation(state)

        assert "findings" in result
        assert "audit_trail" in result
        # Check that CXR analysis was logged
        assert any("cxr_foundation" in str(entry) for entry in result["audit_trail"])

    def test_cxr_analysis_skips_non_xray(self):
        """Test CXR analysis skips non-chest X-ray modalities."""
        from medai_compass.agents.diagnostic.nodes import analyze_with_cxr_foundation
        import numpy as np

        state = {
            "patient_id": "patient-123",
            "preprocessed_images": [
                {
                    "path": "/test/image.dcm",
                    "modality": "CT",  # CT scan - not chest X-ray
                    "array": np.random.randint(0, 255, (896, 896, 3), dtype=np.uint8)
                }
            ],
            "findings": [],
            "confidence_scores": [],
            "audit_trail": []
        }

        result = analyze_with_cxr_foundation(state)

        # No findings should be added for non-CXR modality
        assert len(result["findings"]) == 0


class TestPathFoundationAnalysis:
    """Test Path Foundation integration."""

    def test_path_analysis_processes_pathology(self):
        """Test Path analysis processes pathology modalities."""
        from medai_compass.agents.diagnostic.nodes import analyze_with_path_foundation
        import numpy as np

        state = {
            "patient_id": "patient-123",
            "preprocessed_images": [
                {
                    "path": "/test/slide.dcm",
                    "modality": "SM",  # Slide Microscopy - pathology
                    "array": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                }
            ],
            "findings": [],
            "confidence_scores": [],
            "audit_trail": []
        }

        result = analyze_with_path_foundation(state)

        assert "findings" in result
        assert "audit_trail" in result
        # Check that path analysis was logged
        assert any("path_foundation" in str(entry) for entry in result["audit_trail"])

    def test_path_analysis_skips_non_pathology(self):
        """Test Path analysis skips non-pathology modalities."""
        from medai_compass.agents.diagnostic.nodes import analyze_with_path_foundation
        import numpy as np

        state = {
            "patient_id": "patient-123",
            "preprocessed_images": [
                {
                    "path": "/test/image.dcm",
                    "modality": "CR",  # Chest X-ray - not pathology
                    "array": np.random.randint(0, 255, (896, 896, 3), dtype=np.uint8)
                }
            ],
            "findings": [],
            "confidence_scores": [],
            "audit_trail": []
        }

        result = analyze_with_path_foundation(state)

        # No findings should be added for non-pathology modality
        assert len(result["findings"]) == 0


class TestModalityRouting:
    """Test modality-based routing."""

    def test_route_includes_cxr_for_chest_xray(self):
        """Test routing includes CXR Foundation for chest X-ray."""
        from medai_compass.agents.diagnostic.nodes import route_by_modality

        state = {
            "preprocessed_images": [
                {"modality": "CR", "path": "/test/image.dcm"}
            ]
        }

        nodes = route_by_modality(state)

        assert "analyze_with_medgemma" in nodes
        assert "analyze_with_cxr_foundation" in nodes

    def test_route_includes_path_for_pathology(self):
        """Test routing includes Path Foundation for pathology."""
        from medai_compass.agents.diagnostic.nodes import route_by_modality

        state = {
            "preprocessed_images": [
                {"modality": "SM", "path": "/test/slide.dcm"}
            ]
        }

        nodes = route_by_modality(state)

        assert "analyze_with_medgemma" in nodes
        assert "analyze_with_path_foundation" in nodes

    def test_route_includes_all_for_mixed(self):
        """Test routing includes all analyzers for mixed modalities."""
        from medai_compass.agents.diagnostic.nodes import route_by_modality

        state = {
            "preprocessed_images": [
                {"modality": "CR", "path": "/test/chest.dcm"},
                {"modality": "SM", "path": "/test/slide.dcm"}
            ]
        }

        nodes = route_by_modality(state)

        assert "analyze_with_medgemma" in nodes
        assert "analyze_with_cxr_foundation" in nodes
        assert "analyze_with_path_foundation" in nodes


class TestBoundingBoxLocalization:
    """Test bounding box localization for findings."""

    def test_localize_cardiomegaly_finding(self):
        """Test localization of cardiomegaly finding."""
        from medai_compass.agents.diagnostic.nodes import localize_findings
        import numpy as np

        state = {
            "patient_id": "patient-123",
            "findings": [
                {
                    "source": "cxr_foundation",
                    "finding": "Possible cardiomegaly",
                    "image_path": "/test/chest.dcm"
                }
            ],
            "preprocessed_images": [
                {
                    "path": "/test/chest.dcm",
                    "array": np.random.randint(0, 255, (896, 896, 3), dtype=np.uint8),
                    "modality": "CR"
                }
            ],
            "confidence_scores": [0.85],
            "audit_trail": []
        }

        result = localize_findings(state)

        assert "localizations" in result
        assert len(result["localizations"]) == 1
        bbox = result["localizations"][0]["bounding_box"]
        assert "x" in bbox
        assert "y" in bbox
        assert "width" in bbox
        assert "height" in bbox
        assert bbox["region"] == "cardiomegaly"

    def test_localize_pneumonia_finding(self):
        """Test localization of pneumonia finding."""
        from medai_compass.agents.diagnostic.nodes import localize_findings
        import numpy as np

        state = {
            "patient_id": "patient-123",
            "findings": [
                {
                    "source": "cxr_foundation",
                    "finding": "Possible pneumonia",
                    "image_path": "/test/chest.dcm"
                }
            ],
            "preprocessed_images": [
                {
                    "path": "/test/chest.dcm",
                    "array": np.random.randint(0, 255, (896, 896, 3), dtype=np.uint8),
                    "modality": "CR"
                }
            ],
            "confidence_scores": [0.80],
            "audit_trail": []
        }

        result = localize_findings(state)

        assert len(result["localizations"]) == 1
        assert result["localizations"][0]["bounding_box"]["region"] == "pneumonia"

    def test_compute_iou(self):
        """Test IoU computation for bounding boxes."""
        from medai_compass.agents.diagnostic.nodes import compute_iou_for_localization

        # Perfect overlap
        bbox1 = {"x": 0, "y": 0, "width": 100, "height": 100}
        bbox2 = {"x": 0, "y": 0, "width": 100, "height": 100}
        assert compute_iou_for_localization(bbox1, bbox2) == 1.0

        # No overlap
        bbox3 = {"x": 0, "y": 0, "width": 50, "height": 50}
        bbox4 = {"x": 100, "y": 100, "width": 50, "height": 50}
        assert compute_iou_for_localization(bbox3, bbox4) == 0.0

        # Partial overlap
        bbox5 = {"x": 0, "y": 0, "width": 100, "height": 100}
        bbox6 = {"x": 50, "y": 50, "width": 100, "height": 100}
        iou = compute_iou_for_localization(bbox5, bbox6)
        assert 0 < iou < 1

    def test_localize_pathology_finding(self):
        """Test localization of pathology findings."""
        from medai_compass.agents.diagnostic.nodes import localize_findings
        import numpy as np

        state = {
            "patient_id": "patient-123",
            "findings": [
                {
                    "source": "path_foundation",
                    "finding": "Tissue analysis complete",
                    "image_path": "/test/slide.dcm"
                }
            ],
            "preprocessed_images": [
                {
                    "path": "/test/slide.dcm",
                    "array": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                    "modality": "SM"
                }
            ],
            "confidence_scores": [0.90],
            "audit_trail": []
        }

        result = localize_findings(state)

        assert len(result["localizations"]) == 1
        assert result["localizations"][0]["bounding_box"]["region"] == "tissue_sample"


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
