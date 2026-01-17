"""Diagnostic state management for LangGraph workflow.

Defines the TypedDict state structure for the diagnostic pipeline.
"""

from typing import Any, TypedDict


class DiagnosticState(TypedDict):
    """
    State structure for the diagnostic workflow.
    
    Passed between LangGraph nodes throughout the diagnostic pipeline.
    """
    patient_id: str
    session_id: str
    images: list[str]  # DICOM paths
    preprocessed_images: list[Any]  # Processed image arrays
    findings: list[dict]
    confidence_scores: list[float]
    localizations: list[dict]  # Bounding box localizations for findings
    requires_review: bool
    audit_trail: list[dict]
    fhir_context: dict
    report: str | None


def create_initial_state(
    patient_id: str,
    session_id: str,
    images: list[str] | None = None,
    fhir_context: dict | None = None
) -> DiagnosticState:
    """
    Create initial diagnostic state.
    
    Args:
        patient_id: Patient identifier
        session_id: Workflow session ID
        images: Optional list of DICOM paths
        fhir_context: Optional FHIR patient context
        
    Returns:
        Initialized DiagnosticState
    """
    return DiagnosticState(
        patient_id=patient_id,
        session_id=session_id,
        images=images or [],
        preprocessed_images=[],
        findings=[],
        confidence_scores=[],
        localizations=[],
        requires_review=False,
        audit_trail=[],
        fhir_context=fhir_context or {},
        report=None
    )


def add_finding(
    state: DiagnosticState,
    finding: dict
) -> DiagnosticState:
    """
    Add a finding to the state.
    
    Args:
        state: Current diagnostic state
        finding: Finding to add
        
    Returns:
        Updated state with new finding
    """
    updated_findings = state["findings"].copy()
    updated_findings.append(finding)
    
    return {**state, "findings": updated_findings}


def add_audit_entry(
    state: DiagnosticState,
    entry: dict
) -> DiagnosticState:
    """
    Add an audit trail entry.
    
    Args:
        state: Current diagnostic state
        entry: Audit entry to add
        
    Returns:
        Updated state with audit entry
    """
    updated_trail = state["audit_trail"].copy()
    updated_trail.append(entry)
    
    return {**state, "audit_trail": updated_trail}
