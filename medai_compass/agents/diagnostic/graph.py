"""LangGraph diagnostic workflow graph.

Assembles nodes into a complete diagnostic pipeline with
conditional routing based on confidence scores.
"""

from langgraph.graph import StateGraph, END

from medai_compass.agents.diagnostic.state import DiagnosticState
from medai_compass.agents.diagnostic.nodes import (
    preprocess_images,
    analyze_with_medgemma,
    generate_report,
    confidence_check,
    route_by_confidence,
    human_review,
    finalize
)


def create_diagnostic_graph() -> StateGraph:
    """
    Create the complete diagnostic workflow graph.
    
    Graph structure:
    preprocess_images -> analyze_with_medgemma -> generate_report 
        -> confidence_check -> [high_confidence -> finalize, 
                                low_confidence -> human_review -> finalize]
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create state graph with our state type
    workflow = StateGraph(DiagnosticState)
    
    # Add nodes
    workflow.add_node("preprocess_images", preprocess_images)
    workflow.add_node("analyze_with_medgemma", analyze_with_medgemma)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("confidence_check", confidence_check)
    workflow.add_node("human_review", human_review)
    workflow.add_node("finalize", finalize)
    
    # Set entry point
    workflow.set_entry_point("preprocess_images")
    
    # Add sequential edges
    workflow.add_edge("preprocess_images", "analyze_with_medgemma")
    workflow.add_edge("analyze_with_medgemma", "generate_report")
    workflow.add_edge("generate_report", "confidence_check")
    
    # Add conditional edge based on confidence
    workflow.add_conditional_edges(
        "confidence_check",
        route_by_confidence,
        {
            "finalize": "finalize",
            "human_review": "human_review"
        }
    )
    
    # Human review leads to finalize
    workflow.add_edge("human_review", "finalize")
    
    # Finalize ends the workflow
    workflow.add_edge("finalize", END)
    
    # Compile and return
    return workflow.compile()


async def run_diagnostic_workflow(
    patient_id: str,
    session_id: str,
    image_paths: list[str],
    fhir_context: dict | None = None
) -> DiagnosticState:
    """
    Run the complete diagnostic workflow.
    
    Args:
        patient_id: Patient identifier
        session_id: Session identifier
        image_paths: List of DICOM file paths
        fhir_context: Optional FHIR patient context
        
    Returns:
        Final diagnostic state with results
    """
    from medai_compass.agents.diagnostic.state import create_initial_state
    
    # Create initial state
    initial_state = create_initial_state(
        patient_id=patient_id,
        session_id=session_id,
        images=image_paths,
        fhir_context=fhir_context
    )
    
    # Create and run graph
    graph = create_diagnostic_graph()
    
    # Run asynchronously
    final_state = await graph.ainvoke(initial_state)
    
    return final_state
