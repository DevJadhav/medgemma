"""LangGraph diagnostic workflow graph.

Assembles nodes into a complete diagnostic pipeline with
conditional routing based on confidence scores.

Supports specialized analysis with:
- CXR Foundation for chest X-ray analysis
- Path Foundation for pathology/histology analysis
- PostgreSQL checkpoint persistence for multi-instance support
"""

import os
import logging
from typing import Optional

from langgraph.graph import StateGraph, END

from medai_compass.agents.diagnostic.state import DiagnosticState
from medai_compass.agents.diagnostic.nodes import (
    preprocess_images,
    analyze_with_medgemma,
    analyze_with_cxr_foundation,
    analyze_with_path_foundation,
    route_by_modality,
    localize_findings,
    generate_report,
    confidence_check,
    route_by_confidence,
    human_review,
    finalize
)

logger = logging.getLogger(__name__)


def _get_checkpointer():
    """
    Get PostgreSQL checkpointer for state persistence.
    
    Falls back to memory checkpointer if PostgreSQL is not configured.
    """
    # Check for PostgreSQL configuration
    postgres_host = os.environ.get("POSTGRES_HOST")
    postgres_password = os.environ.get("POSTGRES_PASSWORD")
    
    if postgres_host and postgres_password:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            
            # Build connection string
            postgres_user = os.environ.get("POSTGRES_USER", "medai")
            postgres_db = os.environ.get("POSTGRES_DB", "medai_compass")
            postgres_port = os.environ.get("POSTGRES_PORT", "5432")
            
            conn_string = (
                f"postgresql://{postgres_user}:{postgres_password}"
                f"@{postgres_host}:{postgres_port}/{postgres_db}"
            )
            
            checkpointer = PostgresSaver.from_conn_string(conn_string)
            logger.info("Using PostgreSQL checkpointer for workflow state persistence")
            return checkpointer
            
        except ImportError:
            logger.warning("PostgresSaver not available, using memory checkpointer")
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL: {e}, using memory checkpointer")
    
    # Fallback to memory checkpointer
    try:
        from langgraph.checkpoint.memory import MemorySaver
        logger.info("Using in-memory checkpointer (state will not persist across restarts)")
        return MemorySaver()
    except ImportError:
        logger.warning("No checkpointer available")
        return None


def create_diagnostic_graph(use_checkpointer: bool = True) -> StateGraph:
    """
    Create the complete diagnostic workflow graph.
    
    Graph structure:
    preprocess_images -> analyze_with_medgemma -> localize_findings -> generate_report 
        -> confidence_check -> [high_confidence -> finalize, 
                                low_confidence -> human_review -> finalize]
    
    Args:
        use_checkpointer: Whether to use state persistence checkpointing.
            Set to False for API calls where the full workflow runs in one request.
            Checkpointing can cause serialization issues with numpy arrays.
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create state graph with our state type
    workflow = StateGraph(DiagnosticState)
    
    # Add nodes
    workflow.add_node("preprocess_images", preprocess_images)
    workflow.add_node("analyze_with_medgemma", analyze_with_medgemma)
    workflow.add_node("localize_findings", localize_findings)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("confidence_check", confidence_check)
    workflow.add_node("human_review", human_review)
    workflow.add_node("finalize", finalize)
    
    # Set entry point
    workflow.set_entry_point("preprocess_images")
    
    # Add sequential edges
    # Flow: preprocess -> analyze -> localize -> report -> confidence_check
    workflow.add_edge("preprocess_images", "analyze_with_medgemma")
    workflow.add_edge("analyze_with_medgemma", "localize_findings")
    workflow.add_edge("localize_findings", "generate_report")
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
    
    # Get checkpointer for state persistence (only if requested)
    checkpointer = _get_checkpointer() if use_checkpointer else None
    
    # Compile with checkpointer if available
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


async def run_diagnostic_workflow(
    patient_id: str,
    session_id: str,
    image_paths: list[str],
    fhir_context: dict | None = None,
    thread_id: str | None = None
) -> DiagnosticState:
    """
    Run the complete diagnostic workflow with state persistence.
    
    Args:
        patient_id: Patient identifier
        session_id: Session identifier
        image_paths: List of DICOM file paths
        fhir_context: Optional FHIR patient context
        thread_id: Optional thread ID for checkpoint persistence
        
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
    
    # Create graph
    graph = create_diagnostic_graph()
    
    # Configure thread for checkpointing
    config = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    else:
        # Use session_id as thread_id for persistence
        config["configurable"] = {"thread_id": session_id}
    
    # Run asynchronously with config
    final_state = await graph.ainvoke(initial_state, config=config)
    
    return final_state


async def resume_diagnostic_workflow(
    thread_id: str,
    updates: dict | None = None
) -> DiagnosticState:
    """
    Resume a paused diagnostic workflow from checkpoint.
    
    Args:
        thread_id: Thread ID to resume
        updates: Optional state updates to apply
        
    Returns:
        Final diagnostic state with results
    """
    graph = create_diagnostic_graph()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get current state from checkpoint
    state = await graph.aget_state(config)
    
    if state is None:
        raise ValueError(f"No checkpoint found for thread_id: {thread_id}")
    
    # Apply updates if provided
    if updates:
        await graph.aupdate_state(config, updates)
    
    # Resume execution
    final_state = await graph.ainvoke(None, config=config)
    
    return final_state