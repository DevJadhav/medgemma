"""Diagnostic workflow nodes for LangGraph.

Each function is a node in the diagnostic graph that transforms state.
"""

from datetime import datetime, timezone
from typing import Any
import numpy as np

from medai_compass.agents.diagnostic.state import DiagnosticState


def preprocess_images(state: DiagnosticState) -> dict[str, Any]:
    """
    Preprocess DICOM images for model input.
    
    Node: preprocess_images
    """
    from medai_compass.utils.dicom import (
        parse_dicom_metadata,
        extract_pixel_data,
        resize_for_model,
        ensure_rgb
    )
    
    preprocessed = []
    audit_entries = []
    
    for image_path in state["images"]:
        try:
            # Parse metadata
            metadata = parse_dicom_metadata(image_path)
            
            # Extract and preprocess pixel data
            pixels = extract_pixel_data(image_path, normalize=True)
            pixels_uint8 = (pixels * 255).astype(np.uint8)
            
            # Resize for MedGemma (896x896)
            resized = resize_for_model(pixels_uint8, target_size=(896, 896))
            
            # Ensure RGB
            rgb = ensure_rgb(resized)
            
            preprocessed.append({
                "path": image_path,
                "metadata": metadata,
                "array": rgb,
                "modality": metadata.get("modality", "UNKNOWN")
            })
            
            audit_entries.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "preprocess_image",
                "image_path": image_path,
                "modality": metadata.get("modality")
            })
            
        except Exception as e:
            audit_entries.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "preprocess_error",
                "image_path": image_path,
                "error": str(e)
            })
    
    return {
        "preprocessed_images": preprocessed,
        "audit_trail": state["audit_trail"] + audit_entries
    }


def analyze_with_medgemma(state: DiagnosticState) -> dict[str, Any]:
    """
    Analyze images using MedGemma.
    
    Node: analyze_with_medgemma
    """
    findings = []
    confidence_scores = []
    
    for img_data in state.get("preprocessed_images", []):
        # In production, this would call the actual MedGemma model
        # For now, we return a placeholder result
        finding = {
            "source": "medgemma",
            "finding": "Analysis pending - model inference",
            "location": "unknown",
            "severity": "unknown",
            "image_path": img_data.get("path", ""),
            "modality": img_data.get("modality", "")
        }
        findings.append(finding)
        confidence_scores.append(0.5)  # Placeholder
    
    return {
        "findings": state["findings"] + findings,
        "confidence_scores": state["confidence_scores"] + confidence_scores,
        "audit_trail": state["audit_trail"] + [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "medgemma_analysis",
            "num_images": len(state.get("preprocessed_images", []))
        }]
    }


def generate_report(state: DiagnosticState) -> dict[str, Any]:
    """
    Generate diagnostic report from findings.
    
    Node: generate_report
    """
    findings = state.get("findings", [])
    confidence_scores = state.get("confidence_scores", [])
    
    # Build report sections
    report_lines = [
        f"# Diagnostic Report",
        f"Patient ID: {state.get('patient_id', 'Unknown')}",
        f"Session: {state.get('session_id', 'Unknown')}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Findings",
    ]
    
    for i, finding in enumerate(findings):
        confidence = confidence_scores[i] if i < len(confidence_scores) else 0.0
        report_lines.append(
            f"- {finding.get('finding', 'No finding')} "
            f"(Confidence: {confidence:.1%})"
        )
    
    report_lines.extend([
        "",
        "## Impression",
        "AI-generated preliminary analysis. Requires clinical verification.",
    ])
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    
    return {
        "report": "\n".join(report_lines),
        "audit_trail": state["audit_trail"] + [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "report_generated",
            "num_findings": len(findings),
            "avg_confidence": float(avg_confidence)
        }]
    }


def confidence_check(state: DiagnosticState) -> dict[str, Any]:
    """
    Check confidence and determine if review needed.
    
    Node: confidence_check
    """
    confidence_scores = state.get("confidence_scores", [])
    
    if not confidence_scores:
        requires_review = True
    else:
        avg_confidence = np.mean(confidence_scores)
        min_confidence = np.min(confidence_scores)
        
        # Require review if:
        # - Average confidence < 0.85
        # - Any individual confidence < 0.70
        requires_review = avg_confidence < 0.85 or min_confidence < 0.70
    
    return {"requires_review": requires_review}


def route_by_confidence(state: DiagnosticState) -> str:
    """
    Route based on confidence scores.
    
    Conditional edge function for LangGraph.
    """
    confidence_scores = state.get("confidence_scores", [])
    
    if not confidence_scores:
        return "human_review"
    
    avg_confidence = np.mean(confidence_scores)
    
    if avg_confidence >= 0.85:
        return "finalize"
    else:
        return "human_review"


def human_review(state: DiagnosticState) -> dict[str, Any]:
    """
    Flag for human review.
    
    Node: human_review
    """
    return {
        "requires_review": True,
        "audit_trail": state["audit_trail"] + [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "human_review_triggered",
            "reason": "Low confidence score"
        }]
    }


def finalize(state: DiagnosticState) -> dict[str, Any]:
    """
    Finalize the diagnostic workflow.
    
    Node: finalize
    """
    return {
        "audit_trail": state["audit_trail"] + [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "workflow_complete",
            "requires_review": state.get("requires_review", False)
        }]
    }
