"""Output guardrails for medical AI safety.

Provides:
- Disclaimer addition
- Medical terminology validation
- Hallucination risk detection
- PHI leakage prevention
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """Generic validation result."""
    is_valid: bool
    reason: Optional[str] = None
    details: Optional[dict] = None


@dataclass
class SafetyResult:
    """Safety check result."""
    is_safe: bool
    reason: Optional[str] = None
    issues: list[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


@dataclass
class HallucinationResult:
    """Hallucination risk assessment."""
    risk_level: str  # "low", "medium", "high"
    indicators: list[str]
    score: float


# Clinical disclaimer templates
DISCLAIMERS = {
    "diagnostic": (
        "\n\n---\n**Clinical Disclaimer:** This AI-assisted analysis is for "
        "clinical decision support only. Confidence: {confidence:.1%}. "
        "All findings should be verified by a qualified healthcare professional."
    ),
    "workflow": (
        "\n\n---\n**Note:** This documentation was AI-generated and should be "
        "reviewed for accuracy before use in clinical records."
    ),
    "communication": (
        "\n\n---\n**Disclaimer:** This information is for educational purposes only "
        "and does not constitute medical advice. Please consult a healthcare "
        "provider for personalized recommendations."
    ),
}

# Valid medical terminology patterns (simplified)
VALID_MEDICAL_TERMS = [
    r"\b(bilateral|unilateral)\b",
    r"\b(infiltrate|consolidation|opacity)\b",
    r"\b(effusion|edema|atelectasis)\b",
    r"\b(pneumonia|pneumothorax|cardiomegaly)\b",
    r"\b(nodule|mass|lesion|tumor)\b",
    r"\b(fracture|dislocation)\b",
    r"\b(normal|abnormal|unremarkable)\b",
    r"\b(acute|chronic|subacute)\b",
]

# Hallucination indicators
HALLUCINATION_PATTERNS = [
    (r"\d+\.\d{4,}", "Overly precise measurements"),
    (r"\b(18|19)\d{2}\b", "Historical dates that seem anachronistic"),
    (r"(definitely|certainly|absolutely)\s+\d+%", "Overconfident percentages"),
    (r"study\s+from\s+(18|19)\d{2}", "Possibly fabricated historical references"),
]


def add_disclaimer(
    response: str,
    domain: str,
    confidence: float = 0.0
) -> str:
    """
    Add appropriate disclaimer to AI response.
    
    Args:
        response: AI-generated response
        domain: Domain type (diagnostic, workflow, communication)
        confidence: Model confidence score
        
    Returns:
        Response with disclaimer appended
    """
    template = DISCLAIMERS.get(domain, DISCLAIMERS["communication"])
    disclaimer = template.format(confidence=confidence)
    
    return response + disclaimer


def validate_medical_terms(response: str) -> ValidationResult:
    """
    Validate that response uses appropriate medical terminology.
    
    Args:
        response: AI-generated response
        
    Returns:
        ValidationResult indicating terminology validity
    """
    # Check if any medical terms are present
    found_terms = []
    for pattern in VALID_MEDICAL_TERMS:
        matches = re.findall(pattern, response, re.IGNORECASE)
        found_terms.extend(matches)
    
    # For short responses, we don't require medical terms
    if len(response.split()) < 10:
        return ValidationResult(is_valid=True, reason="Short response, terms not required")
    
    if found_terms:
        return ValidationResult(
            is_valid=True,
            reason="Valid medical terminology found",
            details={"terms_found": list(set(found_terms))}
        )
    
    # Not having medical terms isn't necessarily invalid
    return ValidationResult(
        is_valid=True,
        reason="No specific medical terminology detected",
        details={"terms_found": []}
    )


def check_hallucination_risk(response: str) -> HallucinationResult:
    """
    Check for potential hallucination indicators in response.
    
    Args:
        response: AI-generated response
        
    Returns:
        HallucinationResult with risk assessment
    """
    indicators = []
    
    for pattern, description in HALLUCINATION_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            indicators.append(description)
    
    # Calculate risk score based on indicators
    score = len(indicators) * 0.25
    score = min(1.0, score)
    
    if score >= 0.5:
        risk_level = "high"
    elif score >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return HallucinationResult(
        risk_level=risk_level,
        indicators=indicators,
        score=score
    )


def validate_no_phi_leakage(response: str) -> SafetyResult:
    """
    Validate that response doesn't contain PHI.
    
    Args:
        response: AI-generated response
        
    Returns:
        SafetyResult indicating PHI safety
    """
    from medai_compass.guardrails.phi_detection import detect_phi
    
    detected = detect_phi(response)
    
    if any(detected.values()):
        issues = []
        for phi_type, instances in detected.items():
            for instance in instances:
                issues.append(f"Found {phi_type}: {instance[:20]}...")
        
        return SafetyResult(
            is_safe=False,
            reason="PHI detected in response",
            issues=issues
        )
    
    return SafetyResult(
        is_safe=True,
        reason="No PHI detected"
    )
