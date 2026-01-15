"""Input guardrails for medical AI safety.

Provides:
- Medical scope validation
- Prompt injection detection
- Input sanitization
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScopeValidationResult:
    """Result of scope validation."""
    is_valid: bool
    scope: Optional[str]  # "diagnostic", "workflow", "communication"
    reason: Optional[str]


@dataclass
class InjectionDetectionResult:
    """Result of prompt injection detection."""
    is_safe: bool
    reason: str
    risk_score: float


# Medical scope keywords
MEDICAL_SCOPE_PATTERNS = {
    "diagnostic": [
        r"x-?ray", r"ct\s+scan", r"mri", r"ultrasound", r"mammogram",
        r"pathology", r"histology", r"biopsy", r"radiology",
        r"analyze.*image", r"interpret.*scan", r"findings",
        r"nodule", r"lesion", r"tumor", r"mass", r"infiltrate",
        r"pneumonia", r"fracture", r"hemorrhage",
    ],
    "workflow": [
        r"discharge\s+summary", r"clinical\s+note", r"documentation",
        r"schedule", r"appointment", r"prior\s+auth",
        r"summarize.*record", r"ehr", r"fhir",
    ],
    "communication": [
        r"explain", r"what\s+does.*mean", r"understand",
        r"symptoms?", r"treatment", r"medication",
        r"health\s+information", r"patient\s+education",
    ],
}

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions?",
    r"forget\s+(everything|all|previous)",
    r"disregard\s+(previous|your|all)",
    r"new\s+instructions?:",
    r"system\s+prompt:",
    r"you\s+are\s+now",
    r"act\s+as\s+(if|a)",
    r"pretend\s+(you|to)",
    r"reveal\s+(patient|confidential|private)",
    r"bypass\s+(safety|security|filter)",
    r"jailbreak",
    r"dan\s+mode",
]


def validate_medical_scope(query: str) -> ScopeValidationResult:
    """
    Validate that a query is within medical AI scope.
    
    Args:
        query: User query text
        
    Returns:
        ScopeValidationResult with scope type or rejection reason
    """
    query_lower = query.lower()
    
    # Check each scope
    for scope, patterns in MEDICAL_SCOPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return ScopeValidationResult(
                    is_valid=True,
                    scope=scope,
                    reason=None
                )
    
    # If no medical scope detected
    return ScopeValidationResult(
        is_valid=False,
        scope=None,
        reason="out_of_scope"
    )


def detect_prompt_injection(text: str) -> InjectionDetectionResult:
    """
    Detect potential prompt injection attacks.
    
    Args:
        text: Input text to check
        
    Returns:
        InjectionDetectionResult with safety assessment
    """
    text_lower = text.lower()
    matched_patterns = []
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched_patterns.append(pattern)
    
    if matched_patterns:
        return InjectionDetectionResult(
            is_safe=False,
            reason=f"Potential injection detected: {len(matched_patterns)} suspicious patterns",
            risk_score=min(1.0, len(matched_patterns) * 0.3)
        )
    
    return InjectionDetectionResult(
        is_safe=True,
        reason="No injection patterns detected",
        risk_score=0.0
    )


def sanitize_input(text: str) -> str:
    """
    Sanitize input text by removing potentially dangerous content.
    
    Args:
        text: Raw input text
        
    Returns:
        Sanitized text
    """
    # Remove HTML/script tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
