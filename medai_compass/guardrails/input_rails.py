"""Input guardrails for medical AI safety.

Provides:
- Medical scope validation
- Prompt injection detection
- Jailbreak detection
- Input sanitization
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


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


class JailbreakCategory(Enum):
    """Categories of jailbreak attempts."""
    ROLE_PLAY = "role_play"  # "Pretend you are...", "Act as if..."
    INSTRUCTION_OVERRIDE = "instruction_override"  # "Ignore previous instructions"
    CONTEXT_MANIPULATION = "context_manipulation"  # Fake system prompts
    ENCODING_BYPASS = "encoding_bypass"  # Base64, rot13, etc.
    HYPOTHETICAL_FRAMING = "hypothetical_framing"  # "Hypothetically, if..."
    EMOTIONAL_MANIPULATION = "emotional_manipulation"  # "My grandmother used to..."
    JAILBREAK_TERM = "jailbreak_term"  # Known jailbreak names like DAN
    MEDICAL_BOUNDARY_VIOLATION = "medical_boundary"  # Attempting unauthorized medical advice


@dataclass
class JailbreakDetectionResult:
    """Result of jailbreak detection."""
    is_jailbreak: bool
    category: Optional[JailbreakCategory]
    severity: str  # "low", "medium", "high", "critical"
    matched_patterns: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    recommendation: str = ""


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

# Comprehensive jailbreak patterns by category
JAILBREAK_PATTERNS = {
    JailbreakCategory.ROLE_PLAY: [
        r"pretend\s+(you|you're|to\s+be)",
        r"act\s+as\s+(if|a|an)",
        r"you\s+are\s+now\s+(a|an)",
        r"roleplay\s+as",
        r"behave\s+(like|as)",
        r"from\s+now\s+on\s+you\s+are",
        r"let's\s+play\s+a\s+game",
        r"i\s+want\s+you\s+to\s+be",
        r"imagine\s+you\s+are",
        r"simulate\s+being",
    ],
    JailbreakCategory.INSTRUCTION_OVERRIDE: [
        r"ignore\s+(previous|all|prior|your|any)\s+instructions?",
        r"forget\s+(everything|all|previous|your)",
        r"disregard\s+(previous|your|all|any)",
        r"override\s+(previous|your|all)",
        r"reset\s+(your|all)\s+(instructions?|rules?)",
        r"new\s+instructions?:",
        r"actual\s+instructions?:",
        r"real\s+instructions?:",
        r"the\s+following\s+takes\s+precedence",
        r"cancel\s+(previous|all)\s+directives",
    ],
    JailbreakCategory.CONTEXT_MANIPULATION: [
        r"system\s*[:>]\s*prompt",
        r"\[system\]",
        r"\[assistant\]",
        r"\[user\]",
        r"<\|system\|>",
        r"<\|assistant\|>",
        r"\{system_message\}",
        r"\{\{system\}\}",
        r"###\s*instruction",
        r"beginning\s+of\s+conversation",
        r"end\s+of\s+system\s+prompt",
    ],
    JailbreakCategory.ENCODING_BYPASS: [
        r"decode\s+(this|the\s+following)\s+(base64|rot13|hex)",
        r"base64\s*:",
        r"translate\s+from\s+[a-z]+\s+to\s+english",
        r"what\s+does\s+this\s+mean\s*:\s*[A-Za-z0-9+/=]{20,}",  # Base64 pattern
        r"\\x[0-9a-fA-F]{2}",  # Hex escape sequences
    ],
    JailbreakCategory.HYPOTHETICAL_FRAMING: [
        r"hypothetically\s+(speaking|if)",
        r"for\s+educational\s+purposes\s+only",
        r"in\s+a\s+fictional\s+scenario",
        r"theoretically\s+speaking",
        r"if\s+there\s+were\s+no\s+restrictions",
        r"assume\s+you\s+have\s+no\s+limitations",
        r"if\s+you\s+could\s+do\s+anything",
        r"in\s+an\s+alternate\s+reality",
    ],
    JailbreakCategory.EMOTIONAL_MANIPULATION: [
        r"my\s+(grandmother|grandma|mother|father)\s+used\s+to",
        r"before\s+(she|he)\s+died",
        r"this\s+is\s+very\s+important\s+to\s+me",
        r"i('m|\s+am)\s+going\s+to\s+(die|hurt\s+myself)\s+if\s+you\s+don't",
        r"you're\s+the\s+only\s+one\s+who\s+can\s+help",
        r"please\s+i('m|\s+am)\s+desperate",
    ],
    JailbreakCategory.JAILBREAK_TERM: [
        r"\bdan\b",  # Do Anything Now
        r"\bdeveloper\s+mode\b",
        r"\bjailbreak\b",
        r"\bunlocked\s+mode\b",
        r"\banti[- ]?filter\b",
        r"\bgodmode\b",
        r"\bdevil\s+mode\b",
        r"\bevil\s+mode\b",
        r"\bchaos\s+mode\b",
        r"\bablate\b",  # Remove safety features
    ],
    JailbreakCategory.MEDICAL_BOUNDARY_VIOLATION: [
        r"prescribe\s+(me|a)\s+\w*\s*(medication|drug|medicine)",
        r"prescribe\s+me\s+\w+",  # More general "prescribe me [something]"
        r"diagnose\s+(me|my)\b",
        r"tell\s+me\s+(exactly\s+)?what\s+(i|I)\s+have",
        r"give\s+me\s+a\s+(definitive\s+)?diagnosis",
        r"you('re|\s+are)\s+a\s+(doctor|physician|specialist)",
        r"bypass\s+(medical|clinical)\s+(review|oversight)",
        r"skip\s+the\s+disclaimer",
        r"without\s+(any|the)\s+disclaimer",
        r"don('t|\s+not)\s+tell\s+me\s+to\s+see\s+a\s+doctor",
        r"what\s+(medication|drug)\s+should\s+i\s+take",
    ],
}

# Severity mapping for each category
CATEGORY_SEVERITY = {
    JailbreakCategory.ROLE_PLAY: "medium",
    JailbreakCategory.INSTRUCTION_OVERRIDE: "high",
    JailbreakCategory.CONTEXT_MANIPULATION: "critical",
    JailbreakCategory.ENCODING_BYPASS: "high",
    JailbreakCategory.HYPOTHETICAL_FRAMING: "medium",
    JailbreakCategory.EMOTIONAL_MANIPULATION: "medium",
    JailbreakCategory.JAILBREAK_TERM: "critical",
    JailbreakCategory.MEDICAL_BOUNDARY_VIOLATION: "high",
}


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


def detect_jailbreak(text: str) -> JailbreakDetectionResult:
    """
    Detect potential jailbreak attempts in user input.
    
    Uses pattern matching across multiple categories to identify
    various jailbreak techniques including:
    - Role-play manipulation
    - Instruction override attempts
    - Context manipulation (fake system prompts)
    - Encoding bypass attempts
    - Hypothetical framing
    - Emotional manipulation
    - Known jailbreak terms
    - Medical boundary violations
    
    Args:
        text: User input text to analyze
        
    Returns:
        JailbreakDetectionResult with detection details
    """
    text_lower = text.lower()
    detected_categories: dict[JailbreakCategory, list[str]] = {}
    all_matched_patterns: list[str] = []
    
    # Check each category
    for category, patterns in JAILBREAK_PATTERNS.items():
        matched = []
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched.append(pattern)
                all_matched_patterns.append(pattern)
        if matched:
            detected_categories[category] = matched
    
    # Determine if jailbreak detected
    if not detected_categories:
        return JailbreakDetectionResult(
            is_jailbreak=False,
            category=None,
            severity="none",
            matched_patterns=[],
            risk_score=0.0,
            recommendation="Input appears safe."
        )
    
    # Determine highest severity category
    severity_order = ["critical", "high", "medium", "low"]
    highest_severity = "low"
    primary_category = None
    
    for category in detected_categories.keys():
        cat_severity = CATEGORY_SEVERITY.get(category, "medium")
        if severity_order.index(cat_severity) < severity_order.index(highest_severity):
            highest_severity = cat_severity
            primary_category = category
    
    # Calculate risk score
    # Base score from severity
    severity_scores = {"critical": 0.9, "high": 0.7, "medium": 0.5, "low": 0.3}
    base_score = severity_scores.get(highest_severity, 0.5)
    
    # Add bonus for multiple categories/patterns
    category_bonus = min(0.1 * (len(detected_categories) - 1), 0.1)
    pattern_bonus = min(0.02 * (len(all_matched_patterns) - 1), 0.1)
    risk_score = min(1.0, base_score + category_bonus + pattern_bonus)
    
    # Generate recommendation
    recommendations = {
        "critical": "Block request immediately. Do not process or respond.",
        "high": "Block request. Consider logging for security review.",
        "medium": "Request may contain manipulation attempts. Review before processing.",
        "low": "Minor suspicious patterns detected. Process with caution.",
    }
    recommendation = recommendations.get(highest_severity, "Review request manually.")
    
    return JailbreakDetectionResult(
        is_jailbreak=True,
        category=primary_category,
        severity=highest_severity,
        matched_patterns=all_matched_patterns,
        risk_score=risk_score,
        recommendation=recommendation
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


def apply_input_guardrails(text: str) -> dict:
    """
    Apply all input guardrails and return combined result.
    
    Args:
        text: Raw user input
        
    Returns:
        Dictionary with combined guardrail results
    """
    # Sanitize first
    sanitized = sanitize_input(text)
    
    # Check for jailbreak attempts
    jailbreak_result = detect_jailbreak(sanitized)
    
    # Check for prompt injection
    injection_result = detect_prompt_injection(sanitized)
    
    # Check medical scope
    scope_result = validate_medical_scope(sanitized)
    
    # Determine overall safety
    is_safe = (
        not jailbreak_result.is_jailbreak and
        injection_result.is_safe
    )
    
    # Determine if within scope
    is_valid_request = is_safe and scope_result.is_valid
    
    return {
        "sanitized_input": sanitized,
        "is_safe": is_safe,
        "is_valid_request": is_valid_request,
        "jailbreak": {
            "detected": jailbreak_result.is_jailbreak,
            "category": jailbreak_result.category.value if jailbreak_result.category else None,
            "severity": jailbreak_result.severity,
            "risk_score": jailbreak_result.risk_score,
            "recommendation": jailbreak_result.recommendation,
        },
        "injection": {
            "detected": not injection_result.is_safe,
            "risk_score": injection_result.risk_score,
            "reason": injection_result.reason,
        },
        "scope": {
            "is_valid": scope_result.is_valid,
            "scope": scope_result.scope,
            "reason": scope_result.reason,
        },
    }
