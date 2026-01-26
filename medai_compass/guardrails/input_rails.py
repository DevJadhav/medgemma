"""Input guardrails for medical AI safety.

Provides:
- Medical scope validation
- Prompt injection detection
- Jailbreak detection with encoding bypass detection
- Fuzzy matching for obfuscation attempts
- Input sanitization
"""

import base64
import codecs
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
    # Additional injection patterns
    r"\[\[system\]\]",  # [[system]] bracket notation
    r"\[\[user\]\]",
    r"\[\[assistant\]\]",
    r"override\s+(safety|security|measures)",
    r"do\s+this\s+instead",
    r"base64\s*[:\s]",  # Base64 encoded payloads
    r"decode\s+(this|the\s+following)",
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
        r"ignore\s+all\s+previous\s+instructions\s+and",  # More specific pattern
        r"forget\s+(everything|all|previous|your)",
        r"disregard\s+(previous|your|all|any)",
        r"override\s+(previous|your|all)",
        r"reset\s+(your|all)\s+(instructions?|rules?)",
        r"new\s+instructions?:",
        r"actual\s+instructions?:",
        r"real\s+instructions?:",
        r"the\s+following\s+takes\s+precedence",
        r"cancel\s+(previous|all)\s+directives",
        r"do\s+this\s+instead",  # "and do this instead"
        r"instead\s+of\s+following",
        r"your\s+new\s+instructions?\s+(are|is)",  # "Your new instructions are..."
        r"new\s+instructions?\s+(are|is)\s+to",
        r"your\s+instructions?\s+(are|is)\s+now",
    ],
    JailbreakCategory.CONTEXT_MANIPULATION: [
        r"system\s*[:>]\s*prompt",
        r"\[system\]",
        r"\[assistant\]",
        r"\[user\]",
        r"\[\[system\]\]",  # Double bracket notation
        r"\[\[assistant\]\]",
        r"\[\[user\]\]",
        r"<\|system\|>",
        r"<\|assistant\|>",
        r"\{system_message\}",
        r"\{\{system\}\}",
        r"###\s*system",  # Markdown-style system prompts
        r"###\s*instruction",
        r"beginning\s+of\s+conversation",
        r"end\s+of\s+system\s+prompt",
        r"```system",  # Code block style
        r"```instruction",
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


# =============================================================================
# ENCODING BYPASS DETECTION
# =============================================================================

def decode_base64_segments(text: str) -> list[tuple[str, str]]:
    """
    Find and decode Base64 encoded segments in text.

    Args:
        text: Input text potentially containing Base64

    Returns:
        List of (encoded, decoded) tuples
    """
    # Pattern for Base64 strings (at least 20 chars, valid Base64 alphabet)
    base64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
    decoded_segments = []

    for match in base64_pattern.finditer(text):
        encoded = match.group()
        try:
            # Add padding if needed
            padding = 4 - (len(encoded) % 4)
            if padding != 4:
                encoded_padded = encoded + "=" * padding
            else:
                encoded_padded = encoded

            decoded = base64.b64decode(encoded_padded).decode('utf-8', errors='ignore')
            # Only include if it looks like readable text
            if decoded and len(decoded) > 5 and any(c.isalpha() for c in decoded):
                decoded_segments.append((encoded, decoded))
        except Exception:
            pass

    return decoded_segments


def decode_rot13(text: str) -> str:
    """
    Decode ROT13 encoded text.

    Args:
        text: Input text

    Returns:
        ROT13 decoded text
    """
    return codecs.decode(text, 'rot_13')


def detect_encoded_content(text: str) -> dict:
    """
    Detect and decode potentially encoded malicious content.

    Supports:
    - Base64 encoding
    - ROT13 encoding
    - Hex escape sequences
    - Unicode obfuscation

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with detection results
    """
    results = {
        "has_encoded_content": False,
        "base64_decoded": [],
        "rot13_suspicious": False,
        "hex_sequences": [],
        "unicode_tricks": [],
    }

    # Check for Base64
    base64_decoded = decode_base64_segments(text)
    if base64_decoded:
        results["base64_decoded"] = base64_decoded
        results["has_encoded_content"] = True

    # Check if ROT13 decoding reveals suspicious content
    rot13_decoded = decode_rot13(text)
    suspicious_rot13_patterns = [
        r"ignore\s+instructions",
        r"system\s+prompt",
        r"jailbreak",
        r"bypass\s+safety",
    ]
    for pattern in suspicious_rot13_patterns:
        if re.search(pattern, rot13_decoded, re.IGNORECASE):
            results["rot13_suspicious"] = True
            results["has_encoded_content"] = True
            break

    # Check for hex escape sequences
    hex_pattern = re.compile(r'\\x([0-9a-fA-F]{2})+')
    hex_matches = hex_pattern.findall(text)
    if hex_matches:
        results["hex_sequences"] = hex_matches
        results["has_encoded_content"] = True

    # Check for Unicode tricks (zero-width characters, homoglyphs)
    unicode_tricks = []
    # Zero-width characters
    if re.search(r'[\u200b\u200c\u200d\ufeff]', text):
        unicode_tricks.append("zero_width_characters")
    # Homoglyph detection (Cyrillic/Greek letters that look like Latin)
    homoglyph_pattern = re.compile(r'[\u0430\u0435\u043e\u0440\u0441\u0445]')  # а, е, о, р, с, х
    if homoglyph_pattern.search(text):
        unicode_tricks.append("homoglyphs")

    if unicode_tricks:
        results["unicode_tricks"] = unicode_tricks
        results["has_encoded_content"] = True

    return results


# =============================================================================
# FUZZY MATCHING FOR OBFUSCATION
# =============================================================================

# L33tspeak mappings
LEET_MAPPINGS = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
    '6': 'g', '7': 't', '8': 'b', '9': 'g', '@': 'a',
    '$': 's', '!': 'i', '|': 'l', '+': 't', '(': 'c',
    ')': 'j', '{': 'c', '}': 'j', '<': 'c', '>': 'j',
}


def normalize_leetspeak(text: str) -> str:
    """
    Normalize l33tspeak to regular text.

    Args:
        text: Input text possibly containing l33tspeak

    Returns:
        Normalized text
    """
    result = []
    for char in text.lower():
        result.append(LEET_MAPPINGS.get(char, char))
    return ''.join(result)


def remove_spacing_tricks(text: str) -> str:
    """
    Remove spacing tricks used to evade detection.

    Handles:
    - Extra spaces between letters (i g n o r e)
    - Zero-width spaces
    - Dots/underscores between letters (i.g.n.o.r.e)

    Args:
        text: Input text

    Returns:
        Text with spacing tricks removed
    """
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # Check for single letter spacing (i g n o r e -> ignore)
    # Pattern: letter followed by space followed by letter, repeated
    single_spaced = re.compile(r'\b(\w)\s+(?=\w\s+\w|\w\b)')
    if single_spaced.search(text):
        # Remove single spaces between single letters
        text = re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', text)

    # Remove dots/underscores between single letters
    text = re.sub(r'(?<=\w)[._](?=\w)', '', text)

    return text


def fuzzy_pattern_match(text: str, patterns: list[str]) -> list[str]:
    """
    Perform fuzzy matching against patterns, accounting for obfuscation.

    Handles:
    - L33tspeak (1gn0r3 -> ignore)
    - Spacing tricks (i g n o r e -> ignore)
    - Character substitution

    Args:
        text: Input text to check
        patterns: List of patterns to match against

    Returns:
        List of matched patterns
    """
    matched = []

    # Create normalized versions of the text
    normalized_leet = normalize_leetspeak(text)
    normalized_spacing = remove_spacing_tricks(text)
    normalized_both = remove_spacing_tricks(normalize_leetspeak(text))

    # Check each pattern against all normalized versions
    for pattern in patterns:
        pattern_lower = pattern.lower()

        # Check original
        if re.search(pattern_lower, text.lower()):
            matched.append(pattern)
            continue

        # Check l33tspeak normalized
        if re.search(pattern_lower, normalized_leet):
            matched.append(f"{pattern} (l33tspeak)")
            continue

        # Check spacing normalized
        if re.search(pattern_lower, normalized_spacing.lower()):
            matched.append(f"{pattern} (spacing)")
            continue

        # Check both normalizations
        if re.search(pattern_lower, normalized_both):
            matched.append(f"{pattern} (obfuscated)")
            continue

    return matched


# Key terms to check for obfuscation
OBFUSCATION_CHECK_TERMS = [
    "ignore", "jailbreak", "bypass", "override", "system",
    "instructions", "pretend", "roleplay", "developer mode",
    "dan", "evil", "devil", "chaos", "unlocked",
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


def detect_jailbreak(text: str) -> JailbreakDetectionResult:
    """
    Detect potential jailbreak attempts in user input.

    Uses pattern matching across multiple categories to identify
    various jailbreak techniques including:
    - Role-play manipulation
    - Instruction override attempts
    - Context manipulation (fake system prompts)
    - Encoding bypass attempts (Base64, ROT13, hex)
    - Hypothetical framing
    - Emotional manipulation
    - Known jailbreak terms
    - Medical boundary violations
    - Obfuscation (l33tspeak, spacing tricks)

    Args:
        text: User input text to analyze

    Returns:
        JailbreakDetectionResult with detection details
    """
    text_lower = text.lower()
    detected_categories: dict[JailbreakCategory, list[str]] = {}
    all_matched_patterns: list[str] = []

    # Check each category with standard patterns
    for category, patterns in JAILBREAK_PATTERNS.items():
        matched = []
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched.append(pattern)
                all_matched_patterns.append(pattern)
        if matched:
            detected_categories[category] = matched

    # Check for encoded content (Base64, ROT13, hex)
    encoded_content = detect_encoded_content(text)
    if encoded_content["has_encoded_content"]:
        encoded_matches = []

        # Check decoded Base64 for suspicious content
        for encoded, decoded in encoded_content.get("base64_decoded", []):
            # Re-scan decoded content for jailbreak patterns
            for category, patterns in JAILBREAK_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, decoded.lower(), re.IGNORECASE):
                        encoded_matches.append(f"base64_hidden: {pattern}")
                        if category not in detected_categories:
                            detected_categories[category] = []
                        detected_categories[category].append(f"base64({pattern})")

        if encoded_content.get("rot13_suspicious"):
            encoded_matches.append("rot13_obfuscation")

        if encoded_content.get("hex_sequences"):
            encoded_matches.append("hex_escape_sequences")

        if encoded_content.get("unicode_tricks"):
            encoded_matches.extend([f"unicode:{trick}" for trick in encoded_content["unicode_tricks"]])

        if encoded_matches:
            all_matched_patterns.extend(encoded_matches)
            if JailbreakCategory.ENCODING_BYPASS not in detected_categories:
                detected_categories[JailbreakCategory.ENCODING_BYPASS] = []
            detected_categories[JailbreakCategory.ENCODING_BYPASS].extend(encoded_matches)

    # Check for obfuscated content using fuzzy matching
    obfuscation_matches = fuzzy_pattern_match(text, OBFUSCATION_CHECK_TERMS)
    obfuscation_detected = [m for m in obfuscation_matches if "(l33tspeak)" in m or "(spacing)" in m or "(obfuscated)" in m]
    if obfuscation_detected:
        all_matched_patterns.extend(obfuscation_detected)
        if JailbreakCategory.ENCODING_BYPASS not in detected_categories:
            detected_categories[JailbreakCategory.ENCODING_BYPASS] = []
        detected_categories[JailbreakCategory.ENCODING_BYPASS].extend(obfuscation_detected)

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

    # Add bonus for encoding/obfuscation attempts (indicates sophisticated attack)
    encoding_bonus = 0.1 if JailbreakCategory.ENCODING_BYPASS in detected_categories else 0.0

    risk_score = min(1.0, base_score + category_bonus + pattern_bonus + encoding_bonus)

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
