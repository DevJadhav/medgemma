"""PHI/PII detection and masking for HIPAA compliance.

Implements:
- Pattern-based PHI detection (SSN, MRN, phone, email, DOB, address)
- PHI masking with redaction tokens
- Validation to ensure outputs are PHI-free
"""

import re
from typing import Any


# PHI detection patterns
PHI_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "mrn": re.compile(r'\bMRN[:\s]*(\d{6,10})\b', re.IGNORECASE),
    "phone": re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "email": re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b'),
    "dob": re.compile(r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}\b'),
    "address": re.compile(
        r'\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
        re.IGNORECASE
    ),
}

# Redaction tokens for each PHI type
REDACTION_TOKENS: dict[str, str] = {
    "ssn": "[SSN_REDACTED]",
    "mrn": "[MRN_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "dob": "[DOB_REDACTED]",
    "address": "[ADDRESS_REDACTED]",
}


def detect_phi(text: str) -> dict[str, list[str]]:
    """
    Detect PHI/PII patterns in text.
    
    Args:
        text: Input text to scan for PHI
        
    Returns:
        Dictionary mapping PHI type to list of detected instances
    """
    detected: dict[str, list[str]] = {}
    
    for phi_type, pattern in PHI_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Handle groups in regex
            if isinstance(matches[0], tuple):
                matches = [m[0] if isinstance(m, tuple) else m for m in matches]
            detected[phi_type] = matches
    
    return detected


def mask_phi(
    text: str, 
    return_detected: bool = False
) -> str | tuple[str, list[str]]:
    """
    Mask PHI in text with redaction tokens.
    
    Args:
        text: Input text containing potential PHI
        return_detected: If True, also return list of detected PHI
        
    Returns:
        Masked text (and optionally list of detected PHI)
    """
    masked_text = text
    detected_list: list[str] = []
    
    for phi_type, pattern in PHI_PATTERNS.items():
        token = REDACTION_TOKENS[phi_type]
        
        # Find all matches first
        matches = pattern.findall(text)
        if matches:
            for match in matches:
                # Handle tuple groups
                match_str = match[0] if isinstance(match, tuple) else match
                detected_list.append(f"{phi_type}: {match_str}")
        
        # Replace in text
        masked_text = pattern.sub(token, masked_text)
    
    if return_detected:
        return masked_text, detected_list
    return masked_text


def validate_no_phi(text: str) -> tuple[bool, list[str]]:
    """
    Validate that text contains no PHI.
    
    Args:
        text: Text to validate
        
    Returns:
        Tuple of (is_safe, list_of_issues)
    """
    detected = detect_phi(text)
    
    issues: list[str] = []
    for phi_type, instances in detected.items():
        for instance in instances:
            issues.append(f"Found {phi_type}: {instance}")
    
    is_safe = len(issues) == 0
    return is_safe, issues


class PHIDetector:
    """
    PHI detector class with configurable patterns.
    
    Provides stateful PHI detection with custom patterns
    and audit logging support.
    """
    
    def __init__(self, additional_patterns: dict[str, str] | None = None):
        """
        Initialize PHI detector.
        
        Args:
            additional_patterns: Additional regex patterns to detect
        """
        self.patterns = dict(PHI_PATTERNS)
        
        if additional_patterns:
            for name, pattern in additional_patterns.items():
                self.patterns[name] = re.compile(pattern)
                REDACTION_TOKENS[name] = f"[{name.upper()}_REDACTED]"
    
    def scan(self, text: str) -> dict[str, Any]:
        """
        Scan text for PHI and return detailed report.
        
        Args:
            text: Text to scan
            
        Returns:
            Scan report with detected PHI and risk assessment
        """
        detected = detect_phi(text)
        
        total_instances = sum(len(v) for v in detected.values())
        
        return {
            "detected": detected,
            "total_instances": total_instances,
            "is_safe": total_instances == 0,
            "risk_level": self._assess_risk(detected),
        }
    
    def _assess_risk(self, detected: dict[str, list[str]]) -> str:
        """Assess risk level based on detected PHI types."""
        if not detected:
            return "none"
        
        # SSN is highest risk
        if "ssn" in detected:
            return "critical"
        
        # Multiple types or MRN is high risk
        if len(detected) > 1 or "mrn" in detected:
            return "high"
        
        return "medium"
