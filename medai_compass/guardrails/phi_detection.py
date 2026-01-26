"""PHI/PII detection and masking for HIPAA compliance.

Enhanced implementation with:
- Pattern-based PHI detection (SSN, MRN, phone, email, DOB, address)
- Extended patterns (clinical trial IDs, Medicare, driver's license, etc.)
- NER-based name detection with medical term filtering
- PHI masking with redaction tokens
- Risk assessment and audit logging
- Validation to ensure outputs are PHI-free
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# PHI DETECTION PATTERNS
# =============================================================================

# Core PHI patterns (HIPAA identifiers)
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

# Extended PHI patterns for enhanced detection
EXTENDED_PHI_PATTERNS: dict[str, re.Pattern] = {
    # Clinical trial identifiers (NCT format)
    "clinical_trial": re.compile(r'\b(NCT\d{8})\b', re.IGNORECASE),

    # Driver's license (various state formats)
    "drivers_license": re.compile(
        r'\b(?:DL#?|Driver\'?s?\s*License[:\s#]*)\s*([A-Z]?\d{6,9})\b',
        re.IGNORECASE
    ),

    # Passport numbers (US and international formats)
    "passport": re.compile(
        r'\b(?:Passport[#:\s]*)?([A-Z]{1,2}\d{6,9})\b',
        re.IGNORECASE
    ),

    # Medicare Beneficiary Identifier (MBI format: 1AA1-AA1-AA11)
    "medicare_id": re.compile(
        r'\b(\d[A-Z]{2}\d-[A-Z]{2}\d-[A-Z]{2}\d{2})\b',
        re.IGNORECASE
    ),

    # Medicaid ID (various state formats)
    "medicaid_id": re.compile(
        r'\b(?:Medicaid[#:\s]*)([A-Z]{0,3}\d{8,12})\b',
        re.IGNORECASE
    ),

    # Account/Policy numbers
    "account_number": re.compile(
        r'\b(?:Account|Acct|Policy)[#:\s]*([A-Z]{2,4}-?\d{6,12})\b',
        re.IGNORECASE
    ),

    # IP addresses (device identifiers)
    "ip_address": re.compile(
        r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
    ),

    # Credit card numbers (basic pattern)
    "credit_card": re.compile(
        r'\b(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b'
    ),

    # Bank account numbers
    "bank_account": re.compile(
        r'\b(?:Bank\s*Account|Routing)[#:\s]*(\d{8,17})\b',
        re.IGNORECASE
    ),

    # Health plan beneficiary numbers
    "health_plan_id": re.compile(
        r'\b(?:Health\s*Plan|Insurance|Member)[#:\s]*([A-Z]{2,4}\d{8,12})\b',
        re.IGNORECASE
    ),

    # Vehicle identification numbers (device identifiers per HIPAA)
    "vin": re.compile(
        r'\b[A-HJ-NPR-Z0-9]{17}\b'
    ),

    # Biometric identifiers (fingerprint, voiceprint references)
    "biometric_ref": re.compile(
        r'\b(?:fingerprint|voiceprint|retina|iris)[#:\s]*([A-Z0-9]{8,32})\b',
        re.IGNORECASE
    ),

    # Full face photographs or comparable images (URLs)
    "photo_url": re.compile(
        r'\b(?:photo|image|picture)[#:\s]*(https?://[^\s]+\.(?:jpg|jpeg|png|gif))\b',
        re.IGNORECASE
    ),
}

# Context-aware PHI patterns (detect PHI in natural language context)
CONTEXT_AWARE_PATTERNS: dict[str, re.Pattern] = {
    # "Patient John Smith admitted on 1/15"
    "patient_admission": re.compile(
        r'\b(?:patient|pt\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:admitted|discharged|seen|treated)\s+(?:on\s+)?(\d{1,2}/\d{1,2}(?:/\d{2,4})?)',
        re.IGNORECASE
    ),

    # "John Doe, DOB 01/15/1980"
    "name_with_dob": re.compile(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:DOB|date\s+of\s+birth|born)[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})',
        re.IGNORECASE
    ),

    # "Room 302, bed A: Smith, John"
    "room_patient": re.compile(
        r'(?:room|bed|unit)\s*[#:\s]*\d+[A-Za-z]?[,:\s]+([A-Z][a-z]+,?\s+[A-Z][a-z]+)',
        re.IGNORECASE
    ),

    # "Dr. Smith's patient [Name]"
    "provider_patient": re.compile(
        r"(?:Dr\.?|Doctor)\s+[A-Z][a-z]+(?:'s)?\s+patient[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        re.IGNORECASE
    ),

    # "Contact: (555) 123-4567 for John"
    "contact_for_patient": re.compile(
        r'(?:contact|call|phone|reach)[:\s]+(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\s+(?:for|regarding)\s+([A-Z][a-z]+)',
        re.IGNORECASE
    ),

    # "John's SSN is 123-45-6789"
    "possessive_identifier": re.compile(
        r"([A-Z][a-z]+)(?:'s|s')?\s+(?:SSN|social|MRN|ID|DOB)[:\s]+",
        re.IGNORECASE
    ),

    # Date ranges that might indicate treatment periods
    "treatment_dates": re.compile(
        r'(?:treated|hospitalized|admitted)\s+(?:from\s+)?(\d{1,2}/\d{1,2}/\d{2,4})\s+(?:to|through|-)\s+(\d{1,2}/\d{1,2}/\d{2,4})',
        re.IGNORECASE
    ),

    # Age with context
    "age_context": re.compile(
        r'(\d{1,3})[-\s]?(?:year[-\s]?old|y/?o)\s+(?:male|female|patient|man|woman)',
        re.IGNORECASE
    ),

    # Geographic subdivisions smaller than state
    "geographic": re.compile(
        r'\b(?:zip|zip\s*code|postal)[:\s]*(\d{5}(?:-\d{4})?)\b',
        re.IGNORECASE
    ),
}

# Name detection patterns (title + capitalized words)
NAME_PATTERNS: dict[str, re.Pattern] = {
    # Patient with name
    "patient_name": re.compile(
        r'\b(?:patient|pt\.?)[,:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        re.IGNORECASE
    ),

    # Doctor/provider names
    "provider_name": re.compile(
        r'\b(?:Dr\.?|Doctor|Physician|Nurse|NP|PA)[,:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    ),

    # Title + Name patterns
    "titled_name": re.compile(
        r'\b((?:Mr\.?|Mrs\.?|Ms\.?|Miss|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    ),

    # Name with explicit context
    "explicit_name": re.compile(
        r'\b(?:named?|called|name\s+is)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        re.IGNORECASE
    ),
}

# Medical eponyms and terms that look like names but aren't
MEDICAL_EPONYMS = {
    # Diseases and syndromes
    "parkinson", "alzheimer", "hodgkin", "huntington", "crohn",
    "addison", "cushing", "graves", "hashimoto", "raynaud",
    "tourette", "asperger", "rett", "down", "turner", "klinefelter",
    "marfan", "ehlers", "danlos", "guillain", "barré", "bell",

    # Signs and tests
    "babinski", "murphy", "mcburney", "rovsing", "kernig",
    "brudzinski", "romberg", "trendelenburg", "phalen", "tinel",
    "lasègue", "spurling", "apgar", "glasgow", "ranson",

    # Procedures and equipment
    "foley", "heimlich", "valsalva", "doppler", "holter",

    # Anatomical terms
    "fallopian", "eustachian", "bartholin", "cowper",
}

# Redaction tokens for each PHI type
REDACTION_TOKENS: dict[str, str] = {
    "ssn": "[SSN_REDACTED]",
    "mrn": "[MRN_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "dob": "[DOB_REDACTED]",
    "address": "[ADDRESS_REDACTED]",
    "clinical_trial": "[CLINICAL_TRIAL_REDACTED]",
    "drivers_license": "[DL_REDACTED]",
    "passport": "[PASSPORT_REDACTED]",
    "medicare_id": "[MEDICARE_REDACTED]",
    "medicaid_id": "[MEDICAID_REDACTED]",
    "account_number": "[ACCOUNT_REDACTED]",
    "ip_address": "[IP_REDACTED]",
    "credit_card": "[CC_REDACTED]",
    "bank_account": "[BANK_REDACTED]",
    "health_plan_id": "[HEALTH_PLAN_REDACTED]",
    "vin": "[VIN_REDACTED]",
    "biometric_ref": "[BIOMETRIC_REDACTED]",
    "photo_url": "[PHOTO_URL_REDACTED]",
    "name": "[NAME_REDACTED]",
    "patient_admission": "[PATIENT_ADMISSION_REDACTED]",
    "name_with_dob": "[NAME_DOB_REDACTED]",
    "room_patient": "[ROOM_PATIENT_REDACTED]",
    "provider_patient": "[PROVIDER_PATIENT_REDACTED]",
    "contact_for_patient": "[CONTACT_PATIENT_REDACTED]",
    "possessive_identifier": "[IDENTIFIER_REDACTED]",
    "treatment_dates": "[TREATMENT_DATES_REDACTED]",
    "age_context": "[AGE_REDACTED]",
    "geographic": "[ZIP_REDACTED]",
}


# =============================================================================
# CORE DETECTION FUNCTIONS
# =============================================================================

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


def detect_extended_phi(text: str) -> dict[str, list[str]]:
    """
    Detect extended PHI patterns in text.

    Args:
        text: Input text to scan

    Returns:
        Dictionary mapping PHI type to list of detected instances
    """
    detected: dict[str, list[str]] = {}

    for phi_type, pattern in EXTENDED_PHI_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Handle groups in regex
            if matches and isinstance(matches[0], tuple):
                matches = [m[0] if isinstance(m, tuple) else m for m in matches]
            detected[phi_type] = list(matches)

    return detected


def detect_potential_names(text: str) -> list[str]:
    """
    Detect potential patient/provider names in text.

    Uses pattern matching with medical eponym filtering.

    Args:
        text: Input text to scan

    Returns:
        List of potential names detected
    """
    potential_names: list[str] = []

    for pattern_name, pattern in NAME_PATTERNS.items():
        matches = pattern.findall(text)
        for match in matches:
            name = match.strip() if isinstance(match, str) else match[0].strip()

            # Filter out medical eponyms
            name_lower = name.lower()
            if not any(eponym in name_lower for eponym in MEDICAL_EPONYMS):
                potential_names.append(name)

    return list(set(potential_names))  # Deduplicate


def detect_context_aware_phi(text: str) -> dict[str, list[tuple]]:
    """
    Detect PHI using context-aware patterns.

    These patterns identify PHI that appears in natural language context,
    such as "Patient John admitted on 1/15" or "John's SSN is...".

    Args:
        text: Input text to scan

    Returns:
        Dictionary mapping context type to list of detected instances (as tuples)
    """
    detected: dict[str, list[tuple]] = {}

    for context_type, pattern in CONTEXT_AWARE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            detected[context_type] = matches

    return detected


def mask_context_aware_phi(text: str) -> str:
    """
    Mask context-aware PHI patterns in text.

    Args:
        text: Input text containing potential PHI

    Returns:
        Masked text with context-aware PHI redacted
    """
    masked_text = text

    for context_type, pattern in CONTEXT_AWARE_PATTERNS.items():
        token = REDACTION_TOKENS.get(context_type, f"[{context_type.upper()}_REDACTED]")
        masked_text = pattern.sub(token, masked_text)

    return masked_text


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

    # Mask core PHI patterns
    for phi_type, pattern in PHI_PATTERNS.items():
        token = REDACTION_TOKENS.get(phi_type, f"[{phi_type.upper()}_REDACTED]")

        # Find all matches first
        matches = pattern.findall(text)
        if matches:
            for match in matches:
                # Handle tuple groups
                match_str = match[0] if isinstance(match, tuple) else match
                detected_list.append(f"{phi_type}: {match_str}")

        # Replace in text
        masked_text = pattern.sub(token, masked_text)

    # Mask extended PHI patterns
    for phi_type, pattern in EXTENDED_PHI_PATTERNS.items():
        token = REDACTION_TOKENS.get(phi_type, f"[{phi_type.upper()}_REDACTED]")

        matches = pattern.findall(masked_text)
        if matches:
            for match in matches:
                match_str = match[0] if isinstance(match, tuple) else match
                detected_list.append(f"{phi_type}: {match_str}")

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
    extended = detect_extended_phi(text)

    issues: list[str] = []
    for phi_type, instances in {**detected, **extended}.items():
        for instance in instances:
            issues.append(f"Found {phi_type}: {instance}")

    is_safe = len(issues) == 0
    return is_safe, issues


# =============================================================================
# PHI DETECTOR CLASS
# =============================================================================

@dataclass
class ScanStatistics:
    """Statistics for PHI scanning."""
    total_scans: int = 0
    scans_with_phi: int = 0
    phi_by_type: dict[str, int] = field(default_factory=dict)
    last_scan_time: Optional[datetime] = None


class PHIDetector:
    """
    PHI detector class with configurable patterns.

    Provides stateful PHI detection with custom patterns,
    NER-based name detection, and audit logging support.
    """

    def __init__(
        self,
        additional_patterns: dict[str, str] | None = None,
        use_ner: bool = False,
        include_extended: bool = True,
        include_context_aware: bool = True
    ):
        """
        Initialize PHI detector.

        Args:
            additional_patterns: Additional regex patterns to detect
            use_ner: Whether to use NER-based name detection
            include_extended: Whether to include extended PHI patterns
            include_context_aware: Whether to include context-aware patterns
        """
        self.patterns = dict(PHI_PATTERNS)
        self.use_ner = use_ner
        self.include_extended = include_extended
        self.include_context_aware = include_context_aware
        self._statistics = ScanStatistics()

        # Add extended patterns if requested
        if include_extended:
            self.patterns.update(EXTENDED_PHI_PATTERNS)

        # Add context-aware patterns if requested
        if include_context_aware:
            self.context_patterns = dict(CONTEXT_AWARE_PATTERNS)
        else:
            self.context_patterns = {}

        # Add custom patterns
        if additional_patterns:
            for name, pattern in additional_patterns.items():
                self.patterns[name] = re.compile(pattern)
                REDACTION_TOKENS[name] = f"[{name.upper()}_REDACTED]"

        total_patterns = len(self.patterns) + len(self.context_patterns)
        logger.info(
            f"PHI Detector initialized with {total_patterns} patterns "
            f"({len(self.patterns)} standard, {len(self.context_patterns)} context-aware), "
            f"NER={'enabled' if use_ner else 'disabled'}"
        )

    def scan(self, text: str) -> dict[str, Any]:
        """
        Scan text for PHI and return detailed report.

        Args:
            text: Text to scan

        Returns:
            Scan report with detected PHI and risk assessment
        """
        self._statistics.total_scans += 1
        self._statistics.last_scan_time = datetime.now()

        detected: dict[str, list[str]] = {}
        context_detected: dict[str, list[tuple]] = {}

        # Scan with all standard patterns
        for phi_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Handle groups in regex
                if isinstance(matches[0], tuple):
                    matches = [m[0] if isinstance(m, tuple) else m for m in matches]
                detected[phi_type] = list(matches)

                # Update statistics
                self._statistics.phi_by_type[phi_type] = (
                    self._statistics.phi_by_type.get(phi_type, 0) + len(matches)
                )

        # Scan with context-aware patterns
        if self.include_context_aware:
            for context_type, pattern in self.context_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    context_detected[context_type] = matches
                    # Flatten tuples for counting
                    flat_count = len(matches)
                    self._statistics.phi_by_type[context_type] = (
                        self._statistics.phi_by_type.get(context_type, 0) + flat_count
                    )

        # Detect potential names if NER is enabled
        potential_names: list[str] = []
        has_potential_names = False
        if self.use_ner:
            potential_names = detect_potential_names(text)
            has_potential_names = len(potential_names) > 0
            if potential_names:
                detected["name"] = potential_names

        total_instances = sum(len(v) for v in detected.values()) + sum(len(v) for v in context_detected.values())

        if total_instances > 0:
            self._statistics.scans_with_phi += 1

        return {
            "detected": detected,
            "context_detected": context_detected,
            "total_instances": total_instances,
            "is_safe": total_instances == 0 and not has_potential_names,
            "risk_level": self._assess_risk(detected, context_detected),
            "has_potential_names": has_potential_names,
            "potential_names": potential_names,
        }

    def _assess_risk(
        self,
        detected: dict[str, list[str]],
        context_detected: dict[str, list[tuple]] | None = None
    ) -> str:
        """Assess risk level based on detected PHI types."""
        context_detected = context_detected or {}

        if not detected and not context_detected:
            return "none"

        # SSN is highest risk
        if "ssn" in detected:
            return "critical"

        # Credit card or bank account is critical
        if "credit_card" in detected or "bank_account" in detected:
            return "critical"

        # Passport is critical (international identifier)
        if "passport" in detected:
            return "critical"

        # Context-aware patterns with names are high risk
        if any(k in context_detected for k in ["patient_admission", "name_with_dob", "provider_patient"]):
            return "high"

        # Multiple types or MRN is high risk
        if len(detected) + len(context_detected) > 1 or "mrn" in detected:
            return "high"

        # Medicare ID or driver's license is high risk
        if "medicare_id" in detected or "drivers_license" in detected:
            return "high"

        # Health plan IDs or biometric refs are high risk
        if "health_plan_id" in detected or "biometric_ref" in detected:
            return "high"

        return "medium"

    def mask(self, text: str) -> str:
        """
        Mask all PHI in text.

        Args:
            text: Text to mask

        Returns:
            Masked text
        """
        masked_text = text

        # Mask standard patterns
        for phi_type, pattern in self.patterns.items():
            token = REDACTION_TOKENS.get(phi_type, f"[{phi_type.upper()}_REDACTED]")
            masked_text = pattern.sub(token, masked_text)

        # Mask context-aware patterns
        if self.include_context_aware:
            for context_type, pattern in self.context_patterns.items():
                token = REDACTION_TOKENS.get(context_type, f"[{context_type.upper()}_REDACTED]")
                masked_text = pattern.sub(token, masked_text)

        # Mask detected names if NER is enabled
        if self.use_ner:
            names = detect_potential_names(text)
            for name in names:
                masked_text = masked_text.replace(name, "[NAME_REDACTED]")

        return masked_text

    def get_statistics(self) -> dict[str, Any]:
        """Get scanning statistics."""
        return {
            "total_scans": self._statistics.total_scans,
            "scans_with_phi": self._statistics.scans_with_phi,
            "phi_by_type": dict(self._statistics.phi_by_type),
            "last_scan_time": (
                self._statistics.last_scan_time.isoformat()
                if self._statistics.last_scan_time else None
            ),
            "detection_rate": (
                self._statistics.scans_with_phi / self._statistics.total_scans
                if self._statistics.total_scans > 0 else 0
            ),
        }

    def reset_statistics(self):
        """Reset scanning statistics."""
        self._statistics = ScanStatistics()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hipaa_compliant_detector() -> PHIDetector:
    """Create a fully-configured HIPAA-compliant PHI detector."""
    return PHIDetector(
        use_ner=True,
        include_extended=True,
        include_context_aware=True
    )


def quick_scan(text: str) -> bool:
    """
    Quick check if text contains any PHI.

    Args:
        text: Text to check

    Returns:
        True if PHI detected, False if safe
    """
    for pattern in PHI_PATTERNS.values():
        if pattern.search(text):
            return True
    return False
