# ADR-005: Guardrails Architecture

## Status
Accepted

## Date
2025-12-15

## Context

As a medical AI platform, MedAI Compass requires robust guardrails to:

1. Protect patient privacy (HIPAA compliance)
2. Prevent harmful outputs (medical safety)
3. Block adversarial attacks (jailbreaking)
4. Ensure appropriate escalation (human-in-loop)

### Risk Categories

| Risk | Severity | Frequency | Mitigation Priority |
|------|----------|-----------|---------------------|
| PHI Exposure | Critical | Medium | Highest |
| Jailbreak Attacks | High | Low | High |
| Hallucinated Medical Advice | Critical | Medium | Highest |
| Missed Critical Findings | Critical | Low | Highest |
| Out-of-Scope Queries | Low | High | Medium |

## Decision

We will implement a **layered guardrails architecture** with five components:

### Layer 1: Input Rails (Pre-Processing)

```
User Input → PHI Detection → Jailbreak Detection → Scope Validation → Agent
```

**Components**:
- `phi_detection.py`: 30+ regex patterns + NER-based name detection
- `input_rails.py`: 8 jailbreak categories + encoding bypass detection
- Scope validation: Filter non-medical queries

### Layer 2: NeMo Guardrails Integration

```python
# Colang-based policy enforcement
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./config/guardrails")
rails = LLMRails(config)

response = await rails.generate(
    messages=[{"role": "user", "content": user_input}]
)
```

### Layer 3: Output Rails (Post-Processing)

```
Agent Output → Confidence Check → Disclaimer Injection → PHI Validation → Response
```

**Components**:
- `output_rails.py`: Medical disclaimer injection
- `uncertainty.py`: Confidence scoring and flagging
- PHI re-validation: Ensure no PHI leaks in output

### Layer 4: Escalation Gateway

```python
ESCALATION_TRIGGERS = {
    "critical_findings": ["pneumothorax", "stroke", "mi", "aortic_dissection"],
    "safety_concerns": ["suicidal", "self_harm", "abuse"],
    "low_confidence": {
        "diagnostic": 0.90,
        "workflow": 0.85,
        "communication": 0.80
    },
    "high_uncertainty": 0.20
}
```

### Layer 5: Audit Logging

All guardrail actions are logged with tamper-evident hash chains for HIPAA compliance.

## Implementation

### PHI Detection Patterns

```python
# Core HIPAA identifiers
PHI_PATTERNS = {
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "mrn": r'\bMRN[:\s]*(\d{6,10})\b',
    "phone": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "email": r'\b[\w.-]+@[\w.-]+\.\w+\b',
    "dob": r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}\b',
    "address": r'\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|...)\b',
}

# Extended patterns (11 additional)
EXTENDED_PHI_PATTERNS = {
    "clinical_trial": r'\b(NCT\d{8})\b',
    "drivers_license": r'\bDL#?\s*([A-Z]?\d{6,9})\b',
    "medicare_id": r'\b(\d[A-Z]{2}\d-[A-Z]{2}\d-[A-Z]{2}\d{2})\b',
    # ... 8 more patterns
}

# Context-aware patterns
CONTEXT_PATTERNS = {
    "patient_admission": r'patient\s+([A-Z][a-z]+)\s+admitted\s+on\s+(\d{1,2}/\d{1,2})',
    "name_with_dob": r'([A-Z][a-z]+),?\s+DOB[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})',
    # ... 4 more patterns
}
```

### Jailbreak Detection Categories

```python
JAILBREAK_CATEGORIES = {
    "prompt_injection": ["ignore previous", "disregard instructions", "new task"],
    "role_play": ["pretend you are", "act as if", "you are now"],
    "encoding_bypass": [base64, rot13, hex, url_encoding],
    "instruction_override": ["override", "bypass", "disable safety"],
    "boundary_testing": ["what if", "hypothetically", "in theory"],
    "social_engineering": ["trust me", "this is important", "emergency"],
    "hypothetical": ["imagine that", "suppose", "let's say"],
    "multi_turn": ["remember earlier", "as we discussed", "continuing from"]
}

# Encoding detection
def detect_encoding_bypass(text: str) -> Tuple[bool, str]:
    decoders = [
        ("base64", base64.b64decode),
        ("rot13", codecs.decode(..., "rot_13")),
        ("hex", bytes.fromhex),
        ("url", urllib.parse.unquote),
    ]
    for name, decoder in decoders:
        try:
            decoded = decoder(text)
            if decoded != text:
                return True, f"Encoded content detected: {name}"
        except:
            continue
    return False, ""
```

### Escalation Logic

```python
def should_escalate(response: AgentResponse) -> Tuple[bool, str]:
    """Determine if response requires human review."""

    # Check critical findings
    for finding in response.findings:
        if finding.type in CRITICAL_FINDINGS:
            return True, f"Critical finding: {finding.type}"

    # Check confidence threshold
    threshold = CONFIDENCE_THRESHOLDS[response.domain]
    if response.confidence < threshold:
        return True, f"Low confidence: {response.confidence:.2f}"

    # Check uncertainty
    if response.uncertainty > MAX_UNCERTAINTY:
        return True, f"High uncertainty: {response.uncertainty:.2f}"

    # Check safety concerns
    for keyword in SAFETY_KEYWORDS:
        if keyword in response.content.lower():
            return True, f"Safety concern: {keyword}"

    return False, ""
```

## Consequences

### Positive
- Comprehensive protection against multiple risk categories
- Layered defense provides redundancy
- HIPAA-compliant audit trail
- Configurable thresholds per domain

### Negative
- Latency overhead from multiple checks (~50-100ms)
- Potential for false positives blocking legitimate queries
- Maintenance burden for pattern updates
- Complex testing requirements

### Mitigation
- Parallel execution of independent checks
- Confidence-based bypass for repeat queries
- Regular pattern review and update process
- Comprehensive test suite with known attack patterns

## Performance Impact

| Check | Latency | False Positive Rate |
|-------|---------|---------------------|
| PHI Detection | 5ms | < 0.1% |
| Jailbreak Detection | 10ms | < 1% |
| NeMo Guardrails | 30ms | < 0.5% |
| Confidence Check | 2ms | N/A |
| Total | ~50ms | < 1% |

## Alternatives Considered

### 1. Single LLM-Based Filter
- **Rejected**: Too slow and expensive for high-volume
- Rule-based patterns are faster for common cases

### 2. Post-Hoc Only Filtering
- **Rejected**: PHI should never enter the system
- Pre-processing is essential for HIPAA

### 3. External Guardrails Service
- **Rejected**: Adds network latency and dependency
- Embedded guardrails are faster and more reliable

## References

- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [HIPAA Safe Harbor](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
