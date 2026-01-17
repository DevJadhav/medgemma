"""Human escalation gateway for clinical safety.

Implements:
- Critical finding detection (pneumothorax, stroke, MI, etc.)
- Patient safety concern detection (suicide, self-harm)
- Confidence-based escalation
- Uncertainty-based escalation
- Domain-specific thresholds
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EscalationReason(Enum):
    """Reasons for escalating to human review."""
    CRITICAL_FINDING = "critical_finding"
    SAFETY_CONCERN = "safety_concern"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_UNCERTAINTY = "high_uncertainty"
    OUT_OF_SCOPE = "out_of_scope"
    PATIENT_REQUEST = "patient_request"
    SYSTEM_ERROR = "system_error"


@dataclass
class EscalationDecision:
    """Result of escalation evaluation."""
    should_escalate: bool
    reason: Optional[EscalationReason]
    priority: str  # "immediate", "urgent", "routine", "none"
    details: str


class HumanEscalationGateway:
    """
    Gateway to determine when AI outputs require human review.
    
    Evaluates:
    - Critical medical findings requiring immediate attention
    - Patient safety concerns
    - Model confidence and uncertainty
    """
    
    # Critical findings requiring immediate escalation
    CRITICAL_PATTERNS = [
        r"pneumothorax",
        r"tension\s+pneumo",
        r"pulmonary\s+embol",
        r"aortic\s+dissection",
        r"stroke|CVA",
        r"myocardial\s+infarction|heart\s+attack|\bMI\b",
        r"intracranial\s+hemorrhage",
        r"cardiac\s+arrest",
        r"anaphyla",
        r"septic\s+shock",
    ]
    
    # Safety concern patterns in patient communication
    SAFETY_PATTERNS = [
        r"suicid",
        r"self[- ]?harm",
        r"kill\s+(myself|me)",
        r"end(ing)?\s+(my|it|the)\s+life",
        r"want\s+to\s+die",
        r"hurt(ing)?\s+(myself|me)",
        r"overdose",
        r"hopeless",
    ]
    
    # Domain-specific confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        "diagnostic": 0.90,
        "workflow": 0.85,
        "communication": 0.80,
    }
    
    # Uncertainty threshold for escalation
    UNCERTAINTY_THRESHOLD = 0.20
    
    def __init__(self):
        """Initialize the escalation gateway."""
        self.critical_regex = [
            re.compile(p, re.IGNORECASE) for p in self.CRITICAL_PATTERNS
        ]
        self.safety_regex = [
            re.compile(p, re.IGNORECASE) for p in self.SAFETY_PATTERNS
        ]
    
    def evaluate(
        self,
        response: str,
        confidence: float,
        uncertainty: float,
        domain: str,
        user_input: Optional[str] = None
    ) -> EscalationDecision:
        """
        Evaluate if human escalation is needed.
        
        Args:
            response: AI-generated response text
            confidence: Model confidence score (0-1)
            uncertainty: Model uncertainty score (0-1)
            domain: Domain type ("diagnostic", "workflow", "communication")
            user_input: Optional user input to check for safety concerns
            
        Returns:
            EscalationDecision with escalation details
        """
        # Check for critical findings (highest priority)
        for pattern in self.critical_regex:
            if pattern.search(response):
                return EscalationDecision(
                    should_escalate=True,
                    reason=EscalationReason.CRITICAL_FINDING,
                    priority="immediate",
                    details=f"Critical finding detected: {pattern.pattern}"
                )
        
        # Check for safety concerns in patient communication
        if user_input:
            for pattern in self.safety_regex:
                if pattern.search(user_input):
                    return EscalationDecision(
                        should_escalate=True,
                        reason=EscalationReason.SAFETY_CONCERN,
                        priority="immediate",
                        details="Patient safety concern detected"
                    )
        
        # Check confidence threshold
        threshold = self.CONFIDENCE_THRESHOLDS.get(domain, 0.85)
        
        if confidence < threshold:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                priority="urgent" if confidence < 0.70 else "routine",
                details=f"Confidence {confidence:.1%} below threshold {threshold:.1%}"
            )
        
        # Check uncertainty threshold
        if uncertainty > self.UNCERTAINTY_THRESHOLD:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.HIGH_UNCERTAINTY,
                priority="urgent" if uncertainty > 0.30 else "routine",
                details=f"Uncertainty {uncertainty:.1%} exceeds threshold"
            )
        
        # No escalation needed
        return EscalationDecision(
            should_escalate=False,
            reason=None,
            priority="none",
            details="All checks passed"
        )
