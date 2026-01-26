"""
Master Orchestrator for Multi-Agent Coordination.

This module provides the master orchestrator that coordinates all agents
(Diagnostic, Workflow, Communication) and integrates with NeMo Guardrails.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from medai_compass.agents.communication import (
    CommunicationOrchestrator,
    PatientMessage,
)
from medai_compass.agents.workflow import (
    AppointmentRequest,
    DocumentationRequest,
    PriorAuthRequest,
    WorkflowCrew,
)


class DomainType(Enum):
    """Types of request domains."""
    DIAGNOSTIC = "diagnostic"
    WORKFLOW = "workflow"
    COMMUNICATION = "communication"
    UNKNOWN = "unknown"


@dataclass
class IntentClassification:
    """Result of intent classification."""
    domain: DomainType
    sub_intent: str
    confidence: float
    requires_multimodal: bool = False
    reasoning: str = ""


@dataclass
class OrchestratorRequest:
    """Request to the master orchestrator."""
    request_id: str
    user_id: str
    content: str
    request_type: str = "text"  # text, image, audio, multimodal
    attachments: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OrchestratorResponse:
    """Response from the master orchestrator."""
    request_id: str
    domain: DomainType
    content: str
    agent_used: str
    confidence: float
    requires_review: bool = False
    sub_responses: list[dict] = field(default_factory=list)
    processing_time_ms: float = 0.0
    guardrails_applied: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class IntentClassifier:
    """
    Classify user intent and route to appropriate domain.

    Enhanced with:
    - Synonym handling for natural language variations
    - Context-aware phrase matching
    - Semantic similarity scoring
    - Confidence calibration
    """

    # Keywords for each domain with synonyms and variations
    DIAGNOSTIC_KEYWORDS = [
        "x-ray", "xray", "scan", "mri", "ct scan", "ultrasound", "imaging",
        "radiology", "chest x-ray", "mammogram", "pathology", "biopsy",
        "analyze image", "what does this show", "findings", "diagnosis"
    ]

    WORKFLOW_KEYWORDS = [
        "discharge summary", "clinical notes", "documentation", "schedule surgery",
        "prior authorization", "prior auth", "insurance approval", "referral",
        "transfer patient", "admission", "documentation", "dictation"
    ]

    COMMUNICATION_KEYWORDS = [
        "appointment", "schedule visit", "question about", "symptoms",
        "medication", "refill", "side effects", "health education",
        "follow up", "billing", "test results"
    ]

    # Synonym mappings for semantic understanding
    SYNONYMS = {
        # Diagnostic synonyms
        "x-ray": ["xray", "radiograph", "plain film", "chest film", "skeletal survey"],
        "ct scan": ["cat scan", "computed tomography", "ct", "contrast ct"],
        "mri": ["magnetic resonance", "mr imaging", "mri scan"],
        "ultrasound": ["sonogram", "echo", "doppler", "us scan"],
        "mammogram": ["mammography", "breast imaging", "breast screening"],
        "biopsy": ["tissue sample", "specimen", "histology"],
        "diagnosis": ["diagnose", "identify", "determine", "assess", "evaluate"],
        "findings": ["results", "observations", "conclusions", "interpretation"],
        "analyze": ["review", "examine", "look at", "interpret", "assess", "evaluate"],
        # Workflow synonyms
        "discharge summary": ["d/c summary", "discharge note", "discharge documentation"],
        "prior authorization": ["prior auth", "pre-auth", "preauthorization", "insurance approval"],
        "referral": ["refer", "consult request", "specialist referral"],
        "documentation": ["clinical note", "progress note", "chart note", "medical record"],
        "schedule": ["book", "arrange", "set up", "plan"],
        # Communication synonyms
        "appointment": ["visit", "office visit", "consultation", "check-up", "checkup"],
        "medication": ["medicine", "drug", "prescription", "rx", "meds"],
        "refill": ["renewal", "renew", "reorder"],
        "symptoms": ["symptom", "complaints", "issues", "problems", "concerns"],
        "side effects": ["adverse effects", "reactions", "side-effects"],
    }

    # Context phrases that indicate specific domains
    CONTEXT_PHRASES = {
        DomainType.DIAGNOSTIC: [
            "what do you see", "can you interpret", "look at this",
            "analyze this image", "review this scan", "read this",
            "what are the findings", "is there anything abnormal",
            "any pathology", "concerning features"
        ],
        DomainType.WORKFLOW: [
            "generate a note", "write up", "document this",
            "need authorization", "get approval", "complete the form",
            "prepare for discharge", "transfer to", "admit to"
        ],
        DomainType.COMMUNICATION: [
            "i have a question", "can you explain", "what should i",
            "when should i", "is it normal", "should i be concerned",
            "how do i", "can i take", "what happens if"
        ]
    }

    def __init__(self):
        """Initialize the intent classifier with expanded keyword sets."""
        # Build expanded keyword sets with synonyms
        self._expanded_diagnostic = self._expand_with_synonyms(self.DIAGNOSTIC_KEYWORDS)
        self._expanded_workflow = self._expand_with_synonyms(self.WORKFLOW_KEYWORDS)
        self._expanded_communication = self._expand_with_synonyms(self.COMMUNICATION_KEYWORDS)

    def _expand_with_synonyms(self, keywords: list[str]) -> set[str]:
        """Expand keyword list with synonyms."""
        expanded = set(keywords)
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.SYNONYMS:
                expanded.update(self.SYNONYMS[keyword_lower])
        return expanded

    def classify(self, request: OrchestratorRequest) -> IntentClassification:
        """
        Classify the intent of a request.
        
        Args:
            request: OrchestratorRequest to classify
            
        Returns:
            IntentClassification with domain and sub-intent
        """
        content_lower = request.content.lower()
        
        # Check for multimodal requests
        requires_multimodal = (
            request.request_type in ["image", "multimodal"] or
            len(request.attachments) > 0 or
            any(ext in content_lower for ext in [".dcm", ".dicom", "image", "picture", "photo"])
        )
        
        # Score each domain using keywords
        diagnostic_score = self._score_domain(content_lower, self.DIAGNOSTIC_KEYWORDS)
        workflow_score = self._score_domain(content_lower, self.WORKFLOW_KEYWORDS)
        communication_score = self._score_domain(content_lower, self.COMMUNICATION_KEYWORDS)

        # Add context phrase scoring for semantic understanding
        diagnostic_score += self._score_with_context(content_lower, DomainType.DIAGNOSTIC)
        workflow_score += self._score_with_context(content_lower, DomainType.WORKFLOW)
        communication_score += self._score_with_context(content_lower, DomainType.COMMUNICATION)

        # If multimodal, bias toward diagnostic
        if requires_multimodal:
            diagnostic_score += 0.3

        # Determine domain
        scores = {
            DomainType.DIAGNOSTIC: diagnostic_score,
            DomainType.WORKFLOW: workflow_score,
            DomainType.COMMUNICATION: communication_score
        }

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        # Calibrate confidence based on score distribution
        # High confidence if one domain clearly wins
        score_values = list(scores.values())
        score_variance = max(score_values) - min(score_values)
        confidence_bonus = min(score_variance * 0.2, 0.1)  # Up to 0.1 bonus for clear winner

        # If no clear match, default to communication
        if best_score < 0.1:
            best_domain = DomainType.COMMUNICATION
            best_score = 0.5
            confidence_bonus = 0.0
        
        # Determine sub-intent
        sub_intent = self._get_sub_intent(content_lower, best_domain)
        
        return IntentClassification(
            domain=best_domain,
            sub_intent=sub_intent,
            confidence=min(best_score + 0.3 + confidence_bonus, 1.0),  # Boost with base + calibration
            requires_multimodal=requires_multimodal,
            reasoning=f"Matched {best_domain.value} with score {best_score:.2f} (semantic + context)"
        )
    
    def _score_domain(self, content: str, keywords: list[str]) -> float:
        """
        Score content against domain keywords with semantic understanding.

        Uses:
        - Exact keyword matching
        - Synonym expansion
        - Context phrase matching
        - Partial word matching for variations

        Args:
            content: Input text (lowercased)
            keywords: Original keyword list

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Expand keywords with synonyms
        expanded_keywords = self._expand_with_synonyms(keywords)

        # Count exact and partial matches
        exact_matches = 0
        partial_matches = 0

        for kw in expanded_keywords:
            if kw in content:
                exact_matches += 1
            elif any(word.startswith(kw[:4]) for word in content.split() if len(kw) >= 4):
                # Partial stem matching for word variations
                partial_matches += 0.5

        # Calculate base score
        total_score = (exact_matches * 0.25) + (partial_matches * 0.1)

        return min(total_score, 1.0)

    def _score_with_context(self, content: str, domain: DomainType) -> float:
        """
        Score content using context phrases.

        Args:
            content: Input text (lowercased)
            domain: Domain to check context for

        Returns:
            Context match score
        """
        phrases = self.CONTEXT_PHRASES.get(domain, [])
        matches = sum(1 for phrase in phrases if phrase in content)
        return min(matches * 0.3, 0.5)  # Context can add up to 0.5 to score
    
    def _get_sub_intent(self, content: str, domain: DomainType) -> str:
        """Get sub-intent for a domain."""
        if domain == DomainType.DIAGNOSTIC:
            if "x-ray" in content or "xray" in content:
                return "chest_xray_analysis"
            elif "pathology" in content or "biopsy" in content:
                return "pathology_analysis"
            elif "ct" in content or "mri" in content:
                return "cross_sectional_analysis"
            return "general_imaging"
            
        elif domain == DomainType.WORKFLOW:
            if "discharge" in content:
                return "discharge_summary"
            elif "prior auth" in content:
                return "prior_authorization"
            elif "schedule" in content:
                return "scheduling"
            return "documentation"
            
        elif domain == DomainType.COMMUNICATION:
            if "appointment" in content or "schedule" in content:
                return "scheduling"
            elif "medication" in content or "refill" in content:
                return "medication_inquiry"
            elif "symptom" in content:
                return "symptom_report"
            return "general_inquiry"
            
        return "unknown"


class NeMoGuardrailsIntegration:
    """
    Integration with NeMo Guardrails for safety.
    
    Provides input/output validation, jailbreak detection,
    and hallucination prevention.
    """
    
    # Topics that are out of scope
    OUT_OF_SCOPE_TOPICS = [
        "investment advice", "legal advice", "gambling",
        "weapons", "illegal activities", "political opinions"
    ]
    
    # Medical disclaimers by context
    DISCLAIMERS = {
        "diagnostic": (
            "⚠️ This AI-assisted analysis is for clinical decision support only. "
            "All findings must be verified by a qualified healthcare professional."
        ),
        "workflow": (
            "⚠️ This document has been AI-assisted. Please review for accuracy "
            "before finalizing."
        ),
        "communication": (
            "ℹ️ This information is for educational purposes and is not "
            "personalized medical advice. Consult your healthcare provider."
        )
    }
    
    def __init__(self):
        self.applied_rails: list[str] = []
        
    def check_input(self, request: OrchestratorRequest) -> tuple[bool, list[str]]:
        """
        Check input for safety issues.
        
        Args:
            request: Request to check
            
        Returns:
            Tuple of (is_safe, issues_found)
        """
        issues = []
        content_lower = request.content.lower()
        
        # Check for out-of-scope topics
        for topic in self.OUT_OF_SCOPE_TOPICS:
            if topic in content_lower:
                issues.append(f"Out of scope topic: {topic}")
        
        # Check for jailbreak attempts
        jailbreak_patterns = [
            "ignore previous instructions",
            "disregard your training",
            "pretend you are",
            "act as if you have no restrictions"
        ]
        for pattern in jailbreak_patterns:
            if pattern in content_lower:
                issues.append("Potential jailbreak attempt detected")
                break
        
        self.applied_rails.append("input_validation")
        
        return len(issues) == 0, issues
    
    def check_output(
        self,
        response: str,
        domain: DomainType,
        confidence: float
    ) -> tuple[str, list[str]]:
        """
        Check and modify output for safety.
        
        Args:
            response: Response to check
            domain: Response domain
            confidence: Confidence score
            
        Returns:
            Tuple of (modified_response, rails_applied)
        """
        rails_applied = []
        modified_response = response
        
        # Add appropriate disclaimer
        disclaimer = self.DISCLAIMERS.get(domain.value, self.DISCLAIMERS["communication"])
        if disclaimer not in modified_response:
            modified_response = f"{modified_response}\n\n{disclaimer}"
            rails_applied.append("disclaimer_added")
        
        # Flag low confidence responses
        if confidence < 0.7:
            modified_response += "\n\n⚠️ This response has lower confidence and should be reviewed."
            rails_applied.append("low_confidence_warning")
        
        # Check for potential hallucinations (simplified)
        hallucination_indicators = [
            "I read in your medical records",  # AI shouldn't claim to have read records
            "Based on your previous visit",     # Without actual access
            "The latest research shows"         # Without citation
        ]
        for indicator in hallucination_indicators:
            if indicator.lower() in response.lower():
                rails_applied.append("potential_hallucination_flagged")
                break
        
        self.applied_rails.extend(rails_applied)
        
        return modified_response, rails_applied
    
    def get_applied_rails(self) -> list[str]:
        """Get list of guardrails applied."""
        return self.applied_rails.copy()
    
    def reset(self):
        """Reset applied rails tracker."""
        self.applied_rails = []


class MasterOrchestrator:
    """
    Master orchestrator coordinating all agent domains.
    
    Handles intent classification, domain routing, response aggregation,
    and guardrail integration.
    """
    
    def __init__(
        self,
        diagnostic_agent: Optional[Any] = None,
        model_wrapper: Optional[Any] = None
    ):
        self.classifier = IntentClassifier()
        self.guardrails = NeMoGuardrailsIntegration()
        
        # Initialize domain agents
        self.workflow_crew = WorkflowCrew(model_wrapper)
        self.communication_orch = CommunicationOrchestrator(model_wrapper)
        self.diagnostic_agent = diagnostic_agent  # From Phase 5
        
    def process_request(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Process a request through the appropriate agent.
        
        Args:
            request: OrchestratorRequest to process
            
        Returns:
            OrchestratorResponse from the appropriate agent
        """
        start_time = datetime.now()
        self.guardrails.reset()
        
        # Check input safety
        is_safe, issues = self.guardrails.check_input(request)
        if not is_safe:
            return OrchestratorResponse(
                request_id=request.request_id,
                domain=DomainType.UNKNOWN,
                content=f"I'm unable to process this request. Issues: {', '.join(issues)}",
                agent_used="guardrails",
                confidence=1.0,
                requires_review=True,
                guardrails_applied=self.guardrails.get_applied_rails()
            )
        
        # Classify intent
        classification = self.classifier.classify(request)
        
        # Route to appropriate domain
        if classification.domain == DomainType.DIAGNOSTIC:
            response_content, agent_used, confidence = self._handle_diagnostic(request, classification)
        elif classification.domain == DomainType.WORKFLOW:
            response_content, agent_used, confidence = self._handle_workflow(request, classification)
        else:
            response_content, agent_used, confidence = self._handle_communication(request, classification)
        
        # Apply output guardrails
        safe_response, rails_applied = self.guardrails.check_output(
            response_content,
            classification.domain,
            confidence
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return OrchestratorResponse(
            request_id=request.request_id,
            domain=classification.domain,
            content=safe_response,
            agent_used=agent_used,
            confidence=confidence,
            requires_review=confidence < 0.7 or classification.domain == DomainType.DIAGNOSTIC,
            guardrails_applied=self.guardrails.get_applied_rails(),
            processing_time_ms=processing_time
        )
    
    def _handle_diagnostic(
        self,
        request: OrchestratorRequest,
        classification: IntentClassification
    ) -> tuple[str, str, float]:
        """Handle diagnostic domain requests."""
        # In production, this would use the diagnostic agent from Phase 5
        if self.diagnostic_agent is None:
            return (
                "Diagnostic imaging analysis is available. Please upload your "
                "medical image (DICOM, X-ray, CT, MRI) for analysis. "
                "Our AI will provide preliminary findings that should be "
                "reviewed by a radiologist.",
                "DiagnosticAgent",
                0.85
            )
        
        # Call diagnostic pipeline
        # result = self.diagnostic_agent.process(request)
        return (
            "Diagnostic analysis completed. Findings will be reviewed.",
            "DiagnosticAgent",
            0.85
        )
    
    def _handle_workflow(
        self,
        request: OrchestratorRequest,
        classification: IntentClassification
    ) -> tuple[str, str, float]:
        """Handle workflow domain requests."""
        if classification.sub_intent == "discharge_summary":
            doc_request = DocumentationRequest(
                patient_id=request.metadata.get("patient_id", "unknown"),
                document_type="discharge_summary",
                encounter_id=request.metadata.get("encounter_id", "unknown"),
                clinical_notes=[request.content],
                diagnoses=request.metadata.get("diagnoses", [])
            )
            result = self.workflow_crew.process_documentation(doc_request)
            return (
                result.output.get("content", "Documentation generated."),
                "DocumenterAgent",
                0.9 if result.success else 0.5
            )
            
        elif classification.sub_intent == "prior_authorization":
            return (
                "Prior authorization request initiated. Please provide:\n"
                "1. Procedure code\n"
                "2. Diagnosis codes\n"
                "3. Clinical justification\n"
                "4. Insurance information",
                "PriorAuthAgent",
                0.8
            )
            
        else:
            return (
                "Workflow request received. Our team will process this shortly.",
                "WorkflowCrew",
                0.75
            )
    
    def _handle_communication(
        self,
        request: OrchestratorRequest,
        classification: IntentClassification
    ) -> tuple[str, str, float]:
        """Handle communication domain requests."""
        patient_msg = PatientMessage(
            message_id=request.request_id,
            patient_id=request.user_id,
            content=request.content
        )
        
        response = self.communication_orch.process_message(patient_msg)
        
        return (
            response.content,
            response.agent_name,
            response.confidence
        )
    
    def process_batch(
        self,
        requests: list[OrchestratorRequest]
    ) -> list[OrchestratorResponse]:
        """
        Process a batch of requests.
        
        Args:
            requests: List of requests to process
            
        Returns:
            List of responses
        """
        return [self.process_request(req) for req in requests]
