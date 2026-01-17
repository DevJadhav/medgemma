"""
CrewAI Workflow Agent for Clinical Operations.

This module implements the clinical workflow automation using CrewAI,
including scheduling, documentation, and prior authorization agents.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

# CrewAI imports - will be available when crewai is installed
try:
    from crewai import Agent, Crew, Task
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    Crew = None
    Task = None


class AgentRole(Enum):
    """Roles for workflow agents."""
    SCHEDULER = "scheduler"
    DOCUMENTER = "documenter"
    PRIOR_AUTH = "prior_authorization"


@dataclass
class AppointmentRequest:
    """Request for scheduling an appointment."""
    patient_id: str
    appointment_type: str
    preferred_dates: list[str]
    provider_id: Optional[str] = None
    urgency: str = "routine"  # routine, urgent, emergency
    notes: Optional[str] = None


@dataclass
class DocumentationRequest:
    """Request for clinical documentation generation."""
    patient_id: str
    document_type: str  # discharge_summary, progress_note, referral_letter
    encounter_id: str
    clinical_notes: list[str]
    diagnoses: list[dict]
    procedures: list[dict] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)


@dataclass
class PriorAuthRequest:
    """Request for prior authorization."""
    patient_id: str
    procedure_code: str
    diagnosis_codes: list[str]
    insurance_id: str
    provider_id: str
    clinical_justification: str
    supporting_documents: list[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Result from a workflow agent task."""
    success: bool
    agent_role: AgentRole
    task_id: str
    output: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SchedulerAgent:
    """
    Agent responsible for intelligent appointment scheduling.
    
    Uses MedGemma 4B for understanding scheduling constraints and
    optimizing appointment slots.
    """
    
    def __init__(self, model_wrapper: Optional[Any] = None):
        self.model = model_wrapper
        self.role = AgentRole.SCHEDULER
        
    def process_request(self, request: AppointmentRequest) -> WorkflowResult:
        """
        Process an appointment scheduling request.
        
        Args:
            request: AppointmentRequest with scheduling details
            
        Returns:
            WorkflowResult with scheduling outcome
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Analyze scheduling constraints
            constraints = self._analyze_constraints(request)
            
            # Find optimal slot
            optimal_slot = self._find_optimal_slot(
                request.preferred_dates,
                request.appointment_type,
                request.urgency,
                constraints
            )
            
            # Generate confirmation
            confirmation = self._generate_confirmation(request, optimal_slot)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return WorkflowResult(
                success=True,
                agent_role=self.role,
                task_id=f"sched-{request.patient_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                output={
                    "scheduled_date": optimal_slot["date"],
                    "scheduled_time": optimal_slot["time"],
                    "provider_id": optimal_slot.get("provider_id", request.provider_id),
                    "confirmation_number": confirmation["number"],
                    "instructions": confirmation["instructions"]
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return WorkflowResult(
                success=False,
                agent_role=self.role,
                task_id=f"sched-{request.patient_id}-failed",
                output={},
                errors=[str(e)]
            )
    
    def _analyze_constraints(self, request: AppointmentRequest) -> dict:
        """Analyze scheduling constraints from request."""
        constraints = {
            "urgency_priority": {
                "emergency": 0,
                "urgent": 1,
                "routine": 2
            }.get(request.urgency, 2),
            "requires_specialist": request.appointment_type in [
                "specialist_consult", "surgery", "procedure"
            ],
            "duration_minutes": self._estimate_duration(request.appointment_type)
        }
        return constraints
    
    def _estimate_duration(self, appointment_type: str) -> int:
        """Estimate appointment duration in minutes."""
        durations = {
            "new_patient": 60,
            "follow_up": 30,
            "specialist_consult": 45,
            "procedure": 90,
            "surgery": 180,
            "annual_exam": 45,
            "urgent_care": 30
        }
        return durations.get(appointment_type, 30)
    
    def _find_optimal_slot(
        self,
        preferred_dates: list[str],
        appointment_type: str,
        urgency: str,
        constraints: dict
    ) -> dict:
        """Find the optimal appointment slot."""
        # In production, this would query the scheduling system
        # For now, return a mock optimal slot
        if preferred_dates:
            selected_date = preferred_dates[0]
        else:
            selected_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
        return {
            "date": selected_date,
            "time": "09:00",
            "provider_id": "DR-001",
            "duration_minutes": constraints["duration_minutes"]
        }
    
    def _generate_confirmation(self, request: AppointmentRequest, slot: dict) -> dict:
        """Generate appointment confirmation."""
        return {
            "number": f"APT-{request.patient_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "instructions": self._get_instructions(request.appointment_type)
        }
    
    def _get_instructions(self, appointment_type: str) -> str:
        """Get pre-appointment instructions."""
        instructions = {
            "new_patient": "Please arrive 15 minutes early with your insurance card and ID. Bring a list of current medications.",
            "follow_up": "Please arrive 5 minutes before your scheduled time.",
            "annual_exam": "Please fast for 12 hours before your appointment if blood work is needed.",
            "procedure": "You will receive detailed instructions 24 hours before your procedure. Do not eat or drink after midnight.",
            "surgery": "Pre-operative instructions will be provided separately. A nurse will call you 2 days before."
        }
        return instructions.get(appointment_type, "Please arrive 10 minutes before your scheduled time.")


class DocumenterAgent:
    """
    Agent responsible for clinical documentation generation.
    
    Uses MedGemma 27B for complex medical documentation including
    discharge summaries, progress notes, and referral letters.
    """
    
    def __init__(self, model_wrapper: Optional[Any] = None, use_llm: bool = True):
        self.model = model_wrapper
        self.role = AgentRole.DOCUMENTER
        self.use_llm = use_llm
        self._inference_service = None
    
    async def _get_inference_service(self):
        """Get MedGemma 27B inference service."""
        if self._inference_service is None and self.use_llm:
            try:
                from medai_compass.models.inference_service import MedGemmaInferenceService
                self._inference_service = MedGemmaInferenceService(
                    model_name="google/medgemma-27b-it"
                )
                await self._inference_service.initialize()
            except Exception as e:
                import logging
                logging.warning(f"Could not initialize MedGemma 27B: {e}")
                self._inference_service = None
        return self._inference_service
        
    async def process_request_async(self, request: DocumentationRequest) -> WorkflowResult:
        """
        Process a documentation generation request using MedGemma 27B.
        
        Args:
            request: DocumentationRequest with documentation details
            
        Returns:
            WorkflowResult with generated document
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            inference_service = await self._get_inference_service()
            
            if inference_service:
                # Use MedGemma 27B for generation
                document = await self._generate_with_llm(request, inference_service)
            else:
                # Fall back to template-based generation
                if request.document_type == "discharge_summary":
                    document = self._generate_discharge_summary(request)
                elif request.document_type == "progress_note":
                    document = self._generate_progress_note(request)
                elif request.document_type == "referral_letter":
                    document = self._generate_referral_letter(request)
                else:
                    document = self._generate_generic_document(request)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return WorkflowResult(
                success=True,
                agent_role=self.role,
                task_id=f"doc-{request.encounter_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                output={
                    "document_type": request.document_type,
                    "document_id": document["id"],
                    "content": document["content"],
                    "summary": document["summary"],
                    "word_count": len(document["content"].split()),
                    "generated_by": document.get("generated_by", "template")
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return WorkflowResult(
                success=False,
                agent_role=self.role,
                task_id=f"doc-{request.encounter_id}-failed",
                output={},
                errors=[str(e)]
            )
    
    async def _generate_with_llm(
        self,
        request: DocumentationRequest,
        inference_service
    ) -> dict:
        """Generate document using MedGemma 27B."""
        # Build comprehensive prompt
        diagnoses_text = ", ".join([
            d.get("display", d.get("code", "Unknown")) 
            for d in request.diagnoses
        ])
        procedures_text = ", ".join([
            p.get("display", p.get("code", "None")) 
            for p in request.procedures
        ]) or "None"
        medications_list = [
            m.get("display", m.get("code", "Unknown")) 
            for m in request.medications
        ]
        
        prompt = f"""Generate a professional {request.document_type.replace('_', ' ')} for the following clinical case.

Patient ID: {request.patient_id}
Encounter ID: {request.encounter_id}
Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

CLINICAL NOTES:
{chr(10).join(request.clinical_notes)}

DIAGNOSES:
{diagnoses_text}

PROCEDURES:
{procedures_text}

CURRENT MEDICATIONS:
{chr(10).join(f'- {med}' for med in medications_list) if medications_list else '- None'}

Please generate a complete, professional {request.document_type.replace('_', ' ')} following standard medical documentation format. Include all relevant sections and use appropriate medical terminology."""

        system_prompt = """You are an experienced medical documentation specialist. 
Generate clear, accurate, and comprehensive clinical documents following standard medical documentation practices.
Use professional medical terminology and ensure all required sections are included.
Do not include any information not provided in the clinical notes."""

        result = await inference_service.generate(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.2,
            system_prompt=system_prompt
        )
        
        if result.error:
            # Fall back to template
            return self._generate_discharge_summary(request) if request.document_type == "discharge_summary" else self._generate_generic_document(request)
        
        return {
            "id": f"{request.document_type.upper()[:3]}-{request.encounter_id}",
            "content": result.response,
            "summary": f"AI-generated {request.document_type.replace('_', ' ')} for patient {request.patient_id}",
            "generated_by": f"medgemma-27b ({result.backend})",
            "confidence": result.confidence
        }
        
    def process_request(self, request: DocumentationRequest) -> WorkflowResult:
        """
        Process a documentation generation request (sync version).
        
        Falls back to template-based generation.
        
        Args:
            request: DocumentationRequest with documentation details
            
        Returns:
            WorkflowResult with generated document
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Generate document based on type
            if request.document_type == "discharge_summary":
                document = self._generate_discharge_summary(request)
            elif request.document_type == "progress_note":
                document = self._generate_progress_note(request)
            elif request.document_type == "referral_letter":
                document = self._generate_referral_letter(request)
            else:
                document = self._generate_generic_document(request)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return WorkflowResult(
                success=True,
                agent_role=self.role,
                task_id=f"doc-{request.encounter_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                output={
                    "document_type": request.document_type,
                    "document_id": document["id"],
                    "content": document["content"],
                    "summary": document["summary"],
                    "word_count": len(document["content"].split())
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return WorkflowResult(
                success=False,
                agent_role=self.role,
                task_id=f"doc-{request.encounter_id}-failed",
                output={},
                errors=[str(e)]
            )
    
    def _generate_discharge_summary(self, request: DocumentationRequest) -> dict:
        """Generate a discharge summary."""
        # In production, this would use MedGemma 27B
        diagnoses_text = ", ".join([d.get("display", d.get("code", "Unknown")) for d in request.diagnoses])
        procedures_text = ", ".join([p.get("display", p.get("code", "None")) for p in request.procedures]) or "None"
        medications_text = "\n".join([f"- {m.get('display', m.get('code', 'Unknown'))}" for m in request.medications]) or "- None"
        
        content = f"""
DISCHARGE SUMMARY

Patient ID: {request.patient_id}
Encounter ID: {request.encounter_id}
Date of Discharge: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

ADMISSION DIAGNOSES:
{diagnoses_text}

PROCEDURES PERFORMED:
{procedures_text}

HOSPITAL COURSE:
{' '.join(request.clinical_notes)}

DISCHARGE MEDICATIONS:
{medications_text}

FOLLOW-UP INSTRUCTIONS:
- Follow up with primary care provider within 1 week
- Return to emergency department if symptoms worsen
- Continue prescribed medications as directed

CONDITION AT DISCHARGE: Stable
""".strip()
        
        return {
            "id": f"DS-{request.encounter_id}",
            "content": content,
            "summary": f"Discharge summary for patient {request.patient_id} with diagnoses: {diagnoses_text}"
        }
    
    def _generate_progress_note(self, request: DocumentationRequest) -> dict:
        """Generate a progress note."""
        diagnoses_text = ", ".join([d.get("display", d.get("code", "Unknown")) for d in request.diagnoses])
        
        content = f"""
PROGRESS NOTE

Patient ID: {request.patient_id}
Encounter ID: {request.encounter_id}
Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}

SUBJECTIVE:
{request.clinical_notes[0] if request.clinical_notes else 'No subjective notes provided.'}

OBJECTIVE:
{request.clinical_notes[1] if len(request.clinical_notes) > 1 else 'Vital signs stable.'}

ASSESSMENT:
{diagnoses_text}

PLAN:
{request.clinical_notes[2] if len(request.clinical_notes) > 2 else 'Continue current treatment plan.'}
""".strip()
        
        return {
            "id": f"PN-{request.encounter_id}",
            "content": content,
            "summary": f"Progress note for patient {request.patient_id}"
        }
    
    def _generate_referral_letter(self, request: DocumentationRequest) -> dict:
        """Generate a referral letter."""
        diagnoses_text = ", ".join([d.get("display", d.get("code", "Unknown")) for d in request.diagnoses])
        
        content = f"""
REFERRAL LETTER

Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

RE: {request.patient_id}

Dear Colleague,

I am referring this patient for specialist evaluation.

REASON FOR REFERRAL:
{diagnoses_text}

CLINICAL SUMMARY:
{' '.join(request.clinical_notes)}

CURRENT MEDICATIONS:
{', '.join([m.get('display', 'Unknown') for m in request.medications]) or 'None'}

Please evaluate and advise on management.

Thank you for your assistance with this patient's care.

Sincerely,
[Provider Signature]
""".strip()
        
        return {
            "id": f"REF-{request.encounter_id}",
            "content": content,
            "summary": f"Referral letter for patient {request.patient_id} regarding {diagnoses_text}"
        }
    
    def _generate_generic_document(self, request: DocumentationRequest) -> dict:
        """Generate a generic clinical document."""
        content = f"""
CLINICAL DOCUMENT

Patient ID: {request.patient_id}
Encounter ID: {request.encounter_id}
Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
Document Type: {request.document_type}

CLINICAL NOTES:
{' '.join(request.clinical_notes)}

DIAGNOSES:
{', '.join([d.get('display', d.get('code', 'Unknown')) for d in request.diagnoses])}
""".strip()
        
        return {
            "id": f"DOC-{request.encounter_id}",
            "content": content,
            "summary": f"Clinical document for patient {request.patient_id}"
        }
    
    def summarize_clinical_notes(self, notes: list[str]) -> str:
        """
        Summarize clinical notes using the model.
        
        Args:
            notes: List of clinical notes to summarize
            
        Returns:
            Summarized notes as a single string
        """
        if not notes:
            return "No clinical notes provided."
            
        # In production, this would use MedGemma for summarization
        # For now, combine and truncate
        combined = " ".join(notes)
        if len(combined) > 500:
            return combined[:500] + "..."
        return combined


class PriorAuthAgent:
    """
    Agent responsible for prior authorization processing.
    
    Uses MedGemma 4B for analyzing clinical justifications and
    generating authorization requests.
    """
    
    def __init__(self, model_wrapper: Optional[Any] = None):
        self.model = model_wrapper
        self.role = AgentRole.PRIOR_AUTH
        
        # Common procedures requiring prior auth
        self.auth_required_procedures = {
            "27447": "Total knee replacement",
            "27130": "Total hip arthroplasty",
            "70553": "MRI brain with contrast",
            "74177": "CT abdomen/pelvis with contrast",
            "43239": "Upper GI endoscopy with biopsy"
        }
        
    def process_request(self, request: PriorAuthRequest) -> WorkflowResult:
        """
        Process a prior authorization request.
        
        Args:
            request: PriorAuthRequest with authorization details
            
        Returns:
            WorkflowResult with authorization status
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check if prior auth is required
            auth_required = self._check_auth_required(request.procedure_code)
            
            if not auth_required:
                return WorkflowResult(
                    success=True,
                    agent_role=self.role,
                    task_id=f"auth-{request.patient_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                    output={
                        "status": "not_required",
                        "message": "Prior authorization not required for this procedure"
                    },
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            
            # Validate clinical justification
            justification_score = self._validate_justification(
                request.clinical_justification,
                request.diagnosis_codes,
                request.procedure_code
            )
            
            # Generate authorization request
            auth_request = self._generate_auth_request(request, justification_score)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return WorkflowResult(
                success=True,
                agent_role=self.role,
                task_id=f"auth-{request.patient_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                output={
                    "status": auth_request["status"],
                    "authorization_number": auth_request.get("auth_number"),
                    "justification_score": justification_score,
                    "recommendation": auth_request["recommendation"],
                    "next_steps": auth_request["next_steps"]
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            return WorkflowResult(
                success=False,
                agent_role=self.role,
                task_id=f"auth-{request.patient_id}-failed",
                output={},
                errors=[str(e)]
            )
    
    def _check_auth_required(self, procedure_code: str) -> bool:
        """Check if prior authorization is required for procedure."""
        return procedure_code in self.auth_required_procedures
    
    def _validate_justification(
        self,
        justification: str,
        diagnosis_codes: list[str],
        procedure_code: str
    ) -> float:
        """
        Validate clinical justification.
        
        Returns a score from 0-1 indicating strength of justification.
        """
        score = 0.5  # Base score
        
        # Check justification length
        if len(justification) > 100:
            score += 0.1
        if len(justification) > 300:
            score += 0.1
            
        # Check for diagnosis codes mentioned
        if diagnosis_codes:
            score += 0.1
            
        # Check for key clinical terms
        clinical_terms = [
            "medical necessity", "failed conservative", "clinically indicated",
            "patient symptoms", "diagnostic findings", "treatment failure"
        ]
        justification_lower = justification.lower()
        for term in clinical_terms:
            if term in justification_lower:
                score += 0.05
                
        return min(score, 1.0)
    
    def _generate_auth_request(self, request: PriorAuthRequest, justification_score: float) -> dict:
        """Generate prior authorization request."""
        if justification_score >= 0.8:
            status = "auto_approved"
            recommendation = "Strong clinical justification supports approval"
            next_steps = ["Authorization granted", "Proceed with scheduling procedure"]
            auth_number = f"AUTH-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{request.procedure_code}"
        elif justification_score >= 0.6:
            status = "pending_review"
            recommendation = "Additional documentation may strengthen the request"
            next_steps = ["Request submitted for medical director review", "Expected response within 3 business days"]
            auth_number = f"PEND-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{request.procedure_code}"
        else:
            status = "needs_info"
            recommendation = "Additional clinical justification required"
            next_steps = [
                "Provide additional documentation",
                "Include prior treatment attempts",
                "Document failed conservative measures"
            ]
            auth_number = None
            
        return {
            "status": status,
            "auth_number": auth_number,
            "recommendation": recommendation,
            "next_steps": next_steps
        }


class WorkflowCrew:
    """
    CrewAI-based workflow coordination.
    
    Coordinates multiple agents to handle complex clinical workflows.
    """
    
    def __init__(self, model_wrapper: Optional[Any] = None):
        self.scheduler = SchedulerAgent(model_wrapper)
        self.documenter = DocumenterAgent(model_wrapper)
        self.prior_auth = PriorAuthAgent(model_wrapper)
        
    def process_scheduling(self, request: AppointmentRequest) -> WorkflowResult:
        """Process a scheduling request through the scheduler agent."""
        return self.scheduler.process_request(request)
    
    def process_documentation(self, request: DocumentationRequest) -> WorkflowResult:
        """Process a documentation request through the documenter agent."""
        return self.documenter.process_request(request)
    
    def process_prior_auth(self, request: PriorAuthRequest) -> WorkflowResult:
        """Process a prior authorization through the prior auth agent."""
        return self.prior_auth.process_request(request)
    
    def process_complex_workflow(
        self,
        scheduling_request: Optional[AppointmentRequest] = None,
        documentation_request: Optional[DocumentationRequest] = None,
        prior_auth_request: Optional[PriorAuthRequest] = None
    ) -> dict[str, WorkflowResult]:
        """
        Process a complex workflow involving multiple agents.
        
        Args:
            scheduling_request: Optional scheduling request
            documentation_request: Optional documentation request
            prior_auth_request: Optional prior auth request
            
        Returns:
            Dictionary of results keyed by agent role
        """
        results = {}
        
        # Process prior auth first (may block scheduling)
        if prior_auth_request:
            results["prior_auth"] = self.process_prior_auth(prior_auth_request)
            
            # If prior auth fails, don't proceed with scheduling
            if not results["prior_auth"].success or results["prior_auth"].output.get("status") == "needs_info":
                if scheduling_request:
                    results["scheduling"] = WorkflowResult(
                        success=False,
                        agent_role=AgentRole.SCHEDULER,
                        task_id="blocked",
                        output={},
                        errors=["Scheduling blocked pending prior authorization"]
                    )
        
        # Process scheduling if not blocked
        if scheduling_request and "scheduling" not in results:
            results["scheduling"] = self.process_scheduling(scheduling_request)
            
        # Process documentation
        if documentation_request:
            results["documentation"] = self.process_documentation(documentation_request)
            
        return results
