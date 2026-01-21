"""
AutoGen Communication Agent for Patient Engagement.

This module implements the patient communication system using AutoGen,
including triage, health education, and clinical oversight agents.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

# AutoGen imports - available when autogen-agentchat is installed
try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None
    UserProxyAgent = None
    GroupChat = None
    GroupChatManager = None


class UrgencyLevel(Enum):
    """Patient message urgency levels."""
    EMERGENCY = "emergency"      # Call 911 immediately
    URGENT = "urgent"            # Same-day appointment needed
    SOON = "soon"                # Within 1-2 days
    ROUTINE = "routine"          # Regular scheduling
    INFORMATIONAL = "informational"  # No action needed


class MessageCategory(Enum):
    """Categories of patient messages."""
    SYMPTOM_REPORT = "symptom_report"
    MEDICATION_QUESTION = "medication_question"
    APPOINTMENT_REQUEST = "appointment_request"
    TEST_RESULTS = "test_results"
    BILLING = "billing"
    GENERAL_HEALTH = "general_health"
    MENTAL_HEALTH = "mental_health"
    EMERGENCY = "emergency"


@dataclass
class TriageResult:
    """Result from triage assessment."""
    urgency: UrgencyLevel
    category: MessageCategory
    recommended_action: str
    requires_human_review: bool
    confidence: float
    reasoning: str
    safety_flags: list[str] = field(default_factory=list)


@dataclass
class PatientMessage:
    """Patient communication message."""
    message_id: str
    patient_id: str
    content: str
    language: str = "en"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    attachments: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from communication agent."""
    message_id: str
    agent_name: str
    content: str
    triage_result: Optional[TriageResult] = None
    requires_clinician_review: bool = False
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationContext:
    """Context for ongoing patient conversation."""
    patient_id: str
    session_id: str
    messages: list[dict] = field(default_factory=list)
    patient_history: dict = field(default_factory=dict)
    current_medications: list[str] = field(default_factory=list)
    active_conditions: list[str] = field(default_factory=list)
    last_visit_date: Optional[str] = None


class TriageAgent:
    """
    Agent for triaging patient messages.
    
    Assesses urgency and categorizes patient communications
    to route to appropriate resources.
    """
    
    # Emergency keywords requiring immediate escalation
    EMERGENCY_KEYWORDS = [
        "chest pain", "can't breathe", "difficulty breathing",
        "stroke", "heart attack", "seizure", "unconscious",
        "severe bleeding", "head injury", "overdose",
        "suicidal", "want to die", "kill myself", "self harm"
    ]
    
    # Urgent symptoms requiring same-day attention
    URGENT_KEYWORDS = [
        "high fever", "severe pain", "vomiting blood",
        "can't keep anything down", "sudden vision changes",
        "severe headache", "chest tightness", "racing heart"
    ]
    
    def __init__(self, model_wrapper: Optional[Any] = None):
        self.model = model_wrapper
        self.name = "TriageAgent"
        
    def triage_message(self, message: PatientMessage) -> TriageResult:
        """
        Triage a patient message.
        
        Args:
            message: PatientMessage to triage
            
        Returns:
            TriageResult with urgency and category
        """
        content_lower = message.content.lower()
        safety_flags = []
        
        # Check for emergency keywords
        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in content_lower:
                if "suicidal" in keyword or "kill" in keyword or "self harm" in keyword:
                    safety_flags.append("mental_health_crisis")
                return TriageResult(
                    urgency=UrgencyLevel.EMERGENCY,
                    category=MessageCategory.EMERGENCY,
                    recommended_action="Immediate escalation to emergency services. Advise patient to call 911.",
                    requires_human_review=True,
                    confidence=0.95,
                    reasoning=f"Emergency keyword detected: {keyword}",
                    safety_flags=safety_flags
                )
        
        # Check for urgent keywords
        for keyword in self.URGENT_KEYWORDS:
            if keyword in content_lower:
                return TriageResult(
                    urgency=UrgencyLevel.URGENT,
                    category=MessageCategory.SYMPTOM_REPORT,
                    recommended_action="Schedule same-day appointment or nurse callback",
                    requires_human_review=True,
                    confidence=0.85,
                    reasoning=f"Urgent symptom detected: {keyword}"
                )
        
        # Categorize based on content
        category = self._categorize_message(content_lower)
        urgency = self._assess_urgency(content_lower, category)
        
        return TriageResult(
            urgency=urgency,
            category=category,
            recommended_action=self._get_recommended_action(urgency, category),
            requires_human_review=urgency in [UrgencyLevel.URGENT, UrgencyLevel.EMERGENCY],
            confidence=0.75,
            reasoning=f"Categorized as {category.value} with {urgency.value} urgency"
        )
    
    def _categorize_message(self, content: str) -> MessageCategory:
        """Categorize message based on content."""
        if any(word in content for word in ["medication", "prescription", "refill", "dose", "side effect"]):
            return MessageCategory.MEDICATION_QUESTION
        elif any(word in content for word in ["appointment", "schedule", "visit", "see doctor"]):
            return MessageCategory.APPOINTMENT_REQUEST
        elif any(word in content for word in ["result", "test", "lab", "bloodwork", "imaging"]):
            return MessageCategory.TEST_RESULTS
        elif any(word in content for word in ["bill", "payment", "insurance", "cost"]):
            return MessageCategory.BILLING
        elif any(word in content for word in ["anxious", "depressed", "stressed", "mental", "therapy"]):
            return MessageCategory.MENTAL_HEALTH
        elif any(word in content for word in ["pain", "fever", "cough", "symptom", "sick"]):
            return MessageCategory.SYMPTOM_REPORT
        else:
            return MessageCategory.GENERAL_HEALTH
    
    def _assess_urgency(self, content: str, category: MessageCategory) -> UrgencyLevel:
        """Assess urgency based on content and category."""
        if category == MessageCategory.MENTAL_HEALTH:
            return UrgencyLevel.SOON
        elif category == MessageCategory.SYMPTOM_REPORT:
            return UrgencyLevel.SOON
        elif category == MessageCategory.MEDICATION_QUESTION:
            if "ran out" in content or "missed" in content:
                return UrgencyLevel.SOON
            return UrgencyLevel.ROUTINE
        elif category == MessageCategory.BILLING:
            return UrgencyLevel.ROUTINE
        else:
            return UrgencyLevel.INFORMATIONAL
    
    def _get_recommended_action(self, urgency: UrgencyLevel, category: MessageCategory) -> str:
        """Get recommended action based on urgency and category."""
        actions = {
            (UrgencyLevel.EMERGENCY, MessageCategory.EMERGENCY): "Call 911 immediately",
            (UrgencyLevel.URGENT, MessageCategory.SYMPTOM_REPORT): "Schedule immediate callback from nurse",
            (UrgencyLevel.SOON, MessageCategory.SYMPTOM_REPORT): "Schedule appointment within 24-48 hours",
            (UrgencyLevel.SOON, MessageCategory.MENTAL_HEALTH): "Connect with behavioral health team",
            (UrgencyLevel.ROUTINE, MessageCategory.MEDICATION_QUESTION): "Route to pharmacy team",
            (UrgencyLevel.ROUTINE, MessageCategory.APPOINTMENT_REQUEST): "Send to scheduling",
            (UrgencyLevel.ROUTINE, MessageCategory.BILLING): "Route to billing department",
        }
        return actions.get((urgency, category), "Provide general health information")


class HealthEducatorAgent:
    """
    Agent for providing health education and information.
    
    Responds to patient questions with evidence-based health
    information while maintaining appropriate disclaimers.
    """
    
    # Standard disclaimer
    DISCLAIMER = (
        "\n\n⚠️ This information is for educational purposes only and "
        "is not a substitute for professional medical advice. Please consult "
        "your healthcare provider for personalized medical guidance."
    )
    
    def __init__(self, model_wrapper: Optional[Any] = None):
        self.model = model_wrapper
        self.name = "HealthEducatorAgent"
        
        # Knowledge base for common health topics
        self.health_topics = self._load_health_topics()
        
    def _load_health_topics(self) -> dict:
        """Load health education content."""
        return {
            "diabetes": {
                "overview": "Diabetes is a chronic condition affecting how your body processes blood sugar (glucose).",
                "management": [
                    "Monitor blood sugar levels regularly",
                    "Take medications as prescribed",
                    "Follow a balanced diet",
                    "Exercise regularly",
                    "Attend regular check-ups"
                ],
                "warning_signs": [
                    "Increased thirst and urination",
                    "Unexplained weight loss",
                    "Blurred vision",
                    "Slow-healing wounds"
                ]
            },
            "hypertension": {
                "overview": "High blood pressure is when the force of blood against artery walls is consistently too high.",
                "management": [
                    "Take blood pressure medications as prescribed",
                    "Reduce sodium intake",
                    "Maintain a healthy weight",
                    "Limit alcohol consumption",
                    "Manage stress"
                ],
                "warning_signs": [
                    "Severe headaches",
                    "Chest pain",
                    "Vision problems",
                    "Difficulty breathing"
                ]
            },
            "medication_adherence": {
                "overview": "Taking medications exactly as prescribed is crucial for managing your health conditions.",
                "tips": [
                    "Use a pill organizer",
                    "Set daily reminders",
                    "Keep medications in a visible location",
                    "Refill prescriptions before running out",
                    "Never stop medications without consulting your doctor"
                ]
            }
        }
    
    def respond_to_query(
        self,
        message: PatientMessage,
        context: Optional[ConversationContext] = None
    ) -> AgentResponse:
        """
        Respond to a patient's health question.
        
        Args:
            message: Patient's message
            context: Conversation context
            
        Returns:
            AgentResponse with educational content
        """
        start_time = datetime.now()
        
        content_lower = message.content.lower()
        response_text = ""
        confidence = 0.7
        
        # Match to knowledge base topics
        for topic, info in self.health_topics.items():
            if topic in content_lower:
                response_text = self._format_topic_response(topic, info)
                confidence = 0.85
                break
        
        # If no specific topic matched
        if not response_text:
            response_text = self._generate_general_response(message.content, context)
            confidence = 0.65
        
        # Add disclaimer
        response_text += self.DISCLAIMER
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AgentResponse(
            message_id=f"resp-{message.message_id}",
            agent_name=self.name,
            content=response_text,
            confidence=confidence,
            requires_clinician_review=confidence < 0.7,
            processing_time_ms=processing_time
        )
    
    def _format_topic_response(self, topic: str, info: dict) -> str:
        """Format response for a specific health topic."""
        response = f"**{topic.replace('_', ' ').title()}**\n\n"
        
        if "overview" in info:
            response += f"{info['overview']}\n\n"
            
        if "management" in info:
            response += "**Management Tips:**\n"
            for tip in info["management"]:
                response += f"• {tip}\n"
            response += "\n"
            
        if "tips" in info:
            response += "**Helpful Tips:**\n"
            for tip in info["tips"]:
                response += f"• {tip}\n"
            response += "\n"
            
        if "warning_signs" in info:
            response += "**Warning Signs to Watch For:**\n"
            for sign in info["warning_signs"]:
                response += f"• {sign}\n"
                
        return response
    
    def _generate_general_response(
        self,
        query: str,
        context: Optional[ConversationContext]
    ) -> str:
        """Generate a general response for unmatched queries."""
        return (
            "Thank you for your question. For the most accurate and personalized "
            "advice regarding your health concerns, I recommend discussing this "
            "with your healthcare provider at your next visit.\n\n"
            "In the meantime, if you're experiencing concerning symptoms, please "
            "don't hesitate to contact our office or seek emergency care if needed."
        )


class ClinicalOversightProxy:
    """
    Clinical oversight proxy for human-in-the-loop review.
    
    Ensures clinician review of AI responses when needed.
    """
    
    def __init__(self):
        self.name = "ClinicalOversightProxy"
        self.pending_reviews: list[dict] = []
        
    def flag_for_review(
        self,
        message: PatientMessage,
        agent_response: AgentResponse,
        reason: str
    ) -> str:
        """
        Flag a response for clinician review.
        
        Args:
            message: Original patient message
            agent_response: AI-generated response
            reason: Reason for review
            
        Returns:
            Review ticket ID
        """
        review_id = f"review-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        review_ticket = {
            "review_id": review_id,
            "patient_message": message,
            "agent_response": agent_response,
            "reason": reason,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "reviewed_by": None,
            "review_notes": None
        }
        
        self.pending_reviews.append(review_ticket)
        
        return review_id
    
    def get_pending_reviews(self) -> list[dict]:
        """Get all pending reviews."""
        return [r for r in self.pending_reviews if r["status"] == "pending"]
    
    def complete_review(
        self,
        review_id: str,
        approved: bool,
        reviewer_id: str,
        notes: Optional[str] = None,
        modified_response: Optional[str] = None
    ) -> bool:
        """
        Complete a review.
        
        Args:
            review_id: Review ticket ID
            approved: Whether response is approved
            reviewer_id: ID of reviewing clinician
            notes: Review notes
            modified_response: Modified response if not approved as-is
            
        Returns:
            True if review completed successfully
        """
        for review in self.pending_reviews:
            if review["review_id"] == review_id:
                review["status"] = "approved" if approved else "modified"
                review["reviewed_by"] = reviewer_id
                review["review_notes"] = notes
                review["reviewed_at"] = datetime.now().isoformat()
                if modified_response:
                    review["modified_response"] = modified_response
                return True
        return False


class FollowUpSchedulingAgent:
    """
    Agent for handling follow-up scheduling requests.
    """
    
    def __init__(self):
        self.name = "FollowUpSchedulingAgent"
        
    def process_scheduling_request(
        self,
        message: PatientMessage,
        context: Optional[ConversationContext] = None
    ) -> AgentResponse:
        """Process a scheduling request from patient."""
        start_time = datetime.now()
        
        # Parse scheduling intent
        scheduling_type = self._detect_scheduling_type(message.content)
        
        response_text = self._generate_scheduling_response(scheduling_type, context)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AgentResponse(
            message_id=f"sched-{message.message_id}",
            agent_name=self.name,
            content=response_text,
            confidence=0.85,
            processing_time_ms=processing_time
        )
    
    def _detect_scheduling_type(self, content: str) -> str:
        """Detect type of scheduling request."""
        content_lower = content.lower()
        
        if "cancel" in content_lower:
            return "cancellation"
        elif "reschedule" in content_lower or "change" in content_lower:
            return "reschedule"
        elif "follow up" in content_lower or "follow-up" in content_lower:
            return "follow_up"
        else:
            return "new_appointment"
    
    def _generate_scheduling_response(
        self,
        scheduling_type: str,
        context: Optional[ConversationContext]
    ) -> str:
        """Generate response for scheduling request."""
        responses = {
            "cancellation": (
                "I understand you need to cancel an appointment. To cancel, please:\n\n"
                "1. Call our scheduling line at (555) 123-4567\n"
                "2. Provide your appointment date and time\n"
                "3. Please give at least 24 hours notice when possible\n\n"
                "Would you like me to have someone from scheduling reach out to you?"
            ),
            "reschedule": (
                "I can help you reschedule your appointment. Here are our available options:\n\n"
                "1. Call our scheduling line at (555) 123-4567\n"
                "2. Use our patient portal to view available times\n"
                "3. Reply with your preferred days/times and we'll find an opening\n\n"
                "Which option works best for you?"
            ),
            "follow_up": (
                "Thank you for requesting a follow-up appointment. Based on your care plan, "
                "I can see you may be due for a check-in.\n\n"
                "Our next available appointments are:\n"
                "• Monday-Friday: 9 AM - 5 PM\n"
                "• Some Saturday mornings available\n\n"
                "Please let me know your preferred date and time, and I'll check availability."
            ),
            "new_appointment": (
                "I'd be happy to help you schedule an appointment. To find the best time:\n\n"
                "1. What type of visit do you need? (check-up, specific concern, etc.)\n"
                "2. Do you have a preferred provider?\n"
                "3. What days/times work best for you?\n\n"
                "Please share these details and I'll find available options."
            )
        }
        
        return responses.get(scheduling_type, responses["new_appointment"])


class CommunicationOrchestrator:
    """
    Orchestrates the communication agent team.
    
    Coordinates triage, education, and oversight agents to
    provide comprehensive patient communication support.
    """
    
    def __init__(self, model_wrapper: Optional[Any] = None):
        self.triage_agent = TriageAgent(model_wrapper)
        self.educator_agent = HealthEducatorAgent(model_wrapper)
        self.scheduling_agent = FollowUpSchedulingAgent()
        self.oversight_proxy = ClinicalOversightProxy()
        
        # Conversation history
        self.conversations: dict[str, ConversationContext] = {}
        
    def process_message(
        self,
        message: PatientMessage,
        patient_context: Optional[dict] = None
    ) -> AgentResponse:
        """
        Process a patient message through the agent team.
        
        Args:
            message: Patient's message
            patient_context: Optional patient context
            
        Returns:
            AgentResponse from appropriate agent
        """
        # Get or create conversation context
        context = self._get_or_create_context(message.patient_id, patient_context)
        
        # Add message to history
        context.messages.append({
            "role": "patient",
            "content": message.content,
            "timestamp": message.timestamp
        })
        
        # Triage the message
        triage_result = self.triage_agent.triage_message(message)
        
        # Handle emergencies
        if triage_result.urgency == UrgencyLevel.EMERGENCY:
            response = self._handle_emergency(message, triage_result)
            # Always flag emergencies for review
            self.oversight_proxy.flag_for_review(
                message, response, "Emergency situation detected"
            )
            return response
        
        # Route to appropriate agent based on category
        if triage_result.category == MessageCategory.APPOINTMENT_REQUEST:
            response = self.scheduling_agent.process_scheduling_request(message, context)
        elif triage_result.category in [
            MessageCategory.GENERAL_HEALTH,
            MessageCategory.MEDICATION_QUESTION,
            MessageCategory.SYMPTOM_REPORT
        ]:
            response = self.educator_agent.respond_to_query(message, context)
        else:
            response = self._generate_routing_response(message, triage_result)
        
        # Add triage result to response
        response.triage_result = triage_result
        
        # Flag for review if needed
        if triage_result.requires_human_review or response.confidence < 0.7:
            self.oversight_proxy.flag_for_review(
                message, response,
                f"Requires review: urgency={triage_result.urgency.value}, confidence={response.confidence}"
            )
            response.requires_clinician_review = True
        
        # Add response to conversation history
        context.messages.append({
            "role": "agent",
            "agent": response.agent_name,
            "content": response.content,
            "timestamp": response.timestamp
        })
        
        return response
    
    def _get_or_create_context(
        self,
        patient_id: str,
        patient_context: Optional[dict]
    ) -> ConversationContext:
        """Get or create conversation context for patient."""
        if patient_id not in self.conversations:
            context = ConversationContext(
                patient_id=patient_id,
                session_id=f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            if patient_context:
                context.patient_history = patient_context.get("history", {})
                context.current_medications = patient_context.get("medications", [])
                context.active_conditions = patient_context.get("conditions", [])
                context.last_visit_date = patient_context.get("last_visit")
            self.conversations[patient_id] = context
            
        return self.conversations[patient_id]
    
    def _handle_emergency(
        self,
        message: PatientMessage,
        triage_result: TriageResult
    ) -> AgentResponse:
        """Handle emergency messages."""
        if "mental_health_crisis" in triage_result.safety_flags:
            content = (
                "🆘 **IMMEDIATE HELP AVAILABLE**\n\n"
                "I'm concerned about what you've shared. Please know that help is available:\n\n"
                "• **988 Suicide & Crisis Lifeline**: Call or text 988\n"
                "• **Crisis Text Line**: Text HOME to 741741\n"
                "• **Emergency Services**: Call 911\n\n"
                "You're not alone, and there are people who want to help. "
                "Please reach out to one of these resources right now.\n\n"
                "A member of our care team will also be contacting you shortly."
            )
        else:
            content = (
                "🚨 **EMERGENCY RESPONSE NEEDED**\n\n"
                "Based on what you've described, this may be a medical emergency.\n\n"
                "**Please call 911 immediately** or have someone take you to the "
                "nearest emergency room.\n\n"
                "Do not wait to be seen - seek emergency care now.\n\n"
                "A member of our care team has been notified and will follow up."
            )
        
        return AgentResponse(
            message_id=f"emergency-{message.message_id}",
            agent_name="TriageAgent",
            content=content,
            triage_result=triage_result,
            requires_clinician_review=True,
            confidence=0.95
        )
    
    def _generate_routing_response(
        self,
        message: PatientMessage,
        triage_result: TriageResult
    ) -> AgentResponse:
        """Generate response for messages being routed to other departments."""
        routing_messages = {
            MessageCategory.BILLING: (
                "I see you have a billing-related question. Our billing department "
                "is best equipped to help you with this.\n\n"
                "You can reach them at:\n"
                "• Phone: (555) 123-4568\n"
                "• Email: billing@clinic.example.com\n"
                "• Hours: Monday-Friday, 8 AM - 6 PM\n\n"
                "Would you like me to request a callback from billing?"
            ),
            MessageCategory.TEST_RESULTS: (
                "I understand you're asking about test results. For the security "
                "of your health information, test results are best discussed "
                "directly with your care team.\n\n"
                "You can:\n"
                "• View results in your patient portal\n"
                "• Request a callback from your provider's office\n"
                "• Schedule a follow-up appointment to discuss\n\n"
                "How would you like to proceed?"
            )
        }
        
        content = routing_messages.get(
            triage_result.category,
            "Thank you for your message. I'll make sure it gets to the right team member. "
            "You can expect a response within 1-2 business days."
        )
        
        return AgentResponse(
            message_id=f"route-{message.message_id}",
            agent_name="CommunicationOrchestrator",
            content=content,
            confidence=0.8
        )
    
    def get_conversation_history(self, patient_id: str) -> list[dict]:
        """Get conversation history for a patient."""
        if patient_id in self.conversations:
            return self.conversations[patient_id].messages
        return []

    def get_conversation_summary(self, patient_id: str) -> dict:
        """
        Get a summary of the conversation with a patient.

        Args:
            patient_id: Patient ID

        Returns:
            Dictionary with conversation summary
        """
        if patient_id not in self.conversations:
            return {"error": "No conversation found"}

        context = self.conversations[patient_id]
        messages = context.messages

        # Count message types
        patient_messages = [m for m in messages if m.get("role") == "patient"]
        agent_messages = [m for m in messages if m.get("role") == "agent"]

        # Get unique topics discussed
        topics = set()
        for msg in messages:
            content = msg.get("content", "").lower()
            for topic in ["medication", "appointment", "symptom", "billing", "test result"]:
                if topic in content:
                    topics.add(topic)

        return {
            "patient_id": patient_id,
            "session_id": context.session_id,
            "total_messages": len(messages),
            "patient_messages": len(patient_messages),
            "agent_messages": len(agent_messages),
            "topics_discussed": list(topics),
            "active_conditions": context.active_conditions,
            "current_medications": context.current_medications,
            "last_message_time": messages[-1].get("timestamp") if messages else None
        }

    def clear_conversation(self, patient_id: str) -> bool:
        """
        Clear conversation history for a patient.

        Args:
            patient_id: Patient ID

        Returns:
            True if cleared, False if not found
        """
        if patient_id in self.conversations:
            del self.conversations[patient_id]
            return True
        return False

    def export_conversation(self, patient_id: str) -> dict:
        """
        Export full conversation data for archiving.

        Args:
            patient_id: Patient ID

        Returns:
            Full conversation export
        """
        if patient_id not in self.conversations:
            return {"error": "No conversation found"}

        context = self.conversations[patient_id]

        return {
            "patient_id": context.patient_id,
            "session_id": context.session_id,
            "messages": context.messages,
            "patient_history": context.patient_history,
            "current_medications": context.current_medications,
            "active_conditions": context.active_conditions,
            "last_visit_date": context.last_visit_date,
            "exported_at": datetime.now().isoformat()
        }


class MultiLanguageSupport:
    """
    Multi-language support for patient communication.

    Provides translation and language detection for
    supporting diverse patient populations.
    """

    # Supported languages
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "zh": "Chinese",
        "vi": "Vietnamese",
        "ko": "Korean",
        "tl": "Tagalog",
        "ar": "Arabic",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese"
    }

    # Common medical phrases in multiple languages
    MEDICAL_PHRASES = {
        "emergency": {
            "en": "This is a medical emergency. Please call 911 immediately.",
            "es": "Esta es una emergencia medica. Por favor llame al 911 inmediatamente.",
            "zh": "This is a medical emergency. Please call 911.",
            "vi": "Day la tinh huong khan cap y te. Xin vui long goi 911 ngay lap tuc.",
        },
        "disclaimer": {
            "en": "This information is for educational purposes only. Please consult your healthcare provider.",
            "es": "Esta informacion es solo para fines educativos. Consulte a su proveedor de atencion medica.",
            "zh": "This information is educational only. Please consult your doctor.",
            "vi": "Thong tin nay chi mang tinh chat giao duc. Vui long tham khao y kien bac si.",
        },
        "appointment_confirm": {
            "en": "Your appointment has been scheduled.",
            "es": "Su cita ha sido programada.",
            "zh": "Your appointment is confirmed.",
            "vi": "Cuoc hen cua ban da duoc len lich.",
        }
    }

    def __init__(self):
        self.default_language = "en"

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.

        Args:
            text: Input text

        Returns:
            Language code (default: 'en')
        """
        # Simple detection based on character sets
        text_lower = text.lower()

        # Check for Spanish indicators
        spanish_words = ["hola", "por favor", "gracias", "como", "esta", "dolor", "cita"]
        if any(word in text_lower for word in spanish_words):
            return "es"

        # Check for Chinese characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"

        # Check for Vietnamese diacritics
        vietnamese_chars = "ăâđêôơưàảãạáằẳẵặắầẩẫậấ"
        if any(char in text_lower for char in vietnamese_chars):
            return "vi"

        # Check for Korean characters
        if any('\uac00' <= char <= '\ud7af' for char in text):
            return "ko"

        # Check for Japanese characters
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return "ja"

        # Check for Arabic characters
        if any('\u0600' <= char <= '\u06ff' for char in text):
            return "ar"

        # Default to English
        return "en"

    def get_phrase(self, phrase_key: str, language: str = "en") -> str:
        """
        Get a pre-translated medical phrase.

        Args:
            phrase_key: Key for the phrase
            language: Target language code

        Returns:
            Translated phrase or English fallback
        """
        if phrase_key not in self.MEDICAL_PHRASES:
            return ""

        phrases = self.MEDICAL_PHRASES[phrase_key]
        return phrases.get(language, phrases.get("en", ""))

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()

    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported."""
        return language_code in self.SUPPORTED_LANGUAGES

    def get_language_name(self, language_code: str) -> str:
        """Get full name for a language code."""
        return self.SUPPORTED_LANGUAGES.get(language_code, "Unknown")


class ConversationManager:
    """
    Centralized conversation management with persistence support.

    Manages conversation state, history, and context across
    multiple patient interactions with Redis/PostgreSQL persistence
    for multi-instance deployment.
    """

    def __init__(self, max_history_length: int = 100, use_persistence: bool = True):
        """
        Initialize conversation manager.

        Args:
            max_history_length: Maximum messages to retain per conversation
            use_persistence: Whether to use Redis/PostgreSQL persistence
        """
        self.conversations: dict[str, ConversationContext] = {}
        self.max_history_length = max_history_length
        self.language_support = MultiLanguageSupport()
        self.use_persistence = use_persistence
        self._persistence_store = None
    
    async def _get_persistence_store(self):
        """Get persistence store, creating if needed."""
        if not self.use_persistence:
            return None
        
        if self._persistence_store is None:
            try:
                from medai_compass.utils.persistence import get_conversation_store
                self._persistence_store = get_conversation_store()
            except ImportError:
                self._persistence_store = None
        
        return self._persistence_store

    async def create_session(
        self,
        patient_id: str,
        patient_context: Optional[dict] = None,
        language: str = "en"
    ) -> ConversationContext:
        """
        Create a new conversation session with persistence.

        Args:
            patient_id: Patient identifier
            patient_context: Optional patient context data
            language: Preferred language

        Returns:
            New ConversationContext
        """
        session_id = f"conv-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        context = ConversationContext(
            patient_id=patient_id,
            session_id=session_id
        )

        if patient_context:
            context.patient_history = patient_context.get("history", {})
            context.current_medications = patient_context.get("medications", [])
            context.active_conditions = patient_context.get("conditions", [])
            context.last_visit_date = patient_context.get("last_visit")

        # Store in memory
        self.conversations[patient_id] = context
        
        # Persist to storage
        store = await self._get_persistence_store()
        if store:
            await store.save_conversation(
                session_id=session_id,
                patient_id=patient_id,
                state={
                    "patient_history": context.patient_history,
                    "current_medications": context.current_medications,
                    "active_conditions": context.active_conditions,
                    "last_visit_date": context.last_visit_date,
                    "language": language
                },
                messages=[],
                context=patient_context
            )
        
        return context
    
    def create_session_sync(
        self,
        patient_id: str,
        patient_context: Optional[dict] = None,
        language: str = "en"
    ) -> ConversationContext:
        """
        Synchronous version of create_session for non-async contexts.
        
        Note: This does not persist to storage. Use async version when possible.
        """
        session_id = f"conv-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        context = ConversationContext(
            patient_id=patient_id,
            session_id=session_id
        )

        if patient_context:
            context.patient_history = patient_context.get("history", {})
            context.current_medications = patient_context.get("medications", [])
            context.active_conditions = patient_context.get("conditions", [])
            context.last_visit_date = patient_context.get("last_visit")

        self.conversations[patient_id] = context
        return context

    async def add_message(
        self,
        patient_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
        agent_name: Optional[str] = None,
        triage_result: Optional[dict] = None,
        requires_review: bool = False
    ) -> bool:
        """
        Add a message to conversation history with persistence.

        Args:
            patient_id: Patient identifier
            role: Message role ('patient' or 'agent')
            content: Message content
            metadata: Optional additional metadata
            agent_name: Name of responding agent
            triage_result: Optional triage assessment
            requires_review: Whether message needs clinician review

        Returns:
            True if added successfully
        """
        if patient_id not in self.conversations:
            # Try to load from persistence
            await self._load_conversation(patient_id)
        
        if patient_id not in self.conversations:
            return False

        context = self.conversations[patient_id]

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "requires_review": requires_review
        }

        if metadata:
            message.update(metadata)
        
        if triage_result:
            message["triage_result"] = triage_result

        context.messages.append(message)

        # Trim history if needed
        if len(context.messages) > self.max_history_length:
            context.messages = context.messages[-self.max_history_length:]

        # Persist message
        store = await self._get_persistence_store()
        if store:
            await store.add_message(
                session_id=context.session_id,
                role=role,
                content=content,
                agent_name=agent_name,
                triage_result=triage_result,
                requires_review=requires_review
            )

        return True
    
    def add_message_sync(
        self,
        patient_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Synchronous version of add_message for non-async contexts.
        
        Note: This does not persist to storage. Use async version when possible.
        """
        if patient_id not in self.conversations:
            return False

        context = self.conversations[patient_id]

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        if metadata:
            message.update(metadata)

        context.messages.append(message)

        if len(context.messages) > self.max_history_length:
            context.messages = context.messages[-self.max_history_length:]

        return True
    
    async def _load_conversation(self, patient_id: str) -> bool:
        """Load conversation from persistence store."""
        store = await self._get_persistence_store()
        if not store:
            return False
        
        # Get most recent conversation for patient
        conversations = await store.get_patient_conversations(patient_id, limit=1)
        if not conversations:
            return False
        
        conv_data = conversations[0]
        session_id = conv_data.get("session_id")
        
        # Get full conversation data
        full_data = await store.get_conversation(session_id)
        if not full_data:
            return False
        
        # Reconstruct context
        state = full_data.get("state", {})
        context = ConversationContext(
            patient_id=patient_id,
            session_id=session_id,
            messages=full_data.get("messages", []),
            patient_history=state.get("patient_history", {}),
            current_medications=state.get("current_medications", []),
            active_conditions=state.get("active_conditions", []),
            last_visit_date=state.get("last_visit_date")
        )
        
        self.conversations[patient_id] = context
        return True

    def get_context_for_prompt(self, patient_id: str, max_messages: int = 10) -> str:
        """
        Get formatted context for inclusion in prompts.

        Args:
            patient_id: Patient identifier
            max_messages: Maximum recent messages to include

        Returns:
            Formatted context string
        """
        if patient_id not in self.conversations:
            return ""

        context = self.conversations[patient_id]
        recent_messages = context.messages[-max_messages:]

        context_parts = []

        # Add patient info
        if context.active_conditions:
            context_parts.append(f"Active conditions: {', '.join(context.active_conditions)}")
        if context.current_medications:
            context_parts.append(f"Current medications: {', '.join(context.current_medications)}")

        # Add recent conversation
        if recent_messages:
            context_parts.append("\nRecent conversation:")
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_parts.append(f"  {role}: {content[:200]}...")

        return "\n".join(context_parts)

    async def get_session(self, patient_id: str) -> Optional[ConversationContext]:
        """Get existing session for patient, loading from persistence if needed."""
        if patient_id not in self.conversations:
            await self._load_conversation(patient_id)
        return self.conversations.get(patient_id)
    
    def get_session_sync(self, patient_id: str) -> Optional[ConversationContext]:
        """Synchronous version of get_session."""
        return self.conversations.get(patient_id)

    async def end_session(self, patient_id: str) -> Optional[dict]:
        """
        End a session and return final state.

        Args:
            patient_id: Patient identifier

        Returns:
            Final session data or None
        """
        if patient_id not in self.conversations:
            return None

        context = self.conversations[patient_id]
        final_state = {
            "patient_id": context.patient_id,
            "session_id": context.session_id,
            "message_count": len(context.messages),
            "ended_at": datetime.now().isoformat()
        }

        # Note: We don't delete from persistence to keep history
        del self.conversations[patient_id]
        return final_state
