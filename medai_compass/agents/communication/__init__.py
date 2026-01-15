"""
Communication Agent module for patient engagement.

This module provides agents for:
- Triaging patient messages
- Health education and information
- Appointment scheduling
- Clinical oversight and review
"""

from medai_compass.agents.communication.agents import (
    AgentResponse,
    ClinicalOversightProxy,
    CommunicationOrchestrator,
    ConversationContext,
    FollowUpSchedulingAgent,
    HealthEducatorAgent,
    MessageCategory,
    PatientMessage,
    TriageAgent,
    TriageResult,
    UrgencyLevel,
)

__all__ = [
    "AgentResponse",
    "ClinicalOversightProxy",
    "CommunicationOrchestrator",
    "ConversationContext",
    "FollowUpSchedulingAgent",
    "HealthEducatorAgent",
    "MessageCategory",
    "PatientMessage",
    "TriageAgent",
    "TriageResult",
    "UrgencyLevel",
]
