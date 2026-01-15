"""
Workflow Agent module for clinical operations.

This module provides agents for:
- Scheduling appointments
- Generating clinical documentation
- Processing prior authorizations
"""

from medai_compass.agents.workflow.crew import (
    AgentRole,
    AppointmentRequest,
    DocumentationRequest,
    DocumenterAgent,
    PriorAuthAgent,
    PriorAuthRequest,
    SchedulerAgent,
    WorkflowCrew,
    WorkflowResult,
)

__all__ = [
    "AgentRole",
    "AppointmentRequest",
    "DocumentationRequest",
    "DocumenterAgent",
    "PriorAuthAgent",
    "PriorAuthRequest",
    "SchedulerAgent",
    "WorkflowCrew",
    "WorkflowResult",
]
