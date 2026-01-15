# Multi-Agent Design

MedAI Compass uses a **multi-agent architecture** where specialized agents handle different clinical domains, coordinated by a master orchestrator.

## Agent Framework Choices

| Domain | Framework | Rationale |
|--------|-----------|-----------|
| Diagnostic | **LangGraph** | Stateful workflows, checkpointing for long-running analysis |
| Workflow | **CrewAI** | Task-based coordination, role delegation |
| Communication | **AutoGen** | Conversational agents, human-in-the-loop |

## Diagnostic Agent (LangGraph)

### State Management

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class DiagnosticState(TypedDict):
    image_path: str
    preprocessed_image: bytes
    findings: list[dict]
    confidence: float
    report: str
    requires_review: bool
```

### Workflow Graph

```
START ──▶ preprocess ──▶ analyze ──▶ generate_report
                              │
                              ▼
                    confidence < 0.8?
                         │       │
                        YES     NO
                         │       │
                         ▼       ▼
                   escalate   complete
```

### Nodes

1. **Preprocessing Node**: DICOM loading, windowing, normalization
2. **Analysis Node**: MedGemma/CXR Foundation inference
3. **Report Node**: Structured finding generation
4. **Routing Node**: Confidence-based escalation

---

## Workflow Agent (CrewAI)

### Agent Roles

```python
from medai_compass.agents.workflow import (
    SchedulerAgent,
    DocumenterAgent,
    PriorAuthAgent,
    WorkflowCrew
)

# Initialize crew
crew = WorkflowCrew()

# Process documentation request
result = crew.process_documentation(DocumentationRequest(
    patient_id="P001",
    document_type="discharge_summary",
    encounter_id="ENC001",
    clinical_notes=["Patient stable", "Ready for discharge"],
    diagnoses=[{"code": "I10", "display": "Hypertension"}]
))
```

### Agent Capabilities

| Agent | Responsibilities |
|-------|------------------|
| **SchedulerAgent** | Appointment scheduling, urgency handling, pre-visit instructions |
| **DocumenterAgent** | Discharge summaries, progress notes, referral letters, clinical summarization |
| **PriorAuthAgent** | Prior authorization, clinical justification scoring, auto-approval logic |

### Coordination Logic

```python
# Complex workflow with dependencies
results = crew.process_complex_workflow(
    prior_auth_request=auth_req,      # Must complete first
    scheduling_request=sched_req,     # Blocked if auth fails
    documentation_request=doc_req     # Independent
)
```

---

## Communication Agent (AutoGen)

### Agent Team

```python
from medai_compass.agents.communication import (
    TriageAgent,
    HealthEducatorAgent,
    FollowUpSchedulingAgent,
    ClinicalOversightProxy,
    CommunicationOrchestrator
)

# Initialize orchestrator
comm = CommunicationOrchestrator()

# Process patient message
response = comm.process_message(PatientMessage(
    message_id="msg-001",
    patient_id="pat-001",
    content="I've been having chest pain"
))
```

### Triage Classification

| Urgency | Response Time | Examples |
|---------|---------------|----------|
| **EMERGENCY** | Immediate (911) | Chest pain, stroke symptoms, suicidal ideation |
| **URGENT** | Same-day | High fever, severe pain, vomiting blood |
| **SOON** | 24-48 hours | Symptom reports, mental health concerns |
| **ROUTINE** | Standard scheduling | Medication questions, refills |
| **INFORMATIONAL** | Self-service | General health education |

### Safety Features

- **Emergency Detection**: Keywords for chest pain, stroke, mental health crisis
- **Disclaimer Injection**: All responses include medical advice disclaimers
- **Human Review Queue**: Low-confidence or flagged responses

---

## Inter-Agent Communication

### Message Flow

```
Patient ──▶ Master Orchestrator
                    │
                    ├──▶ Intent Classification
                    │
                    ├──▶ Domain Routing
                    │         │
                    │    ┌────┴────┬────────────┐
                    │    ▼         ▼            ▼
                    │ Diagnostic  Workflow  Communication
                    │    │         │            │
                    │    └────┬────┴────────────┘
                    │         ▼
                    ├──▶ Response Aggregation
                    │
                    └──▶ Output Guardrails
                              │
                              ▼
                          Patient
```

### Cross-Domain Handoffs

```python
# Example: Diagnostic finding triggers communication
if diagnostic_result.critical_finding:
    # Escalate to communication for patient notification
    comm.process_message(PatientMessage(
        content=f"Important finding: {diagnostic_result.summary}"
    ))
    
    # Also trigger workflow for documentation
    workflow.process_documentation(DocumentationRequest(
        document_type="critical_finding",
        clinical_notes=[diagnostic_result.report]
    ))
```
