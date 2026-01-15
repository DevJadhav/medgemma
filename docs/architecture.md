# System Architecture

MedAI Compass is a **multi-agent medical AI platform** built on Google's Health AI Developer Foundations (HAI-DEF) models, designed for clinical decision support, workflow automation, and patient engagement.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MedAI Compass Platform                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │   Patient   │───▶│                 Master Orchestrator                  │ │
│  │   Portal    │    │  ┌───────────────┬──────────────┬────────────────┐  │ │
│  └─────────────┘    │  │ Intent        │ NeMo         │ Response       │  │ │
│                     │  │ Classifier    │ Guardrails   │ Aggregator     │  │ │
│  ┌─────────────┐    │  └───────────────┴──────────────┴────────────────┘  │ │
│  │   Clinician │───▶│                          │                          │ │
│  │   Dashboard │    └──────────────────────────┼──────────────────────────┘ │
│  └─────────────┘                               │                             │
│                                                ▼                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Agent Domain Layer                             │  │
│  ├──────────────────┬──────────────────┬────────────────────────────────┤  │
│  │  Diagnostic      │  Workflow        │  Communication                 │  │
│  │  Agent           │  Agent           │  Agent                         │  │
│  │  (LangGraph)     │  (CrewAI)        │  (AutoGen)                     │  │
│  │                  │                  │                                │  │
│  │  • Image         │  • Scheduler     │  • Triage                      │  │
│  │    Preprocessing │  • Documenter    │  • Health Educator             │  │
│  │  • MedGemma      │  • Prior Auth    │  • Follow-up Scheduler         │  │
│  │    Inference     │  • Coordination  │  • Clinical Oversight          │  │
│  │  • Report Gen    │                  │                                │  │
│  └──────────────────┴──────────────────┴────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Model & Safety Layer                           │  │
│  ├──────────────────┬──────────────────┬────────────────────────────────┤  │
│  │  Model Wrappers  │  Guardrails      │  Human Escalation               │  │
│  │  • MedGemma      │  • PHI Detection │  • Critical Findings            │  │
│  │  • CXR Foundation│  • Input/Output  │  • Uncertainty Threshold        │  │
│  │  • Path Found.   │  • Uncertainty   │  • Clinician Review Queue       │  │
│  └──────────────────┴──────────────────┴────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Infrastructure Layer                           │  │
│  ├──────────────────┬──────────────────┬────────────────────────────────┤  │
│  │  Inference       │  Storage         │  Monitoring                     │  │
│  │  • vLLM          │  • PostgreSQL    │  • Prometheus                   │  │
│  │  • Triton        │  • MinIO (DICOM) │  • Grafana                      │  │
│  │                  │  • Redis         │  • Audit Logging                │  │
│  └──────────────────┴──────────────────┴────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. Master Orchestrator

The central coordination layer that:
- **Classifies intent** to route requests to appropriate agents
- **Applies guardrails** for safety and compliance
- **Aggregates responses** from multiple agents

```python
from medai_compass.orchestrator import MasterOrchestrator, OrchestratorRequest

orch = MasterOrchestrator()
response = orch.process_request(OrchestratorRequest(
    request_id="req-001",
    user_id="user-001",
    content="Analyze this chest X-ray"
))
```

### 2. Diagnostic Agent (LangGraph)

Stateful workflow for medical image analysis:
- **Preprocessing Node**: DICOM loading, normalization
- **Inference Node**: MedGemma/CXR Foundation analysis
- **Report Node**: Structured finding generation
- **Routing**: Confidence-based escalation

### 3. Workflow Agent (CrewAI)

Clinical operations automation:
- **SchedulerAgent**: Appointment management
- **DocumenterAgent**: Discharge summaries, progress notes
- **PriorAuthAgent**: Insurance authorization with justification scoring

### 4. Communication Agent (AutoGen)

Patient engagement with safety:
- **TriageAgent**: Emergency detection, urgency routing
- **HealthEducatorAgent**: Evidence-based education
- **ClinicalOversightProxy**: Human-in-the-loop review

## Data Flow

```
Patient/Clinician Request
         │
         ▼
┌────────────────────┐
│  Input Guardrails  │──▶ Block jailbreak/out-of-scope
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Intent Classifier  │──▶ Diagnostic / Workflow / Communication
└────────────────────┘
         │
         ▼
┌────────────────────┐
│   Domain Agent     │──▶ Process with HAI-DEF models
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Output Guardrails  │──▶ Add disclaimers, flag low confidence
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Human Review?     │──▶ Queue for clinician if needed
└────────────────────┘
         │
         ▼
      Response
```

## HAI-DEF Model Integration

| Model | Use Case | Agent |
|-------|----------|-------|
| MedGemma 4B | Fast inference, triage | Diagnostic, Communication |
| MedGemma 27B | Complex documentation | Workflow (Documenter) |
| CXR Foundation | Chest X-ray analysis | Diagnostic |
| Path Foundation | Pathology WSI analysis | Diagnostic |
| MedASR | Clinical dictation | Workflow |

## HIPAA Compliance

- **PHI Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Audit Logging**: Tamper-evident logs with blockchain anchoring
- **Access Control**: Role-based with session management
- **Data Isolation**: Row-level security in PostgreSQL
