# System Design Document

This document outlines the system design choices, architectural patterns, and design rationale for MedAI Compass.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Multi-Agent Design](#multi-agent-design)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Scalability Design](#scalability-design)
6. [Security Architecture](#security-architecture)
7. [Observability Design](#observability-design)
8. [Configuration Management](#configuration-management)

---

## Design Philosophy

### Core Principles

1. **Safety First**: All medical AI outputs must pass through guardrails before reaching users
2. **Human-in-the-Loop**: Low confidence decisions require clinical review
3. **HIPAA by Design**: PHI protection is built into every component
4. **Horizontal Scalability**: Stateless services with shared state in databases
5. **Graceful Degradation**: System continues functioning when components fail
6. **Explainability**: All decisions can be traced and audited

### Design Trade-offs

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| Multi-agent over monolithic | Complexity vs. Specialization | Specialized agents optimize for domain-specific tasks |
| LangGraph for diagnostics | Learning curve vs. Checkpointing | State persistence critical for resumable workflows |
| CrewAI for workflows | Framework lock-in vs. Role-based agents | Natural fit for clinical role delegation |
| PostgreSQL checkpointing | Latency vs. Durability | Multi-instance support requires shared state |

---

## Architecture Overview

### High-Level System Architecture

```
                                    ┌──────────────────────────────────┐
                                    │         Load Balancer            │
                                    │     (nginx / Cloud LB)           │
                                    └─────────────┬────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
         ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
         │   API Server     │         │   API Server     │         │   API Server     │
         │   (FastAPI)      │         │   (FastAPI)      │         │   (FastAPI)      │
         │   Replica 1      │         │   Replica 2      │         │   Replica N      │
         └────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
                  │                            │                            │
                  └────────────────────────────┼────────────────────────────┘
                                               │
                  ┌────────────────────────────┼────────────────────────────┐
                  │                            │                            │
                  ▼                            ▼                            ▼
       ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
       │     Redis       │         │   PostgreSQL    │         │     MinIO       │
       │   (Sessions/    │         │   (Checkpoints/ │         │   (DICOM/       │
       │    Cache)       │         │    Audit Logs)  │         │    Images)      │
       └─────────────────┘         └─────────────────┘         └─────────────────┘
                                               │
                  ┌────────────────────────────┼────────────────────────────┐
                  │                            │                            │
                  ▼                            ▼                            ▼
       ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
       │  Celery Worker  │         │  Celery Worker  │         │  Celery Worker  │
       │   (Inference    │         │   (Data         │         │   (Background   │
       │    Tasks)       │         │    Ingestion)   │         │    Jobs)        │
       └─────────────────┘         └─────────────────┘         └─────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │  GPU Inference  │
                                    │  (Local/Modal)  │
                                    └─────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| API Server | Request routing, validation, orchestration | FastAPI, Pydantic |
| Redis | Session storage, caching, rate limiting | Redis 7.x |
| PostgreSQL | Workflow checkpoints, audit logs, metadata | PostgreSQL 15+ |
| MinIO | DICOM image storage, model artifacts | S3-compatible |
| Celery | Async task execution, background jobs | Celery 5.x |
| GPU Inference | Model inference (local or cloud) | PyTorch, vLLM, Modal |

---

## Multi-Agent Design

### Agent Framework Selection

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Master Orchestrator                              │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Intent Classification                          │ │
│  │  • Semantic keyword matching                                       │ │
│  │  • Synonym expansion                                               │ │
│  │  • Context phrase detection                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│         ┌──────────────────────────┼──────────────────────────┐        │
│         │                          │                          │        │
│         ▼                          ▼                          ▼        │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐   │
│  │  DIAGNOSTIC │           │  WORKFLOW   │           │COMMUNICATION│   │
│  │  (LangGraph)│           │  (CrewAI)   │           │  (AutoGen)  │   │
│  └─────────────┘           └─────────────┘           └─────────────┘   │
│                                                                         │
│  Why LangGraph:              Why CrewAI:              Why AutoGen:      │
│  • State machine for        • Role-based agents      • Conversational  │
│    complex workflows        • Task delegation        • Group chat      │
│  • Checkpoint support       • Crew coordination      • Async messaging │
│  • Conditional routing      • Backstory/goals        • Oversight proxy │
│  • PostgreSQL persistence   • Natural for clinical   • Human-in-loop   │
│                               roles                                     │
└────────────────────────────────────────────────────────────────────────┘
```

### Diagnostic Agent (LangGraph)

**Design Rationale**: Medical image analysis requires a structured, checkpointable workflow with conditional routing based on confidence levels.

```python
# State machine design
DiagnosticState = TypedDict:
    patient_id: str
    session_id: str
    images: List[bytes]
    image_metadata: Dict
    findings: List[Finding]
    report: str
    confidence_score: float
    requires_review: bool
    fhir_context: Optional[Dict]
```

**Workflow Nodes**:
1. `preprocess_images`: DICOM loading, normalization, quality validation
2. `analyze_with_medgemma`: Multi-modal model inference
3. `localize_findings`: Bounding box detection
4. `generate_report`: Structured report generation
5. `confidence_check`: Uncertainty quantification
6. `human_review`: Queue for clinician review (conditional)
7. `finalize`: Output formatting and persistence

**Conditional Routing**:
```
IF confidence_score >= 0.90:
    ROUTE TO finalize
ELSE:
    ROUTE TO human_review THEN finalize
```

### Workflow Agent (CrewAI)

**Design Rationale**: Clinical workflows naturally map to roles (scheduler, documenter, prior auth specialist) with task delegation.

```python
# Crew composition
WorkflowCrew:
    agents:
        - SchedulerAgent(role="Appointment Scheduler")
        - DocumenterAgent(role="Clinical Documentation Specialist")
        - PriorAuthAgent(role="Prior Authorization Specialist")

    tasks:
        - SchedulingTask → SchedulerAgent
        - DocumentationTask → DocumenterAgent
        - PriorAuthTask → PriorAuthAgent
```

**Task Flow**:
```
Request → Task Classification → Agent Assignment → Task Execution → Result Aggregation
```

### Communication Agent (AutoGen)

**Design Rationale**: Patient communication requires conversational agents with oversight for safety-critical responses.

```python
# AutoGen team structure
CommunicationTeam:
    agents:
        - TriageAgent(urgency classification)
        - HealthEducatorAgent(patient education)
        - FollowUpSchedulingAgent(appointment coordination)
        - ClinicalOversightProxy(human-in-loop review)

    group_chat:
        manager: ClinicalOversightProxy
        participants: [TriageAgent, HealthEducatorAgent, FollowUpSchedulingAgent]
```

---

## Data Flow Architecture

### Request Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

1. INGRESS
   User Request → Load Balancer → API Server
                                      │
2. AUTHENTICATION                     ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  JWT Validation → Session Lookup (Redis) → Rate Limiting → Request Log  │
   └──────────────────────────────────────────────────────────────────────────┘
                                      │
3. INPUT GUARDRAILS                   ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  PHI Detection → PHI Masking → Jailbreak Detection → Scope Validation   │
   └──────────────────────────────────────────────────────────────────────────┘
                                      │
4. ORCHESTRATION                      ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Intent Classification → Domain Routing → Agent Selection → Execution   │
   └──────────────────────────────────────────────────────────────────────────┘
                                      │
5. AGENT EXECUTION                    ▼
   ┌─────────────┬─────────────────────┬──────────────────────────────────────┐
   │ DIAGNOSTIC  │     WORKFLOW        │         COMMUNICATION                │
   │ (LangGraph) │     (CrewAI)        │         (AutoGen)                    │
   │             │                     │                                      │
   │ Preprocess  │  Task Assignment    │  Triage Classification               │
   │ Analyze     │  Agent Execution    │  Education Generation                │
   │ Localize    │  Result Aggregation │  Oversight Review                    │
   │ Report      │                     │                                      │
   └─────────────┴─────────────────────┴──────────────────────────────────────┘
                                      │
6. OUTPUT GUARDRAILS                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Disclaimer Injection → Confidence Check → Hallucination Filter → PHI   │
   └──────────────────────────────────────────────────────────────────────────┘
                                      │
7. ESCALATION CHECK                   ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Critical Finding? → Low Confidence? → Safety Concern? → Human Review?  │
   └──────────────────────────────────────────────────────────────────────────┘
                                      │
8. RESPONSE                           ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Audit Logging → Response Formatting → Metrics Update → Return Response │
   └──────────────────────────────────────────────────────────────────────────┘
```

### State Persistence Strategy

```
┌──────────────────────────────────────────────────────────────────────┐
│                    STATE MANAGEMENT LAYERS                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │     REDIS       │    │   POSTGRESQL    │    │     MinIO       │  │
│  │                 │    │                 │    │                 │  │
│  │ • Sessions      │    │ • Checkpoints   │    │ • DICOM files   │  │
│  │ • Cache         │    │ • Audit logs    │    │ • Model weights │  │
│  │ • Rate limits   │    │ • User data     │    │ • Artifacts     │  │
│  │ • Pub/Sub       │    │ • Escalations   │    │ • Reports       │  │
│  │                 │    │                 │    │                 │  │
│  │ TTL: Minutes    │    │ TTL: 6 years    │    │ TTL: Configurable│  │
│  │ to Hours        │    │ (HIPAA)         │    │                 │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                                                       │
│  Use Cases:                                                          │
│  • Session lookup: Redis (sub-ms latency)                            │
│  • Workflow resume: PostgreSQL (durability)                          │
│  • Image storage: MinIO (object storage)                             │
│  • Audit queries: PostgreSQL (SQL flexibility)                       │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Scalability Design

### Horizontal Scaling Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         HORIZONTAL SCALING                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  STATELESS SERVICES (Scale horizontally):                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  API Servers → Auto-scale based on request rate                         │ │
│  │  Celery Workers → Auto-scale based on queue depth                       │ │
│  │  Ray Serve Replicas → Auto-scale based on latency                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  STATEFUL SERVICES (Scale vertically, then shard):                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  PostgreSQL → Read replicas, then sharding by tenant                    │ │
│  │  Redis → Redis Cluster for horizontal sharding                          │ │
│  │  MinIO → Distributed mode with erasure coding                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  GPU INFERENCE (Scale based on throughput):                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Modal Cloud → Serverless auto-scaling (H100)                           │ │
│  │  Ray Serve → Replica-based scaling with autoscaler                      │ │
│  │  Triton → Dynamic batching with multiple instances                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Scaling Metrics

| Component | Metric | Scale Trigger |
|-----------|--------|---------------|
| API Server | Request rate | > 1000 req/sec per instance |
| API Server | P99 latency | > 500ms |
| Celery Worker | Queue depth | > 100 pending tasks |
| GPU Inference | Queue depth | > 10 pending requests |
| GPU Inference | Latency | > 2000ms P95 |
| Redis | Memory usage | > 80% |
| PostgreSQL | Connection count | > 80% max_connections |

### Ray Serve Autoscaling

```yaml
ray_serve_config:
  deployment:
    autoscaling_config:
      min_replicas: 1
      max_replicas: 10
      target_num_ongoing_requests_per_replica: 5
      upscale_delay_s: 30
      downscale_delay_s: 300

    ray_actor_options:
      num_cpus: 1
      num_gpus: 1  # For GPU inference
```

---

## Security Architecture

### Defense in Depth

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY LAYERS                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  LAYER 1: NETWORK PERIMETER                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • TLS 1.3 termination at load balancer                                 │ │
│  │  • WAF rules for common attacks                                         │ │
│  │  • DDoS protection                                                      │ │
│  │  • IP allowlisting for admin endpoints                                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  LAYER 2: APPLICATION SECURITY                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • JWT authentication with RSA-256                                      │ │
│  │  • Role-based access control (RBAC)                                     │ │
│  │  • Rate limiting per user/IP                                            │ │
│  │  • Input validation (Pydantic)                                          │ │
│  │  • CSRF protection                                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  LAYER 3: DATA PROTECTION                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • PHI encryption at rest (AES-256-GCM)                                 │ │
│  │  • Automatic key rotation                                               │ │
│  │  • Row-level security in PostgreSQL                                     │ │
│  │  • PHI masking in logs                                                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  LAYER 4: AI SAFETY                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Jailbreak detection (8 categories)                                   │ │
│  │  • PHI detection (30+ patterns)                                         │ │
│  │  • Output guardrails (disclaimers)                                      │ │
│  │  • Confidence-based escalation                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  LAYER 5: AUDIT & COMPLIANCE                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  • Tamper-evident audit logs (hash chains)                              │ │
│  │  • 6-year retention (HIPAA)                                             │ │
│  │  • SIEM integration (Splunk, ELK, CloudWatch)                           │ │
│  │  • Compliance reporting                                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Encryption Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ENCRYPTION DESIGN                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  KEY HIERARCHY:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │  ┌───────────────────┐                                                  │ │
│  │  │   Master Key      │  ← Stored in HSM / Key Management Service        │ │
│  │  │   (KEK)           │                                                  │ │
│  │  └─────────┬─────────┘                                                  │ │
│  │            │                                                             │ │
│  │            ▼                                                             │ │
│  │  ┌───────────────────┐    ┌───────────────────┐                         │ │
│  │  │   PHI Encryption  │    │   Session Key     │                         │ │
│  │  │   Key (DEK)       │    │   (DEK)           │                         │ │
│  │  └───────────────────┘    └───────────────────┘                         │ │
│  │                                                                          │ │
│  │  Rotation Schedule:                                                     │ │
│  │  • Master Key: Annually                                                 │ │
│  │  • DEKs: Monthly (automated)                                            │ │
│  │  • Session Keys: Per-session                                            │ │
│  │                                                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ENCRYPTION MODES:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  At Rest: AES-256-GCM (AEAD)                                            │ │
│  │  In Transit: TLS 1.3                                                    │ │
│  │  In Use: Encrypted fields in memory (where supported)                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Observability Design

### Three Pillars of Observability

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY STACK                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  METRICS (Prometheus + Grafana)                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Application Metrics:                                                   │ │
│  │  • medai_request_total (counter)                                        │ │
│  │  • medai_request_duration_seconds (histogram)                           │ │
│  │  • medai_inference_latency_seconds (histogram)                          │ │
│  │  • medai_escalation_total (counter)                                     │ │
│  │  • medai_confidence_score (gauge)                                       │ │
│  │  • medai_phi_detection_total (counter)                                  │ │
│  │                                                                          │ │
│  │  Infrastructure Metrics:                                                │ │
│  │  • GPU utilization, memory                                              │ │
│  │  • CPU, memory, disk, network                                           │ │
│  │  • Database connections, query latency                                  │ │
│  │  • Redis operations, memory                                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  LOGS (Elasticsearch + Kibana)                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Structured Logging:                                                    │ │
│  │  • JSON format with correlation IDs                                     │ │
│  │  • Request/response logging (PHI masked)                                │ │
│  │  • Error tracking with stack traces                                     │ │
│  │  • Audit logging (tamper-evident)                                       │ │
│  │                                                                          │ │
│  │  Log Levels:                                                            │ │
│  │  • DEBUG: Development only                                              │ │
│  │  • INFO: Normal operations                                              │ │
│  │  • WARNING: Potential issues                                            │ │
│  │  • ERROR: Failures                                                      │ │
│  │  • CRITICAL: System failures                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  TRACES (OpenTelemetry + Jaeger)                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Distributed Tracing:                                                   │ │
│  │  • Request flow across services                                         │ │
│  │  • Latency breakdown per component                                      │ │
│  │  • Error propagation tracking                                           │ │
│  │  • Agent execution spans                                                │ │
│  │                                                                          │ │
│  │  Trace Context:                                                         │ │
│  │  • trace_id: Unique per request                                         │ │
│  │  • span_id: Unique per operation                                        │ │
│  │  • parent_span_id: Call hierarchy                                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Alerting Rules

```yaml
# Production alerts
alerts:
  - name: HighErrorRate
    condition: error_rate > 0.01  # 1%
    severity: critical

  - name: HighLatency
    condition: p99_latency > 2000ms
    severity: warning

  - name: LowConfidence
    condition: avg_confidence < 0.7
    severity: warning

  - name: EscalationBacklog
    condition: pending_escalations > 10
    severity: critical

  - name: CriticalFinding
    condition: critical_finding_detected
    severity: critical
    notify: on_call_pager
```

---

## Configuration Management

### Hierarchical Configuration

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION HIERARCHY                                    │
│                    (Highest priority on top)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. COMMAND-LINE ARGUMENTS                                                   │
│     └─ Hydra overrides: --training.args.learning_rate=1e-5                  │
│                                                                               │
│  2. ENVIRONMENT VARIABLES                                                    │
│     └─ MEDGEMMA_MODEL_NAME=medgemma-27b                                     │
│     └─ MEDAI_ENVIRONMENT=production                                         │
│                                                                               │
│  3. YAML CONFIGURATION FILES                                                 │
│     └─ config/hydra/config.yaml (defaults)                                  │
│     └─ config/hydra/experiment/production.yaml (overrides)                  │
│                                                                               │
│  4. DEFAULT VALUES                                                           │
│     └─ Hardcoded in dataclasses                                             │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Hydra Configuration Structure

```
config/hydra/
├── config.yaml           # Main entry point
├── model/
│   ├── medgemma_4b.yaml  # 4B model settings
│   └── medgemma_27b.yaml # 27B model settings (default)
├── training/
│   ├── lora.yaml         # LoRA fine-tuning
│   ├── qlora.yaml        # QLoRA (4-bit)
│   ├── dora.yaml         # DoRA
│   ├── dpo.yaml          # Direct Preference Optimization
│   ├── grpo.yaml         # Group Relative Policy Optimization
│   └── deepspeed/
│       ├── zero1.yaml
│       ├── zero2.yaml
│       ├── zero3_offload.yaml
│       └── zero_infinity.yaml
├── compute/
│   ├── modal_h100.yaml   # Modal cloud GPU
│   ├── modal_a100.yaml   # Modal A100
│   └── local.yaml        # Local GPU
├── tuning/
│   ├── asha.yaml         # ASHA scheduler
│   ├── pbt.yaml          # Population-Based Training
│   └── hyperband.yaml    # Hyperband
└── experiment/
    ├── production.yaml   # Production profile
    └── quick_test.yaml   # Quick testing
```

### Configuration Validation

```python
# Pydantic models ensure configuration validity
@dataclass
class ModelConfig:
    name: str
    hf_model_id: str
    max_seq_length: int = 8192

    def __post_init__(self):
        assert self.max_seq_length > 0
        assert self.name in ["medgemma-4b", "medgemma-27b"]
```

---

## Summary

MedAI Compass is designed with:

1. **Multi-Agent Architecture**: Three specialized agents for diagnostics, workflows, and communication
2. **Safety-First Design**: Guardrails at input and output, with human-in-the-loop escalation
3. **Horizontal Scalability**: Stateless services with shared state in databases
4. **Defense in Depth**: Five security layers from network to audit
5. **Full Observability**: Metrics, logs, and traces with alerting
6. **Flexible Configuration**: Hydra-based hierarchical configuration

These design choices enable a production-ready medical AI platform that is safe, scalable, and maintainable.
