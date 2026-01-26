# MedAI Compass Documentation

Welcome to the MedAI Compass documentation. This comprehensive guide covers architecture, algorithms, API usage, deployment, training, inference, and operational guidance.

## Table of Contents

### Getting Started
- [Quick Start Guide](guides/quickstart.md)
- [Configuration](guides/configuration.md)
- [GPU Inference Setup](guides/gpu_inference.md)

### Architecture & Design
- [System Architecture](architecture.md)
- [System Design Document](SYSTEM_DESIGN.md) - Design patterns, data flow, scalability
- [Multi-Agent Design](agents.md)
- [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md)

### Algorithm Documentation
- [Algorithm Deep Dive](ALGORITHMS.md) - Comprehensive algorithm explanations including:
  - Multi-agent orchestration
  - Intent classification
  - Diagnostic workflow algorithms
  - Training algorithms (LoRA, QLoRA, DoRA, DPO, GRPO)
  - Inference optimization (Flash Attention, CUDA Graphs, KV Cache)
  - Guardrails and safety algorithms
  - Distributed computing (5D Parallelism, DeepSpeed ZeRO)

### Architectural Decision Records (ADR)
- [ADR Index](adr/README.md)
- [ADR-001: Multi-Agent Framework Selection](adr/ADR-001-multi-agent-framework.md)
- [ADR-002: State Persistence Strategy](adr/ADR-002-state-persistence.md)
- [ADR-003: GPU Inference Backend Selection](adr/ADR-003-inference-backend.md)
- [ADR-004: Distributed Training Strategy](adr/ADR-004-training-strategy.md)
- [ADR-005: Guardrails Architecture](adr/ADR-005-guardrails-architecture.md)
- [ADR-006: Configuration Management with Hydra](adr/ADR-006-configuration-management.md)

### Operations Guides
- [Production Deployment Guide](operations/PRODUCTION_DEPLOYMENT.md) - End-to-end production setup
- [Training Guide](operations/TRAINING_GUIDE.md) - Fine-tuning MedGemma models
- [Inference Guide](operations/INFERENCE_GUIDE.md) - Production inference and optimization

### API Reference
- [Workflow Agent API](api/workflow.md)
- [Communication Agent API](api/communication.md)
- [Master Orchestrator API](api/orchestrator.md)

### Deployment
- [Docker Deployment](deployment/docker.md)
- [Modal Cloud Integration](deployment/MODAL_INTEGRATION.md)
- [Persistence Configuration](guides/persistence.md)

### Security & Compliance
- [HIPAA Compliance](HIPAA_COMPLIANCE.md)
- [Security Assessment Report](SECURITY_ASSESSMENT_REPORT.md)

### Development & Testing
- [Testing Guide](guides/testing.md)

### Competition Materials
- [Technical Writeup](competition/TECHNICAL_WRITEUP.md)
- [Video Script](competition/VIDEO_SCRIPT.md)

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [README](../README.md) | Project overview |
| [CLAUDE.md](../CLAUDE.md) | AI assistant guidelines |
| [.env.example](../.env.example) | Configuration template |
| [docker-compose.yml](../docker-compose.yml) | Deployment config |
| [config/hydra](../config/hydra/) | Hydra configuration files |

---

## Documentation Map

```
docs/
├── README.md                    # This file
├── ALGORITHMS.md               # Algorithm explanations
├── SYSTEM_DESIGN.md            # System design choices
├── architecture.md             # High-level architecture
├── agents.md                   # Multi-agent design
├── ARCHITECTURE_DIAGRAMS.md    # Visual diagrams
│
├── adr/                        # Architectural Decision Records
│   ├── README.md              # ADR index
│   ├── ADR-001-*.md           # Multi-agent framework
│   ├── ADR-002-*.md           # State persistence
│   ├── ADR-003-*.md           # Inference backend
│   ├── ADR-004-*.md           # Training strategy
│   ├── ADR-005-*.md           # Guardrails
│   └── ADR-006-*.md           # Configuration management
│
├── operations/                 # Operational guides
│   ├── PRODUCTION_DEPLOYMENT.md  # Full production setup
│   ├── TRAINING_GUIDE.md         # Training documentation
│   └── INFERENCE_GUIDE.md        # Inference documentation
│
├── api/                        # API reference
│   ├── workflow.md
│   ├── communication.md
│   └── orchestrator.md
│
├── deployment/                 # Deployment guides
│   ├── docker.md
│   └── MODAL_INTEGRATION.md
│
├── guides/                     # User guides
│   ├── quickstart.md
│   ├── configuration.md
│   ├── gpu_inference.md
│   ├── persistence.md
│   └── testing.md
│
├── HIPAA_COMPLIANCE.md        # Compliance documentation
├── SECURITY_ASSESSMENT_REPORT.md
│
└── competition/               # Kaggle competition
    ├── TECHNICAL_WRITEUP.md
    └── VIDEO_SCRIPT.md
```

---

## Key Concepts

### Multi-Agent System
MedAI Compass uses three specialized agent frameworks:
- **LangGraph** (Diagnostic): Stateful graph-based workflows for medical image analysis
- **CrewAI** (Workflow): Role-based agents for clinical operations
- **AutoGen** (Communication): Conversational agents for patient engagement

### Training Methods
- **LoRA/QLoRA/DoRA**: Parameter-efficient fine-tuning
- **DPO/GRPO**: Alignment with clinical preferences
- **DeepSpeed ZeRO**: Distributed training for large models
- **Ray Tune**: Hyperparameter optimization

### Inference Optimization
- **vLLM**: High-throughput serving with PagedAttention
- **Flash Attention 2**: Memory-efficient attention
- **CUDA Graphs**: Reduced kernel launch overhead
- **Modal**: Serverless H100 GPU access

### Safety & Compliance
- **PHI Detection**: 30+ patterns for HIPAA compliance
- **Jailbreak Prevention**: 8 detection categories
- **Human-in-Loop**: Confidence-based escalation
- **Audit Logging**: 6-year retention with tamper detection

---

## Competition Context

MedAI Compass is built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) using Google's Health AI Developer Foundations (HAI-DEF) models.

**Deadline**: February 24, 2026

### HAI-DEF Models Used
- MedGemma 4B/27B - Clinical reasoning
- CXR Foundation - Chest X-ray analysis
- Path Foundation - Pathology analysis
- MedASR - Clinical dictation
