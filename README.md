<p align="center">
  <h1 align="center">🏥 MedAI Compass</h1>
  <p align="center">
    <strong>Multi-Agent Medical AI Platform</strong><br>
    Built on Google's Health AI Developer Foundations (HAI-DEF)
  </p>
  <p align="center">
    <a href="https://www.kaggle.com/competitions/medgemma-impact-challenge">Kaggle MedGemma Impact Challenge</a> •
    <a href="docs/README.md">Documentation</a> •
    <a href="docs/guides/quickstart.md">Quick Start</a>
  </p>
</p>

---

## 🎯 Overview

MedAI Compass is a **HIPAA-compliant multi-agent platform** that integrates Google's [HAI-DEF models](https://developers.google.com/health-ai-developer-foundations) for clinical decision support, workflow automation, and patient engagement.

### Key Features

| Domain | Agent | Capabilities |
|--------|-------|--------------|
| **Diagnostic** | LangGraph | X-ray/CT/MRI analysis, pathology, report generation |
| **Workflow** | CrewAI | Scheduling, documentation, prior authorization |
| **Communication** | AutoGen | Triage, health education, patient messaging |

### HAI-DEF Models Used

- **MedGemma 4B/27B** - Clinical reasoning and documentation
- **CXR Foundation** - Chest X-ray analysis
- **Path Foundation** - Pathology WSI analysis
- **MedASR** - Clinical dictation

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/DevJadhav/medgemma.git
cd medgemma

# Install
pip install uv && uv sync

# Configure
cp .env.example .env
# Edit .env with your HF_TOKEN

# Test
uv run pytest tests/ -v
# Expected: 207 tests passing

# Run (Docker)
docker-compose up -d
```

---

## 📁 Project Structure

```
medai_compass/
├── agents/
│   ├── diagnostic/     # LangGraph diagnostic workflow
│   ├── workflow/       # CrewAI workflow agents
│   └── communication/  # AutoGen patient agents
├── guardrails/         # PHI detection, safety
├── models/             # HAI-DEF model wrappers
├── orchestrator/       # Master orchestrator
├── utils/              # DICOM, FHIR, MedASR
└── security/           # HIPAA compliance

docs/                   # Documentation
├── api/                # API reference
├── guides/             # User guides
└── deployment/         # Deployment guides

tests/                  # 207 passing tests
docker/                 # Prometheus, Grafana, Postgres
```

---

## 💡 Usage Examples

### Patient Communication

```python
from medai_compass.agents.communication import (
    CommunicationOrchestrator,
    PatientMessage
)

comm = CommunicationOrchestrator()
response = comm.process_message(PatientMessage(
    message_id="msg-001",
    patient_id="pat-001",
    content="How do I manage my diabetes?"
))

print(response.content)
# Health education with medical disclaimer
```

### Clinical Documentation

```python
from medai_compass.agents.workflow import WorkflowCrew, DocumentationRequest

crew = WorkflowCrew()
result = crew.process_documentation(DocumentationRequest(
    patient_id="P001",
    document_type="discharge_summary",
    encounter_id="ENC001",
    clinical_notes=["Patient stable"],
    diagnoses=[{"display": "Hypertension"}]
))

print(result.output["content"])
```

### Master Orchestrator

```python
from medai_compass.orchestrator import MasterOrchestrator, OrchestratorRequest

orch = MasterOrchestrator()
response = orch.process_request(OrchestratorRequest(
    request_id="req-001",
    user_id="clinician-001",
    content="Generate a discharge summary"
))

print(f"Domain: {response.domain}")
print(f"Guardrails: {response.guardrails_applied}")
```

---

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up -d

# Services:
# - API:        http://localhost:8000
# - Grafana:    http://localhost:3000
# - Prometheus: http://localhost:9090
# - MinIO:      http://localhost:9001
```

### GPU Inference (Optional)

```bash
docker-compose --profile gpu up -d
# Starts vLLM and Triton inference servers
```

---

## 🧪 Testing

```bash
# All tests
uv run pytest tests/ -v

# Specific module
uv run pytest tests/test_workflow_agent.py -v

# With coverage
uv run pytest tests/ --cov=medai_compass
```

**Test Summary: 207 tests passing** ✅

---

## 🔒 HIPAA Compliance

- **PHI Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Audit Logging**: Tamper-evident with blockchain anchoring
- **Access Control**: Role-based with session management
- **Data Isolation**: Row-level security in PostgreSQL

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design |
| [Agents](docs/agents.md) | Multi-agent design |
| [Quick Start](docs/guides/quickstart.md) | Getting started |
| [Workflow API](docs/api/workflow.md) | Workflow agent API |
| [Communication API](docs/api/communication.md) | Patient agent API |
| [Orchestrator API](docs/api/orchestrator.md) | Master orchestrator API |
| [Docker Deployment](docs/deployment/docker.md) | Container deployment |

---

## 🏆 Competition

Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge).

**Deadline**: February 24, 2026

### Evaluation Criteria
- Effective use of HAI-DEF models (20%)
- Problem domain importance (15%)
- Impact potential (15%)
- Product feasibility (20%)
- Execution and communication (30%)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AutoGen](https://github.com/microsoft/autogen)