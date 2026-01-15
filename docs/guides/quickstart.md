# Quick Start Guide

Get MedAI Compass running in under 5 minutes.

## Prerequisites

- Python 3.11+
- Docker Desktop (for database/services)
- HuggingFace account with HAI-DEF access

## Installation

```bash
# Clone repository
git clone https://github.com/DevJadhav/medgemma.git
cd medgemma

# Install dependencies with uv
pip install uv
uv sync

# Copy environment template
cp .env.example .env
# Edit .env with your credentials
```

## Configure Credentials

Edit `.env` with your API keys:

```bash
# Required
HF_TOKEN=hf_your_huggingface_token

# Optional (for production)
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
```

## Run Tests

```bash
# Verify installation
uv run pytest tests/ -v

# Expected: 207 tests passing
```

## Basic Usage

### 1. Patient Communication

```python
from medai_compass.agents.communication import (
    CommunicationOrchestrator,
    PatientMessage
)

# Initialize
comm = CommunicationOrchestrator()

# Process patient message
response = comm.process_message(PatientMessage(
    message_id="msg-001",
    patient_id="pat-001",
    content="What should I know about managing my diabetes?"
))

print(response.content)
# Includes health education + disclaimer
```

### 2. Clinical Documentation

```python
from medai_compass.agents.workflow import (
    WorkflowCrew,
    DocumentationRequest
)

crew = WorkflowCrew()

result = crew.process_documentation(DocumentationRequest(
    patient_id="P001",
    document_type="discharge_summary",
    encounter_id="ENC001",
    clinical_notes=["Patient stable", "Vitals normal"],
    diagnoses=[{"code": "I10", "display": "Hypertension"}]
))

print(result.output["content"])
```

### 3. Master Orchestrator

```python
from medai_compass.orchestrator import (
    MasterOrchestrator,
    OrchestratorRequest
)

orch = MasterOrchestrator()

response = orch.process_request(OrchestratorRequest(
    request_id="req-001",
    user_id="clinician-001",
    content="Generate a discharge summary for patient P001"
))

print(f"Domain: {response.domain}")
print(f"Response: {response.content}")
```

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

## Next Steps

- [Installation Guide](installation.md) - Detailed setup
- [Configuration](configuration.md) - All options
- [Architecture](../architecture.md) - System design
- [API Reference](../api/) - Full API docs
