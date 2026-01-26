<p align="center">
  <h1 align="center">🏥 MedAI Compass</h1>
  <p align="center">
    <strong>Production-Grade Multi-Agent Medical AI Platform</strong><br>
    Built on Google's Health AI Developer Foundations (HAI-DEF)
  </p>
  <p align="center">
    <a href="https://www.kaggle.com/competitions/medgemma-impact-challenge"><img src="https://img.shields.io/badge/Kaggle-MedGemma%20Challenge-20BEFF?style=flat&logo=kaggle" alt="Kaggle"></a>
    <img src="https://img.shields.io/badge/Tests-1900%2B%20passing-brightgreen" alt="Tests">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT%202.0-green" alt="License">
  </p>
  <p align="center">
    <a href="docs/README.md">Documentation</a> •
    <a href="docs/guides/quickstart.md">Quick Start</a> •
    <a href="docs/HIPAA_COMPLIANCE.md">HIPAA Compliance</a>
  </p>
</p>

---

## 🎯 Overview

MedAI Compass is a **HIPAA-compliant multi-agent platform** that integrates Google's [HAI-DEF models](https://developers.google.com/health-ai-developer-foundations) for clinical decision support, workflow automation, and patient engagement.

### Key Features

| Domain | Agent | Capabilities |
|--------|-------|--------------|
| **Diagnostic** | LangGraph | X-ray/CT/MRI analysis, pathology, bounding box localization, report generation |
| **Workflow** | CrewAI | Scheduling, documentation, prior authorization, clinical dictation |
| **Communication** | AutoGen | Triage, health education, multi-language support, patient messaging |

### HAI-DEF Models Used

- **MedGemma 4B/27B** - Clinical reasoning and documentation
- **CXR Foundation** - Chest X-ray analysis
- **Path Foundation** - Pathology WSI analysis
- **MedASR** - Clinical dictation

### Safety & Compliance

- **Guardrails**: PHI detection, jailbreak protection, hallucination prevention
- **Audit Logging**: HIPAA-compliant with 6-year retention
- **Encryption**: AES-256 at rest, TLS 1.3 in transit

---

## 🚀 Quick Start

### Option 1: Production with Modal GPU (Recommended)

Run MedGemma 27B on NVIDIA H100 GPUs via Modal cloud:

```bash
# 1. Clone and install
git clone https://github.com/DevJadhav/medgemma.git
cd medgemma && uv sync

# 2. Setup Modal (cloud GPU)
uv run modal token new                                    # Authenticate
uv run modal secret create huggingface-secret HF_TOKEN=hf_xxx  # Add HF token
uv run modal volume create medgemma-model-cache           # Create volumes
uv run modal volume create medgemma-checkpoints
uv run modal deploy medai_compass/modal/app.py            # Deploy

# 3. Configure environment
python scripts/generate_secrets.py
# Edit .env: Add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET from ~/.modal.toml

# 4. Start services
docker compose up -d

# 5. Test inference
curl http://localhost:8000/api/v1/inference/status
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the symptoms of pneumonia?", "max_tokens": 256}'
```

**Full setup guide**: [docs/operations/PRODUCTION_DEPLOYMENT.md](docs/operations/PRODUCTION_DEPLOYMENT.md)

### Option 2: Local Development

```bash
# Clone and install
git clone https://github.com/DevJadhav/medgemma.git
cd medgemma && uv sync

# Configure
cp .env.example .env
# Edit .env with your HF_TOKEN

# Test
uv run pytest tests/ -v
# Expected: 340 tests passing

# Run API locally
uv run python -m medai_compass.api.main
# API available at http://localhost:8000
```

### Production Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI backend |
| Frontend | 3001 | Next.js UI |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |
| Grafana | 3000 | Monitoring dashboards |
| Prometheus | 9090 | Metrics |

---

## 📁 Project Structure

```
medai_compass/
├── api/                # FastAPI application with Prometheus metrics
├── agents/
│   ├── diagnostic/     # LangGraph diagnostic workflow
│   ├── workflow/       # CrewAI workflow agents
│   └── communication/  # AutoGen patient agents
├── guardrails/         # PHI detection, jailbreak protection, safety
├── models/             # HAI-DEF model wrappers
├── orchestrator/       # Master orchestrator
├── utils/              # DICOM, FHIR, MedASR
├── security/           # HIPAA compliance, encryption
└── evaluation/         # Metrics and drift detection

docs/                   # Documentation
├── api/                # API reference
├── guides/             # User guides
└── deployment/         # Deployment guides

tests/                  # 340 passing tests
├── load/               # Load testing with Locust
docker/                 # Prometheus, Grafana, PostgreSQL, ELK
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

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/health/ready` | GET | Kubernetes readiness probe |
| `/health/live` | GET | Kubernetes liveness probe |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/diagnostic/analyze` | POST | Analyze medical images |
| `/api/v1/workflow/process` | POST | Process workflow requests |
| `/api/v1/communication/message` | POST | Patient communication |
| `/api/v1/orchestrator/process` | POST | Auto-route to agents |
| `/api/v1/session/{id}` | GET/DELETE | Session management |

### API Example

```bash
# Health check
curl http://localhost:8000/health

# Patient communication
curl -X POST http://localhost:8000/api/v1/communication/message \
  -H "Content-Type: application/json" \
  -d '{"message": "What are symptoms of diabetes?", "patient_id": "P001"}'
```

---

## 🐳 Docker Deployment

```bash
# Start core services
docker-compose up -d

# Services:
# - API:           http://localhost:8000
# - Grafana:       http://localhost:3000
# - Prometheus:    http://localhost:9090
# - MinIO:         http://localhost:9001
# - PostgreSQL:    localhost:5432
# - Redis:         localhost:6379

# With GPU inference (requires NVIDIA GPU)
docker-compose --profile gpu up -d

# With Modal cloud GPU (H100)
docker-compose --profile modal up -d

# With log aggregation (ELK stack)
docker-compose --profile logging up -d
```

### GPU Options

MedAI Compass supports multiple GPU backends with automatic fallback:

| Backend | Hardware | Best For |
|---------|----------|----------|
| Local CUDA | NVIDIA A100/H100 80GB | Production, low latency |
| Local MPS | Apple Silicon M1/M2/M3 | Development |
| Modal (Cloud) | H100 80GB on-demand | No local GPU |
| CPU | Any | Testing only |

```python
# GPU auto-detection
from medai_compass.utils.gpu import get_inference_config

config = get_inference_config()
print(f"Backend: {config['backend']}")  # local, modal, or cpu
print(f"Device: {config['device']}")     # cuda:0, mps, or cpu
```

### Ray Optimization Infrastructure

MedAI Compass includes a comprehensive Ray-based ML infrastructure for distributed training and serving.

**Hydra Configuration**:
```bash
# Use pre-configured experiment profiles
config/hydra/
├── config.yaml           # Main entry point
├── model/                # MedGemma 4B/27B configs
├── training/             # LoRA, QLoRA, full fine-tuning
├── training/deepspeed/   # ZeRO-2, ZeRO-3 with offloading
├── tuning/               # ASHA, PBT, Hyperband HPO
├── compute/              # Modal H100, A100, local
└── experiment/           # Production, quick-test profiles
```

```python
# Load configuration
from medai_compass.config.hydra_config import load_config

cfg = load_config("config/hydra")
print(cfg.model.name)     # google/medgemma-4b-it
print(cfg.training.method) # lora
```

**Ray Tune Hyperparameter Optimization**:
```python
from medai_compass.tuning import run_hyperparameter_tuning

# ASHA scheduler for efficient early stopping
results = run_hyperparameter_tuning(
    config=cfg,
    scheduler="asha",
    num_samples=50,
)
print(f"Best config: {results.best_config}")
```

**Ray Serve Inference**:
```python
from medai_compass.inference import deploy_medgemma

# Deploy with autoscaling
manager = deploy_medgemma(
    model_name="google/medgemma-4b-it",
    num_replicas=1,
    autoscaling=True,
)
# Available at http://localhost:8000/generate
```

See [docs/operations/TRAINING_GUIDE.md](docs/operations/TRAINING_GUIDE.md) for the complete training optimization guide.

### Environment Variables

```bash
# Required
HUGGING_FACE_HUB_TOKEN=your_huggingface_token
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
JWT_SECRET=your_jwt_secret
PHI_ENCRYPTION_KEY=your_fernet_key

# Model Configuration (defaults to medgemma-27b)
MEDGEMMA_MODEL_NAME=medgemma-27b  # Options: medgemma-4b, medgemma-27b

# Modal (optional - for cloud GPU)
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret

# Configuration Overrides (optional)
MEDAI_ENVIRONMENT=production     # development, staging, production
MEDAI_CONFIDENCE_HIGH=0.90       # High confidence threshold
MEDAI_INFERENCE_TIMEOUT=90       # Inference timeout in seconds
MEDAI_LOG_LEVEL=INFO             # DEBUG, INFO, WARNING, ERROR

# SIEM Integration (optional)
SPLUNK_HEC_URL=https://your-splunk:8088/services/collector
SPLUNK_HEC_TOKEN=your-hec-token
ELASTICSEARCH_HOSTS=http://localhost:9200
AWS_REGION=us-east-1             # For CloudWatch

# Optional
MINIO_SECRET_KEY=your_minio_secret
GRAFANA_PASSWORD=admin
```

### Configuration Management

MedAI Compass uses a centralized configuration system. Values can be set via:
1. Environment variables (highest priority)
2. Configuration file (`config/medai_config.yaml`)
3. Default values

```python
from medai_compass.config import get_config, get_model_name

config = get_config()
print(f"Model: {get_model_name()}")  # medgemma-27b
print(f"Environment: {config.environment}")
print(f"Confidence threshold: {config.confidence.high_confidence}")
```

See `config/medai_config.yaml` for all configurable options.

### Modal Setup (Optional Cloud GPU)

The `medai_compass/modal/` folder provides optional H100 GPU access via Modal:

```bash
# Install Modal
uv pip install modal

# Authenticate
modal setup

# Deploy MedGemma inference
modal deploy medai_compass/modal/app.py

# Use in code
from medai_compass.modal.client import MedGemmaModalClient

client = MedGemmaModalClient()
result = await client.generate("Analyze this medical image...")
```

**Note**: The modal folder is entirely optional. Delete it to run fully locally.

---

## 🧪 Testing

```bash
# All tests (1,900+ passing)
uv run pytest tests/ -v

# Specific modules
uv run pytest tests/test_api.py -v                    # API tests (98 tests)
uv run pytest tests/test_guardrails_phi.py -v         # PHI detection (41 tests)
uv run pytest tests/test_penetration.py -v            # Security tests (57 tests)
uv run pytest tests/test_training_strategy_selector.py -v  # Training tests (28 tests)

# With coverage
uv run pytest tests/ --cov=medai_compass --cov-report=html

# Load testing (requires running API)
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

**Test Summary: 1,900+ tests passing** ✅

### Test Categories
- **Unit Tests**: Core functionality, models, utilities
- **Integration Tests**: API endpoints, agent orchestration
- **Security Tests**: Penetration testing, PHI detection, jailbreak prevention
- **Training Tests**: Strategy selection, distributed training, optimizations
- **E2E Tests**: Complete workflow testing

---

## 🔒 HIPAA Compliance

- **PHI Detection**: 30+ patterns including SSN, MRN, names, DOB, phone, email, passport, driver's license, Medicare/Medicaid IDs, and context-aware detection
- **PHI Encryption**: AES-256-GCM at rest with key rotation, TLS 1.3 in transit
- **Audit Logging**: SIEM integration (Splunk, ELK, CloudWatch) with tamper-evident hash chains, 6-year retention enforcement
- **Access Control**: JWT + role-based with session management (Redis)
- **Data Isolation**: Row-level security in PostgreSQL
- **Guardrails**: Jailbreak detection (8 categories + encoding bypass + fuzzy matching), hallucination prevention, uncertainty quantification
- **Key Management**: Automatic key rotation, secure storage, audit logging

See [HIPAA Compliance Documentation](docs/HIPAA_COMPLIANCE.md) for full details.

---

## 📊 Monitoring

- **Prometheus**: Metrics collection with custom medical AI metrics
- **Grafana**: Pre-configured dashboards for API, agents, and clinical alerts
- **Alerting**: 12 production alerts (error rates, latency, escalations, critical findings)
- **Elasticsearch/Kibana**: Optional centralized logging (use `--profile logging`)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design |
| [Agents](docs/agents.md) | Multi-agent design |
| [Quick Start](docs/guides/quickstart.md) | Getting started |
| [HIPAA Compliance](docs/HIPAA_COMPLIANCE.md) | Security & compliance |
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

Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AutoGen](https://github.com/microsoft/autogen)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Prometheus](https://prometheus.io/) / [Grafana](https://grafana.com/)

---

## 📈 Project Status

| Component | Status |
|-----------|--------|
| Diagnostic Agent | ✅ Complete |
| Workflow Agent | ✅ Complete |
| Communication Agent | ✅ Complete |
| Master Orchestrator | ✅ Complete |
| Guardrails | ✅ Complete |
| NeMo Guardrails Integration | ✅ Complete |
| API | ✅ Complete |
| GPU Detection & Modal | ✅ Complete |
| Conversation Persistence | ✅ Complete |
| Triton Model Repository | ✅ Complete |
| Monitoring | ✅ Complete |
| HIPAA Compliance | ✅ Complete |
| Load Testing | ✅ Complete |
| Documentation | ✅ Complete |
| Distributed Training | ✅ Complete |
| Security Hardening | ✅ Complete |
| Configuration Management | ✅ Complete |
| Audit Logging & SIEM | ✅ Complete |
| Hydra Configuration | ✅ Complete |
| Ray Tune HPO | ✅ Complete |
| Ray Serve Inference | ✅ Complete |
| Ray Actors & Workflows | ✅ Complete |

### New in Latest Release (v2.0)

**Security & Compliance**:
- **Enhanced PHI Detection**: 30+ patterns with context-aware detection
- **Jailbreak Prevention**: Base64/ROT13 decoding, l33tspeak detection, fuzzy matching
- **SIEM Integration**: Splunk, Elasticsearch, CloudWatch audit log backends
- **Tamper-Evident Audit**: Hash chain verification for audit log integrity
- **Key Rotation**: Automatic encryption key rotation with audit logging

**Configuration & Infrastructure**:
- **Centralized Configuration**: YAML/JSON config files + environment variables
- **Model Selection**: Environment variable (`MEDGEMMA_MODEL_NAME`) for UI configurability
- **Thread-Safe Metrics**: Fixed race conditions in concurrent request tracking
- **Intent Classification**: Semantic understanding with synonym expansion

**Training & Inference**:
- **Distributed Training**: DeepSpeed ZeRO, Megatron-LM, FSDP2, 5D Parallelism
- **Kernel Optimizations**: Fused cross-entropy, RoPE, SwiGLU (4x memory reduction)
- **Quality Gates**: Automated benchmarking for training and inference
- **MedGemma 27B**: Default model with H100 optimization

**Ray Optimization Infrastructure** (NEW):
- **Hydra Configuration**: Hierarchical YAML configs for models, training, tuning, compute
- **Ray Tune HPO**: ASHA, PBT, Hyperband schedulers for hyperparameter optimization
- **Ray Serve**: Production inference deployment with autoscaling and health monitoring
- **Ray Actors**: Evaluation, metrics aggregation, checkpoint management actors
