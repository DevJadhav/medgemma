# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

MedAI Compass is a HIPAA-compliant multi-agent medical AI platform built for the Kaggle MedGemma Impact Challenge. It integrates Google's HAI-DEF models (MedGemma, CXR Foundation, Path Foundation) using three agent frameworks:
- **LangGraph** for diagnostic workflows (stateful, checkpointing)
- **CrewAI** for workflow coordination (scheduling, documentation, prior auth)
- **AutoGen** for patient communication (triage, health education)

## Essential Commands

### Setup & Installation
```bash
# Install dependencies
pip install uv && uv sync

# Configure environment
cp .env.example .env
# Edit .env with required tokens (HF_TOKEN is critical)
```

### Testing
```bash
# Run all tests (207 tests expected to pass)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_workflow_agent.py -v

# Run with coverage
uv run pytest tests/ --cov=medai_compass --cov-report=html

# Skip slow tests
uv run pytest tests/ -v -m "not slow"

# Run specific test markers
uv run pytest tests/ -v -m integration
uv run pytest tests/ -v -m gpu
```

### Code Quality
```bash
# Run linting
ruff check medai_compass/ tests/

# Auto-fix linting issues
ruff check --fix medai_compass/ tests/

# Format code
ruff format medai_compass/ tests/

# Check formatting without changes
ruff format --check medai_compass/ tests/

# Type checking
mypy medai_compass/ --ignore-missing-imports
```

### Docker
```bash
# Start all services (API, Postgres, Redis, MinIO, Prometheus, Grafana)
docker-compose up -d

# Start with GPU inference (vLLM + Triton)
docker-compose --profile gpu up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Development Server
```bash
# Run API server (default port 8000)
uvicorn medai_compass.api.main:app --reload

# Services available at:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - MinIO: http://localhost:9001
```

## Architecture

### Multi-Agent System
The system uses a **master orchestrator** (`medai_compass/orchestrator/master.py`) that routes requests to three specialized agent domains:

1. **Diagnostic Agent** (LangGraph)
   - Location: `medai_compass/agents/diagnostic/`
   - Key files: `graph.py` (workflow), `nodes.py` (processing steps), `state.py` (state definition)
   - Flow: preprocess → analyze → generate_report → confidence_check → [human_review if needed] → finalize
   - Uses: MedGemma for clinical reasoning, CXR Foundation for chest X-rays, Path Foundation for pathology

2. **Workflow Agent** (CrewAI)
   - Location: `medai_compass/agents/workflow/`
   - Three specialized agents: SchedulerAgent, DocumenterAgent, PriorAuthAgent
   - Handles: clinical documentation, appointment scheduling, prior authorization
   - Uses task-based coordination with role delegation

3. **Communication Agent** (AutoGen)
   - Location: `medai_compass/agents/communication/`
   - Team: TriageAgent, HealthEducatorAgent, FollowUpSchedulingAgent, ClinicalOversightProxy
   - Triage levels: EMERGENCY, URGENT, SOON, ROUTINE, INFORMATIONAL
   - All responses include medical disclaimers

### Request Flow
```
User Input → Master Orchestrator
  ↓
Intent Classification (domain routing)
  ↓
Input Guardrails (PHI detection, safety checks)
  ↓
[Diagnostic | Workflow | Communication] Agent
  ↓
Output Guardrails (uncertainty, escalation)
  ↓
Human Escalation Gateway (if critical/low confidence)
  ↓
Response to User
```

### Guardrails System
Location: `medai_compass/guardrails/`
- `input_rails.py` - Jailbreak detection, out-of-scope filtering
- `output_rails.py` - Medical disclaimer injection, hallucination prevention
- `phi_detection.py` - PHI scrubbing (SSN, MRN, names, DOB)
- `uncertainty.py` - Confidence scoring, uncertainty quantification
- `escalation.py` - Critical finding detection, safety-based escalation

Escalation triggers:
- Critical findings: pneumothorax, stroke, MI, aortic dissection, etc.
- Safety concerns: suicidal ideation, self-harm
- Low confidence: <90% diagnostic, <85% workflow, <80% communication
- High uncertainty: >20%

### Model Integration
Location: `medai_compass/models/`
- `medgemma.py` - MedGemma 4B/27B wrapper with quantization (4-bit/8-bit)
- `cxr_foundation.py` - CXR Foundation for chest X-ray analysis
- `path_foundation.py` - Path Foundation for pathology/histology

All models support:
- Quantization for memory efficiency
- Batch processing
- Confidence extraction

### Utilities
Location: `medai_compass/utils/`
- `dicom.py` - DICOM loading, windowing, normalization
- `fhir.py` - FHIR client, resource conversion
- `medasr.py` - MedASR clinical dictation processing
- `data_pipeline.py` - Data loading and preprocessing
- `datasets.py` - Dataset utilities

### Security & Compliance
Location: `medai_compass/security/`
- `encryption.py` - AES-256 PHI encryption
- `auth.py` - JWT authentication, role-based access
- `audit.py` - Tamper-evident audit logging with blockchain anchoring

## Testing Guidelines

### Test Organization
- Test files follow pattern: `test_<module_name>.py`
- 207 tests total (as of current status)
- Use pytest fixtures in `conftest.py` for shared test data
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.gpu`

### Writing Tests
- Use descriptive test names: `test_<method>_<expected_behavior>`
- Mock GPU models and external services to avoid dependencies
- Use `tmp_path` fixture for file operations
- Async tests automatically detected with `asyncio_mode = "auto"`

### Critical Test Coverage
Always test:
- Agent coordination and handoffs
- Guardrail triggers (critical findings, safety concerns)
- PHI detection and scrubbing
- Confidence thresholds and escalation logic
- FHIR/DICOM processing edge cases

## Configuration

### Environment Variables
Required in `.env`:
- `HF_TOKEN` - HuggingFace token for model access (critical)
- `POSTGRES_*` - Database connection
- `REDIS_*` - Cache connection
- `MINIO_*` - Object storage
- `PHI_ENCRYPTION_KEY` - Fernet key for PHI encryption
- `JWT_SECRET` - Authentication secret

Optional:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` - Backup LLMs
- `MODAL_TOKEN_*` - GPU inference via Modal
- `PHYSIONET_*` - MIMIC dataset access

### Code Quality Standards
Configured in `pyproject.toml`:
- Line length: 100 characters
- Python version: >=3.10
- Ruff linting: E, F, I, N, W, UP rules (ignore E501 for line length)
- MyPy type checking enabled
- Coverage: omit tests and pycache

## Development Workflow

### When Making Changes

1. **Before editing**:
   - Understand the agent framework being modified (LangGraph/CrewAI/AutoGen)
   - Check if guardrails need updates for new functionality
   - Review related test files

2. **While editing**:
   - Follow existing patterns (each agent framework has distinct idioms)
   - Add type hints for all function signatures
   - Update docstrings for public methods
   - Maintain HIPAA compliance (encrypt PHI, audit trail)

3. **After editing**:
   ```bash
   # Format and lint
   ruff format medai_compass/ tests/
   ruff check --fix medai_compass/ tests/
   
   # Type check
   mypy medai_compass/ --ignore-missing-imports
   
   # Run relevant tests
   uv run pytest tests/test_<relevant_module>.py -v
   
   # Run full test suite
   uv run pytest tests/ -v
   ```

### Agent-Specific Patterns

**LangGraph (Diagnostic)**:
- Define state in `DiagnosticState` TypedDict
- Nodes are functions that take/return state
- Use `workflow.add_conditional_edges()` for branching
- Compile graph before invoking: `graph = workflow.compile()`

**CrewAI (Workflow)**:
- Agents have roles, goals, and backstories
- Tasks have descriptions and expected outputs
- Use `crew.kickoff()` to execute workflow
- Coordinate with agent delegation

**AutoGen (Communication)**:
- Agents are conversational with `generate_reply()`
- Use `GroupChat` for multi-agent coordination
- `ClinicalOversightProxy` provides human-in-loop

### Common Pitfalls

1. **Don't bypass guardrails** - All agent outputs must pass through output rails and escalation gateway
2. **Don't hardcode secrets** - Use environment variables for all credentials
3. **Don't assume model availability** - Mock models in tests, handle loading failures gracefully
4. **Don't skip PHI encryption** - Any patient data must be encrypted at rest
5. **Don't ignore confidence scores** - Low confidence triggers human review

## Integration Points

### FHIR Resources
Use `medai_compass/utils/fhir.py` for:
- Patient demographics
- Observations (vital signs, labs)
- Diagnostic reports
- Encounters and procedures

### DICOM Processing
Use `medai_compass/utils/dicom.py` for:
- Loading .dcm files
- Windowing (lung, mediastinum, bone)
- Normalization for model input
- Metadata extraction

### External Services
- **PostgreSQL**: Patient records, audit logs (row-level security enabled)
- **Redis**: Session state, caching
- **MinIO**: DICOM/image object storage
- **Prometheus/Grafana**: Metrics and monitoring

## Production Considerations

### HIPAA Compliance
- All PHI is AES-256 encrypted
- TLS 1.3 for transit
- Audit logs are tamper-evident
- Access control via JWT + role-based policies
- Data isolation with row-level security

### Performance
- Use quantization (4-bit/8-bit) for memory efficiency
- Batch processing for multiple requests
- Redis caching for frequent queries
- GPU inference via vLLM/Triton for production scale

### Monitoring
- Health check: `http://localhost:8000/health`
- Metrics: Prometheus on port 9090
- Dashboards: Grafana on port 3000
- Audit logs in PostgreSQL `audit_logs` table

## Additional Resources

- Architecture: `docs/architecture.md`
- Agents: `docs/agents.md`
- API docs: `docs/api/` (workflow, communication, orchestrator)
- Deployment: `docs/deployment/docker.md`
- Testing: `docs/guides/testing.md`
- Kaggle Competition: https://www.kaggle.com/competitions/medgemma-impact-challenge
