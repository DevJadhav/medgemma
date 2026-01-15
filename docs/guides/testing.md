# Testing Guide

Comprehensive guide for testing MedAI Compass.

## Overview

MedAI Compass uses **pytest** with **Test-Driven Development (TDD)** methodology.

**Current Status: 207 tests passing** ✅

## Running Tests

### All Tests

```bash
uv run pytest tests/ -v
```

### Specific Test Files

```bash
# Workflow Agent
uv run pytest tests/test_workflow_agent.py -v

# Communication Agent
uv run pytest tests/test_communication_agent.py -v

# Orchestrator
uv run pytest tests/test_orchestrator.py -v

# Data Pipeline
uv run pytest tests/test_data_pipeline.py -v

# MedASR
uv run pytest tests/test_datasets_medasr.py -v
```

### With Coverage

```bash
uv run pytest tests/ --cov=medai_compass --cov-report=html
open htmlcov/index.html
```

### Fast (No Slow Tests)

```bash
uv run pytest tests/ -v -m "not slow"
```

## Test Organization

```
tests/
├── test_data_pipeline.py       # Data loading tests
├── test_datasets_medasr.py     # Dataset & dictation tests
├── test_workflow_agent.py      # CrewAI agent tests
├── test_communication_agent.py # AutoGen agent tests
├── test_orchestrator.py        # Master orchestrator tests
├── test_diagnostic_agent.py    # LangGraph diagnostic tests
├── test_guardrails.py          # Safety guardrails tests
├── test_fhir_client.py         # FHIR client tests
└── test_dicom_utils.py         # DICOM utilities tests
```

## Writing Tests

### Naming Convention

```python
class TestClassName:
    """Tests for ClassName."""
    
    def test_method_name_expected_behavior(self):
        """Test that method does expected thing."""
        pass
```

### Using Fixtures

```python
import pytest

@pytest.fixture
def sample_patient_message():
    return PatientMessage(
        message_id="msg-001",
        patient_id="pat-001",
        content="Test message"
    )

def test_triage_message(sample_patient_message):
    agent = TriageAgent()
    result = agent.triage_message(sample_patient_message)
    assert result.urgency is not None
```

### Temporary Directories

```python
def test_with_files(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("content")
    assert file_path.exists()
```

## Test Categories

### Unit Tests

Test individual components in isolation.

```python
def test_estimate_duration():
    agent = SchedulerAgent()
    assert agent._estimate_duration("new_patient") == 60
    assert agent._estimate_duration("follow_up") == 30
```

### Integration Tests

Test component interactions.

```python
def test_workflow_crew_coordination():
    crew = WorkflowCrew()
    results = crew.process_complex_workflow(
        prior_auth_request=auth_req,
        scheduling_request=sched_req
    )
    assert "prior_auth" in results
    assert "scheduling" in results
```

### Safety Tests

Test guardrails and safety features.

```python
def test_emergency_detection():
    comm = CommunicationOrchestrator()
    response = comm.process_message(PatientMessage(
        content="I'm having a heart attack"
    ))
    assert response.triage_result.urgency == UrgencyLevel.EMERGENCY
    assert "911" in response.content
```

## Mocking

### Mock GPU Models

```python
from unittest.mock import Mock, patch

@patch('medai_compass.models.medgemma.MedGemmaWrapper.load_model')
def test_with_mocked_model(mock_load):
    mock_load.return_value = True
    wrapper = MedGemmaWrapper()
    assert wrapper.load_model() is True
```

### Mock External Services

```python
@patch('requests.get')
def test_with_mocked_api(mock_get):
    mock_get.return_value.json.return_value = {"data": "test"}
    # Test code that makes HTTP requests
```

## CI/CD

Tests run automatically on:
- Push to main/develop
- Pull requests

See `.github/workflows/test.yml` for configuration.
