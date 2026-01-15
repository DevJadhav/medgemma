# Workflow Agent API

The Workflow Agent handles clinical operations using CrewAI coordination.

## Classes

### WorkflowCrew

Main coordinator for workflow agents.

```python
from medai_compass.agents.workflow import WorkflowCrew

crew = WorkflowCrew(model_wrapper=None)  # Optional model for real inference
```

#### Methods

| Method | Description |
|--------|-------------|
| `process_scheduling(request)` | Process appointment scheduling |
| `process_documentation(request)` | Generate clinical documents |
| `process_prior_auth(request)` | Handle prior authorization |
| `process_complex_workflow(...)` | Coordinate multiple agents |

---

### AppointmentRequest

Request for scheduling an appointment.

```python
from medai_compass.agents.workflow import AppointmentRequest

request = AppointmentRequest(
    patient_id="P001",
    appointment_type="follow_up",      # new_patient, follow_up, specialist_consult, procedure
    preferred_dates=["2026-02-01", "2026-02-02"],
    provider_id="DR-001",              # Optional
    urgency="routine",                 # routine, urgent, emergency
    notes="Annual checkup"             # Optional
)
```

---

### DocumentationRequest

Request for clinical documentation.

```python
from medai_compass.agents.workflow import DocumentationRequest

request = DocumentationRequest(
    patient_id="P001",
    document_type="discharge_summary",  # discharge_summary, progress_note, referral_letter
    encounter_id="ENC001",
    clinical_notes=[
        "Patient admitted with pneumonia",
        "Treated with antibiotics",
        "Symptoms resolved"
    ],
    diagnoses=[
        {"code": "J18.9", "display": "Pneumonia"}
    ],
    procedures=[],                      # Optional
    medications=[                       # Optional
        {"code": "123", "display": "Amoxicillin 500mg"}
    ]
)
```

---

### PriorAuthRequest

Request for prior authorization.

```python
from medai_compass.agents.workflow import PriorAuthRequest

request = PriorAuthRequest(
    patient_id="P001",
    procedure_code="27447",             # CPT code
    diagnosis_codes=["M17.11"],         # ICD-10 codes
    insurance_id="INS-001",
    provider_id="DR-001",
    clinical_justification="Patient has failed conservative treatment for 6 months. Medical necessity established.",
    supporting_documents=[]             # Optional file paths
)
```

---

### WorkflowResult

Result from workflow processing.

```python
@dataclass
class WorkflowResult:
    success: bool
    agent_role: AgentRole               # SCHEDULER, DOCUMENTER, PRIOR_AUTH
    task_id: str
    output: dict[str, Any]
    errors: list[str]
    processing_time_ms: float
    timestamp: str
```

---

## Usage Examples

### Generate Discharge Summary

```python
result = crew.process_documentation(DocumentationRequest(
    patient_id="P001",
    document_type="discharge_summary",
    encounter_id="ENC001",
    clinical_notes=["Patient stable"],
    diagnoses=[{"display": "Hypertension"}]
))

if result.success:
    print(result.output["content"])
```

### Prior Authorization with Justification Scoring

```python
result = crew.process_prior_auth(PriorAuthRequest(
    patient_id="P001",
    procedure_code="27447",  # Total knee replacement
    diagnosis_codes=["M17.11"],
    insurance_id="INS-001",
    provider_id="DR-001",
    clinical_justification="Medical necessity confirmed. Failed conservative treatment."
))

print(f"Status: {result.output['status']}")
print(f"Justification Score: {result.output['justification_score']}")
# Scores >= 0.8 may get auto-approved
```

### Complex Workflow (Auth → Scheduling)

```python
results = crew.process_complex_workflow(
    prior_auth_request=auth_request,
    scheduling_request=sched_request
)

# If auth fails, scheduling is blocked
if results["prior_auth"].output.get("status") == "needs_info":
    print("Scheduling blocked - need more documentation")
```
