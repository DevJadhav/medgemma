# Communication Agent API

The Communication Agent handles patient engagement with safety-first design.

## Classes

### CommunicationOrchestrator

Main coordinator for communication agents.

```python
from medai_compass.agents.communication import CommunicationOrchestrator

comm = CommunicationOrchestrator(model_wrapper=None)
```

#### Methods

| Method | Description |
|--------|-------------|
| `process_message(message, context)` | Process patient message through agent team |
| `get_conversation_history(patient_id)` | Get conversation history |

---

### PatientMessage

Incoming patient message.

```python
from medai_compass.agents.communication import PatientMessage

message = PatientMessage(
    message_id="msg-001",
    patient_id="pat-001",
    content="I've been having headaches for a week",
    attachments=[]  # Optional file paths
)
```

---

### AgentResponse

Response from communication agents.

```python
@dataclass
class AgentResponse:
    message_id: str
    agent_name: str
    content: str
    triage_result: Optional[TriageResult]
    requires_clinician_review: bool
    confidence: float
    processing_time_ms: float
    timestamp: str
```

---

### TriageResult

Result from triage assessment.

```python
@dataclass
class TriageResult:
    urgency: UrgencyLevel           # EMERGENCY, URGENT, SOON, ROUTINE, INFORMATIONAL
    category: MessageCategory       # SYMPTOM_REPORT, MEDICATION_QUESTION, etc.
    recommended_action: str
    requires_human_review: bool
    confidence: float
    reasoning: str
    safety_flags: list[str]         # e.g., ["mental_health_crisis"]
```

---

### UrgencyLevel

```python
class UrgencyLevel(Enum):
    EMERGENCY = "emergency"         # Call 911
    URGENT = "urgent"               # Same-day appointment
    SOON = "soon"                   # Within 1-2 days
    ROUTINE = "routine"             # Regular scheduling
    INFORMATIONAL = "informational" # No action needed
```

---

### MessageCategory

```python
class MessageCategory(Enum):
    SYMPTOM_REPORT = "symptom_report"
    MEDICATION_QUESTION = "medication_question"
    APPOINTMENT_REQUEST = "appointment_request"
    TEST_RESULTS = "test_results"
    BILLING = "billing"
    GENERAL_HEALTH = "general_health"
    MENTAL_HEALTH = "mental_health"
    EMERGENCY = "emergency"
```

---

## Individual Agents

### TriageAgent

```python
from medai_compass.agents.communication import TriageAgent

triage = TriageAgent()
result = triage.triage_message(message)

if result.urgency == UrgencyLevel.EMERGENCY:
    # Immediate escalation
    pass
```

### HealthEducatorAgent

```python
from medai_compass.agents.communication import HealthEducatorAgent

educator = HealthEducatorAgent()
response = educator.respond_to_query(message)
# Response includes health info + disclaimer
```

### ClinicalOversightProxy

```python
from medai_compass.agents.communication import ClinicalOversightProxy

proxy = ClinicalOversightProxy()

# Flag for review
review_id = proxy.flag_for_review(message, response, "Low confidence")

# Complete review
proxy.complete_review(
    review_id=review_id,
    approved=True,
    reviewer_id="DR-001",
    notes="Response appropriate"
)
```

---

## Usage Examples

### Process Patient Message

```python
comm = CommunicationOrchestrator()

response = comm.process_message(PatientMessage(
    message_id="msg-001",
    patient_id="pat-001",
    content="What should I know about managing my diabetes?"
))

print(f"Agent: {response.agent_name}")
print(f"Urgency: {response.triage_result.urgency}")
print(f"Response:\n{response.content}")
```

### Handle Emergency

```python
response = comm.process_message(PatientMessage(
    message_id="msg-002",
    patient_id="pat-002",
    content="I'm having severe chest pain and can't breathe"
))

# Emergency detection
assert response.triage_result.urgency == UrgencyLevel.EMERGENCY
assert "911" in response.content
assert response.requires_clinician_review is True
```

### Mental Health Crisis

```python
response = comm.process_message(PatientMessage(
    message_id="msg-003",
    patient_id="pat-003",
    content="I've been having thoughts about hurting myself"
))

# Safety response
assert "988" in response.content  # Suicide hotline
assert "mental_health_crisis" in response.triage_result.safety_flags
```
