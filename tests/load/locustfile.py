"""
Load Testing for MedAI Compass API.

Run with: locust -f tests/load/locustfile.py --host=http://localhost:8000

Usage:
    # Start API first
    docker-compose up -d api redis postgres
    
    # Run load test
    locust -f tests/load/locustfile.py --host=http://localhost:8000
    
    # Headless mode (CI/CD)
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
           --headless -u 10 -r 2 -t 60s
"""

import json
import random
from locust import HttpUser, task, between


class MedAIUser(HttpUser):
    """Simulates a user interacting with MedAI Compass API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts - verify health."""
        self.client.get("/health")
    
    @task(10)
    def health_check(self):
        """Health check - most common endpoint."""
        self.client.get("/health")
    
    @task(5)
    def health_ready(self):
        """Readiness probe."""
        self.client.get("/health/ready")
    
    @task(5)
    def health_live(self):
        """Liveness probe."""
        self.client.get("/health/live")
    
    @task(3)
    def communication_message(self):
        """Patient communication - common use case."""
        messages = [
            "What are the symptoms of diabetes?",
            "How do I manage high blood pressure?",
            "When should I see a doctor for a headache?",
            "What medications are used for cholesterol?",
            "How often should I exercise?",
        ]
        
        self.client.post(
            "/api/v1/communication/message",
            json={
                "message": random.choice(messages),
                "patient_id": f"load-test-patient-{random.randint(1, 100)}",
                "language": "en"
            },
            headers={"Content-Type": "application/json"}
        )
    
    @task(2)
    def workflow_scheduling(self):
        """Workflow scheduling request."""
        self.client.post(
            "/api/v1/workflow/process",
            json={
                "request_type": "scheduling",
                "patient_id": f"load-test-patient-{random.randint(1, 100)}",
                "data": {
                    "appointment_type": random.choice(["follow_up", "new_patient", "consultation"]),
                    "urgency": random.choice(["routine", "urgent", "emergency"]),
                    "notes": "Load test appointment"
                }
            },
            headers={"Content-Type": "application/json"}
        )
    
    @task(2)
    def workflow_documentation(self):
        """Workflow documentation request."""
        self.client.post(
            "/api/v1/workflow/process",
            json={
                "request_type": "documentation",
                "patient_id": f"load-test-patient-{random.randint(1, 100)}",
                "encounter_id": f"ENC-{random.randint(1000, 9999)}",
                "data": {
                    "document_type": random.choice(["progress_note", "discharge_summary"]),
                    "clinical_notes": ["Patient stable", "Continue current medications"]
                }
            },
            headers={"Content-Type": "application/json"}
        )
    
    @task(1)
    def diagnostic_analyze(self):
        """Diagnostic analysis - resource intensive."""
        self.client.post(
            "/api/v1/diagnostic/analyze",
            json={
                "image_path": "/test/load-test-image.dcm",
                "image_type": random.choice(["cxr", "ct", "mri"]),
                "patient_id": f"load-test-patient-{random.randint(1, 100)}",
                "clinical_context": "Load test diagnostic request"
            },
            headers={"Content-Type": "application/json"}
        )
    
    @task(1)
    def orchestrator_process(self):
        """Master orchestrator - routes to appropriate agent."""
        messages = [
            "I need to schedule a follow-up appointment",
            "Generate a discharge summary for patient",
            "What are the side effects of metformin?",
            "Analyze chest X-ray for patient",
        ]
        
        self.client.post(
            "/api/v1/orchestrator/process",
            params={
                "message": random.choice(messages),
                "patient_id": f"load-test-patient-{random.randint(1, 100)}"
            }
        )


class HealthCheckUser(HttpUser):
    """User that only checks health endpoints - simulates monitoring."""
    
    wait_time = between(5, 10)
    
    @task
    def health_check(self):
        """Health check."""
        self.client.get("/health")
    
    @task
    def metrics(self):
        """Prometheus metrics scrape."""
        self.client.get("/metrics")


# Load test configuration
"""
Recommended test scenarios:

1. Smoke Test (verify system works):
   locust --headless -u 1 -r 1 -t 30s

2. Load Test (normal load):
   locust --headless -u 10 -r 2 -t 5m

3. Stress Test (find breaking point):
   locust --headless -u 50 -r 5 -t 10m

4. Spike Test (sudden traffic surge):
   locust --headless -u 100 -r 50 -t 2m

5. Soak Test (sustained load):
   locust --headless -u 20 -r 2 -t 1h

Expected Results (on modest hardware):
- Health endpoints: < 10ms p95
- Communication: < 500ms p95
- Workflow: < 1s p95
- Diagnostic: < 5s p95 (depends on model)
- Throughput: 50-100 req/s (without GPU inference)
"""
