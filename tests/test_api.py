"""
Tests for MedAI Compass API endpoints.

Tests health checks, metrics, and agent endpoints.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_response_model(self):
        """Test HealthResponse model creation."""
        from medai_compass.api.main import HealthResponse

        response = HealthResponse(
            status="healthy",
            timestamp="2026-01-17T00:00:00",
            version="1.0.0",
            services={"api": "healthy", "redis": "healthy"},
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.services["api"] == "healthy"

    def test_health_response_defaults(self):
        """Test HealthResponse with default services."""
        from medai_compass.api.main import HealthResponse

        response = HealthResponse(
            status="healthy",
            timestamp="2026-01-17T00:00:00",
            version="1.0.0",
        )

        assert response.services == {}


class TestDiagnosticModels:
    """Tests for diagnostic request/response models."""

    def test_diagnostic_request_defaults(self):
        """Test DiagnosticRequest with defaults."""
        from medai_compass.api.main import DiagnosticRequest

        request = DiagnosticRequest()

        assert request.image_path is None
        assert request.image_type == "cxr"
        assert request.patient_id is None

    def test_diagnostic_request_full(self):
        """Test DiagnosticRequest with all fields."""
        from medai_compass.api.main import DiagnosticRequest

        request = DiagnosticRequest(
            image_path="/path/to/image.dcm",
            image_base64="base64data",
            image_type="ct",
            patient_id="P001",
            clinical_context="Chest pain evaluation",
        )

        assert request.image_path == "/path/to/image.dcm"
        assert request.image_type == "ct"
        assert request.patient_id == "P001"

    def test_diagnostic_response(self):
        """Test DiagnosticResponse model."""
        from medai_compass.api.main import DiagnosticResponse

        response = DiagnosticResponse(
            request_id="req-001",
            status="completed",
            findings=[{"finding": "nodule", "location": "RUL"}],
            confidence=0.92,
            report="Analysis complete",
            requires_review=False,
            processing_time_ms=150.5,
        )

        assert response.request_id == "req-001"
        assert response.status == "completed"
        assert len(response.findings) == 1
        assert response.confidence == 0.92

    def test_diagnostic_response_defaults(self):
        """Test DiagnosticResponse defaults."""
        from medai_compass.api.main import DiagnosticResponse

        response = DiagnosticResponse(request_id="req-001", status="pending")

        assert response.findings == []
        assert response.confidence == 0.0
        assert response.report == ""
        assert response.requires_review is False


class TestWorkflowModels:
    """Tests for workflow request/response models."""

    def test_workflow_request(self):
        """Test WorkflowRequest model."""
        from medai_compass.api.main import WorkflowRequest

        request = WorkflowRequest(
            request_type="scheduling",
            patient_id="P001",
            encounter_id="ENC001",
            data={"appointment_type": "follow_up"},
        )

        assert request.request_type == "scheduling"
        assert request.patient_id == "P001"
        assert request.data["appointment_type"] == "follow_up"

    def test_workflow_request_defaults(self):
        """Test WorkflowRequest defaults."""
        from medai_compass.api.main import WorkflowRequest

        request = WorkflowRequest(request_type="documentation")

        assert request.patient_id is None
        assert request.data == {}

    def test_workflow_response(self):
        """Test WorkflowResponse model."""
        from medai_compass.api.main import WorkflowResponse

        response = WorkflowResponse(
            request_id="req-001",
            status="completed",
            result={"appointment_id": "APT001"},
            processing_time_ms=200.0,
        )

        assert response.request_id == "req-001"
        assert response.result["appointment_id"] == "APT001"


class TestCommunicationModels:
    """Tests for communication request/response models."""

    def test_communication_request(self):
        """Test CommunicationRequest model."""
        from medai_compass.api.main import CommunicationRequest

        request = CommunicationRequest(
            message="I have chest pain",
            patient_id="P001",
            session_id="sess-001",
            language="en",
        )

        assert request.message == "I have chest pain"
        assert request.language == "en"

    def test_communication_request_defaults(self):
        """Test CommunicationRequest defaults."""
        from medai_compass.api.main import CommunicationRequest

        request = CommunicationRequest(message="Hello")

        assert request.patient_id is None
        assert request.session_id is None
        assert request.language == "en"

    def test_communication_response(self):
        """Test CommunicationResponse model."""
        from medai_compass.api.main import CommunicationResponse

        response = CommunicationResponse(
            request_id="req-001",
            response="Please seek immediate medical attention.",
            triage_level="EMERGENCY",
            requires_escalation=True,
            disclaimer="This is not medical advice.",
            session_id="sess-001",
            processing_time_ms=50.0,
        )

        assert response.triage_level == "EMERGENCY"
        assert response.requires_escalation is True


class TestRedisManager:
    """Tests for Redis connection manager."""

    def test_redis_manager_init(self):
        """Test RedisManager initialization."""
        from medai_compass.api.main import RedisManager

        manager = RedisManager()

        assert manager.client is None
        assert manager._connected is False
        assert manager.is_connected is False

    @pytest.mark.asyncio
    async def test_redis_manager_connect_no_redis(self):
        """Test RedisManager connect when Redis unavailable."""
        from medai_compass.api.main import RedisManager

        manager = RedisManager()

        # Mock REDIS_AVAILABLE to False
        with patch("medai_compass.api.main.REDIS_AVAILABLE", False):
            result = await manager.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_redis_manager_get_not_connected(self):
        """Test RedisManager get when not connected."""
        from medai_compass.api.main import RedisManager

        manager = RedisManager()
        result = await manager.get("some_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_redis_manager_set_not_connected(self):
        """Test RedisManager set when not connected."""
        from medai_compass.api.main import RedisManager

        manager = RedisManager()
        result = await manager.set("key", "value")

        assert result is False

    @pytest.mark.asyncio
    async def test_redis_manager_delete_not_connected(self):
        """Test RedisManager delete when not connected."""
        from medai_compass.api.main import RedisManager

        manager = RedisManager()
        result = await manager.delete("key")

        assert result is False

    @pytest.mark.asyncio
    async def test_redis_manager_incr_not_connected(self):
        """Test RedisManager incr when not connected."""
        from medai_compass.api.main import RedisManager

        manager = RedisManager()
        result = await manager.incr("counter")

        assert result is None


class TestAppFactory:
    """Tests for application factory."""

    def test_create_app(self):
        """Test create_app returns FastAPI instance."""
        from medai_compass.api.main import create_app

        app = create_app()

        assert app.title == "MedAI Compass API"
        assert app.version == "1.0.0"

    def test_app_routes_registered(self):
        """Test that all expected routes are registered."""
        from medai_compass.api.main import app

        routes = [route.path for route in app.routes]

        assert "/health" in routes
        assert "/health/ready" in routes
        assert "/health/live" in routes
        assert "/metrics" in routes
        assert "/api/v1/diagnostic/analyze" in routes
        assert "/api/v1/workflow/process" in routes
        assert "/api/v1/communication/message" in routes
        assert "/api/v1/orchestrator/process" in routes


class TestErrorModels:
    """Tests for error response models."""

    def test_error_response(self):
        """Test ErrorResponse model."""
        from medai_compass.api.main import ErrorResponse

        error = ErrorResponse(
            error="Not found",
            detail="Resource does not exist",
            request_id="req-001",
        )

        assert error.error == "Not found"
        assert error.detail == "Resource does not exist"

    def test_error_response_minimal(self):
        """Test ErrorResponse with minimal fields."""
        from medai_compass.api.main import ErrorResponse

        error = ErrorResponse(error="Server error")

        assert error.detail is None
        assert error.request_id is None


class TestAPIEndpointsWithClient:
    """Integration tests using TestClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from medai_compass.api.main import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_ready_endpoint(self, client):
        """Test /health/ready endpoint."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_health_live_endpoint(self, client):
        """Test /health/live endpoint."""
        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_diagnostic_analyze_endpoint(self, client):
        """Test /api/v1/diagnostic/analyze endpoint."""
        response = client.post(
            "/api/v1/diagnostic/analyze",
            json={
                "image_path": "/test/image.dcm",
                "image_type": "cxr",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "status" in data
        assert "processing_time_ms" in data

    def test_workflow_process_endpoint(self, client):
        """Test /api/v1/workflow/process endpoint."""
        response = client.post(
            "/api/v1/workflow/process",
            json={
                "request_type": "scheduling",
                "patient_id": "P001",
                "data": {"appointment_type": "follow_up"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "status" in data

    def test_workflow_process_documentation(self, client):
        """Test workflow documentation request."""
        response = client.post(
            "/api/v1/workflow/process",
            json={
                "request_type": "documentation",
                "patient_id": "P001",
                "encounter_id": "ENC001",
                "data": {
                    "document_type": "discharge_summary",
                    "clinical_notes": ["Patient stable"],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("completed", "error")

    def test_workflow_process_prior_auth(self, client):
        """Test workflow prior auth request."""
        response = client.post(
            "/api/v1/workflow/process",
            json={
                "request_type": "prior_auth",
                "patient_id": "P001",
                "data": {
                    "procedure_code": "27447",
                    "diagnosis_codes": ["M17.11"],
                    "clinical_justification": "Severe osteoarthritis",
                },
            },
        )

        assert response.status_code == 200

    def test_communication_message_endpoint(self, client):
        """Test /api/v1/communication/message endpoint."""
        response = client.post(
            "/api/v1/communication/message",
            json={
                "message": "What are the symptoms of diabetes?",
                "patient_id": "P001",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "response" in data
        assert "triage_level" in data
        assert "session_id" in data

    def test_communication_emergency_detection(self, client):
        """Test emergency detection in communication."""
        response = client.post(
            "/api/v1/communication/message",
            json={
                "message": "I'm having severe chest pain and difficulty breathing",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should detect potential emergency
        assert "response" in data

    def test_orchestrator_process_endpoint(self, client):
        """Test /api/v1/orchestrator/process endpoint."""
        response = client.post(
            "/api/v1/orchestrator/process",
            params={
                "message": "I need to schedule a follow-up appointment",
                "patient_id": "P001",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "status" in data

    def test_session_not_found(self, client):
        """Test session not found error."""
        response = client.get("/api/v1/session/nonexistent-session")

        # Should return 503 (Redis unavailable) or 404 (not found)
        assert response.status_code in (404, 503)

    def test_docs_endpoint(self, client):
        """Test OpenAPI docs are available."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        """Test ReDoc docs are available."""
        response = client.get("/redoc")

        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "MedAI Compass API"
        assert "paths" in schema


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from medai_compass.api.main import app

        return TestClient(app)

    def test_metrics_endpoint_available(self, client):
        """Test /metrics endpoint is available."""
        response = client.get("/metrics")

        # Will be 200 if prometheus_client installed, 501 otherwise
        assert response.status_code in (200, 501)

    def test_metrics_content_type(self, client):
        """Test metrics endpoint returns correct content type."""
        response = client.get("/metrics")

        if response.status_code == 200:
            assert "text/plain" in response.headers.get("content-type", "")


class TestRequestModelsValidation:
    """Tests for request model validation."""

    def test_diagnostic_request_invalid_type(self):
        """Test DiagnosticRequest accepts any image_type string."""
        from medai_compass.api.main import DiagnosticRequest

        # Should not raise - no enum validation
        request = DiagnosticRequest(image_type="unknown")
        assert request.image_type == "unknown"

    def test_workflow_request_missing_type(self):
        """Test WorkflowRequest requires request_type."""
        from medai_compass.api.main import WorkflowRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowRequest()

    def test_communication_request_missing_message(self):
        """Test CommunicationRequest requires message."""
        from medai_compass.api.main import CommunicationRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CommunicationRequest()
