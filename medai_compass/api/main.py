"""
MedAI Compass API - Production FastAPI Application.

HIPAA-compliant API for the multi-agent medical AI system.
Uses Redis for session management and caching.
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import escalation store early for proper patching in tests
from medai_compass.utils.escalation_store import escalation_store

# Prometheus metrics
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    from starlette.responses import Response

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Redis client
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# =============================================================================
# Prometheus Metrics
# =============================================================================
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "medai_requests_total",
        "Total number of requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "medai_request_latency_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    ACTIVE_REQUESTS = Gauge(
        "medai_active_requests",
        "Number of active requests",
    )
    AGENT_CALLS = Counter(
        "medai_agent_calls_total",
        "Total agent invocations",
        ["agent_type", "status"],
    )
    AGENT_LATENCY = Histogram(
        "medai_agent_latency_seconds",
        "Agent processing latency",
        ["agent_type"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    )
    MODEL_INFERENCE_LATENCY = Histogram(
        "medai_model_inference_seconds",
        "Model inference latency",
        ["model_name"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    ESCALATIONS = Counter(
        "medai_escalations_total",
        "Total escalations to human review",
        ["reason"],
    )
    CRITICAL_FINDINGS = Counter(
        "medai_critical_findings_total",
        "Critical findings detected",
        ["finding_type"],
    )


# =============================================================================
# Pydantic Models
# =============================================================================
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    services: dict[str, str] = Field(default_factory=dict, description="Service statuses")


class DiagnosticRequest(BaseModel):
    """Diagnostic analysis request."""

    image_path: Optional[str] = Field(None, description="Path to image file")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_type: str = Field("cxr", description="Image type: cxr, ct, mri, pathology")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    clinical_context: Optional[str] = Field(None, description="Clinical context")


class DiagnosticResponse(BaseModel):
    """Diagnostic analysis response."""

    request_id: str
    status: str
    findings: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    report: str = ""
    requires_review: bool = False
    processing_time_ms: float = 0.0


class WorkflowRequest(BaseModel):
    """Workflow processing request."""

    request_type: str = Field(..., description="Type: scheduling, documentation, prior_auth")
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    data: dict = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    """Workflow processing response."""

    request_id: str
    status: str
    result: dict = Field(default_factory=dict)
    processing_time_ms: float = 0.0


class CommunicationRequest(BaseModel):
    """Patient communication request."""

    message: str = Field(..., description="Patient message")
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    language: str = Field("en", description="Language code")


class CommunicationResponse(BaseModel):
    """Patient communication response."""

    request_id: str
    response: str
    triage_level: str = "INFORMATIONAL"
    requires_escalation: bool = False
    disclaimer: str = ""
    session_id: str = ""
    processing_time_ms: float = 0.0


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


# =============================================================================
# Escalation Models
# =============================================================================
class EscalationCreateRequest(BaseModel):
    """Create escalation request."""

    request_id: str = Field(..., description="Original request ID")
    reason: str = Field(..., description="Escalation reason")
    priority: str = Field("medium", description="Priority: high, medium, low")
    patient_id: Optional[str] = None
    original_message: Optional[str] = None
    diagnostic_result: Optional[dict] = None
    communication_result: Optional[dict] = None
    workflow_result: Optional[dict] = None
    agent_type: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Optional[dict] = None


class ReviewRequest(BaseModel):
    """Review decision request."""

    decision: str = Field(..., description="Decision: approve, reject, modify")
    notes: str = Field(..., min_length=1, description="Review notes")
    reviewer_id: str = Field(..., description="Reviewer identifier")
    modified_response: Optional[str] = Field(None, description="Modified response for modify decision")


class EscalationResponse(BaseModel):
    """Single escalation response."""

    id: str
    request_id: str
    timestamp: str
    patient_id: Optional[str] = None
    reason: str
    priority: str
    status: str
    original_message: Optional[str] = None
    diagnostic_result: Optional[dict] = None
    communication_result: Optional[dict] = None
    agent_type: Optional[str] = None
    confidence_score: Optional[float] = None
    assigned_to: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None
    modified_response: Optional[str] = None


class EscalationListResponse(BaseModel):
    """List escalations response."""

    escalations: list[dict] = Field(default_factory=list)
    total: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EscalationStatsResponse(BaseModel):
    """Escalation statistics response."""

    total_pending: int = 0
    total_in_review: int = 0
    total_approved_today: int = 0
    total_rejected_today: int = 0
    average_review_time_ms: float = 0.0


# =============================================================================
# Redis Connection Manager
# =============================================================================
class RedisManager:
    """Async Redis connection manager."""

    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            return False

        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD", None)

            self.client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True,
                socket_timeout=5,
            )
            await self.client.ping()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self._connected or not self.client:
            return None
        try:
            return await self.client.get(key)
        except Exception:
            return None

    async def set(
        self, key: str, value: str, expire_seconds: Optional[int] = None
    ) -> bool:
        """Set value in Redis."""
        if not self._connected or not self.client:
            return False
        try:
            await self.client.set(key, value, ex=expire_seconds)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._connected or not self.client:
            return False
        try:
            await self.client.delete(key)
            return True
        except Exception:
            return False

    async def incr(self, key: str) -> Optional[int]:
        """Increment counter."""
        if not self._connected or not self.client:
            return None
        try:
            return await self.client.incr(key)
        except Exception:
            return None


# Global Redis manager
redis_manager = RedisManager()


# =============================================================================
# Application Factory
# =============================================================================
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        await redis_manager.connect()
        yield
        # Shutdown
        await redis_manager.disconnect()

    application = FastAPI(
        title="MedAI Compass API",
        description="HIPAA-compliant Multi-Agent Medical AI Platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "").split(",") or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @application.middleware("http")
    async def timing_middleware(request: Request, call_next):
        """Add timing and metrics to requests."""
        start_time = time.time()

        if PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.inc()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            duration = time.time() - start_time

            if PROMETHEUS_AVAILABLE:
                ACTIVE_REQUESTS.dec()
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=status_code,
                ).inc()
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=request.url.path,
                ).observe(duration)

        response.headers["X-Process-Time"] = str(duration)
        return response

    return application


# Create the app instance
app = create_app()


# =============================================================================
# Health & Metrics Endpoints
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns service status and dependent service health.
    """
    services = {
        "api": "healthy",
        "redis": "healthy" if redis_manager.is_connected else "unavailable",
    }

    # Check Redis with ping
    if redis_manager.is_connected:
        try:
            await redis_manager.client.ping()
        except Exception:
            services["redis"] = "unhealthy"

    overall_status = "healthy" if all(
        s in ("healthy", "unavailable") for s in services.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        services=services,
    )


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes readiness probe."""
    if not redis_manager.is_connected:
        # Still ready even without Redis (graceful degradation)
        pass
    return {"status": "ready"}


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Prometheus client not installed",
        )

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# Diagnostic Agent Endpoints
# =============================================================================
@app.post(
    "/api/v1/diagnostic/analyze",
    response_model=DiagnosticResponse,
    tags=["Diagnostic"],
)
async def analyze_image(request: DiagnosticRequest) -> DiagnosticResponse:
    """
    Analyze medical image using the diagnostic agent.

    Supports chest X-rays, CT scans, MRI, and pathology images.
    """
    import uuid

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Import diagnostic agent
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        from medai_compass.agents.diagnostic.state import create_initial_state

        # Create initial state
        state = create_initial_state()
        state["image_path"] = request.image_path or ""
        state["image_type"] = request.image_type

        # Run diagnostic workflow
        graph = create_diagnostic_graph()
        result = graph.invoke(state)

        processing_time = (time.time() - start_time) * 1000

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="diagnostic", status="success").inc()
            AGENT_LATENCY.labels(agent_type="diagnostic").observe(processing_time / 1000)

            if result.get("requires_review", False):
                ESCALATIONS.labels(reason="low_confidence").inc()

        return DiagnosticResponse(
            request_id=request_id,
            status="completed",
            findings=result.get("findings", []),
            confidence=result.get("confidence", 0.0),
            report=result.get("report", ""),
            requires_review=result.get("requires_review", False),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="diagnostic", status="error").inc()

        return DiagnosticResponse(
            request_id=request_id,
            status="error",
            report=f"Analysis failed: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


# =============================================================================
# Workflow Agent Endpoints
# =============================================================================
@app.post(
    "/api/v1/workflow/process",
    response_model=WorkflowResponse,
    tags=["Workflow"],
)
async def process_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """
    Process workflow request (scheduling, documentation, prior auth).
    """
    import uuid

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        from medai_compass.agents.workflow.crew import WorkflowCrew

        crew = WorkflowCrew()
        result: dict[str, Any] = {}

        if request.request_type == "scheduling":
            from medai_compass.agents.workflow.crew import SchedulingRequest

            sched_req = SchedulingRequest(
                patient_id=request.patient_id or "",
                appointment_type=request.data.get("appointment_type", "follow_up"),
                urgency=request.data.get("urgency", "routine"),
                preferred_dates=request.data.get("preferred_dates", []),
                notes=request.data.get("notes", ""),
            )
            result = crew.process_scheduling(sched_req)

        elif request.request_type == "documentation":
            from medai_compass.agents.workflow.crew import DocumentationRequest

            doc_req = DocumentationRequest(
                patient_id=request.patient_id or "",
                document_type=request.data.get("document_type", "progress_note"),
                encounter_id=request.encounter_id or "",
                clinical_notes=request.data.get("clinical_notes", []),
                diagnoses=request.data.get("diagnoses", []),
            )
            result = crew.process_documentation(doc_req)

        elif request.request_type == "prior_auth":
            from medai_compass.agents.workflow.crew import PriorAuthRequest

            auth_req = PriorAuthRequest(
                patient_id=request.patient_id or "",
                procedure_code=request.data.get("procedure_code", ""),
                diagnosis_codes=request.data.get("diagnosis_codes", []),
                clinical_justification=request.data.get("clinical_justification", ""),
                urgency=request.data.get("urgency", "routine"),
            )
            result = crew.process_prior_auth(auth_req)

        processing_time = (time.time() - start_time) * 1000

        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="workflow", status="success").inc()
            AGENT_LATENCY.labels(agent_type="workflow").observe(processing_time / 1000)

        return WorkflowResponse(
            request_id=request_id,
            status="completed",
            result=result if isinstance(result, dict) else {"output": str(result)},
            processing_time_ms=processing_time,
        )

    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="workflow", status="error").inc()

        return WorkflowResponse(
            request_id=request_id,
            status="error",
            result={"error": str(e)},
            processing_time_ms=(time.time() - start_time) * 1000,
        )


# =============================================================================
# Communication Agent Endpoints
# =============================================================================
@app.post(
    "/api/v1/communication/message",
    response_model=CommunicationResponse,
    tags=["Communication"],
)
async def process_message(request: CommunicationRequest) -> CommunicationResponse:
    """
    Process patient communication message.

    Handles triage, health education, and follow-up scheduling.
    """
    import uuid

    start_time = time.time()
    request_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())

    try:
        from medai_compass.agents.communication.agents import (
            CommunicationOrchestrator,
            PatientMessage,
        )

        orchestrator = CommunicationOrchestrator()

        patient_msg = PatientMessage(
            message_id=request_id,
            patient_id=request.patient_id or "anonymous",
            content=request.message,
            language=request.language,
        )

        result = orchestrator.process_message(patient_msg)

        processing_time = (time.time() - start_time) * 1000

        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="communication", status="success").inc()
            AGENT_LATENCY.labels(agent_type="communication").observe(
                processing_time / 1000
            )

            if result.requires_escalation:
                ESCALATIONS.labels(reason="communication_escalation").inc()

        # Store session in Redis
        if redis_manager.is_connected:
            await redis_manager.set(
                f"session:{session_id}",
                request.message,
                expire_seconds=3600,  # 1 hour
            )

        return CommunicationResponse(
            request_id=request_id,
            response=result.response,
            triage_level=result.triage_level.value if hasattr(result.triage_level, 'value') else str(result.triage_level),
            requires_escalation=result.requires_escalation,
            disclaimer=result.disclaimer,
            session_id=session_id,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="communication", status="error").inc()

        return CommunicationResponse(
            request_id=request_id,
            response=f"Unable to process message: {str(e)}",
            triage_level="INFORMATIONAL",
            requires_escalation=True,
            disclaimer="Please consult a healthcare provider.",
            session_id=session_id,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


# =============================================================================
# Orchestrator Endpoint
# =============================================================================
@app.post("/api/v1/orchestrator/process", tags=["Orchestrator"])
async def orchestrator_process(
    message: str,
    patient_id: Optional[str] = None,
    context: Optional[dict] = None,
):
    """
    Process request through master orchestrator.

    Automatically routes to appropriate agent based on intent classification.
    """
    import uuid

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        from medai_compass.orchestrator.master import MasterOrchestrator

        orchestrator = MasterOrchestrator()
        result = orchestrator.process_request(
            message=message,
            patient_id=patient_id,
            context=context or {},
        )

        processing_time = (time.time() - start_time) * 1000

        return {
            "request_id": request_id,
            "status": "completed",
            "result": result,
            "processing_time_ms": processing_time,
        }

    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e),
            "processing_time_ms": (time.time() - start_time) * 1000,
        }


# =============================================================================
# Session Management Endpoints
# =============================================================================
@app.get("/api/v1/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """Get session data from Redis."""
    if not redis_manager.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session storage unavailable",
        )

    data = await redis_manager.get(f"session:{session_id}")
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return {"session_id": session_id, "data": data}


@app.delete("/api/v1/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """Delete session from Redis."""
    if not redis_manager.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session storage unavailable",
        )

    await redis_manager.delete(f"session:{session_id}")
    return {"status": "deleted", "session_id": session_id}


# =============================================================================
# Escalation Endpoints for Clinician Review
# =============================================================================


@app.get(
    "/api/v1/escalations",
    response_model=EscalationListResponse,
    tags=["Escalations"],
)
async def list_escalations(
    priority: Optional[str] = None,
    reason: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List pending escalations for clinician review.
    
    Supports filtering by priority, reason, and status.
    Results are ordered by priority (high first) then timestamp.
    """
    try:
        escalations = escalation_store.list_pending(
            priority=priority,
            reason=reason,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        
        return EscalationListResponse(
            escalations=escalations,
            total=len(escalations),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list escalations: {str(e)}",
        )


@app.get(
    "/api/v1/escalations/stats",
    response_model=EscalationStatsResponse,
    tags=["Escalations"],
)
async def get_escalation_stats():
    """
    Get escalation statistics for dashboard.
    
    Returns counts of pending, in-review, approved, and rejected escalations.
    """
    try:
        stats = escalation_store.get_stats()
        return EscalationStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


@app.get(
    "/api/v1/escalations/{escalation_id}",
    response_model=EscalationResponse,
    tags=["Escalations"],
)
async def get_escalation(escalation_id: str):
    """
    Get single escalation by ID.
    """
    escalation = escalation_store.get_by_id(escalation_id)
    
    if not escalation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Escalation {escalation_id} not found",
        )
    
    return EscalationResponse(**escalation)


@app.post(
    "/api/v1/escalations",
    response_model=EscalationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Escalations"],
)
async def create_escalation(request: EscalationCreateRequest):
    """
    Create a new escalation for clinician review.
    
    Called internally when agents detect conditions requiring human review.
    """
    try:
        # Validate reason and priority
        valid_reasons = {"critical_finding", "low_confidence", "safety_concern", "manual_request"}
        valid_priorities = {"high", "medium", "low"}
        
        if request.reason not in valid_reasons:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid reason. Must be one of: {valid_reasons}",
            )
        
        if request.priority not in valid_priorities:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid priority. Must be one of: {valid_priorities}",
            )
        
        escalation = escalation_store.create(
            request_id=request.request_id,
            reason=request.reason,
            priority=request.priority,
            patient_id=request.patient_id,
            original_message=request.original_message,
            diagnostic_result=request.diagnostic_result,
            communication_result=request.communication_result,
            workflow_result=request.workflow_result,
            agent_type=request.agent_type,
            confidence_score=request.confidence_score,
            metadata=request.metadata,
        )
        
        # Record metric
        if PROMETHEUS_AVAILABLE:
            ESCALATIONS.labels(reason=request.reason).inc()
        
        return EscalationResponse(**escalation)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create escalation: {str(e)}",
        )


@app.post(
    "/api/v1/escalations/{escalation_id}/review",
    response_model=EscalationResponse,
    tags=["Escalations"],
)
async def submit_escalation_review(escalation_id: str, request: ReviewRequest):
    """
    Submit review decision for an escalation.
    
    Clinicians can approve, reject, or modify and approve escalations.
    """
    # Validate escalation exists
    escalation = escalation_store.get_by_id(escalation_id)
    if not escalation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Escalation {escalation_id} not found",
        )
    
    # Validate decision
    valid_decisions = {"approve", "reject", "modify"}
    if request.decision not in valid_decisions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid decision. Must be one of: {valid_decisions}",
        )
    
    # Require modified_response for modify decision
    if request.decision == "modify" and not request.modified_response:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="modified_response required for modify decision",
        )
    
    try:
        updated = escalation_store.submit_review(
            escalation_id=escalation_id,
            decision=request.decision,
            reviewer_id=request.reviewer_id,
            notes=request.notes,
            modified_response=request.modified_response,
        )
        
        return EscalationResponse(**updated)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit review: {str(e)}",
        )


# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None,
        ).model_dump(),
    )


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "medai_compass.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )
