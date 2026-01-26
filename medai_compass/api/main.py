"""
MedAI Compass API - Production FastAPI Application.

HIPAA-compliant API for the multi-agent medical AI system.
Uses Redis for session management and caching.
"""

import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import escalation store early for proper patching in tests
from medai_compass.utils.escalation_store import escalation_store
from medai_compass.api.guardrails import router as guardrails_router
from medai_compass.api.rag import router as rag_router

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
    from prometheus_client import REGISTRY
    
    # Helper to get or create a metric
    def get_or_create_counter(name, description, labels):
        """Get existing counter or create a new one."""
        try:
            return Counter(name, description, labels)
        except ValueError:
            # Already exists, get it from registry
            return REGISTRY._names_to_collectors.get(name, Counter(name, description, labels))
    
    def get_or_create_histogram(name, description, labels, buckets):
        """Get existing histogram or create a new one."""
        try:
            return Histogram(name, description, labels, buckets=buckets)
        except ValueError:
            return REGISTRY._names_to_collectors.get(name, Histogram(name, description, labels, buckets=buckets))
    
    def get_or_create_gauge(name, description):
        """Get existing gauge or create a new one."""
        try:
            return Gauge(name, description)
        except ValueError:
            return REGISTRY._names_to_collectors.get(name, Gauge(name, description))
    
    REQUEST_COUNT = get_or_create_counter(
        "medai_requests_total",
        "Total number of requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = get_or_create_histogram(
        "medai_request_latency_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    ACTIVE_REQUESTS = get_or_create_gauge(
        "medai_active_requests",
        "Number of active requests",
    )
    AGENT_CALLS = get_or_create_counter(
        "medai_agent_calls_total",
        "Total agent invocations",
        ["agent_type", "status"],
    )
    AGENT_LATENCY = get_or_create_histogram(
        "medai_agent_latency_seconds",
        "Agent processing latency",
        ["agent_type"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    )
    MODEL_INFERENCE_LATENCY = get_or_create_histogram(
        "medai_model_inference_seconds",
        "Model inference latency",
        ["model_name"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    ESCALATIONS = get_or_create_counter(
        "medai_escalations_total",
        "Total escalations to human review",
        ["reason"],
    )
    CRITICAL_FINDINGS = get_or_create_counter(
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


class SystemMetricsResponse(BaseModel):
    """System metrics for analytics dashboard."""
    
    timestamp: str
    uptime_seconds: float
    active_sessions: int
    total_requests_today: int
    avg_response_time_ms: float
    model_status: str  # "online", "warming", "offline"
    gpu_available: bool
    gpu_name: Optional[str] = None
    redis_connected: bool
    postgres_connected: bool
    modal_connected: bool
    inference_queue_size: int
    recent_errors: int


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
        # SECURITY: Don't allow credentials with wildcard origins
        allow_credentials=bool(os.getenv("CORS_ORIGINS")),
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
app.include_router(guardrails_router)
app.include_router(rag_router)


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
# Thread-Safe Metrics (Fixed from global mutable state)
# =============================================================================

class ThreadSafeMetrics:
    """
    Thread-safe metrics counter for concurrent request tracking.

    Solves race condition issues with global mutable state under load.
    Uses locks for atomic operations and bounded deque for memory efficiency.
    """

    def __init__(self, max_request_history: int = 1000):
        """
        Initialize thread-safe metrics.

        Args:
            max_request_history: Maximum number of request times to keep
        """
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._request_times: deque[float] = deque(maxlen=max_request_history)
        self._error_count = 0
        self._total_requests = 0

    @property
    def uptime(self) -> float:
        """Get API uptime in seconds."""
        return time.time() - self._start_time

    def record_request(self, response_time_ms: float) -> None:
        """
        Record a request with its response time.

        Args:
            response_time_ms: Response time in milliseconds
        """
        with self._lock:
            self._request_times.append(response_time_ms)
            self._total_requests += 1

    def record_error(self) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._error_count += 1

    def get_metrics(self) -> dict:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary with metrics data
        """
        with self._lock:
            request_times = list(self._request_times)
            error_count = self._error_count
            total_requests = self._total_requests

        avg_response_time = sum(request_times) / len(request_times) if request_times else 0.0

        return {
            "uptime": self.uptime,
            "total_requests": total_requests,
            "error_count": error_count,
            "avg_response_time_ms": avg_response_time,
            "recent_request_count": len(request_times),
            "error_rate": error_count / total_requests if total_requests > 0 else 0.0
        }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._request_times.clear()
            self._error_count = 0
            self._total_requests = 0


# Global thread-safe metrics instance
_metrics = ThreadSafeMetrics()

# Legacy aliases for backwards compatibility (deprecated)
_api_start_time = _metrics._start_time
_request_times: deque[float] = _metrics._request_times
_error_count = 0  # Deprecated: use _metrics.record_error()


@app.get(
    "/api/v1/system/metrics",
    response_model=SystemMetricsResponse,
    tags=["Monitoring"],
)
async def get_system_metrics() -> SystemMetricsResponse:
    """
    Get real-time system metrics for the analytics dashboard.

    Returns comprehensive system health and performance data including:
    - Service uptime and active sessions
    - Model and GPU status
    - Database and cache connectivity
    - Request throughput and latency
    """
    # Use thread-safe metrics
    metrics_snapshot = _metrics.get_metrics()
    uptime = metrics_snapshot["uptime"]
    
    # Check Redis connection and get active sessions
    active_sessions = 0
    redis_connected = redis_manager.is_connected
    if redis_connected and redis_manager.client:
        try:
            # Count session keys
            keys = await redis_manager.client.keys("session:*")
            active_sessions = len(keys) if keys else 0
        except Exception:
            pass
    
    # Check PostgreSQL connection using asyncpg
    postgres_connected = False
    try:
        import asyncpg
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "")
        if postgres_host and postgres_password:
            conn = await asyncpg.connect(
                host=postgres_host,
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "medai"),
                password=postgres_password,
                database=os.getenv("POSTGRES_DB", "medai_compass"),
                timeout=2
            )
            await conn.close()
            postgres_connected = True
    except Exception:
        pass
    
    # Check Modal connectivity and GPU status
    modal_connected = False
    model_status = "offline"
    gpu_available = False
    gpu_name = None
    
    try:
        # Check if Modal is configured and accessible
        import modal
        modal_token_id = os.getenv("MODAL_TOKEN_ID")
        modal_token_secret = os.getenv("MODAL_TOKEN_SECRET")
        
        if modal_token_id and modal_token_secret:
            # Modal credentials are configured
            modal_connected = True
            model_status = "online"
            gpu_available = True
            gpu_name = "NVIDIA H100 (Modal)"
    except ImportError:
        # Modal SDK not installed
        pass
    except Exception:
        pass
    
    # Fallback: check local GPU if Modal not available
    if not gpu_available:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                model_status = "online"
        except Exception:
            pass
    
    # Get request metrics from Prometheus if available
    total_requests_today = 0
    avg_response_time_ms = 0.0
    
    if PROMETHEUS_AVAILABLE:
        try:
            # Get total request count (approximate for today)
            total_requests_today = int(REQUEST_COUNT._metrics.get((), None) or 0) if hasattr(REQUEST_COUNT, '_metrics') else 0
        except Exception:
            pass
    
    # Calculate average response time from recent requests
    if _request_times:
        avg_response_time_ms = sum(_request_times[-100:]) / len(_request_times[-100:]) * 1000
    
    # Get inference queue size (approximate)
    inference_queue_size = 0
    if PROMETHEUS_AVAILABLE:
        try:
            inference_queue_size = int(ACTIVE_REQUESTS._value._value) if hasattr(ACTIVE_REQUESTS, '_value') else 0
        except Exception:
            pass
    
    return SystemMetricsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=uptime,
        active_sessions=active_sessions,
        total_requests_today=total_requests_today,
        avg_response_time_ms=avg_response_time_ms,
        model_status=model_status,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        redis_connected=redis_connected,
        postgres_connected=postgres_connected,
        modal_connected=modal_connected,
        inference_queue_size=inference_queue_size,
        recent_errors=_error_count
    )


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
    import base64
    import tempfile
    import os as os_module

    start_time = time.time()
    request_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    patient_id = request.patient_id or "anonymous"
    temp_file_path = None

    try:
        # Handle image input - either path or base64
        image_path = request.image_path
        
        if request.image_base64 and not image_path:
            # Decode base64 and save to temp file
            try:
                image_bytes = base64.b64decode(request.image_base64)
                # Determine extension based on image type
                ext = ".png"  # Default
                if request.image_type == "cxr":
                    ext = ".dcm"  # DICOM for chest X-rays
                
                # Create temp file
                fd, temp_file_path = tempfile.mkstemp(suffix=ext)
                with os_module.fdopen(fd, 'wb') as f:
                    f.write(image_bytes)
                image_path = temp_file_path
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image: {str(e)}")
        
        if not image_path:
            raise ValueError("No image provided. Please provide image_path or image_base64")

        # Import diagnostic agent
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        from medai_compass.agents.diagnostic.state import create_initial_state

        # Create initial state with required patient_id and session_id
        state = create_initial_state(
            patient_id=patient_id,
            session_id=session_id,
            images=[image_path]
        )
        state["image_path"] = image_path
        state["image_type"] = request.image_type

        # Run diagnostic workflow without checkpointer (avoids numpy serialization issues)
        graph = create_diagnostic_graph(use_checkpointer=False)
        result = graph.invoke(state)

        processing_time = (time.time() - start_time) * 1000
        requires_review = result.get("requires_review", False)
        confidence = result.get("confidence", 0.0)

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="diagnostic", status="success").inc()
            AGENT_LATENCY.labels(agent_type="diagnostic").observe(processing_time / 1000)

            if requires_review:
                ESCALATIONS.labels(reason="low_confidence").inc()

        # Create escalation record for analytics tracking
        if requires_review or confidence < 0.7:
            try:
                escalation_store.create(
                    request_id=request_id,
                    reason=f"Diagnostic review needed: confidence={confidence:.2f}",
                    priority="medium" if confidence < 0.5 else "low",
                    patient_id=patient_id,
                    diagnostic_result={
                        "findings": result.get("findings", []),
                        "confidence": confidence,
                        "image_path": image_path
                    },
                    agent_type="diagnostic",
                    confidence_score=confidence
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to store diagnostic escalation: {e}")

        return DiagnosticResponse(
            request_id=request_id,
            status="completed",
            findings=result.get("findings", []),
            confidence=confidence,
            report=result.get("report", ""),
            requires_review=requires_review,
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
    
    finally:
        # Clean up temp file if created
        if temp_file_path and os_module.path.exists(temp_file_path):
            try:
                os_module.remove(temp_file_path)
            except Exception:
                pass  # Ignore cleanup errors


# =============================================================================
# Async Diagnostic Endpoints (Task 4.1: Celery Queuing)
# =============================================================================
class AsyncJobResponse(BaseModel):
    """Response for async job submission."""
    job_id: str
    status: str
    message: str = ""


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: str
    progress: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@app.post(
    "/api/v1/diagnostic/analyze-async",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Diagnostic"],
)
async def analyze_image_async(request: DiagnosticRequest) -> AsyncJobResponse:
    """
    Submit async diagnostic image analysis.

    Returns a job ID for polling status. Use for long-running analyses.
    """
    from medai_compass.workers.inference_tasks import (
        create_job,
        analyze_diagnostic_async,
    )

    # Create job
    job_id = create_job(
        "diagnostic",
        metadata={
            "image_type": request.image_type,
            "patient_id": request.patient_id,
        }
    )

    # Submit to Celery (or run inline for testing without Celery)
    try:
        analyze_diagnostic_async.delay(
            job_id=job_id,
            image_path=request.image_path,
            image_base64=request.image_base64,
            image_type=request.image_type,
            patient_id=request.patient_id,
            clinical_context=request.clinical_context,
        )
    except Exception:
        # Celery not available, run synchronously
        analyze_diagnostic_async(
            job_id=job_id,
            image_path=request.image_path,
            image_base64=request.image_base64,
            image_type=request.image_type,
            patient_id=request.patient_id,
            clinical_context=request.clinical_context,
        )

    return AsyncJobResponse(
        job_id=job_id,
        status="accepted",
        message="Analysis job submitted. Poll /api/v1/jobs/{job_id} for status.",
    )


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get status of an async job.

    Poll this endpoint to check job progress and retrieve results.
    """
    from medai_compass.workers.inference_tasks import get_job

    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job.get("created_at", ""),
        updated_at=job.get("updated_at", ""),
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
        
        # Map AgentResponse fields to CommunicationResponse
        requires_escalation = getattr(result, 'requires_clinician_review', False) or getattr(result, 'requires_escalation', False)
        triage_level = "INFORMATIONAL"
        if hasattr(result, 'triage_result') and result.triage_result:
            # Get the value and convert to uppercase to match frontend expectations
            raw_level = result.triage_result.urgency.value if hasattr(result.triage_result.urgency, 'value') else str(result.triage_result.urgency)
            triage_level = raw_level.upper()

        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent_type="communication", status="success").inc()
            AGENT_LATENCY.labels(agent_type="communication").observe(
                processing_time / 1000
            )

            if requires_escalation:
                ESCALATIONS.labels(reason="communication_escalation").inc()

        # Create escalation record if needed for analytics tracking
        if requires_escalation:
            try:
                escalation_store.create(
                    request_id=request_id,
                    reason=f"Communication escalation: {triage_level}",
                    priority="medium" if triage_level in ["soon", "urgent"] else "low",
                    patient_id=request.patient_id,
                    original_message=request.message,
                    communication_result={
                        "response": result.content,
                        "triage_level": triage_level,
                        "confidence": getattr(result, 'confidence', 0.0)
                    },
                    agent_type="communication",
                    confidence_score=getattr(result, 'confidence', 0.0)
                )
            except Exception as e:
                # Log but don't fail the request if escalation storage fails
                import logging
                logging.getLogger(__name__).warning(f"Failed to store escalation: {e}")

        # Store session in Redis
        if redis_manager.is_connected:
            await redis_manager.set(
                f"session:{session_id}",
                request.message,
                expire_seconds=3600,  # 1 hour
            )

        return CommunicationResponse(
            request_id=request_id,
            response=result.content,
            triage_level=triage_level,
            requires_escalation=requires_escalation,
            disclaimer="This is for informational purposes only. Please consult a healthcare provider for medical advice.",
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


@app.post(
    "/api/v1/analytics/seed-demo",
    tags=["Analytics"],
)
async def seed_demo_analytics():
    """
    Seed the escalation store with demo data for analytics dashboard.
    
    Creates sample escalations across different statuses, priorities,
    and agent types to populate the analytics visualizations.
    """
    import uuid
    from datetime import timedelta
    
    demo_escalations = [
        # Pending escalations
        {
            "request_id": str(uuid.uuid4()),
            "reason": "critical_finding",
            "priority": "high",
            "patient_id": "DEMO-001",
            "agent_type": "diagnostic",
            "confidence_score": 0.92,
            "diagnostic_result": {
                "findings": [{"finding": "Possible pneumothorax", "severity": "high", "confidence": 0.92}],
                "report": "AI-generated report indicating potential pneumothorax requiring urgent review."
            },
        },
        {
            "request_id": str(uuid.uuid4()),
            "reason": "low_confidence",
            "priority": "medium",
            "patient_id": "DEMO-002",
            "agent_type": "diagnostic",
            "confidence_score": 0.65,
            "diagnostic_result": {
                "findings": [{"finding": "Unclear opacity in right lower lobe", "severity": "medium", "confidence": 0.65}],
                "report": "Inconclusive findings requiring specialist review."
            },
        },
        {
            "request_id": str(uuid.uuid4()),
            "reason": "safety_concern",
            "priority": "high",
            "patient_id": "DEMO-003",
            "agent_type": "communication",
            "confidence_score": 0.88,
            "original_message": "Patient reported severe chest pain and difficulty breathing",
            "communication_result": {"triage_level": "EMERGENCY", "response": "Emergency protocol initiated"},
        },
        {
            "request_id": str(uuid.uuid4()),
            "reason": "manual_request",
            "priority": "low",
            "patient_id": "DEMO-004",
            "agent_type": "workflow",
            "confidence_score": 0.95,
            "workflow_result": {"type": "prior_auth", "status": "requires_review"},
        },
        # More pending for variety
        {
            "request_id": str(uuid.uuid4()),
            "reason": "critical_finding",
            "priority": "high",
            "patient_id": "DEMO-005",
            "agent_type": "diagnostic",
            "confidence_score": 0.89,
            "diagnostic_result": {
                "findings": [{"finding": "Suspected cardiomegaly", "severity": "high", "confidence": 0.89}],
            },
        },
        {
            "request_id": str(uuid.uuid4()),
            "reason": "low_confidence",
            "priority": "medium",
            "patient_id": "DEMO-006",
            "agent_type": "diagnostic",
            "confidence_score": 0.58,
        },
    ]
    
    created_count = 0
    reviewed_count = 0
    
    try:
        # Create demo escalations
        for demo in demo_escalations:
            escalation_store.create(**demo)
            created_count += 1
        
        # Get all escalations and review some to create variety in stats
        all_escalations = escalation_store.list_pending(limit=100)
        
        # Review approximately half to show approved/rejected stats
        for i, esc in enumerate(all_escalations[:4]):
            if i < 2:
                # Approve first 2
                escalation_store.submit_review(
                    escalation_id=esc["id"],
                    decision="approve",
                    reviewer_id="DEMO-CLINICIAN-001",
                    notes="Demo approval for analytics testing"
                )
                reviewed_count += 1
            elif i < 3:
                # Reject 1
                escalation_store.submit_review(
                    escalation_id=esc["id"],
                    decision="reject",
                    reviewer_id="DEMO-CLINICIAN-002",
                    notes="Demo rejection - false positive"
                )
                reviewed_count += 1
            # Leave the rest as pending - no need for in_review status
        
        stats = escalation_store.get_stats()
        
        return {
            "message": "Demo analytics data seeded successfully",
            "created_escalations": created_count,
            "reviewed_escalations": reviewed_count,
            "current_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to seed demo data: {str(e)}",
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
# Inference Models
# =============================================================================
class InferenceRequest(BaseModel):
    """Inference generation request."""
    
    prompt: str = Field(..., description="User prompt for inference")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")


class InferenceResponse(BaseModel):
    """Inference generation response."""
    
    request_id: str
    response: str
    confidence: float
    model: str
    model_source: Optional[str] = Field(None, description="Model source: trained or huggingface")
    backend: str = Field(..., description="Backend used: modal or local")
    device: str = Field(..., description="Device used: H100, cuda, cpu, etc.")
    tokens_generated: int = 0
    processing_time_ms: float = 0.0


class ImageAnalysisRequest(BaseModel):
    """Image analysis request."""
    
    prompt: str = Field(..., description="Analysis prompt")
    image_base64: str = Field(..., description="Base64 encoded image")
    max_tokens: int = Field(1024, description="Maximum tokens to generate")


class InferenceStatusResponse(BaseModel):
    """Inference service status response."""
    
    status: str
    backend: str
    model_source: Optional[str]
    model_path: Optional[str]
    is_trained_model: bool
    modal_available: bool
    device: Optional[str]


# =============================================================================
# Inference Endpoints
# =============================================================================
@app.post(
    "/api/v1/inference/generate",
    response_model=InferenceResponse,
    tags=["Inference"],
)
async def generate_inference(request: InferenceRequest, http_request: Request):
    """
    Generate text response using MedGemma model.
    
    Uses Modal H100 GPU by default with trained model priority.
    Falls back to local GPU or HuggingFace model if Modal unavailable.
    """
    import time
    import uuid
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        from medai_compass.models.inference_service import get_inference_service
        
        service = get_inference_service()
        result = await service.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            MODEL_INFERENCE_LATENCY.labels(model_name=result.model).observe(
                processing_time_ms / 1000
            )
        
        return InferenceResponse(
            request_id=request_id,
            response=result.response,
            confidence=result.confidence,
            model=result.model,
            model_source=result.model_source,
            backend=result.backend,
            device=result.device,
            tokens_generated=result.tokens_generated,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post(
    "/api/v1/inference/analyze-image",
    response_model=InferenceResponse,
    tags=["Inference"],
)
async def analyze_image_inference(request: ImageAnalysisRequest):
    """
    Analyze medical image using MedGemma model.
    
    Supports multimodal analysis of medical images (CXR, CT, MRI, etc.).
    """
    import base64
    import time
    import uuid
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        from medai_compass.models.inference_service import get_inference_service
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid base64 image data"
            )
        
        service = get_inference_service()
        result = await service.analyze_image(
            image=image_bytes,
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            MODEL_INFERENCE_LATENCY.labels(model_name=result.model).observe(
                processing_time_ms / 1000
            )
        
        return InferenceResponse(
            request_id=request_id,
            response=result.response,
            confidence=result.confidence,
            model=result.model,
            model_source=result.model_source,
            backend=result.backend,
            device=result.device,
            tokens_generated=result.tokens_generated,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis failed: {str(e)}"
        )


@app.get(
    "/api/v1/inference/status",
    response_model=InferenceStatusResponse,
    tags=["Inference"],
)
async def get_inference_status():
    """
    Get inference service status and model information.
    
    Returns current backend, model source, and configuration.
    """
    try:
        from medai_compass.models.inference_service import get_inference_service
        
        service = get_inference_service()
        model_info = await service.get_model_info()
        backend_info = service.get_backend_info()
        
        # Check Modal availability
        modal_available = False
        try:
            from medai_compass.modal.client import MedGemmaModalClient
            client = MedGemmaModalClient()
            modal_available = client.is_available()
        except Exception:
            pass
        
        return InferenceStatusResponse(
            status="ready" if backend_info.get("initialized") else "initializing",
            backend="modal" if backend_info.get("use_modal") else "local",
            model_source=model_info.get("model_source"),
            model_path=model_info.get("model_path"),
            is_trained_model=model_info.get("is_trained_model", False),
            modal_available=modal_available,
            device=model_info.get("device") or backend_info.get("config", {}).get("device")
        )
        
    except Exception as e:
        return InferenceStatusResponse(
            status="error",
            backend="unknown",
            model_source=None,
            model_path=None,
            is_trained_model=False,
            modal_available=False,
            device=None
        )


# =============================================================================
# Settings Models and Endpoints
# =============================================================================
class SettingsResponse(BaseModel):
    """Current settings response."""

    model_name: str = Field(..., description="Current model name")
    available_models: list[str] = Field(..., description="Available model options")
    inference_backend: str = Field(..., description="Current inference backend")
    available_backends: list[str] = Field(..., description="Available inference backends")
    training_strategy: str = Field(..., description="Current training strategy")
    available_strategies: list[str] = Field(..., description="Available training strategies")
    prefer_modal: bool = Field(..., description="Prefer Modal GPU")
    environment: str = Field(..., description="Current environment")


class SettingsUpdateRequest(BaseModel):
    """Settings update request."""

    model_name: Optional[str] = Field(None, description="Model name to use")
    inference_backend: Optional[str] = Field(None, description="Inference backend")
    training_strategy: Optional[str] = Field(None, description="Training strategy")
    prefer_modal: Optional[bool] = Field(None, description="Prefer Modal GPU")


@app.get(
    "/api/v1/settings",
    response_model=SettingsResponse,
    tags=["Settings"],
)
async def get_settings():
    """
    Get current system settings for the UI dashboard.

    Returns model selection, inference backend, and training strategy settings.
    These can be configured via environment variables for persistence.
    """
    from medai_compass.inference import InferenceStrategySelector
    from medai_compass.training import TrainingStrategySelector

    # Get current model from environment
    current_model = os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-27b")

    # Available models
    available_models = [
        "medgemma-4b",
        "medgemma-27b",
        "google/medgemma-4b-it",
        "google/medgemma-27b-it",
    ]

    # Get inference backends
    inference_selector = InferenceStrategySelector()
    available_backends = inference_selector.list_backends()
    current_backend = os.environ.get("INFERENCE_BACKEND", "vllm")

    # Get training strategies
    training_selector = TrainingStrategySelector()
    available_strategies = training_selector.list_strategies()
    current_strategy = os.environ.get("TRAINING_STRATEGY", "deepspeed_zero3")

    # Modal preference
    prefer_modal = os.environ.get("PREFER_MODAL_GPU", "true").lower() == "true"

    # Environment
    environment = os.environ.get("ENVIRONMENT", "development")

    return SettingsResponse(
        model_name=current_model,
        available_models=available_models,
        inference_backend=current_backend,
        available_backends=available_backends,
        training_strategy=current_strategy,
        available_strategies=available_strategies,
        prefer_modal=prefer_modal,
        environment=environment,
    )


@app.put(
    "/api/v1/settings",
    response_model=SettingsResponse,
    tags=["Settings"],
)
async def update_settings(request: SettingsUpdateRequest):
    """
    Update system settings from the UI dashboard.

    Note: Changes are applied at runtime but require environment variable
    updates for persistence across restarts. The dashboard should update
    the .env file or deployment configuration for permanent changes.
    """
    # Update environment variables (runtime only)
    if request.model_name:
        os.environ["MEDGEMMA_MODEL_NAME"] = request.model_name

    if request.inference_backend:
        os.environ["INFERENCE_BACKEND"] = request.inference_backend

    if request.training_strategy:
        os.environ["TRAINING_STRATEGY"] = request.training_strategy

    if request.prefer_modal is not None:
        os.environ["PREFER_MODAL_GPU"] = "true" if request.prefer_modal else "false"

    # Return updated settings
    return await get_settings()


@app.get(
    "/api/v1/settings/models",
    tags=["Settings"],
)
async def get_available_models():
    """
    Get list of available models with their configurations.

    Returns detailed information about each available model including
    memory requirements, recommended strategies, and capabilities.
    """
    models = [
        {
            "name": "medgemma-4b",
            "hf_model_id": "google/medgemma-4b-it",
            "parameters": "4B",
            "memory_required_gb": 16,
            "recommended_strategy": "single_gpu",
            "capabilities": ["text", "vision"],
            "description": "Lightweight model for fast inference and single-GPU training",
        },
        {
            "name": "medgemma-27b",
            "hf_model_id": "google/medgemma-27b-it",
            "parameters": "27B",
            "memory_required_gb": 80,
            "recommended_strategy": "deepspeed_zero3",
            "capabilities": ["text", "vision", "medical_reasoning"],
            "description": "Full-scale model for production medical AI applications",
        },
    ]

    return {
        "models": models,
        "current_model": os.environ.get("MEDGEMMA_MODEL_NAME", "medgemma-27b"),
    }


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
