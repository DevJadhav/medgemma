"""
Tracing module for MedAI Compass (Phase 9).

Provides OpenTelemetry-based distributed tracing for:
- FastAPI endpoints
- Agent calls
- Model inference
- External service calls
"""

from medai_compass.tracing.tracer import (
    TracerConfig,
    MedAITracer,
    get_tracer,
    get_current_trace_id,
    trace,
    trace_agent_call,
    trace_model_inference,
    trace_guardrail,
    inject_context,
    extract_context,
)
from medai_compass.tracing.instrumentation import (
    FastAPIInstrumentor,
    RedisInstrumentor,
    SQLAlchemyInstrumentor,
    instrument_fastapi,
    instrument_redis,
    instrument_sqlalchemy,
    instrument_all,
)

__all__ = [
    # Tracer
    "TracerConfig",
    "MedAITracer",
    "get_tracer",
    "get_current_trace_id",
    # Decorators
    "trace",
    "trace_agent_call",
    "trace_model_inference",
    "trace_guardrail",
    # Context propagation
    "inject_context",
    "extract_context",
    # Instrumentation
    "FastAPIInstrumentor",
    "RedisInstrumentor",
    "SQLAlchemyInstrumentor",
    "instrument_fastapi",
    "instrument_redis",
    "instrument_sqlalchemy",
    "instrument_all",
]
