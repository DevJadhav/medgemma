"""
OpenTelemetry tracer implementation for MedAI Compass.

Provides distributed tracing with support for:
- Multiple exporters (OTLP, Console, Jaeger)
- Configurable sampling
- Context propagation
- Decorator-based instrumentation
"""

import os
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, TypeVar, Union
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class ExporterType(Enum):
    """Supported trace exporters."""
    CONSOLE = "console"
    OTLP = "otlp"
    JAEGER = "jaeger"
    NONE = "none"


class SamplingStrategy(Enum):
    """Trace sampling strategies."""
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    PARENT_BASED = "parent_based"
    RATIO_BASED = "ratio_based"


@dataclass
class TracerConfig:
    """Configuration for the OpenTelemetry tracer."""
    
    service_name: str = "medai-compass"
    enabled: bool = True
    exporter_type: str = "console"
    otlp_endpoint: Optional[str] = None
    jaeger_host: Optional[str] = None
    jaeger_port: int = 6831
    
    # Sampling
    sampling_rate: float = 1.0
    sampling_strategy: str = "always_on"
    always_sample_critical: bool = True
    
    # Batch processing
    batch_export: bool = True
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000
    
    # Resource attributes
    deployment_environment: str = "development"
    service_version: str = "0.1.0"
    
    @classmethod
    def from_env(cls) -> "TracerConfig":
        """Load configuration from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "medai-compass"),
            enabled=os.getenv("OTEL_ENABLED", "true").lower() == "true",
            exporter_type=os.getenv("OTEL_EXPORTER_TYPE", "console"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            jaeger_host=os.getenv("JAEGER_AGENT_HOST"),
            jaeger_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            sampling_rate=float(os.getenv("OTEL_SAMPLING_RATE", "1.0")),
            sampling_strategy=os.getenv("OTEL_SAMPLING_STRATEGY", "always_on"),
            deployment_environment=os.getenv("DEPLOYMENT_ENV", "development"),
            service_version=os.getenv("SERVICE_VERSION", "0.1.0"),
        )


# Global tracer instance
_TRACER_INSTANCE: Optional["MedAITracer"] = None


class NoOpSpan:
    """No-op span for when tracing is disabled."""
    
    def __init__(self, name: str = "noop"):
        self.name = name
        self._attributes: Dict[str, Any] = {}
        self._events: list = []
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute (no-op)."""
        self._attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event (no-op)."""
        self._events.append({"name": name, "attributes": attributes})
    
    def record_exception(self, exception: Exception) -> None:
        """Record exception (no-op)."""
        pass
    
    def set_status(self, status: Any) -> None:
        """Set status (no-op)."""
        pass
    
    def end(self) -> None:
        """End span (no-op)."""
        pass
    
    def __enter__(self) -> "NoOpSpan":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class MedAITracer:
    """
    OpenTelemetry tracer wrapper for MedAI Compass.
    
    Provides a unified interface for distributed tracing with
    support for multiple backends and graceful degradation.
    """
    
    def __init__(self, config: Optional[TracerConfig] = None):
        """
        Initialize the tracer.
        
        Args:
            config: Tracer configuration
        """
        self.config = config or TracerConfig()
        self._tracer = None
        self._provider = None
        self._initialized = False
        
        if self.config.enabled:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize OpenTelemetry components."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            
            # Create resource with service info
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.deployment_environment,
            })
            
            # Create tracer provider
            self._provider = TracerProvider(resource=resource)
            
            # Add exporter based on config
            self._add_exporter()
            
            # Set as global tracer provider
            trace.set_tracer_provider(self._provider)
            
            # Get tracer
            self._tracer = trace.get_tracer(
                self.config.service_name,
                self.config.service_version,
            )
            
            self._initialized = True
            logger.info(
                f"OpenTelemetry tracer initialized: {self.config.service_name} "
                f"(exporter: {self.config.exporter_type})"
            )
            
        except ImportError as e:
            logger.warning(f"OpenTelemetry not available: {e}. Tracing disabled.")
            self.config.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}. Tracing disabled.")
            self.config.enabled = False
    
    def _add_exporter(self) -> None:
        """Add span exporter based on configuration."""
        try:
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                SimpleSpanProcessor,
            )
            
            exporter = self._create_exporter()
            if exporter is None:
                return
            
            # Use batch or simple processor
            if self.config.batch_export:
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.export_timeout_millis,
                )
            else:
                processor = SimpleSpanProcessor(exporter)
            
            self._provider.add_span_processor(processor)
            
        except Exception as e:
            logger.warning(f"Failed to add exporter: {e}")
    
    def _create_exporter(self):
        """Create span exporter based on type."""
        exporter_type = self.config.exporter_type.lower()
        
        if exporter_type == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()
        
        elif exporter_type == "otlp":
            if not self.config.otlp_endpoint:
                logger.warning("OTLP endpoint not configured, using console exporter")
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                return ConsoleSpanExporter()
            
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                return OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            except ImportError:
                logger.warning("OTLP exporter not available, using console")
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                return ConsoleSpanExporter()
        
        elif exporter_type == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                return JaegerExporter(
                    agent_host_name=self.config.jaeger_host or "localhost",
                    agent_port=self.config.jaeger_port,
                )
            except ImportError:
                logger.warning("Jaeger exporter not available, using console")
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                return ConsoleSpanExporter()
        
        elif exporter_type == "none":
            return None
        
        else:
            logger.warning(f"Unknown exporter type: {exporter_type}, using console")
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()
    
    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[Any] = None,
    ):
        """
        Start a new span.
        
        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Span kind (client, server, etc.)
            
        Yields:
            Span object
        """
        if not self.config.enabled or self._tracer is None:
            yield NoOpSpan(name)
            return
        
        span = None
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import SpanKind
            
            span_kind = kind or SpanKind.INTERNAL
            
            span = self._tracer.start_span(
                name,
                kind=span_kind,
                attributes=attributes,
            )
            
            # Set as current span
            token = otel_trace.set_span_in_context(span)
            
            try:
                yield span
            except Exception as e:
                # Record exception on span
                if span:
                    span.record_exception(e)
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                # End span
                if span:
                    span.end()
                
        except ImportError as e:
            logger.debug(f"OpenTelemetry not available: {e}")
            yield NoOpSpan(name)
        except Exception as e:
            if span is None:
                # Error before span was created
                logger.debug(f"Error creating span: {e}")
                yield NoOpSpan(name)
            else:
                # Re-raise if span was created (exception already recorded)
                raise
    
    def get_current_span(self):
        """Get the current active span."""
        if not self.config.enabled:
            return NoOpSpan()
        
        try:
            from opentelemetry import trace
            return trace.get_current_span()
        except Exception:
            return NoOpSpan()
    
    def shutdown(self) -> None:
        """Shutdown the tracer and flush pending spans."""
        if self._provider is not None:
            try:
                self._provider.shutdown()
                logger.info("Tracer shutdown complete")
            except Exception as e:
                logger.error(f"Error during tracer shutdown: {e}")


def get_tracer() -> MedAITracer:
    """
    Get the global tracer instance.
    
    Returns:
        MedAITracer instance (singleton)
    """
    global _TRACER_INSTANCE
    
    if _TRACER_INSTANCE is None:
        config = TracerConfig.from_env()
        _TRACER_INSTANCE = MedAITracer(config=config)
    
    return _TRACER_INSTANCE


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID if available.
    
    Returns:
        Trace ID as hex string, or None
    """
    tracer = get_tracer()
    span = tracer.get_current_span()
    
    if isinstance(span, NoOpSpan):
        return None
    
    try:
        from opentelemetry import trace
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    
    return None


def trace(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a function.
    
    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to span
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_span(span_name, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_span(span_name, attributes=attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def trace_agent_call(
    agent_type: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to trace agent calls.
    
    Args:
        agent_type: Type of agent (diagnostic, communication, workflow)
        attributes: Additional attributes
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        base_attrs = {
            "agent.type": agent_type,
            "component": "agent",
        }
        if attributes:
            base_attrs.update(attributes)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = f"agent.{agent_type}.{func.__name__}"
            
            with tracer.start_span(span_name, attributes=base_attrs) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("agent.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("agent.success", False)
                    span.set_attribute("agent.error", str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper  # type: ignore
    
    return decorator


def trace_model_inference(
    model_name: str,
    record_tokens: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to trace model inference calls.
    
    Args:
        model_name: Name of the model (medgemma-4b-it, medgemma-27b-it)
        record_tokens: Whether to record token counts
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = f"model.inference.{model_name}"
            
            attributes = {
                "model.name": model_name,
                "model.type": "llm",
                "component": "inference",
            }
            
            # Determine model size
            if "4b" in model_name.lower():
                attributes["model.size"] = "4B"
            elif "27b" in model_name.lower():
                attributes["model.size"] = "27B"
            
            with tracer.start_span(span_name, attributes=attributes) as span:
                import time
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record timing
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("model.inference_time_ms", elapsed_ms)
                    
                    # Record tokens if enabled and available
                    if record_tokens and isinstance(result, dict):
                        if "input_tokens" in result:
                            span.set_attribute("model.input_tokens", result["input_tokens"])
                        if "output_tokens" in result:
                            span.set_attribute("model.output_tokens", result["output_tokens"])
                    
                    span.set_attribute("model.success", True)
                    return result
                    
                except Exception as e:
                    span.set_attribute("model.success", False)
                    span.set_attribute("model.error", str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper  # type: ignore
    
    return decorator


def trace_guardrail(
    guardrail_type: str,
) -> Callable[[F], F]:
    """
    Decorator to trace guardrail checks.
    
    Args:
        guardrail_type: Type of guardrail (phi_detection, escalation, etc.)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = f"guardrail.{guardrail_type}"
            
            attributes = {
                "guardrail.type": guardrail_type,
                "component": "guardrail",
            }
            
            with tracer.start_span(span_name, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    
                    # Record guardrail result if dict
                    if isinstance(result, dict):
                        for key in ["passed", "blocked", "should_escalate", "phi_detected"]:
                            if key in result:
                                span.set_attribute(f"guardrail.{key}", result[key])
                    
                    span.set_attribute("guardrail.success", True)
                    return result
                    
                except Exception as e:
                    span.set_attribute("guardrail.success", False)
                    span.record_exception(e)
                    raise
        
        return wrapper  # type: ignore
    
    return decorator


def inject_context(carrier: Dict[str, str]) -> None:
    """
    Inject trace context into a carrier (e.g., HTTP headers).
    
    Args:
        carrier: Dictionary to inject context into
    """
    tracer = get_tracer()
    if not tracer.config.enabled:
        return
    
    try:
        from opentelemetry import propagate
        propagate.inject(carrier)
    except Exception as e:
        logger.debug(f"Failed to inject context: {e}")


def extract_context(carrier: Dict[str, str]):
    """
    Extract trace context from a carrier.
    
    Args:
        carrier: Dictionary containing trace context
        
    Returns:
        Context object or None
    """
    tracer = get_tracer()
    if not tracer.config.enabled:
        return None
    
    try:
        from opentelemetry import propagate
        return propagate.extract(carrier)
    except Exception as e:
        logger.debug(f"Failed to extract context: {e}")
        return None
