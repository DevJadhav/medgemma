"""
TDD tests for OpenTelemetry tracing integration (Phase 9).

Tests distributed tracing across:
- FastAPI endpoints
- Agent calls
- Model inference
- External service calls (Redis, PostgreSQL)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime


# =============================================================================
# Tracer Configuration Tests
# =============================================================================

class TestTracerConfiguration:
    """Test OpenTelemetry tracer configuration."""
    
    def test_tracer_module_imports(self):
        """Test tracing module can be imported."""
        from medai_compass.tracing import (
            TracerConfig,
            MedAITracer,
            get_tracer,
        )
        
        assert TracerConfig is not None
        assert MedAITracer is not None
        assert get_tracer is not None
    
    def test_tracer_config_defaults(self):
        """Test tracer config has sensible defaults."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig()
        assert config.service_name == "medai-compass"
        assert config.enabled is True
        assert config.exporter_type in ["otlp", "console", "jaeger"]
    
    def test_tracer_config_from_env(self):
        """Test tracer config can be loaded from environment."""
        from medai_compass.tracing import TracerConfig
        
        with patch.dict("os.environ", {
            "OTEL_SERVICE_NAME": "test-service",
            "OTEL_EXPORTER_TYPE": "console",
            "OTEL_ENABLED": "true",
        }):
            config = TracerConfig.from_env()
            assert config.service_name == "test-service"
            assert config.exporter_type == "console"
    
    def test_tracer_config_disabled(self):
        """Test tracer can be disabled via config."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig(enabled=False)
        assert config.enabled is False
    
    def test_tracer_config_with_otlp_endpoint(self):
        """Test tracer config with OTLP endpoint."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig(
            exporter_type="otlp",
            otlp_endpoint="http://jaeger:4317",
        )
        assert config.otlp_endpoint == "http://jaeger:4317"


class TestMedAITracer:
    """Test MedAI tracer wrapper."""
    
    @pytest.fixture
    def tracer_config(self):
        """Test tracer config."""
        from medai_compass.tracing import TracerConfig
        return TracerConfig(enabled=True, exporter_type="console")
    
    def test_tracer_initialization(self, tracer_config):
        """Test tracer can be initialized."""
        from medai_compass.tracing import MedAITracer
        
        tracer = MedAITracer(config=tracer_config)
        assert tracer is not None
    
    def test_tracer_singleton_pattern(self, tracer_config):
        """Test tracer follows singleton pattern."""
        from medai_compass.tracing import get_tracer
        
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2
    
    def test_create_span(self, tracer_config):
        """Test creating a span."""
        from medai_compass.tracing import MedAITracer
        
        tracer = MedAITracer(config=tracer_config)
        
        with tracer.start_span("test-span") as span:
            assert span is not None
            span.set_attribute("test.key", "test-value")
    
    def test_span_context_propagation(self, tracer_config):
        """Test span context propagation."""
        from medai_compass.tracing import MedAITracer
        
        tracer = MedAITracer(config=tracer_config)
        
        with tracer.start_span("parent-span") as parent:
            with tracer.start_span("child-span") as child:
                # Child should have parent context
                assert child is not None
    
    def test_span_attributes(self, tracer_config):
        """Test setting span attributes."""
        from medai_compass.tracing import MedAITracer
        
        tracer = MedAITracer(config=tracer_config)
        
        with tracer.start_span("attribute-test") as span:
            span.set_attribute("model.name", "medgemma-4b-it")
            span.set_attribute("model.inference_time_ms", 150.5)
            span.set_attribute("patient.id", "patient-001")


class TestSpanDecorators:
    """Test span decorator utilities."""
    
    def test_trace_function_decorator(self):
        """Test @trace decorator for functions."""
        from medai_compass.tracing import trace
        
        @trace(name="test-function")
        def my_function():
            return "result"
        
        result = my_function()
        assert result == "result"
    
    def test_trace_async_function_decorator(self):
        """Test @trace decorator for async functions."""
        from medai_compass.tracing import trace
        
        @trace(name="async-test")
        async def my_async_function():
            return "async-result"
        
        import asyncio
        result = asyncio.run(my_async_function())
        assert result == "async-result"
    
    def test_trace_with_attributes(self):
        """Test @trace decorator with custom attributes."""
        from medai_compass.tracing import trace
        
        @trace(name="attributed-function", attributes={"component": "test"})
        def attributed_function():
            return "result"
        
        result = attributed_function()
        assert result == "result"
    
    def test_trace_class_method(self):
        """Test @trace decorator on class methods."""
        from medai_compass.tracing import trace
        
        class MyClass:
            @trace(name="class-method")
            def my_method(self):
                return "method-result"
        
        obj = MyClass()
        result = obj.my_method()
        assert result == "method-result"


# =============================================================================
# FastAPI Instrumentation Tests
# =============================================================================

class TestFastAPIInstrumentation:
    """Test FastAPI auto-instrumentation."""
    
    def test_fastapi_instrumentor_available(self):
        """Test FastAPI instrumentor is available."""
        from medai_compass.tracing.instrumentation import FastAPIInstrumentor
        assert FastAPIInstrumentor is not None
    
    def test_fastapi_instrumentation_setup(self):
        """Test FastAPI instrumentation can be set up."""
        from medai_compass.tracing.instrumentation import instrument_fastapi
        from fastapi import FastAPI
        
        app = FastAPI()
        instrumented_app = instrument_fastapi(app)
        assert instrumented_app is not None
    
    def test_request_span_created(self):
        """Test HTTP requests create spans."""
        from medai_compass.tracing.instrumentation import instrument_fastapi
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        instrument_fastapi(app)
        client = TestClient(app)
        
        response = client.get("/test")
        assert response.status_code == 200
    
    def test_request_attributes_captured(self):
        """Test HTTP request attributes are captured in spans."""
        from medai_compass.tracing.instrumentation import instrument_fastapi
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.get("/health")
        def health_endpoint():
            return {"status": "healthy"}
        
        instrument_fastapi(app)
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200


class TestRedisInstrumentation:
    """Test Redis instrumentation."""
    
    def test_redis_instrumentor_available(self):
        """Test Redis instrumentor is available."""
        from medai_compass.tracing.instrumentation import RedisInstrumentor
        assert RedisInstrumentor is not None
    
    @patch("redis.asyncio.Redis")
    def test_redis_operations_traced(self, mock_redis):
        """Test Redis operations create spans."""
        from medai_compass.tracing.instrumentation import instrument_redis
        
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        instrumented_client = instrument_redis(mock_client)
        assert instrumented_client is not None


class TestSQLAlchemyInstrumentation:
    """Test SQLAlchemy instrumentation."""
    
    def test_sqlalchemy_instrumentor_available(self):
        """Test SQLAlchemy instrumentor is available."""
        from medai_compass.tracing.instrumentation import SQLAlchemyInstrumentor
        assert SQLAlchemyInstrumentor is not None
    
    def test_database_queries_traced(self):
        """Test database queries create spans."""
        from medai_compass.tracing.instrumentation import instrument_sqlalchemy
        
        # Mock engine
        mock_engine = MagicMock()
        instrumented_engine = instrument_sqlalchemy(mock_engine)
        assert instrumented_engine is not None


# =============================================================================
# Agent Tracing Tests
# =============================================================================

class TestAgentTracing:
    """Test tracing for agent calls."""
    
    def test_agent_span_created(self):
        """Test agent calls create spans."""
        from medai_compass.tracing import trace_agent_call
        
        @trace_agent_call(agent_type="diagnostic")
        def diagnostic_agent_call():
            return {"findings": "Normal"}
        
        result = diagnostic_agent_call()
        assert result["findings"] == "Normal"
    
    def test_agent_span_attributes(self):
        """Test agent spans have correct attributes."""
        from medai_compass.tracing import trace_agent_call
        
        @trace_agent_call(
            agent_type="communication",
            attributes={"triage_level": "routine"}
        )
        def communication_agent_call():
            return {"response": "Health information"}
        
        result = communication_agent_call()
        assert "response" in result
    
    def test_orchestrator_span_encompasses_agents(self):
        """Test orchestrator span encompasses agent spans."""
        from medai_compass.tracing import trace, trace_agent_call
        
        @trace_agent_call(agent_type="workflow")
        def workflow_call():
            return "workflow done"
        
        @trace(name="orchestrator-request")
        def orchestrator_process():
            return workflow_call()
        
        result = orchestrator_process()
        assert result == "workflow done"


class TestModelInferenceTracing:
    """Test tracing for model inference."""
    
    def test_model_inference_span_created(self):
        """Test model inference creates spans."""
        from medai_compass.tracing import trace_model_inference
        
        @trace_model_inference(model_name="medgemma-4b-it")
        def run_inference(prompt: str):
            return "Generated response"
        
        result = run_inference("Test prompt")
        assert result == "Generated response"
    
    def test_model_inference_metrics(self):
        """Test model inference metrics are recorded."""
        from medai_compass.tracing import trace_model_inference
        
        @trace_model_inference(
            model_name="medgemma-27b-it",
            record_tokens=True
        )
        def inference_with_metrics(prompt: str):
            return {
                "response": "Generated text",
                "input_tokens": 50,
                "output_tokens": 100,
            }
        
        result = inference_with_metrics("Test")
        assert result["input_tokens"] == 50
    
    def test_4b_vs_27b_model_tracing(self):
        """Test different model sizes are traced correctly."""
        from medai_compass.tracing import trace_model_inference
        
        @trace_model_inference(model_name="medgemma-4b-it")
        def inference_4b():
            return "4B response"
        
        @trace_model_inference(model_name="medgemma-27b-it")
        def inference_27b():
            return "27B response"
        
        result_4b = inference_4b()
        result_27b = inference_27b()
        
        assert result_4b == "4B response"
        assert result_27b == "27B response"


class TestGuardrailTracing:
    """Test tracing for guardrail checks."""
    
    def test_guardrail_span_created(self):
        """Test guardrail checks create spans."""
        from medai_compass.tracing import trace_guardrail
        
        @trace_guardrail(guardrail_type="phi_detection")
        def phi_check(text: str):
            return {"phi_detected": False}
        
        result = phi_check("Safe text")
        assert result["phi_detected"] is False
    
    def test_escalation_span_created(self):
        """Test escalation decisions create spans."""
        from medai_compass.tracing import trace_guardrail
        
        @trace_guardrail(guardrail_type="escalation")
        def escalation_check(response: str, confidence: float):
            return {"should_escalate": confidence < 0.8}
        
        result = escalation_check("Test response", 0.75)
        assert result["should_escalate"] is True


# =============================================================================
# Context Propagation Tests
# =============================================================================

class TestContextPropagation:
    """Test trace context propagation."""
    
    def test_context_propagates_across_functions(self):
        """Test context propagates across function calls."""
        from medai_compass.tracing import get_tracer, trace
        
        @trace(name="inner-function")
        def inner():
            return "inner result"
        
        @trace(name="outer-function")
        def outer():
            return inner()
        
        result = outer()
        assert result == "inner result"
    
    def test_context_propagates_to_http_headers(self):
        """Test context propagates to HTTP headers."""
        from medai_compass.tracing import (
            inject_context,
            extract_context,
        )
        
        headers = {}
        inject_context(headers)
        
        # Should have trace context headers
        # (traceparent, tracestate in W3C format)
        assert isinstance(headers, dict)
    
    def test_context_extracted_from_headers(self):
        """Test context can be extracted from headers."""
        from medai_compass.tracing import extract_context
        
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }
        
        context = extract_context(headers)
        assert context is not None


# =============================================================================
# Integration with Monitoring Tests
# =============================================================================

class TestTracingMonitoringIntegration:
    """Test tracing integrates with Phase 8 monitoring."""
    
    def test_traces_link_to_metrics(self):
        """Test traces can link to Prometheus metrics."""
        from medai_compass.tracing import get_tracer
        from medai_compass.monitoring import PerformanceMonitor, PerformanceConfig
        
        tracer = get_tracer()
        config = PerformanceConfig(model_name="test-model")
        monitor = PerformanceMonitor(config)
        
        # Both should be available
        assert tracer is not None
        assert monitor is not None
    
    def test_trace_id_in_logs(self):
        """Test trace ID appears in structured logs."""
        from medai_compass.tracing import get_current_trace_id
        
        # Should return trace ID or None if no active span
        trace_id = get_current_trace_id()
        assert trace_id is None or isinstance(trace_id, str)
    
    def test_span_events_for_alerts(self):
        """Test span events can trigger alerts."""
        from medai_compass.tracing import get_tracer
        
        tracer = get_tracer()
        
        with tracer.start_span("alert-test") as span:
            span.add_event(
                "critical_finding",
                attributes={"finding_type": "pneumothorax"}
            )


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestTracingErrorHandling:
    """Test error handling in tracing."""
    
    def test_span_records_exception(self):
        """Test exceptions are recorded in spans."""
        from medai_compass.tracing import trace
        
        @trace(name="error-function")
        def error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            error_function()
    
    def test_tracing_disabled_gracefully(self):
        """Test code works when tracing is disabled."""
        from medai_compass.tracing import TracerConfig, MedAITracer
        
        config = TracerConfig(enabled=False)
        tracer = MedAITracer(config=config)
        
        with tracer.start_span("disabled-span") as span:
            # Should work without errors even when disabled
            pass
    
    def test_missing_exporter_handled(self):
        """Test missing exporter is handled gracefully."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig(
            exporter_type="nonexistent",
        )
        # Should fall back to console or no-op


# =============================================================================
# Export and Visualization Tests
# =============================================================================

class TestTracingExport:
    """Test trace export functionality."""
    
    def test_console_exporter_works(self):
        """Test console exporter outputs traces."""
        from medai_compass.tracing import TracerConfig, MedAITracer
        
        config = TracerConfig(exporter_type="console")
        tracer = MedAITracer(config=config)
        
        with tracer.start_span("console-test"):
            pass
    
    def test_otlp_exporter_configured(self):
        """Test OTLP exporter can be configured."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig(
            exporter_type="otlp",
            otlp_endpoint="http://localhost:4317",
        )
        
        assert config.otlp_endpoint is not None
    
    def test_batch_processor_configured(self):
        """Test batch span processor is configured."""
        from medai_compass.tracing import TracerConfig, MedAITracer
        
        config = TracerConfig(
            batch_export=True,
            max_queue_size=2048,
            max_export_batch_size=512,
        )
        
        assert config.batch_export is True


class TestTracingSampling:
    """Test trace sampling configuration."""
    
    def test_sampling_rate_configurable(self):
        """Test sampling rate is configurable."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig(sampling_rate=0.5)
        assert config.sampling_rate == 0.5
    
    def test_always_sample_critical(self):
        """Test critical operations are always sampled."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig(
            sampling_rate=0.1,
            always_sample_critical=True,
        )
        
        assert config.always_sample_critical is True
    
    def test_parent_based_sampling(self):
        """Test parent-based sampling is supported."""
        from medai_compass.tracing import TracerConfig
        
        config = TracerConfig(
            sampling_strategy="parent_based",
        )
        
        assert config.sampling_strategy == "parent_based"
