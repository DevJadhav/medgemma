"""
Auto-instrumentation for external libraries.

Provides instrumentation for:
- FastAPI (HTTP endpoints)
- Redis (caching)
- SQLAlchemy (database)
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FastAPIInstrumentor:
    """
    FastAPI instrumentation for OpenTelemetry.
    
    Automatically traces incoming HTTP requests with:
    - Request method, path, status code
    - Request/response headers (configurable)
    - Latency
    """
    
    _instance: Optional["FastAPIInstrumentor"] = None
    _instrumented = False
    
    def __new__(cls) -> "FastAPIInstrumentor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def instrument(self, app: Any) -> Any:
        """
        Instrument a FastAPI application.
        
        Args:
            app: FastAPI application instance
            
        Returns:
            Instrumented app
        """
        if self._instrumented:
            return app
        
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor as OTELFastAPI
            
            OTELFastAPI.instrument_app(
                app,
                excluded_urls="/health,/health/live,/health/ready,/metrics",
            )
            
            self._instrumented = True
            logger.info("FastAPI instrumentation enabled")
            
        except ImportError:
            logger.warning(
                "opentelemetry-instrumentation-fastapi not installed. "
                "Install with: pip install opentelemetry-instrumentation-fastapi"
            )
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
        
        return app
    
    def uninstrument(self, app: Any) -> None:
        """Remove instrumentation from FastAPI app."""
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor as OTELFastAPI
            OTELFastAPI.uninstrument_app(app)
            self._instrumented = False
        except Exception as e:
            logger.debug(f"Failed to uninstrument FastAPI: {e}")


class RedisInstrumentor:
    """
    Redis instrumentation for OpenTelemetry.
    
    Traces Redis commands with:
    - Command name
    - Key (configurable)
    - Latency
    """
    
    _instance: Optional["RedisInstrumentor"] = None
    _instrumented = False
    
    def __new__(cls) -> "RedisInstrumentor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def instrument(self, client: Optional[Any] = None) -> Any:
        """
        Instrument Redis client or global instrumentation.
        
        Args:
            client: Optional Redis client instance
            
        Returns:
            Instrumented client
        """
        if self._instrumented:
            return client
        
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor as OTELRedis
            
            OTELRedis().instrument()
            
            self._instrumented = True
            logger.info("Redis instrumentation enabled")
            
        except ImportError:
            logger.warning(
                "opentelemetry-instrumentation-redis not installed. "
                "Install with: pip install opentelemetry-instrumentation-redis"
            )
        except Exception as e:
            logger.error(f"Failed to instrument Redis: {e}")
        
        return client
    
    def uninstrument(self) -> None:
        """Remove Redis instrumentation."""
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor as OTELRedis
            OTELRedis().uninstrument()
            self._instrumented = False
        except Exception as e:
            logger.debug(f"Failed to uninstrument Redis: {e}")


class SQLAlchemyInstrumentor:
    """
    SQLAlchemy instrumentation for OpenTelemetry.
    
    Traces database queries with:
    - Statement type (SELECT, INSERT, etc.)
    - Table name
    - Latency
    """
    
    _instance: Optional["SQLAlchemyInstrumentor"] = None
    _instrumented = False
    
    def __new__(cls) -> "SQLAlchemyInstrumentor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def instrument(self, engine: Optional[Any] = None) -> Any:
        """
        Instrument SQLAlchemy engine.
        
        Args:
            engine: Optional SQLAlchemy engine
            
        Returns:
            Instrumented engine
        """
        if self._instrumented:
            return engine
        
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor as OTELSQL
            
            if engine is not None:
                OTELSQL().instrument(engine=engine)
            else:
                OTELSQL().instrument()
            
            self._instrumented = True
            logger.info("SQLAlchemy instrumentation enabled")
            
        except ImportError:
            logger.warning(
                "opentelemetry-instrumentation-sqlalchemy not installed. "
                "Install with: pip install opentelemetry-instrumentation-sqlalchemy"
            )
        except Exception as e:
            logger.error(f"Failed to instrument SQLAlchemy: {e}")
        
        return engine
    
    def uninstrument(self) -> None:
        """Remove SQLAlchemy instrumentation."""
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor as OTELSQL
            OTELSQL().uninstrument()
            self._instrumented = False
        except Exception as e:
            logger.debug(f"Failed to uninstrument SQLAlchemy: {e}")


class HTTPXInstrumentor:
    """
    HTTPX instrumentation for OpenTelemetry.
    
    Traces outgoing HTTP requests with:
    - URL, method, status
    - Request/response headers
    - Latency
    """
    
    _instance: Optional["HTTPXInstrumentor"] = None
    _instrumented = False
    
    def __new__(cls) -> "HTTPXInstrumentor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def instrument(self) -> None:
        """Instrument HTTPX globally."""
        if self._instrumented:
            return
        
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            
            HTTPXClientInstrumentor().instrument()
            
            self._instrumented = True
            logger.info("HTTPX instrumentation enabled")
            
        except ImportError:
            logger.warning(
                "opentelemetry-instrumentation-httpx not installed. "
                "Install with: pip install opentelemetry-instrumentation-httpx"
            )
        except Exception as e:
            logger.error(f"Failed to instrument HTTPX: {e}")
    
    def uninstrument(self) -> None:
        """Remove HTTPX instrumentation."""
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().uninstrument()
            self._instrumented = False
        except Exception as e:
            logger.debug(f"Failed to uninstrument HTTPX: {e}")


# Convenience functions

def instrument_fastapi(app: Any) -> Any:
    """
    Instrument a FastAPI application.
    
    Args:
        app: FastAPI application
        
    Returns:
        Instrumented app
    """
    return FastAPIInstrumentor().instrument(app)


def instrument_redis(client: Any) -> Any:
    """
    Instrument Redis client.
    
    Args:
        client: Redis client
        
    Returns:
        Instrumented client
    """
    return RedisInstrumentor().instrument(client)


def instrument_sqlalchemy(engine: Any) -> Any:
    """
    Instrument SQLAlchemy engine.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        Instrumented engine
    """
    return SQLAlchemyInstrumentor().instrument(engine)


def instrument_all(
    app: Optional[Any] = None,
    redis_client: Optional[Any] = None,
    db_engine: Optional[Any] = None,
) -> dict:
    """
    Instrument all available libraries.
    
    Args:
        app: Optional FastAPI app
        redis_client: Optional Redis client
        db_engine: Optional SQLAlchemy engine
        
    Returns:
        Dictionary of instrumented components
    """
    result = {}
    
    if app is not None:
        result["fastapi"] = instrument_fastapi(app)
    
    if redis_client is not None:
        result["redis"] = instrument_redis(redis_client)
    else:
        # Try global instrumentation
        RedisInstrumentor().instrument()
        result["redis"] = "global"
    
    if db_engine is not None:
        result["sqlalchemy"] = instrument_sqlalchemy(db_engine)
    else:
        # Try global instrumentation
        SQLAlchemyInstrumentor().instrument()
        result["sqlalchemy"] = "global"
    
    # Always try HTTPX
    HTTPXInstrumentor().instrument()
    result["httpx"] = "global"
    
    logger.info(f"Instrumentation complete: {list(result.keys())}")
    return result


def setup_tracing(
    service_name: str = "medai-compass",
    otlp_endpoint: Optional[str] = None,
    app: Optional[Any] = None,
) -> "MedAITracer":
    """
    Complete tracing setup with instrumentation.
    
    Args:
        service_name: Name of the service
        otlp_endpoint: OTLP collector endpoint
        app: Optional FastAPI app to instrument
        
    Returns:
        Configured tracer
    """
    from medai_compass.tracing.tracer import TracerConfig, MedAITracer
    
    # Determine exporter type
    exporter_type = "otlp" if otlp_endpoint else "console"
    
    config = TracerConfig(
        service_name=service_name,
        enabled=True,
        exporter_type=exporter_type,
        otlp_endpoint=otlp_endpoint,
    )
    
    tracer = MedAITracer(config=config)
    
    # Instrument libraries
    instrument_all(app=app)
    
    return tracer
