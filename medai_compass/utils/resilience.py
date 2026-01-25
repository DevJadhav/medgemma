"""
Resilience patterns for production reliability.

Provides:
- Request timeouts with async support
- Exponential backoff retry logic
- Circuit breaker pattern

All patterns follow TDD development approach.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# TIMEOUT CONFIGURATION
# =============================================================================
TIMEOUT_CONFIG = {
    "diagnostic": 120,      # 2 minutes for image analysis
    "workflow": 60,         # 1 minute for workflow operations
    "communication": 30,    # 30 seconds for chat responses
    "inference": 90,        # 90 seconds for model inference
    "modal_cold_start": 45, # 45 seconds for Modal container startup
    "database": 10,         # 10 seconds for database operations
    "redis": 5,             # 5 seconds for cache operations
    "default": 30,          # Default timeout
}


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
class TimeoutExceededError(Exception):
    """Raised when an operation exceeds its timeout."""

    def __init__(self, operation_name: str, timeout_seconds: float):
        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Operation '{operation_name}' timed out after {timeout_seconds} seconds"
        )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejects requests."""

    def __init__(self, circuit_name: str, reset_time: float):
        self.circuit_name = circuit_name
        self.reset_time = reset_time
        super().__init__(
            f"Circuit '{circuit_name}' is open. Will reset in {reset_time:.1f} seconds"
        )


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, operation_name: str, attempts: int, last_error: Exception):
        self.operation_name = operation_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Operation '{operation_name}' failed after {attempts} attempts. "
            f"Last error: {last_error}"
        )


# =============================================================================
# TIMEOUT IMPLEMENTATION
# =============================================================================
async def with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    operation_name: str = "operation"
) -> T:
    """
    Execute an async operation with a timeout.

    Args:
        coro: The coroutine to execute
        timeout_seconds: Maximum time to wait in seconds
        operation_name: Name for error messages

    Returns:
        The result of the coroutine

    Raises:
        TimeoutExceededError: If the operation exceeds the timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            f"Timeout exceeded for {operation_name}: {timeout_seconds}s",
            extra={
                "operation": operation_name,
                "timeout_seconds": timeout_seconds,
            }
        )
        raise TimeoutExceededError(operation_name, timeout_seconds)


def timeout(
    seconds: Optional[float] = None,
    operation_type: Optional[str] = None
):
    """
    Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds (overrides operation_type)
        operation_type: Key from TIMEOUT_CONFIG

    Example:
        @timeout(seconds=30)
        async def my_operation():
            ...

        @timeout(operation_type="diagnostic")
        async def analyze_image():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            timeout_value = seconds
            if timeout_value is None:
                timeout_value = TIMEOUT_CONFIG.get(
                    operation_type or "default",
                    TIMEOUT_CONFIG["default"]
                )

            return await with_timeout(
                func(*args, **kwargs),
                timeout_seconds=timeout_value,
                operation_name=func.__name__
            )
        return wrapper
    return decorator


# =============================================================================
# RETRY IMPLEMENTATION
# =============================================================================
async def with_retry(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: str = "operation"
) -> T:
    """
    Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute (callable that returns awaitable)
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        jitter: Whether to add random jitter to delays
        retry_exceptions: Tuple of exception types to retry on
        operation_name: Name for logging

    Returns:
        The result of the function

    Raises:
        The last exception if all retries fail
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = await func()
            if attempt > 1:
                logger.info(
                    f"Operation {operation_name} succeeded on attempt {attempt}",
                    extra={"operation": operation_name, "attempt": attempt}
                )
            return result

        except retry_exceptions as e:
            last_exception = e

            if attempt == max_attempts:
                logger.error(
                    f"Operation {operation_name} failed after {max_attempts} attempts",
                    extra={
                        "operation": operation_name,
                        "attempts": max_attempts,
                        "error": str(e),
                    }
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            # Add jitter if enabled (0.5 to 1.5 multiplier)
            if jitter:
                delay *= 0.5 + random.random()

            logger.warning(
                f"Operation {operation_name} failed (attempt {attempt}/{max_attempts}), "
                f"retrying in {delay:.2f}s: {e}",
                extra={
                    "operation": operation_name,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "delay": delay,
                    "error": str(e),
                }
            )

            await asyncio.sleep(delay)

        except Exception as e:
            # Non-retryable exception
            logger.error(
                f"Operation {operation_name} failed with non-retryable error: {e}",
                extra={"operation": operation_name, "error": str(e)}
            )
            raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no result and no exception")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator to add retry logic to async functions.

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        async def call_external_service():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await with_retry(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter=jitter,
                retry_exceptions=retry_exceptions,
                operation_name=func.__name__
            )
        return wrapper
    return decorator


# =============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# =============================================================================
class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit tripped, requests fail fast
    - HALF_OPEN: Testing if service recovered

    Args:
        failure_threshold: Number of failures before opening
        reset_timeout: Seconds before attempting reset (half-open)
        name: Circuit breaker name for logging
    """
    failure_threshold: int = 5
    reset_timeout: float = 60.0
    name: str = "default"

    # Internal state
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: Optional[float] = field(default=None, init=False, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)

    @property
    def state(self) -> str:
        """Get current state, checking for half-open transition."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                logger.info(f"Circuit {self.name} transitioning to half-open")
        return self._state.value

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.reset_timeout

    def _record_success(self):
        """Record a successful call."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        logger.debug(f"Circuit {self.name} success recorded, failures reset")

    def _record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failure in half-open means back to open
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} reopened after half-open failure")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit {self.name} opened after {self._failure_count} failures"
            )
        else:
            logger.debug(
                f"Circuit {self.name} failure recorded: {self._failure_count}/{self.failure_threshold}"
            )

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        fallback: Optional[Callable[..., Awaitable[T]]] = None,
        *args,
        **kwargs
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            fallback: Optional fallback function if circuit is open
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of function or fallback

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        # Check current state (triggers half-open transition if needed)
        current_state = self.state

        if current_state == "open":
            if fallback:
                logger.info(f"Circuit {self.name} open, using fallback")
                return await fallback(*args, **kwargs)

            time_until_reset = (
                self._last_failure_time + self.reset_timeout - time.time()
                if self._last_failure_time else 0
            )
            raise CircuitOpenError(self.name, max(0, time_until_reset))

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            self._record_failure()
            raise

    def reset(self):
        """Manually reset the circuit breaker."""
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        logger.info(f"Circuit {self.name} manually reset")


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                name=name
            )
        return self._breakers[name]

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_status(self) -> dict[str, dict]:
        """Get status of all circuit breakers."""
        return {
            name: {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "failure_threshold": breaker.failure_threshold,
            }
            for name, breaker in self._breakers.items()
        }


# Global registry
circuit_breakers = CircuitBreakerRegistry()


# =============================================================================
# PRE-CONFIGURED CIRCUIT BREAKERS
# =============================================================================
def get_modal_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Modal GPU service."""
    return circuit_breakers.get(
        "modal_gpu",
        failure_threshold=3,
        reset_timeout=30.0  # Try again after 30 seconds
    )


def get_postgres_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for PostgreSQL."""
    return circuit_breakers.get(
        "postgres",
        failure_threshold=5,
        reset_timeout=15.0
    )


def get_redis_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Redis."""
    return circuit_breakers.get(
        "redis",
        failure_threshold=5,
        reset_timeout=10.0
    )


# =============================================================================
# COMBINED RESILIENT CALL
# =============================================================================
async def resilient_call(
    func: Callable[..., Awaitable[T]],
    timeout_seconds: Optional[float] = None,
    max_retries: int = 3,
    circuit_breaker: Optional[CircuitBreaker] = None,
    fallback: Optional[Callable[..., Awaitable[T]]] = None,
    operation_name: str = "operation"
) -> T:
    """
    Execute function with combined resilience patterns.

    Applies in order:
    1. Circuit breaker (fail fast if service is down)
    2. Timeout (don't wait forever)
    3. Retry (handle transient failures)

    Args:
        func: Async function to execute
        timeout_seconds: Timeout for each attempt
        max_retries: Number of retry attempts
        circuit_breaker: Optional circuit breaker to use
        fallback: Optional fallback if everything fails
        operation_name: Name for logging

    Returns:
        Result of the function
    """
    async def attempt():
        # Apply timeout
        if timeout_seconds:
            return await with_timeout(
                func(),
                timeout_seconds=timeout_seconds,
                operation_name=operation_name
            )
        return await func()

    async def attempt_with_circuit():
        if circuit_breaker:
            return await circuit_breaker.call(attempt, fallback=fallback)
        return await attempt()

    # Apply retry
    try:
        return await with_retry(
            attempt_with_circuit,
            max_attempts=max_retries,
            base_delay=1.0,
            operation_name=operation_name,
            retry_exceptions=(TimeoutExceededError, ConnectionError, OSError)
        )
    except Exception as e:
        if fallback and not circuit_breaker:
            logger.info(f"Using fallback for {operation_name} after error: {e}")
            return await fallback()
        raise
