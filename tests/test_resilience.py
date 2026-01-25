"""
Tests for resilience patterns: timeouts, retries, and circuit breakers.

TDD approach: Tests written first, then implementation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time


# =============================================================================
# TIMEOUT TESTS
# =============================================================================
class TestRequestTimeouts:
    """Tests for request timeout functionality."""

    @pytest.mark.asyncio
    async def test_async_timeout_succeeds_within_limit(self):
        """Verify async operation completes within timeout."""
        from medai_compass.utils.resilience import with_timeout

        async def fast_operation():
            await asyncio.sleep(0.1)
            return "success"

        result = await with_timeout(fast_operation(), timeout_seconds=1.0)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_timeout_raises_on_exceeded(self):
        """Verify TimeoutError is raised when timeout exceeded."""
        from medai_compass.utils.resilience import with_timeout, TimeoutExceededError

        async def slow_operation():
            await asyncio.sleep(5.0)
            return "never reached"

        with pytest.raises(TimeoutExceededError) as exc_info:
            await with_timeout(slow_operation(), timeout_seconds=0.1)

        assert "timed out after 0.1" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_timeout_preserves_result_type(self):
        """Verify timeout wrapper preserves return type."""
        from medai_compass.utils.resilience import with_timeout

        async def return_dict():
            return {"key": "value", "count": 42}

        result = await with_timeout(return_dict(), timeout_seconds=1.0)
        assert result == {"key": "value", "count": 42}

    @pytest.mark.asyncio
    async def test_timeout_with_custom_error_message(self):
        """Verify custom error message is included."""
        from medai_compass.utils.resilience import with_timeout, TimeoutExceededError

        async def slow_op():
            await asyncio.sleep(5.0)

        with pytest.raises(TimeoutExceededError) as exc_info:
            await with_timeout(
                slow_op(),
                timeout_seconds=0.1,
                operation_name="diagnostic_analysis"
            )

        assert "diagnostic_analysis" in str(exc_info.value)


# =============================================================================
# RETRY TESTS
# =============================================================================
class TestRetryLogic:
    """Tests for exponential backoff retry functionality."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_attempt(self):
        """Verify no retry if first attempt succeeds."""
        from medai_compass.utils.resilience import with_retry

        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await with_retry(successful_operation, max_attempts=3)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_third_attempt(self):
        """Verify retry logic succeeds after transient failures."""
        from medai_compass.utils.resilience import with_retry

        attempt = 0

        async def flaky_operation():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = await with_retry(flaky_operation, max_attempts=3, base_delay=0.01)
        assert result == "success"
        assert attempt == 3

    @pytest.mark.asyncio
    async def test_retry_respects_max_attempts(self):
        """Verify retry gives up after max attempts."""
        from medai_compass.utils.resilience import with_retry

        attempt = 0

        async def always_fails():
            nonlocal attempt
            attempt += 1
            raise ValueError("Persistent error")

        with pytest.raises(ValueError) as exc_info:
            await with_retry(always_fails, max_attempts=3, base_delay=0.01)

        assert "Persistent error" in str(exc_info.value)
        assert attempt == 3

    @pytest.mark.asyncio
    async def test_retry_with_specific_exceptions(self):
        """Verify retry only on specified exceptions."""
        from medai_compass.utils.resilience import with_retry

        attempt = 0

        async def fails_with_value_error():
            nonlocal attempt
            attempt += 1
            raise ValueError("Not retryable")

        # Should not retry ValueError, only ConnectionError
        with pytest.raises(ValueError):
            await with_retry(
                fails_with_value_error,
                max_attempts=3,
                base_delay=0.01,
                retry_exceptions=(ConnectionError,)
            )

        assert attempt == 1  # No retry

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Verify exponential backoff increases delay."""
        from medai_compass.utils.resilience import with_retry

        timestamps = []

        async def record_time_and_fail():
            timestamps.append(time.time())
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            await with_retry(
                record_time_and_fail,
                max_attempts=4,
                base_delay=0.1,
                jitter=False  # Disable jitter for predictable timing
            )

        # Check delays: should be ~0.1, ~0.2, ~0.4 (exponential)
        delays = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        assert len(delays) == 3
        # Allow some tolerance for timing
        assert 0.08 < delays[0] < 0.15  # ~0.1
        assert 0.15 < delays[1] < 0.30  # ~0.2
        assert 0.30 < delays[2] < 0.60  # ~0.4

    @pytest.mark.asyncio
    async def test_retry_with_jitter(self):
        """Verify jitter adds randomness to delays."""
        from medai_compass.utils.resilience import with_retry

        timestamps_run1 = []
        timestamps_run2 = []

        async def record_time(storage):
            storage.append(time.time())
            raise ConnectionError("Fail")

        # Run 1
        with pytest.raises(ConnectionError):
            await with_retry(
                lambda: record_time(timestamps_run1),
                max_attempts=3,
                base_delay=0.1,
                jitter=True
            )

        # Run 2
        with pytest.raises(ConnectionError):
            await with_retry(
                lambda: record_time(timestamps_run2),
                max_attempts=3,
                base_delay=0.1,
                jitter=True
            )

        # Delays should be different due to jitter
        delays1 = [timestamps_run1[i+1] - timestamps_run1[i] for i in range(len(timestamps_run1)-1)]
        delays2 = [timestamps_run2[i+1] - timestamps_run2[i] for i in range(len(timestamps_run2)-1)]

        # With jitter, delays should not be exactly equal
        # (there's a tiny chance they could be, but very unlikely)
        assert delays1 != delays2 or len(delays1) == 0

    @pytest.mark.asyncio
    async def test_retry_respects_max_delay(self):
        """Verify delay is capped at max_delay."""
        from medai_compass.utils.resilience import with_retry

        timestamps = []

        async def record_and_fail():
            timestamps.append(time.time())
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            await with_retry(
                record_and_fail,
                max_attempts=5,
                base_delay=0.1,
                max_delay=0.2,  # Cap at 0.2
                jitter=False
            )

        delays = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        # All delays should be <= max_delay (with tolerance)
        for delay in delays:
            assert delay < 0.3  # 0.2 + tolerance


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================
class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    @pytest.mark.asyncio
    async def test_circuit_starts_closed(self):
        """Verify circuit breaker starts in closed state."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_allows_requests_when_closed(self):
        """Verify requests pass through when circuit is closed."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        async def successful_op():
            return "success"

        result = await cb.call(successful_op)
        assert result == "success"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Verify circuit opens after threshold failures."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        async def failing_op():
            raise ConnectionError("Service down")

        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(failing_op)

        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self):
        """Verify requests are rejected when circuit is open."""
        from medai_compass.utils.resilience import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        async def failing_op():
            raise ConnectionError("Service down")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(failing_op)

        # Now circuit should reject
        async def would_succeed():
            return "success"

        with pytest.raises(CircuitOpenError):
            await cb.call(would_succeed)

    @pytest.mark.asyncio
    async def test_circuit_half_opens_after_timeout(self):
        """Verify circuit enters half-open after reset timeout."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.1)

        async def failing_op():
            raise ConnectionError("Service down")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(failing_op)

        assert cb.state == "open"

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Should be half-open now
        assert cb.state == "half-open"

    @pytest.mark.asyncio
    async def test_circuit_closes_on_success_in_half_open(self):
        """Verify circuit closes when request succeeds in half-open."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.1)

        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Service down")
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(flaky_op)

        assert cb.state == "open"

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Now in half-open, next request should succeed and close circuit
        result = await cb.call(flaky_op)
        assert result == "success"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_failure_in_half_open(self):
        """Verify circuit reopens when request fails in half-open."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.1)

        async def always_fails():
            raise ConnectionError("Service down")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(always_fails)

        # Wait for reset timeout
        await asyncio.sleep(0.15)
        assert cb.state == "half-open"

        # Fail in half-open
        with pytest.raises(ConnectionError):
            await cb.call(always_fails)

        # Should be back to open
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_circuit_tracks_failure_count(self):
        """Verify failure count is tracked correctly."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5, reset_timeout=60)

        async def failing_op():
            raise ConnectionError("Fail")

        for i in range(4):
            with pytest.raises(ConnectionError):
                await cb.call(failing_op)
            assert cb.failure_count == i + 1
            assert cb.state == "closed"  # Not yet at threshold

        # One more failure should open
        with pytest.raises(ConnectionError):
            await cb.call(failing_op)
        assert cb.failure_count == 5
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_circuit_resets_failure_count_on_success(self):
        """Verify success resets failure count."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5, reset_timeout=60)

        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Fail")
            return "success"

        # Two failures
        with pytest.raises(ConnectionError):
            await cb.call(flaky_op)
        with pytest.raises(ConnectionError):
            await cb.call(flaky_op)
        assert cb.failure_count == 2

        # Success should reset
        result = await cb.call(flaky_op)
        assert result == "success"
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_with_fallback(self):
        """Verify fallback is called when circuit is open."""
        from medai_compass.utils.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        async def failing_op():
            raise ConnectionError("Service down")

        async def fallback_op():
            return "fallback result"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(failing_op)

        # Use fallback
        result = await cb.call(failing_op, fallback=fallback_op)
        assert result == "fallback result"


# =============================================================================
# COMBINED RESILIENCE TESTS
# =============================================================================
class TestCombinedResilience:
    """Tests for combining timeout, retry, and circuit breaker."""

    @pytest.mark.asyncio
    async def test_retry_with_timeout(self):
        """Verify retry works with timeout on each attempt."""
        from medai_compass.utils.resilience import with_retry, with_timeout

        attempt = 0

        async def slow_then_fast():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                await asyncio.sleep(0.5)  # Will timeout
            return "success"

        async def operation_with_timeout():
            return await with_timeout(slow_then_fast(), timeout_seconds=0.1)

        result = await with_retry(
            operation_with_timeout,
            max_attempts=3,
            base_delay=0.01
        )
        assert result == "success"
        assert attempt == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self):
        """Verify circuit breaker integrates with retry."""
        from medai_compass.utils.resilience import CircuitBreaker, with_retry

        cb = CircuitBreaker(failure_threshold=5, reset_timeout=60)

        attempt = 0

        async def flaky_external_service():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError("Transient failure")
            return "success"

        async def resilient_call():
            return await cb.call(flaky_external_service)

        result = await with_retry(resilient_call, max_attempts=5, base_delay=0.01)
        assert result == "success"
        assert cb.state == "closed"  # Success resets failures


# =============================================================================
# API INTEGRATION TESTS
# =============================================================================
class TestAPITimeouts:
    """Tests for API endpoint timeouts."""

    @pytest.mark.asyncio
    async def test_diagnostic_endpoint_timeout_config(self):
        """Verify diagnostic endpoint has timeout configured."""
        from medai_compass.utils.resilience import TIMEOUT_CONFIG

        assert "diagnostic" in TIMEOUT_CONFIG
        assert TIMEOUT_CONFIG["diagnostic"] >= 60  # At least 60 seconds for image processing

    @pytest.mark.asyncio
    async def test_workflow_endpoint_timeout_config(self):
        """Verify workflow endpoint has timeout configured."""
        from medai_compass.utils.resilience import TIMEOUT_CONFIG

        assert "workflow" in TIMEOUT_CONFIG
        assert TIMEOUT_CONFIG["workflow"] >= 30

    @pytest.mark.asyncio
    async def test_communication_endpoint_timeout_config(self):
        """Verify communication endpoint has timeout configured."""
        from medai_compass.utils.resilience import TIMEOUT_CONFIG

        assert "communication" in TIMEOUT_CONFIG
        assert TIMEOUT_CONFIG["communication"] >= 30
