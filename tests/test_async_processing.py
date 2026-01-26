"""TDD tests for async processing with Celery.

Tests for:
- Task 4.1: Add Request Queuing with Celery
- Async diagnostic analysis
- Job status polling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestAsyncProcessingTDD:
    """TDD tests for async processing - written first, then implementation."""

    def test_long_running_task_returns_job_id(self):
        """Verify long tasks return job ID for polling."""
        from medai_compass.workers.inference_tasks import (
            create_job,
            get_job,
            JobStatus,
        )

        job_id = create_job("diagnostic", metadata={"image_type": "cxr"})

        assert job_id is not None
        assert len(job_id) == 36  # UUID format

        job = get_job(job_id)
        assert job is not None
        assert job["status"] == JobStatus.PENDING
        assert job["job_type"] == "diagnostic"

    def test_can_poll_job_status(self):
        """Verify job status can be polled."""
        from medai_compass.workers.inference_tasks import (
            create_job,
            get_job,
            update_job,
            JobStatus,
        )

        job_id = create_job("diagnostic")

        # Initially pending
        job = get_job(job_id)
        assert job["status"] == JobStatus.PENDING

        # Update to processing
        update_job(job_id, status=JobStatus.PROCESSING, progress=50)
        job = get_job(job_id)
        assert job["status"] == JobStatus.PROCESSING
        assert job["progress"] == 50

        # Update to completed
        update_job(job_id, status=JobStatus.COMPLETED, progress=100, result={"findings": []})
        job = get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED
        assert job["result"] is not None

    def test_job_status_includes_progress(self):
        """Verify job status includes progress percentage."""
        from medai_compass.workers.inference_tasks import (
            create_job,
            get_job,
            update_job,
        )

        job_id = create_job("diagnostic")
        update_job(job_id, progress=75)

        job = get_job(job_id)
        assert "progress" in job
        assert job["progress"] == 75

    def test_job_failure_includes_error(self):
        """Verify failed jobs include error message."""
        from medai_compass.workers.inference_tasks import (
            create_job,
            update_job,
            get_job,
            JobStatus,
        )

        job_id = create_job("diagnostic")
        update_job(job_id, status=JobStatus.FAILED, error="Image processing failed")

        job = get_job(job_id)
        assert job["status"] == JobStatus.FAILED
        assert job["error"] == "Image processing failed"

    def test_cleanup_removes_old_jobs(self):
        """Verify old jobs are cleaned up."""
        from datetime import datetime, timedelta
        from medai_compass.workers.inference_tasks import (
            _job_store,
            cleanup_old_jobs,
            JobStatus,
        )

        # Create old job
        old_job_id = "old-job-123"
        old_time = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        _job_store[old_job_id] = {
            "job_id": old_job_id,
            "status": JobStatus.COMPLETED,
            "created_at": old_time,
            "updated_at": old_time,
        }

        # Cleanup jobs older than 24 hours
        removed = cleanup_old_jobs(max_age_hours=24)

        assert removed >= 1
        assert old_job_id not in _job_store


class TestAsyncDiagnosticTask:
    """Tests for async diagnostic analysis task."""

    @patch("medai_compass.agents.diagnostic.graph.create_diagnostic_graph")
    @patch("medai_compass.agents.diagnostic.state.create_initial_state")
    def test_analyze_diagnostic_async_success(
        self, mock_create_state, mock_create_graph
    ):
        """Verify async diagnostic analysis completes successfully."""
        from medai_compass.workers.inference_tasks import (
            analyze_diagnostic_async,
            create_job,
            get_job,
            JobStatus,
        )

        # Mock the diagnostic graph
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "findings": [{"finding": "Normal chest X-ray"}],
            "confidence": 0.95,
            "report": "No abnormalities detected",
            "requires_review": False,
        }
        mock_create_graph.return_value = mock_graph
        mock_create_state.return_value = {}

        job_id = create_job("diagnostic")

        # Run task
        result = analyze_diagnostic_async(
            job_id=job_id,
            image_path="/tmp/test.dcm",
            image_type="cxr",
        )

        assert result["status"] == "completed"
        assert len(result["findings"]) > 0
        assert result["confidence"] > 0

        job = get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED

    def test_analyze_diagnostic_async_no_image_fails(self):
        """Verify task fails when no image is provided."""
        from medai_compass.workers.inference_tasks import (
            analyze_diagnostic_async,
            create_job,
            get_job,
            JobStatus,
        )

        job_id = create_job("diagnostic")

        result = analyze_diagnostic_async(
            job_id=job_id,
            image_path=None,
            image_base64=None,
        )

        assert result["status"] == "failed"
        assert "No image provided" in result["error"]

        job = get_job(job_id)
        assert job["status"] == JobStatus.FAILED


class TestAsyncGenerationTask:
    """Tests for async text generation task."""

    @patch("medai_compass.models.inference_service.get_inference_service")
    def test_generate_async_success(self, mock_get_service):
        """Verify async generation completes successfully."""
        from medai_compass.workers.inference_tasks import (
            generate_async,
            create_job,
            get_job,
            JobStatus,
        )

        # Mock inference service
        mock_result = MagicMock()
        mock_result.response = "This is the generated response"
        mock_result.confidence = 0.9
        mock_result.model = "medgemma-27b"
        mock_result.backend = "modal"
        mock_result.device = "H100"
        mock_result.tokens_generated = 50

        mock_service = MagicMock()
        mock_service.generate = AsyncMock(return_value=mock_result)
        mock_get_service.return_value = mock_service

        job_id = create_job("generation")

        result = generate_async(
            job_id=job_id,
            prompt="What are symptoms of diabetes?",
            max_tokens=100,
        )

        assert result["status"] == "completed"
        assert result["response"] == "This is the generated response"

        job = get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED


class TestCeleryConfiguration:
    """Tests for Celery configuration."""

    def test_celery_app_configured(self):
        """Verify Celery app is properly configured."""
        from medai_compass.workers.celery import app

        assert app is not None
        assert "medai_compass.workers.ingestion_tasks" in app.conf.include

    def test_task_routes_configured(self):
        """Verify task routing is configured."""
        from medai_compass.workers.celery import app

        routes = app.conf.task_routes
        assert routes is not None
        assert "medai_compass.workers.ingestion_tasks.*" in routes

    def test_result_backend_configured(self):
        """Verify result backend is configured."""
        from medai_compass.workers.celery import app

        assert app.conf.result_backend is not None
