"""Inference Celery tasks for async processing.

Provides background tasks for:
- Async diagnostic image analysis
- Long-running inference jobs
- Batch processing
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from medai_compass.workers.celery import app

logger = logging.getLogger(__name__)


# Job status constants
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job store (use Redis in production)
_job_store: Dict[str, Dict[str, Any]] = {}


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status by ID."""
    return _job_store.get(job_id)


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id in _job_store:
        _job_store[job_id].update(kwargs)
        _job_store[job_id]["updated_at"] = datetime.utcnow().isoformat()


def create_job(job_type: str, metadata: Optional[Dict] = None) -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "job_id": job_id,
        "job_type": job_type,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
        "progress": 0,
        "metadata": metadata or {},
    }
    return job_id


@app.task(
    bind=True,
    name="medai_compass.workers.inference_tasks.analyze_diagnostic_async",
    time_limit=300,  # 5 minute hard limit
    soft_time_limit=240,  # 4 minute soft limit
)
def analyze_diagnostic_async(
    self,
    job_id: str,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    image_type: str = "cxr",
    patient_id: Optional[str] = None,
    clinical_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async diagnostic image analysis task.

    Args:
        job_id: Job identifier for tracking
        image_path: Path to image file
        image_base64: Base64 encoded image
        image_type: Type of image (cxr, ct, mri, pathology)
        patient_id: Optional patient identifier
        clinical_context: Optional clinical context

    Returns:
        Analysis results
    """
    import base64
    import tempfile
    import os as os_module

    start_time = time.time()
    temp_file_path = None

    try:
        # Update job to processing
        update_job(job_id, status=JobStatus.PROCESSING, progress=10)
        logger.info(f"Starting async diagnostic analysis for job {job_id}")

        # Handle image input
        if image_base64 and not image_path:
            try:
                image_bytes = base64.b64decode(image_base64)
                ext = ".dcm" if image_type == "cxr" else ".png"
                fd, temp_file_path = tempfile.mkstemp(suffix=ext)
                with os_module.fdopen(fd, 'wb') as f:
                    f.write(image_bytes)
                image_path = temp_file_path
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image: {str(e)}")

        if not image_path:
            raise ValueError("No image provided")

        update_job(job_id, progress=30)

        # Import and run diagnostic agent
        from medai_compass.agents.diagnostic.graph import create_diagnostic_graph
        from medai_compass.agents.diagnostic.state import create_initial_state

        session_id = str(uuid.uuid4())
        state = create_initial_state(
            patient_id=patient_id or "anonymous",
            session_id=session_id,
            images=[image_path]
        )
        state["image_path"] = image_path
        state["image_type"] = image_type

        update_job(job_id, progress=50)

        # Run diagnostic workflow
        graph = create_diagnostic_graph(use_checkpointer=False)
        result = graph.invoke(state)

        update_job(job_id, progress=90)

        processing_time = (time.time() - start_time) * 1000

        # Prepare result
        analysis_result = {
            "job_id": job_id,
            "status": "completed",
            "findings": result.get("findings", []),
            "confidence": result.get("confidence", 0.0),
            "report": result.get("report", ""),
            "requires_review": result.get("requires_review", False),
            "processing_time_ms": processing_time,
        }

        # Update job with result
        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result=analysis_result
        )

        logger.info(f"Completed async diagnostic analysis for job {job_id}")
        return analysis_result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Async diagnostic analysis failed for job {job_id}: {error_msg}")

        update_job(
            job_id,
            status=JobStatus.FAILED,
            error=error_msg
        )

        return {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    finally:
        # Clean up temp file
        if temp_file_path and os_module.path.exists(temp_file_path):
            try:
                os_module.remove(temp_file_path)
            except Exception:
                pass


@app.task(
    bind=True,
    name="medai_compass.workers.inference_tasks.generate_async",
    time_limit=180,  # 3 minute hard limit
    soft_time_limit=150,  # 2.5 minute soft limit
)
def generate_async(
    self,
    job_id: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async text generation task.

    Args:
        job_id: Job identifier
        prompt: User prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt

    Returns:
        Generation result
    """
    import asyncio

    start_time = time.time()

    try:
        update_job(job_id, status=JobStatus.PROCESSING, progress=10)
        logger.info(f"Starting async generation for job {job_id}")

        # Import inference service
        from medai_compass.models.inference_service import get_inference_service

        update_job(job_id, progress=30)

        # Run async generation in sync context
        async def run_generate():
            service = get_inference_service()
            return await service.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )

        result = asyncio.run(run_generate())

        update_job(job_id, progress=90)

        processing_time = (time.time() - start_time) * 1000

        generation_result = {
            "job_id": job_id,
            "status": "completed",
            "response": result.response,
            "confidence": result.confidence,
            "model": result.model,
            "backend": result.backend,
            "device": result.device,
            "tokens_generated": result.tokens_generated,
            "processing_time_ms": processing_time,
        }

        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result=generation_result
        )

        return generation_result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Async generation failed for job {job_id}: {error_msg}")

        update_job(
            job_id,
            status=JobStatus.FAILED,
            error=error_msg
        )

        return {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }


def cleanup_old_jobs(max_age_hours: int = 24):
    """Remove jobs older than max_age_hours."""
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    removed = 0

    for job_id in list(_job_store.keys()):
        job = _job_store[job_id]
        created = datetime.fromisoformat(job["created_at"])
        if created < cutoff:
            del _job_store[job_id]
            removed += 1

    logger.info(f"Cleaned up {removed} old jobs")
    return removed
