"""Celery application configuration for MedAI Compass.

Provides:
- Redis broker configuration
- Task routing
- Scheduled tasks (beat)
"""

import os
from celery import Celery
from celery.schedules import crontab

# Redis configuration from environment
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

# Build Redis URL
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Create Celery app
app = Celery(
    "medai_compass",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "medai_compass.workers.ingestion_tasks",
        "medai_compass.workers.download_tasks",
    ],
)

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "medai_compass.workers.ingestion_tasks.*": {"queue": "ingestion"},
        "medai_compass.workers.download_tasks.*": {"queue": "downloads"},
    },
    
    # Rate limiting
    task_annotations={
        "medai_compass.workers.download_tasks.download_dataset": {
            "rate_limit": "1/m",  # 1 download per minute
        },
    },
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Don't prefetch (for long tasks)
    worker_concurrency=4,
    
    # Task time limits
    task_time_limit=3600 * 24,  # 24 hours max
    task_soft_time_limit=3600 * 23,  # Soft limit at 23 hours
)

# Celery Beat schedule
app.conf.beat_schedule = {
    # Clean up old ingestion jobs
    "cleanup-old-jobs": {
        "task": "medai_compass.workers.ingestion_tasks.cleanup_old_jobs",
        "schedule": crontab(hour=3, minute=0),  # Daily at 3 AM
    },
    # Check dataset integrity
    "verify-datasets": {
        "task": "medai_compass.workers.ingestion_tasks.verify_dataset_integrity",
        "schedule": crontab(hour=4, minute=0, day_of_week=0),  # Weekly on Sunday
    },
}

if __name__ == "__main__":
    app.start()
