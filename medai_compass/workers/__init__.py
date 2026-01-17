"""Workers package for MedAI Compass."""

from medai_compass.workers.celery import app as celery_app

__all__ = ["celery_app"]
