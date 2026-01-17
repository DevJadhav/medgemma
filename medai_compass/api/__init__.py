"""
MedAI Compass API Module.

Production-grade FastAPI application for the MedAI Compass multi-agent system.
"""

from medai_compass.api.main import app, create_app

__all__ = ["app", "create_app"]
