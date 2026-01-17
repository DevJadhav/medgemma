"""Modal GPU Inference Module for MedAI Compass.

This module provides cloud GPU inference using Modal when local GPU is unavailable.
It is designed as an optional add-on - the application works without this folder.

Key Features:
- H100 GPU inference for MedGemma models
- Automatic fallback from local GPU
- Asynchronous inference support
- Batch processing optimization

Usage:
    from modal_inference import MedGemmaModalClient
    
    client = MedGemmaModalClient()
    result = await client.generate("Analyze this chest X-ray...")

Note: This module requires Modal tokens in environment variables:
    - MODAL_TOKEN_ID
    - MODAL_TOKEN_SECRET
"""

from medai_compass.modal.client import MedGemmaModalClient, ModalInferenceError

__all__ = ["MedGemmaModalClient", "ModalInferenceError"]
