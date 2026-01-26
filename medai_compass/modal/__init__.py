"""Modal GPU Inference & Training Module for MedAI Compass.

This module provides cloud GPU inference and training using Modal with H100 GPUs.
It is designed as an optional add-on - the application works without this folder.

Key Features:
- H100 GPU inference for MedGemma models (app.py)
- Distributed training on 8x H100 GPUs (training_app.py)
- Medical dataset downloading (data_download.py)
- Model evaluation with quality gates (evaluation_app.py)
- Automatic fallback from local GPU
- Asynchronous inference support
- Batch processing optimization

Usage:
    # Inference
    from medai_compass.modal import MedGemmaModalClient
    client = MedGemmaModalClient()
    result = await client.generate("Analyze this chest X-ray...")

    # Training (via CLI)
    modal run medai_compass/modal/data_download.py  # Download datasets
    modal run medai_compass/modal/training_app.py   # Train model
    modal run medai_compass/modal/evaluation_app.py # Evaluate model

Note: This module requires Modal tokens in environment variables:
    - MODAL_TOKEN_ID
    - MODAL_TOKEN_SECRET
    - HF_TOKEN (for HuggingFace model access)
"""

from medai_compass.modal.client import MedGemmaModalClient, ModalInferenceError

__all__ = ["MedGemmaModalClient", "ModalInferenceError"]
