"""MedAI Compass Utilities.

Provides common utilities for:
- DICOM processing
- FHIR client
- GPU detection
- Data pipelines
- Logging
"""

from medai_compass.utils.gpu import (
    GPUBackend,
    GPUInfo,
    detect_local_gpu,
    get_optimal_device,
    is_modal_available,
    should_use_modal,
    get_inference_config,
)

__all__ = [
    "GPUBackend",
    "GPUInfo",
    "detect_local_gpu",
    "get_optimal_device",
    "is_modal_available",
    "should_use_modal",
    "get_inference_config",
]
