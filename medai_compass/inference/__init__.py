"""
Inference module for MedGemma models.

Provides optimized inference services for H100 GPUs with:
- Flash Attention 2 integration
- CUDA Graphs for reduced latency
- Production serving: vLLM, Ray Serve, Triton
- Optimized DICOM processing
"""

from .optimized import (
    # Configuration
    H100InferenceConfig,
    # Core components
    OptimizedModelLoader,
    CUDAGraphRunner,
    KVCacheManager,
    DynamicBatcher,
    # Production Serving Engines
    VLLMInferenceEngine,
    RayServeEngine,
    TritonInferenceEngine,
    ProductionServingFactory,
    # DICOM processing
    ParallelDICOMLoader,
    DICOMPreprocessingCache,
    GPUImagePreprocessor,
    # Main service
    OptimizedInferenceService,
    # Benchmarking
    ThroughputBenchmark,
    # Utilities
    check_flash_attention_available,
)

# Strategy Selector for Inference Backends
from .strategy_selector import (
    InferenceStrategySelector,
    InferenceBackend,
    InferenceBackendConfig,
    BackendType,
    select_inference_backend,
    auto_select_inference_backend,
)

__all__ = [
    # Configuration
    "H100InferenceConfig",
    # Core components
    "OptimizedModelLoader",
    "CUDAGraphRunner",
    "KVCacheManager",
    "DynamicBatcher",
    # Production Serving Engines
    "VLLMInferenceEngine",
    "RayServeEngine",
    "TritonInferenceEngine",
    "ProductionServingFactory",
    # DICOM processing
    "ParallelDICOMLoader",
    "DICOMPreprocessingCache",
    "GPUImagePreprocessor",
    # Main service
    "OptimizedInferenceService",
    # Benchmarking
    "ThroughputBenchmark",
    # Utilities
    "check_flash_attention_available",
    # Strategy Selector
    "InferenceStrategySelector",
    "InferenceBackend",
    "InferenceBackendConfig",
    "BackendType",
    "select_inference_backend",
    "auto_select_inference_backend",
]
