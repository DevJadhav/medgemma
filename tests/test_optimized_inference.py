"""
Tests for Optimized Inference Service for H100 GPUs.

TDD approach: Tests written first for all optimizations:
- Flash Attention 2 integration
- CUDA Graphs for reduced kernel launch overhead
- Optimized KV cache management
- Batch processing with dynamic batching
- vLLM integration for high-throughput inference
- DICOM-optimized preprocessing pipeline
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


def check_cuda_available() -> bool:
    """Check if CUDA is available for tests."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_cuda = pytest.mark.skipif(
    not check_cuda_available(),
    reason="CUDA not available"
)


# =============================================================================
# Optimized Inference Config Tests
# =============================================================================

class TestOptimizedInferenceConfig:
    """Tests for H100-optimized inference configuration."""

    def test_h100_config_creation(self):
        """Verify H100-specific config can be created."""
        from medai_compass.inference.optimized import H100InferenceConfig

        config = H100InferenceConfig()
        assert config is not None
        assert config.use_flash_attention_2 is True
        assert config.use_cuda_graphs is True

    def test_config_enables_tensor_cores(self):
        """Verify config enables tensor core optimizations."""
        from medai_compass.inference.optimized import H100InferenceConfig

        config = H100InferenceConfig()
        assert config.use_tensor_cores is True
        assert config.compute_dtype == "bfloat16"

    def test_config_kv_cache_settings(self):
        """Verify KV cache optimization settings."""
        from medai_compass.inference.optimized import H100InferenceConfig

        config = H100InferenceConfig()
        assert hasattr(config, "kv_cache_dtype")
        assert hasattr(config, "max_kv_cache_length")
        assert config.kv_cache_dtype == "fp8"  # H100 FP8 support

    def test_config_batch_settings(self):
        """Verify batch processing settings."""
        from medai_compass.inference.optimized import H100InferenceConfig

        config = H100InferenceConfig()
        assert hasattr(config, "max_batch_size")
        assert hasattr(config, "dynamic_batching")
        assert config.max_batch_size >= 8

    def test_config_for_model_size(self):
        """Verify config adjusts for model size."""
        from medai_compass.inference.optimized import H100InferenceConfig

        config_4b = H100InferenceConfig.for_model("medgemma-4b")
        config_27b = H100InferenceConfig.for_model("medgemma-27b")

        assert config_4b.max_batch_size > config_27b.max_batch_size
        assert config_27b.use_tensor_parallelism is True


# =============================================================================
# Flash Attention 2 Tests
# =============================================================================

class TestFlashAttention2:
    """Tests for Flash Attention 2 integration."""

    def test_flash_attention_available_check(self):
        """Verify Flash Attention 2 availability check."""
        from medai_compass.inference.optimized import check_flash_attention_available

        result = check_flash_attention_available()
        assert isinstance(result, bool)

    @requires_cuda
    def test_model_loads_with_flash_attention(self):
        """Verify model can be configured with Flash Attention 2.

        Note: This test requires CUDA and Flash Attention to be installed.
        On systems without GPU, Flash Attention is unavailable and SDPA is used.
        """
        from medai_compass.inference.optimized import OptimizedModelLoader, check_flash_attention_available

        loader = OptimizedModelLoader(use_flash_attention_2=True)
        assert loader.use_flash_attention_2 is True
        assert hasattr(loader, "get_attn_implementation")
        # Only expect flash_attention_2 if it's actually available
        if check_flash_attention_available():
            assert loader.get_attn_implementation() == "flash_attention_2"
        else:
            assert loader.get_attn_implementation() == "sdpa"

    def test_fallback_to_sdpa_if_unavailable(self):
        """Verify fallback to SDPA if Flash Attention unavailable."""
        from medai_compass.inference.optimized import OptimizedModelLoader

        with patch("medai_compass.inference.optimized.check_flash_attention_available", return_value=False):
            loader = OptimizedModelLoader(use_flash_attention_2=True)
            assert loader.get_attn_implementation() == "sdpa"


# =============================================================================
# CUDA Graphs Tests
# =============================================================================

class TestCUDAGraphs:
    """Tests for CUDA Graph optimizations."""

    def test_cuda_graph_capture_available(self):
        """Verify CUDA graph capture functionality."""
        from medai_compass.inference.optimized import CUDAGraphRunner

        runner = CUDAGraphRunner()
        assert hasattr(runner, "capture_graph")
        assert hasattr(runner, "replay_graph")

    def test_cuda_graph_for_fixed_batch_size(self):
        """Verify CUDA graphs work with fixed batch sizes."""
        from medai_compass.inference.optimized import CUDAGraphRunner

        runner = CUDAGraphRunner(batch_sizes=[1, 4, 8, 16])
        assert 1 in runner.supported_batch_sizes
        assert 8 in runner.supported_batch_sizes

    def test_cuda_graph_warmup(self):
        """Verify CUDA graph warmup process."""
        from medai_compass.inference.optimized import CUDAGraphRunner

        runner = CUDAGraphRunner()
        assert hasattr(runner, "warmup")
        assert runner.is_warmed_up is False


# =============================================================================
# KV Cache Optimization Tests
# =============================================================================

class TestKVCacheOptimization:
    """Tests for KV cache optimizations."""

    def test_kv_cache_manager_creation(self):
        """Verify KV cache manager can be created."""
        from medai_compass.inference.optimized import KVCacheManager

        manager = KVCacheManager(max_length=8192, dtype="fp8")
        assert manager is not None
        assert manager.max_length == 8192

    def test_kv_cache_fp8_quantization(self):
        """Verify FP8 quantization for KV cache on H100."""
        from medai_compass.inference.optimized import KVCacheManager

        manager = KVCacheManager(dtype="fp8")
        assert manager.dtype == "fp8"
        assert hasattr(manager, "quantize_cache")

    def test_kv_cache_paged_attention(self):
        """Verify paged attention support for efficient memory."""
        from medai_compass.inference.optimized import KVCacheManager

        manager = KVCacheManager(use_paged_attention=True)
        assert manager.use_paged_attention is True
        assert hasattr(manager, "page_size")

    def test_kv_cache_reuse(self):
        """Verify KV cache can be reused across requests."""
        from medai_compass.inference.optimized import KVCacheManager

        manager = KVCacheManager()
        assert hasattr(manager, "get_cache")
        assert hasattr(manager, "release_cache")


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestBatchProcessing:
    """Tests for optimized batch processing."""

    def test_dynamic_batcher_creation(self):
        """Verify dynamic batcher can be created."""
        from medai_compass.inference.optimized import DynamicBatcher

        batcher = DynamicBatcher(max_batch_size=16, max_wait_ms=50)
        assert batcher is not None
        assert batcher.max_batch_size == 16

    def test_batch_request_grouping(self):
        """Verify requests are grouped efficiently."""
        from medai_compass.inference.optimized import DynamicBatcher

        batcher = DynamicBatcher(max_batch_size=4)
        assert hasattr(batcher, "add_request")
        assert hasattr(batcher, "get_batch")

    def test_batch_padding_optimization(self):
        """Verify batch padding is optimized."""
        from medai_compass.inference.optimized import DynamicBatcher

        batcher = DynamicBatcher()
        assert hasattr(batcher, "pad_to_multiple")
        # Padding to multiples of 8 for tensor core efficiency
        assert batcher.pad_to_multiple == 8

    def test_continuous_batching(self):
        """Verify continuous batching support."""
        from medai_compass.inference.optimized import DynamicBatcher

        batcher = DynamicBatcher(continuous_batching=True)
        assert batcher.continuous_batching is True


# =============================================================================
# vLLM Integration Tests
# =============================================================================

class TestVLLMIntegration:
    """Tests for vLLM high-throughput inference."""

    def test_vllm_engine_creation(self):
        """Verify vLLM engine can be created."""
        from medai_compass.inference.optimized import VLLMInferenceEngine

        engine = VLLMInferenceEngine(model_name="medgemma-4b")
        assert engine is not None
        assert hasattr(engine, "generate")

    def test_vllm_tensor_parallelism(self):
        """Verify vLLM tensor parallelism for large models."""
        from medai_compass.inference.optimized import VLLMInferenceEngine

        engine = VLLMInferenceEngine(
            model_name="medgemma-27b",
            tensor_parallel_size=4
        )
        assert engine.tensor_parallel_size == 4

    def test_vllm_speculative_decoding(self):
        """Verify speculative decoding support."""
        from medai_compass.inference.optimized import VLLMInferenceEngine

        engine = VLLMInferenceEngine(use_speculative_decoding=True)
        assert engine.use_speculative_decoding is True

    def test_vllm_prefix_caching(self):
        """Verify prefix caching for repeated prompts."""
        from medai_compass.inference.optimized import VLLMInferenceEngine

        engine = VLLMInferenceEngine(enable_prefix_caching=True)
        assert engine.enable_prefix_caching is True


# =============================================================================
# Ray Serve Integration Tests
# =============================================================================

class TestRayServeIntegration:
    """Tests for Ray Serve production serving."""

    def test_ray_serve_engine_creation(self):
        """Verify Ray Serve engine can be created."""
        from medai_compass.inference.optimized import RayServeEngine

        engine = RayServeEngine(model_name="medgemma-27b")
        assert engine is not None
        assert hasattr(engine, "generate")
        assert hasattr(engine, "deploy")

    def test_ray_serve_num_replicas(self):
        """Verify Ray Serve replica configuration."""
        from medai_compass.inference.optimized import RayServeEngine

        engine = RayServeEngine(num_replicas=4)
        assert engine.num_replicas == 4

    def test_ray_serve_max_concurrent_queries(self):
        """Verify max concurrent queries configuration."""
        from medai_compass.inference.optimized import RayServeEngine

        engine = RayServeEngine(max_concurrent_queries=200)
        assert engine.max_concurrent_queries == 200

    def test_ray_serve_autoscaling_config(self):
        """Verify autoscaling configuration."""
        from medai_compass.inference.optimized import RayServeEngine

        autoscaling = {
            "min_replicas": 2,
            "max_replicas": 8,
            "target_num_ongoing_requests_per_replica": 10,
        }
        engine = RayServeEngine(autoscaling_config=autoscaling)
        assert engine.autoscaling_config["min_replicas"] == 2
        assert engine.autoscaling_config["max_replicas"] == 8

    def test_ray_serve_actor_options(self):
        """Verify Ray actor options."""
        from medai_compass.inference.optimized import RayServeEngine

        actor_options = {"num_gpus": 2, "num_cpus": 8}
        engine = RayServeEngine(ray_actor_options=actor_options)
        assert engine.ray_actor_options["num_gpus"] == 2
        assert engine.ray_actor_options["num_cpus"] == 8


# =============================================================================
# Triton Inference Server Tests
# =============================================================================

class TestTritonIntegration:
    """Tests for Triton Inference Server integration."""

    def test_triton_engine_creation(self):
        """Verify Triton engine can be created."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        engine = TritonInferenceEngine(model_name="medgemma-27b")
        assert engine is not None
        assert hasattr(engine, "generate")
        assert hasattr(engine, "is_server_ready")
        assert hasattr(engine, "is_model_ready")

    def test_triton_model_repository(self):
        """Verify model repository configuration."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        engine = TritonInferenceEngine(model_repository="/opt/triton/models")
        assert engine.model_repository == "/opt/triton/models"

    def test_triton_grpc_url(self):
        """Verify gRPC URL configuration."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        engine = TritonInferenceEngine(grpc_url="triton-server:8001")
        assert engine.grpc_url == "triton-server:8001"

    def test_triton_http_url(self):
        """Verify HTTP URL configuration."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        engine = TritonInferenceEngine(http_url="triton-server:8000")
        assert engine.http_url == "triton-server:8000"

    def test_triton_max_batch_size(self):
        """Verify max batch size configuration."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        engine = TritonInferenceEngine(max_batch_size=64)
        assert engine.max_batch_size == 64

    def test_triton_preferred_batch_sizes(self):
        """Verify preferred batch sizes configuration."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        batch_sizes = [1, 2, 4, 8]
        engine = TritonInferenceEngine(preferred_batch_sizes=batch_sizes)
        assert engine.preferred_batch_sizes == [1, 2, 4, 8]

    def test_triton_queue_delay(self):
        """Verify queue delay configuration."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        engine = TritonInferenceEngine(max_queue_delay_microseconds=50000)
        assert engine.max_queue_delay_microseconds == 50000

    def test_triton_model_config_generation(self):
        """Verify model config can be generated."""
        from medai_compass.inference.optimized import TritonInferenceEngine

        engine = TritonInferenceEngine(model_name="medgemma-27b", max_batch_size=16)
        config = engine._create_model_config()
        assert config["name"] == "medgemma-27b"
        assert config["max_batch_size"] == 16
        assert "dynamic_batching" in config


# =============================================================================
# Production Serving Factory Tests
# =============================================================================

class TestProductionServingFactory:
    """Tests for production serving factory."""

    def test_factory_create_vllm(self):
        """Verify factory creates vLLM engine."""
        from medai_compass.inference.optimized import ProductionServingFactory, VLLMInferenceEngine

        engine = ProductionServingFactory.create("vllm")
        assert isinstance(engine, VLLMInferenceEngine)

    def test_factory_create_ray_serve(self):
        """Verify factory creates Ray Serve engine."""
        from medai_compass.inference.optimized import ProductionServingFactory, RayServeEngine

        engine = ProductionServingFactory.create("ray_serve")
        assert isinstance(engine, RayServeEngine)

    def test_factory_create_triton(self):
        """Verify factory creates Triton engine."""
        from medai_compass.inference.optimized import ProductionServingFactory, TritonInferenceEngine

        engine = ProductionServingFactory.create("triton")
        assert isinstance(engine, TritonInferenceEngine)

    def test_factory_with_config(self):
        """Verify factory uses config settings."""
        from medai_compass.inference.optimized import ProductionServingFactory, H100InferenceConfig

        config = H100InferenceConfig(
            model_name="medgemma-27b",
            tensor_parallel_size=4,
        )
        engine = ProductionServingFactory.create("vllm", config=config)
        assert engine.tensor_parallel_size == 4

    def test_factory_with_kwargs(self):
        """Verify factory accepts additional kwargs."""
        from medai_compass.inference.optimized import ProductionServingFactory

        engine = ProductionServingFactory.create(
            "ray_serve",
            num_replicas=8,
            max_concurrent_queries=500,
        )
        assert engine.num_replicas == 8
        assert engine.max_concurrent_queries == 500

    def test_factory_invalid_backend(self):
        """Verify factory raises error for invalid backend."""
        from medai_compass.inference.optimized import ProductionServingFactory
        import pytest

        with pytest.raises(ValueError):
            ProductionServingFactory.create("invalid_backend")

    def test_factory_get_recommended_backend(self):
        """Verify recommended backend selection."""
        from medai_compass.inference.optimized import ProductionServingFactory, H100InferenceConfig

        # Tensor parallelism -> vLLM
        config = H100InferenceConfig(use_tensor_parallelism=True, tensor_parallel_size=4)
        assert ProductionServingFactory.get_recommended_backend(config) == "vllm"

        # Autoscaling -> Ray Serve
        config = H100InferenceConfig(
            ray_serve_autoscaling_min=1,
            ray_serve_autoscaling_max=10,
        )
        assert ProductionServingFactory.get_recommended_backend(config) == "ray_serve"

    def test_factory_backends_list(self):
        """Verify available backends."""
        from medai_compass.inference.optimized import ProductionServingFactory

        assert "vllm" in ProductionServingFactory.BACKENDS
        assert "ray_serve" in ProductionServingFactory.BACKENDS
        assert "triton" in ProductionServingFactory.BACKENDS


# =============================================================================
# DICOM Optimized Pipeline Tests
# =============================================================================

class TestDICOMOptimizedPipeline:
    """Tests for optimized DICOM processing pipeline."""

    def test_parallel_dicom_loader(self):
        """Verify parallel DICOM loading."""
        from medai_compass.inference.optimized import ParallelDICOMLoader

        loader = ParallelDICOMLoader(num_workers=4)
        assert loader is not None
        assert loader.num_workers == 4

    def test_dicom_preprocessing_cache(self):
        """Verify DICOM preprocessing cache."""
        from medai_compass.inference.optimized import DICOMPreprocessingCache

        cache = DICOMPreprocessingCache(max_size_gb=10)
        assert cache is not None
        assert hasattr(cache, "get")
        assert hasattr(cache, "put")

    @requires_cuda
    def test_gpu_accelerated_preprocessing(self):
        """Verify GPU-accelerated image preprocessing.

        Note: This test requires CUDA. On CPU-only systems, use_gpu will be False.
        Run on Modal with GPU for full verification.
        """
        from medai_compass.inference.optimized import GPUImagePreprocessor

        preprocessor = GPUImagePreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, "preprocess_batch")
        # use_gpu depends on CUDA availability
        if check_cuda_available():
            assert preprocessor.use_gpu is True
        else:
            assert preprocessor.use_gpu is False

    def test_dicom_batch_processing(self):
        """Verify batch DICOM processing."""
        from medai_compass.inference.optimized import ParallelDICOMLoader

        loader = ParallelDICOMLoader()
        assert hasattr(loader, "load_batch")
        assert hasattr(loader, "preprocess_batch")

    def test_dicom_memory_mapping(self):
        """Verify memory-mapped DICOM loading for large files."""
        from medai_compass.inference.optimized import ParallelDICOMLoader

        loader = ParallelDICOMLoader(use_mmap=True)
        assert loader.use_mmap is True


# =============================================================================
# Optimized Inference Service Tests
# =============================================================================

class TestOptimizedInferenceService:
    """Tests for the complete optimized inference service."""

    def test_service_creation(self):
        """Verify optimized service can be created."""
        from medai_compass.inference.optimized import OptimizedInferenceService

        service = OptimizedInferenceService()
        assert service is not None

    def test_service_auto_optimization(self):
        """Verify service auto-detects and applies optimizations."""
        from medai_compass.inference.optimized import OptimizedInferenceService

        service = OptimizedInferenceService(auto_optimize=True)
        assert service.auto_optimize is True
        assert hasattr(service, "detected_optimizations")

    def test_service_h100_detection(self):
        """Verify service detects H100 GPU."""
        from medai_compass.inference.optimized import OptimizedInferenceService

        service = OptimizedInferenceService()
        assert hasattr(service, "detect_gpu_type")
        # Should return GPU info dict

    def test_service_generates_with_optimizations(self):
        """Verify generation uses optimizations."""
        from medai_compass.inference.optimized import OptimizedInferenceService

        service = OptimizedInferenceService()
        assert hasattr(service, "generate")
        assert hasattr(service, "generate_batch")

    def test_service_analyze_dicom(self):
        """Verify DICOM analysis with optimizations."""
        from medai_compass.inference.optimized import OptimizedInferenceService

        service = OptimizedInferenceService()
        assert hasattr(service, "analyze_dicom")
        assert hasattr(service, "analyze_dicom_batch")

    def test_service_metrics_reporting(self):
        """Verify service reports performance metrics."""
        from medai_compass.inference.optimized import OptimizedInferenceService

        service = OptimizedInferenceService()
        assert hasattr(service, "get_metrics")
        # Should include throughput, latency, memory usage


# =============================================================================
# Inference Throughput Tests
# =============================================================================

class TestInferenceThroughput:
    """Tests for inference throughput benchmarking."""

    def test_throughput_benchmark_runner(self):
        """Verify throughput benchmark can be run."""
        from medai_compass.inference.optimized import ThroughputBenchmark

        benchmark = ThroughputBenchmark()
        assert benchmark is not None
        assert hasattr(benchmark, "run")

    def test_throughput_measures_tokens_per_second(self):
        """Verify tokens per second measurement."""
        from medai_compass.inference.optimized import ThroughputBenchmark

        benchmark = ThroughputBenchmark()
        assert hasattr(benchmark, "tokens_per_second")

    def test_throughput_measures_latency_percentiles(self):
        """Verify latency percentile measurements."""
        from medai_compass.inference.optimized import ThroughputBenchmark

        benchmark = ThroughputBenchmark()
        assert hasattr(benchmark, "get_latency_percentiles")
        # Should return p50, p90, p99 latencies

    def test_throughput_comparison_baseline(self):
        """Verify throughput comparison with baseline."""
        from medai_compass.inference.optimized import ThroughputBenchmark

        benchmark = ThroughputBenchmark()
        assert hasattr(benchmark, "compare_with_baseline")
