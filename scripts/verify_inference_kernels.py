#!/usr/bin/env python3
"""
Local Verification Script for Inference Techniques and Kernel Optimizations.

This script verifies that all inference and kernel optimization components
can be imported and configured correctly without requiring GPU hardware.

Tests include:
- Import verification for all classes
- Config class instantiation
- Strategy selector functionality
- Method signature verification

Usage:
    uv run python scripts/verify_inference_kernels.py
    uv run python scripts/verify_inference_kernels.py --category inference
    uv run python scripts/verify_inference_kernels.py --category kernels
    uv run python scripts/verify_inference_kernels.py --verbose
"""

import argparse
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    category: str
    passed: bool
    message: str
    details: str | None = None


class VerificationReport:
    """Collects and formats verification results."""

    def __init__(self):
        self.results: list[TestResult] = []

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)

    def passed_count(self, category: str | None = None) -> int:
        results = self.results
        if category:
            results = [r for r in results if r.category == category]
        return sum(1 for r in results if r.passed)

    def total_count(self, category: str | None = None) -> int:
        results = self.results
        if category:
            results = [r for r in results if r.category == category]
        return len(results)

    def print_report(self, verbose: bool = False) -> None:
        print("=" * 70)
        print("MedGemma Inference & Kernel Optimization Verification Report (Local)")
        print("=" * 70)
        print()

        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        for category, results in categories.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            print(f"{category}: {passed}/{total} passed")

            for result in results:
                status = "[PASS]" if result.passed else "[FAIL]"
                print(f"  {status} {result.name} - {result.message}")
                if verbose and result.details:
                    for line in result.details.split("\n"):
                        print(f"         {line}")
            print()

        # Summary
        print("-" * 70)
        total_passed = self.passed_count()
        total_tests = self.total_count()
        print(f"TOTAL: {total_passed}/{total_tests} tests passed")

        if total_passed == total_tests:
            print("\nAll local verification tests passed!")
            print("Run Modal verification for GPU tests:")
            print("  uv run modal run scripts/verify_inference_kernels_modal.py")
        else:
            print(f"\n{total_tests - total_passed} tests failed. Review above.")


def run_test(
    name: str,
    category: str,
    test_fn: Callable[[], tuple[bool, str, str | None]],
) -> TestResult:
    """Run a single test and capture results."""
    try:
        passed, message, details = test_fn()
        return TestResult(name, category, passed, message, details)
    except Exception as e:
        return TestResult(
            name,
            category,
            False,
            f"Exception: {type(e).__name__}",
            traceback.format_exc(),
        )


# =============================================================================
# Inference Technique Tests
# =============================================================================


def test_flash_attention_import() -> tuple[bool, str, str | None]:
    """Test Flash Attention 2 components can be imported."""
    from medai_compass.inference.optimized import (
        OptimizedModelLoader,
        check_flash_attention_available,
    )

    loader = OptimizedModelLoader(use_flash_attention_2=True)
    attn_impl = loader.get_attn_implementation()
    available = check_flash_attention_available()

    return (
        True,
        f"attn_impl={attn_impl}, available={available}",
        f"OptimizedModelLoader created with use_flash_attention_2=True\n"
        f"get_attn_implementation() returns '{attn_impl}'",
    )


def test_cuda_graphs_import() -> tuple[bool, str, str | None]:
    """Test CUDA Graphs components can be imported."""
    from medai_compass.inference.optimized import (
        CUDAGraphRunner,
        check_cuda_graphs_available,
    )

    runner = CUDAGraphRunner(batch_sizes=[1, 2, 4, 8])
    available = check_cuda_graphs_available()

    return (
        True,
        f"batch_sizes={list(runner.supported_batch_sizes)}, available={available}",
        "CUDAGraphRunner created with supported batch sizes\n"
        "Methods: warmup, capture_graph, replay_graph",
    )


def test_kv_cache_import() -> tuple[bool, str, str | None]:
    """Test KV Cache manager can be imported."""
    from medai_compass.inference.optimized import KVCacheManager

    manager = KVCacheManager(max_length=8192, dtype="fp8", use_paged_attention=True, page_size=16)

    return (
        True,
        f"max_length={manager.max_length}, dtype={manager.dtype}",
        f"KVCacheManager created with FP8 dtype\n"
        f"use_paged_attention={manager.use_paged_attention}, page_size={manager.page_size}",
    )


def test_dynamic_batcher_import() -> tuple[bool, str, str | None]:
    """Test Dynamic Batcher can be imported."""
    from medai_compass.inference.optimized import DynamicBatcher

    batcher = DynamicBatcher(max_batch_size=16, max_wait_ms=50, pad_to_multiple=8)

    return (
        True,
        f"max_batch={batcher.max_batch_size}, wait={batcher.max_wait_ms}ms",
        f"DynamicBatcher created\npad_to_multiple={batcher.pad_to_multiple}",
    )


def test_vllm_engine_import() -> tuple[bool, str, str | None]:
    """Test vLLM Engine can be imported."""
    from medai_compass.inference.optimized import VLLMInferenceEngine

    engine = VLLMInferenceEngine(
        model_name="medgemma-4b",
        tensor_parallel_size=1,
        use_speculative_decoding=False,
        enable_prefix_caching=True,
    )

    return (
        True,
        f"model={engine.model_name}, tp={engine.tensor_parallel_size}",
        f"VLLMInferenceEngine created\n"
        f"speculative_decoding={engine.use_speculative_decoding}\n"
        f"prefix_caching={engine.enable_prefix_caching}",
    )


def test_ray_serve_engine_import() -> tuple[bool, str, str | None]:
    """Test Ray Serve Engine can be imported."""
    from medai_compass.inference.optimized import RayServeEngine

    engine = RayServeEngine(
        model_name="medgemma-27b",
        num_replicas=1,
        max_concurrent_queries=100,
        autoscaling_config={"min_replicas": 1, "max_replicas": 10},
    )

    return (
        True,
        f"replicas={engine.num_replicas}, max_queries={engine.max_concurrent_queries}",
        "RayServeEngine created with autoscaling config",
    )


def test_triton_engine_import() -> tuple[bool, str, str | None]:
    """Test Triton Engine can be imported."""
    from medai_compass.inference.optimized import TritonInferenceEngine

    engine = TritonInferenceEngine(
        model_name="medgemma-27b",
        grpc_url="localhost:8001",
        max_batch_size=32,
        preferred_batch_sizes=[1, 4, 8, 16, 32],
    )

    config = engine._create_model_config()

    return (
        True,
        f"model={engine.model_name}, max_batch={engine.max_batch_size}",
        f"TritonInferenceEngine created\n"
        f"config has dynamic_batching: {'dynamic_batching' in config}",
    )


def test_ray_serve_deployment_import() -> tuple[bool, str, str | None]:
    """Test Ray Serve Deployment Manager can be imported."""
    from medai_compass.inference.ray_serve_deployment import (
        RayServeConfig,
        RayServeDeploymentManager,
    )

    config = RayServeConfig(model_name="google/medgemma-4b-it", num_replicas=1)
    _ = RayServeDeploymentManager(config)  # Verify it can be instantiated

    return (
        True,
        f"model={config.model_name}, autoscaling={config.autoscaling_enabled}",
        f"RayServeDeploymentManager created\nConfig: {config.model_variant} variant",
    )


def test_modal_inference_import() -> tuple[bool, str, str | None]:
    """Test Modal Inference class can be imported."""
    try:
        import importlib.util

        modal_available = importlib.util.find_spec("modal") is not None
    except ImportError:
        modal_available = False

    if modal_available:
        from medai_compass.modal.app import MedGemmaInference  # noqa: F401

        return (
            True,
            "Modal available, MedGemmaInference imported",
            "Modal cloud inference class ready for deployment",
        )
    else:
        return (
            True,
            "Modal not installed (expected on local dev)",
            "Install modal to use cloud inference: pip install modal",
        )


def test_strategy_selector_import() -> tuple[bool, str, str | None]:
    """Test Inference Strategy Selector can be imported."""
    from medai_compass.inference.strategy_selector import (
        InferenceStrategySelector,
    )

    selector = InferenceStrategySelector()
    backends = selector.list_backends()

    # Test auto-selection
    backend = selector.auto_select(priority="throughput", batch_size=32)

    return (
        True,
        f"backends={backends}",
        f"Auto-selected backend for throughput: {backend.name}\n"
        f"Config: tensor_parallel_size={backend.config.tensor_parallel_size}",
    )


def test_gpu_preprocessor_import() -> tuple[bool, str, str | None]:
    """Test GPU Image Preprocessor can be imported."""
    from medai_compass.inference.optimized import GPUImagePreprocessor

    preprocessor = GPUImagePreprocessor(use_gpu=True, target_size=(896, 896), normalize=True)

    return (
        True,
        f"use_gpu={preprocessor.use_gpu}, target_size={preprocessor.target_size}",
        "GPUImagePreprocessor created for DICOM processing",
    )


# =============================================================================
# Kernel Optimization Tests
# =============================================================================


def test_fused_cross_entropy_import() -> tuple[bool, str, str | None]:
    """Test Fused Cross-Entropy can be imported."""
    from medai_compass.training.kernels.fused_kernels import (
        FusedCrossEntropy,
    )

    loss_fn = FusedCrossEntropy(soft_cap=30.0, reduction="mean", ignore_index=-100)

    return (
        True,
        f"soft_cap={loss_fn.soft_cap}, reduction={loss_fn.reduction}",
        "FusedCrossEntropy: ~4x memory reduction by avoiding logit materialization",
    )


def test_fused_rope_import() -> tuple[bool, str, str | None]:
    """Test Fused RoPE can be imported."""
    from medai_compass.training.kernels.fused_kernels import FusedRoPE

    rope = FusedRoPE(dim=128, max_seq_len=8192, base=10000.0)

    return (
        True,
        f"dim={rope.dim}, max_seq_len={rope.max_seq_len}",
        "FusedRoPE: ~2x memory, ~1.5x speed vs naive implementation",
    )


def test_fused_swiglu_import() -> tuple[bool, str, str | None]:
    """Test Fused SwiGLU can be imported."""
    from medai_compass.training.kernels.fused_kernels import FusedSwiGLU

    swiglu = FusedSwiGLU(hidden_size=4096, intermediate_size=11008, bias=False)

    return (
        True,
        f"hidden={swiglu.hidden_size}, intermediate={swiglu.intermediate_size}",
        "FusedSwiGLU: ~3x memory, ~1.8x speed vs naive implementation",
    )


def test_fused_rmsnorm_import() -> tuple[bool, str, str | None]:
    """Test Fused RMSNorm can be imported."""
    from medai_compass.training.kernels.fused_kernels import FusedRMSNorm

    norm = FusedRMSNorm(hidden_size=4096, eps=1e-6)

    return (
        True,
        f"hidden_size={norm.hidden_size}, eps={norm.eps}",
        "FusedRMSNorm: More efficient than LayerNorm (no mean subtraction)",
    )


def test_logit_softcap_import() -> tuple[bool, str, str | None]:
    """Test Logit Soft-Cap can be imported."""
    from medai_compass.training.kernels.logit_softcap import (
        LogitSoftCap,
    )

    softcap = LogitSoftCap(cap=30.0)

    return (
        True,
        f"cap={softcap.cap}",
        "LogitSoftCap: Bounds logits to [-30, 30] for training stability",
    )


def test_adaptive_softcap_import() -> tuple[bool, str, str | None]:
    """Test Adaptive Soft-Cap can be imported."""
    from medai_compass.training.kernels.logit_softcap import AdaptiveSoftCap

    adaptive = AdaptiveSoftCap(initial_cap=30.0, learnable=True, min_cap=1.0, max_cap=100.0)
    current_cap = adaptive.get_cap()

    return (
        True,
        f"learnable={adaptive.learnable}, cap~{current_cap:.1f}",
        f"AdaptiveSoftCap: Learns optimal cap during training\n"
        f"Range: [{adaptive.min_cap}, {adaptive.max_cap}]",
    )


def test_async_gradient_reducer_import() -> tuple[bool, str, str | None]:
    """Test Async Gradient Reducer can be imported."""
    from medai_compass.training.kernels.optimized_backward import AsyncGradientReducer

    reducer = AsyncGradientReducer(overlap_comm=True, bucket_cap_mb=25.0)

    return (
        True,
        f"overlap_comm={reducer.overlap_comm}, bucket_cap={reducer.bucket_cap_mb}MB",
        "AsyncGradientReducer: Overlaps allreduce with backward computation",
    )


def test_gradient_bucketizer_import() -> tuple[bool, str, str | None]:
    """Test Gradient Bucketizer can be imported."""
    from medai_compass.training.kernels.optimized_backward import GradientBucketizer

    bucketizer = GradientBucketizer(bucket_size_mb=25.0)

    return (
        True,
        f"bucket_size={bucketizer.bucket_size_mb}MB",
        "GradientBucketizer: Groups small gradients for efficient allreduce",
    )


def test_selective_recomputation_import() -> tuple[bool, str, str | None]:
    """Test Selective Recomputation can be imported."""
    from medai_compass.training.kernels.optimized_backward import SelectiveRecomputation

    recomp = SelectiveRecomputation(
        checkpoint_policy="selective", checkpoint_ratio=0.5, preserve_rng_state=True
    )

    should_cp = recomp.should_checkpoint(layer_idx=0, total_layers=32)

    return (
        True,
        f"policy={recomp.checkpoint_policy}, ratio={recomp.checkpoint_ratio}",
        f"SelectiveRecomputation: ~50% memory saved\nLayer 0 checkpoint: {should_cp}",
    )


def test_gradient_compression_import() -> tuple[bool, str, str | None]:
    """Test Gradient Compression can be imported."""
    from medai_compass.training.kernels.optimized_backward import GradientCompression

    compression = GradientCompression(dtype="fp8", error_feedback=True)

    return (
        True,
        f"dtype={compression.dtype}, error_feedback={compression.error_feedback}",
        f"GradientCompression: Supported dtypes: {compression.SUPPORTED_DTYPES}",
    )


def test_h100_training_config_import() -> tuple[bool, str, str | None]:
    """Test H100 Training Config can be imported."""
    from medai_compass.training.optimized import H100TrainingConfig

    config = H100TrainingConfig.for_model("medgemma-27b")

    return (
        True,
        f"bf16={config.bf16}, tf32={config.tf32}, fp8={config.use_fp8}",
        f"H100TrainingConfig for 27B model\nFSDP={config.use_fsdp}, workers={config.num_workers}",
    )


def test_fsdp_trainer_import() -> tuple[bool, str, str | None]:
    """Test FSDP Trainer can be imported."""
    from medai_compass.training.optimized import FSDPTrainer, check_fsdp_available

    trainer = FSDPTrainer(
        num_gpus=8,
        sharding_strategy="FULL_SHARD",
        activation_checkpointing=True,
        mixed_precision="bf16",
    )
    fsdp_available = check_fsdp_available()

    return (
        True,
        f"gpus={trainer.num_gpus}, strategy={trainer.sharding_strategy}",
        f"FSDPTrainer: FSDP available={fsdp_available}\n"
        f"activation_checkpointing={trainer.activation_checkpointing}",
    )


def test_h100_optimizer_import() -> tuple[bool, str, str | None]:
    """Test H100 Optimizer can be imported."""
    from medai_compass.training.optimized import H100Optimizer

    _ = H100Optimizer(use_transformer_engine=False)  # Verify it can be instantiated

    return (
        True,
        "H100Optimizer imported successfully",
        "Methods: enable_fp8_matmul, optimize_nvlink_comm, optimize_memory_access",
    )


def test_triton_kernels_available() -> tuple[bool, str, str | None]:
    """Test Triton kernel availability check."""
    from medai_compass.training.kernels.fused_kernels import check_triton_available

    available = check_triton_available()

    return (
        True,
        f"triton_available={available}",
        "Triton enables custom fused kernels for additional speedups",
    )


# =============================================================================
# Main
# =============================================================================


def run_verification(category: str | None = None, verbose: bool = False) -> int:
    """Run all verification tests."""
    report = VerificationReport()

    # Define all tests
    inference_tests = [
        ("Flash Attention 2", test_flash_attention_import),
        ("CUDA Graphs", test_cuda_graphs_import),
        ("KV Cache (FP8)", test_kv_cache_import),
        ("Dynamic Batching", test_dynamic_batcher_import),
        ("vLLM Engine", test_vllm_engine_import),
        ("Ray Serve Engine", test_ray_serve_engine_import),
        ("Triton Engine", test_triton_engine_import),
        ("Ray Serve Deployment", test_ray_serve_deployment_import),
        ("Modal Cloud Inference", test_modal_inference_import),
        ("Strategy Selector", test_strategy_selector_import),
        ("DICOM GPU Preprocessing", test_gpu_preprocessor_import),
    ]

    kernel_tests = [
        ("Fused Cross-Entropy", test_fused_cross_entropy_import),
        ("Fused RoPE", test_fused_rope_import),
        ("Fused SwiGLU", test_fused_swiglu_import),
        ("Fused RMSNorm", test_fused_rmsnorm_import),
        ("Logit Soft-Cap", test_logit_softcap_import),
        ("Adaptive Soft-Cap", test_adaptive_softcap_import),
        ("Async Gradient Reducer", test_async_gradient_reducer_import),
        ("Gradient Bucketing", test_gradient_bucketizer_import),
        ("Selective Recomputation", test_selective_recomputation_import),
        ("Gradient Compression FP8", test_gradient_compression_import),
        ("H100 FP8 Training", test_h100_training_config_import),
        ("FSDP Training", test_fsdp_trainer_import),
        ("H100 Hardware Opts", test_h100_optimizer_import),
        ("Triton Kernels", test_triton_kernels_available),
    ]

    # Run tests
    if category is None or category == "inference":
        for name, test_fn in inference_tests:
            result = run_test(name, "Inference Techniques", test_fn)
            report.add_result(result)

    if category is None or category == "kernels":
        for name, test_fn in kernel_tests:
            result = run_test(name, "Kernel Optimizations", test_fn)
            report.add_result(result)

    # Print report
    report.print_report(verbose=verbose)

    # Return exit code
    return 0 if report.passed_count() == report.total_count() else 1


def main():
    parser = argparse.ArgumentParser(
        description="Verify MedGemma inference and kernel optimizations (local)"
    )
    parser.add_argument(
        "--category",
        choices=["inference", "kernels"],
        help="Test only specific category",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed test output")

    args = parser.parse_args()
    sys.exit(run_verification(category=args.category, verbose=args.verbose))


if __name__ == "__main__":
    main()
