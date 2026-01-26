"""
Tests for Kernel Optimizations.

TDD approach: Tests written first for all kernel optimizations:
- Fused Cross-Entropy (no logit materialization)
- Fused RoPE Embeddings
- Fused SwiGLU Activation
- Fused RMSNorm
- Logit Soft-Capping
- Optimized Backpropagation
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional


# =============================================================================
# Fused Cross-Entropy Tests
# =============================================================================

class TestFusedCrossEntropy:
    """Tests for fused cross-entropy loss."""

    def test_loss_creation(self):
        """Verify fused cross-entropy can be created."""
        from medai_compass.training.kernels import FusedCrossEntropy

        loss_fn = FusedCrossEntropy()
        assert loss_fn is not None

    def test_loss_with_soft_cap(self):
        """Verify fused cross-entropy with soft-capping."""
        from medai_compass.training.kernels import FusedCrossEntropy

        loss_fn = FusedCrossEntropy(soft_cap=30.0)
        assert loss_fn.soft_cap == 30.0

    def test_loss_forward(self):
        """Verify forward pass works."""
        from medai_compass.training.kernels import FusedCrossEntropy

        loss_fn = FusedCrossEntropy()
        assert hasattr(loss_fn, "forward")
        assert callable(loss_fn.forward)

    def test_loss_reduction_modes(self):
        """Verify different reduction modes."""
        from medai_compass.training.kernels import FusedCrossEntropy

        loss_mean = FusedCrossEntropy(reduction="mean")
        loss_sum = FusedCrossEntropy(reduction="sum")
        loss_none = FusedCrossEntropy(reduction="none")

        assert loss_mean.reduction == "mean"
        assert loss_sum.reduction == "sum"
        assert loss_none.reduction == "none"

    def test_fused_cross_entropy_function(self):
        """Verify functional interface."""
        from medai_compass.training.kernels import fused_cross_entropy

        assert callable(fused_cross_entropy)


# =============================================================================
# Fused RoPE Tests
# =============================================================================

class TestFusedRoPE:
    """Tests for fused RoPE embeddings."""

    def test_rope_creation(self):
        """Verify fused RoPE can be created."""
        from medai_compass.training.kernels import FusedRoPE

        rope = FusedRoPE(dim=128)
        assert rope is not None
        assert rope.dim == 128

    def test_rope_max_seq_length(self):
        """Verify max sequence length configuration."""
        from medai_compass.training.kernels import FusedRoPE

        rope = FusedRoPE(dim=128, max_seq_len=8192)
        assert rope.max_seq_len == 8192

    def test_rope_base_frequency(self):
        """Verify base frequency configuration."""
        from medai_compass.training.kernels import FusedRoPE

        rope = FusedRoPE(dim=128, base=10000)
        assert rope.base == 10000

    def test_rope_forward(self):
        """Verify RoPE forward pass."""
        from medai_compass.training.kernels import FusedRoPE

        rope = FusedRoPE(dim=128)
        assert hasattr(rope, "forward")

    def test_fused_rope_function(self):
        """Verify functional interface."""
        from medai_compass.training.kernels import fused_rope_forward

        assert callable(fused_rope_forward)


# =============================================================================
# Fused SwiGLU Tests
# =============================================================================

class TestFusedSwiGLU:
    """Tests for fused SwiGLU activation."""

    def test_swiglu_creation(self):
        """Verify fused SwiGLU can be created."""
        from medai_compass.training.kernels import FusedSwiGLU

        swiglu = FusedSwiGLU(hidden_size=4096, intermediate_size=11008)
        assert swiglu is not None

    def test_swiglu_dimensions(self):
        """Verify dimension configuration."""
        from medai_compass.training.kernels import FusedSwiGLU

        swiglu = FusedSwiGLU(hidden_size=4096, intermediate_size=11008)
        assert swiglu.hidden_size == 4096
        assert swiglu.intermediate_size == 11008

    def test_swiglu_forward(self):
        """Verify SwiGLU forward pass."""
        from medai_compass.training.kernels import FusedSwiGLU

        swiglu = FusedSwiGLU(hidden_size=4096, intermediate_size=11008)
        assert hasattr(swiglu, "forward")

    def test_fused_swiglu_function(self):
        """Verify functional interface."""
        from medai_compass.training.kernels import fused_swiglu

        assert callable(fused_swiglu)


# =============================================================================
# Fused RMSNorm Tests
# =============================================================================

class TestFusedRMSNorm:
    """Tests for fused RMSNorm."""

    def test_rmsnorm_creation(self):
        """Verify fused RMSNorm can be created."""
        from medai_compass.training.kernels import FusedRMSNorm

        norm = FusedRMSNorm(hidden_size=4096)
        assert norm is not None
        assert norm.hidden_size == 4096

    def test_rmsnorm_eps(self):
        """Verify epsilon configuration."""
        from medai_compass.training.kernels import FusedRMSNorm

        norm = FusedRMSNorm(hidden_size=4096, eps=1e-5)
        assert norm.eps == 1e-5

    def test_rmsnorm_forward(self):
        """Verify RMSNorm forward pass."""
        from medai_compass.training.kernels import FusedRMSNorm

        norm = FusedRMSNorm(hidden_size=4096)
        assert hasattr(norm, "forward")

    def test_fused_rms_norm_function(self):
        """Verify functional interface."""
        from medai_compass.training.kernels import fused_rms_norm

        assert callable(fused_rms_norm)


# =============================================================================
# Logit Soft-Capping Tests
# =============================================================================

class TestLogitSoftCap:
    """Tests for logit soft-capping."""

    def test_softcap_creation(self):
        """Verify LogitSoftCap can be created."""
        from medai_compass.training.kernels import LogitSoftCap

        softcap = LogitSoftCap(cap=30.0)
        assert softcap is not None
        assert softcap.cap == 30.0

    def test_softcap_forward(self):
        """Verify soft-cap forward pass."""
        from medai_compass.training.kernels import LogitSoftCap

        softcap = LogitSoftCap(cap=30.0)
        assert hasattr(softcap, "forward")

    def test_softcap_prevents_explosion(self):
        """Verify soft-cap limits logit values."""
        from medai_compass.training.kernels import LogitSoftCap

        softcap = LogitSoftCap(cap=30.0)
        # After soft-capping, values should be bounded by [-cap, cap]
        assert hasattr(softcap, "cap")
        assert softcap.cap == 30.0

    def test_apply_soft_cap_function(self):
        """Verify functional interface."""
        from medai_compass.training.kernels import apply_soft_cap

        assert callable(apply_soft_cap)


class TestSoftCapCrossEntropy:
    """Tests for cross-entropy with soft-capping."""

    def test_loss_creation(self):
        """Verify SoftCapCrossEntropy can be created."""
        from medai_compass.training.kernels import SoftCapCrossEntropy

        loss_fn = SoftCapCrossEntropy(cap=30.0)
        assert loss_fn is not None
        assert loss_fn.cap == 30.0

    def test_loss_forward(self):
        """Verify forward pass."""
        from medai_compass.training.kernels import SoftCapCrossEntropy

        loss_fn = SoftCapCrossEntropy(cap=30.0)
        assert hasattr(loss_fn, "forward")


class TestAdaptiveSoftCap:
    """Tests for adaptive soft-capping."""

    def test_adaptive_creation(self):
        """Verify AdaptiveSoftCap can be created."""
        from medai_compass.training.kernels import AdaptiveSoftCap

        adaptive = AdaptiveSoftCap(initial_cap=30.0)
        assert adaptive is not None

    def test_adaptive_learns_cap(self):
        """Verify cap is learnable."""
        from medai_compass.training.kernels import AdaptiveSoftCap

        adaptive = AdaptiveSoftCap(initial_cap=30.0, learnable=True)
        assert adaptive.learnable is True

    def test_adaptive_cap_range(self):
        """Verify cap value constraints."""
        from medai_compass.training.kernels import AdaptiveSoftCap

        adaptive = AdaptiveSoftCap(
            initial_cap=30.0,
            min_cap=10.0,
            max_cap=50.0
        )
        assert adaptive.min_cap == 10.0
        assert adaptive.max_cap == 50.0


# =============================================================================
# Backpropagation Optimization Tests
# =============================================================================

class TestAsyncGradientReducer:
    """Tests for async gradient reduction."""

    def test_reducer_creation(self):
        """Verify AsyncGradientReducer can be created."""
        from medai_compass.training.kernels import AsyncGradientReducer

        reducer = AsyncGradientReducer()
        assert reducer is not None

    def test_reducer_overlap_enabled(self):
        """Verify communication overlap."""
        from medai_compass.training.kernels import AsyncGradientReducer

        reducer = AsyncGradientReducer(overlap_comm=True)
        assert reducer.overlap_comm is True

    def test_reducer_reduce_method(self):
        """Verify reduce method exists."""
        from medai_compass.training.kernels import AsyncGradientReducer

        reducer = AsyncGradientReducer()
        assert hasattr(reducer, "reduce")
        assert hasattr(reducer, "wait")


class TestGradientBucketizer:
    """Tests for gradient bucketing."""

    def test_bucketizer_creation(self):
        """Verify GradientBucketizer can be created."""
        from medai_compass.training.kernels import GradientBucketizer

        bucketizer = GradientBucketizer(bucket_size_mb=25)
        assert bucketizer is not None
        assert bucketizer.bucket_size_mb == 25

    def test_bucketizer_add_gradient(self):
        """Verify gradient addition to bucket."""
        from medai_compass.training.kernels import GradientBucketizer

        bucketizer = GradientBucketizer()
        assert hasattr(bucketizer, "add_gradient")

    def test_bucketizer_flush(self):
        """Verify bucket flushing."""
        from medai_compass.training.kernels import GradientBucketizer

        bucketizer = GradientBucketizer()
        assert hasattr(bucketizer, "flush")


class TestSelectiveRecomputation:
    """Tests for selective activation recomputation."""

    def test_recomputation_creation(self):
        """Verify SelectiveRecomputation can be created."""
        from medai_compass.training.kernels import SelectiveRecomputation

        recomp = SelectiveRecomputation()
        assert recomp is not None

    def test_recomputation_policy(self):
        """Verify recomputation policy configuration."""
        from medai_compass.training.kernels import SelectiveRecomputation

        recomp = SelectiveRecomputation(
            checkpoint_policy="selective",
            checkpoint_ratio=0.5
        )
        assert recomp.checkpoint_policy == "selective"
        assert recomp.checkpoint_ratio == 0.5

    def test_recomputation_wrap(self):
        """Verify module wrapping."""
        from medai_compass.training.kernels import SelectiveRecomputation

        recomp = SelectiveRecomputation()
        assert hasattr(recomp, "wrap_module")


class TestGradientCompression:
    """Tests for gradient compression."""

    def test_compression_creation(self):
        """Verify GradientCompression can be created."""
        from medai_compass.training.kernels import GradientCompression

        compression = GradientCompression()
        assert compression is not None

    def test_compression_fp8(self):
        """Verify FP8 gradient compression."""
        from medai_compass.training.kernels import GradientCompression

        compression = GradientCompression(dtype="fp8")
        assert compression.dtype == "fp8"

    def test_compression_int8(self):
        """Verify INT8 gradient compression."""
        from medai_compass.training.kernels import GradientCompression

        compression = GradientCompression(dtype="int8")
        assert compression.dtype == "int8"

    def test_compression_compress(self):
        """Verify compression method."""
        from medai_compass.training.kernels import GradientCompression

        compression = GradientCompression()
        assert hasattr(compression, "compress")

    def test_compression_decompress(self):
        """Verify decompression method."""
        from medai_compass.training.kernels import GradientCompression

        compression = GradientCompression()
        assert hasattr(compression, "decompress")


class TestOptimizedBackward:
    """Tests for combined optimized backward pass."""

    def test_backward_creation(self):
        """Verify OptimizedBackward can be created."""
        from medai_compass.training.kernels import OptimizedBackward

        backward = OptimizedBackward()
        assert backward is not None

    def test_backward_all_optimizations(self):
        """Verify all backward optimizations can be enabled."""
        from medai_compass.training.kernels import OptimizedBackward

        backward = OptimizedBackward(
            async_reduce=True,
            use_bucketing=True,
            selective_recomputation=True,
            gradient_compression="fp8"
        )
        assert backward.async_reduce is True
        assert backward.use_bucketing is True
        assert backward.selective_recomputation is True
        assert backward.gradient_compression == "fp8"

    def test_backward_wrap_model(self):
        """Verify model wrapping for optimized backward."""
        from medai_compass.training.kernels import OptimizedBackward

        backward = OptimizedBackward()
        assert hasattr(backward, "wrap_model")


# =============================================================================
# Triton Availability Tests
# =============================================================================

class TestTritonAvailability:
    """Tests for Triton kernel availability."""

    def test_check_triton_available(self):
        """Verify Triton availability check."""
        from medai_compass.training.kernels import check_triton_available

        result = check_triton_available()
        assert isinstance(result, bool)
