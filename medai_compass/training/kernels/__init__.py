"""
Kernel Optimizations Module for MedGemma.

Provides Unsloth-style fused kernel optimizations:
- Fused Cross-Entropy (no logit materialization)
- Fused RoPE Embeddings
- Fused SwiGLU Activation
- Fused RMSNorm
- Logit Soft-Capping
- Optimized Backpropagation

Memory savings: Up to 4x reduction in gradient memory
Speed improvement: Up to 2x faster backward pass

Example:
    >>> from medai_compass.training.kernels import FusedCrossEntropy, LogitSoftCap
    >>> loss_fn = FusedCrossEntropy(soft_cap=30.0)
    >>> loss = loss_fn(logits, labels)
"""

from .fused_kernels import (
    FusedCrossEntropy,
    fused_cross_entropy,
    FusedRoPE,
    fused_rope_forward,
    FusedSwiGLU,
    fused_swiglu,
    FusedRMSNorm,
    fused_rms_norm,
    check_triton_available,
)
from .logit_softcap import (
    LogitSoftCap,
    SoftCapCrossEntropy,
    AdaptiveSoftCap,
    apply_soft_cap,
)
from .optimized_backward import (
    AsyncGradientReducer,
    GradientBucketizer,
    SelectiveRecomputation,
    GradientCompression,
    OptimizedBackward,
)

__all__ = [
    # Fused Kernels
    "FusedCrossEntropy",
    "fused_cross_entropy",
    "FusedRoPE",
    "fused_rope_forward",
    "FusedSwiGLU",
    "fused_swiglu",
    "FusedRMSNorm",
    "fused_rms_norm",
    "check_triton_available",
    # Logit Soft-Capping
    "LogitSoftCap",
    "SoftCapCrossEntropy",
    "AdaptiveSoftCap",
    "apply_soft_cap",
    # Backprop Optimizations
    "AsyncGradientReducer",
    "GradientBucketizer",
    "SelectiveRecomputation",
    "GradientCompression",
    "OptimizedBackward",
]
