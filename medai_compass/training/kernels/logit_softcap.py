"""
Logit Soft-Capping for MedGemma Training.

Prevents logit explosion during training by applying a soft upper bound
to logit values. This improves training stability especially for large models.

Formula: soft_cap * tanh(logits / soft_cap)

This bounds logits to [-soft_cap, soft_cap] while maintaining differentiability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LogitSoftCap(nn.Module):
    """
    Logit Soft-Capping module.

    Applies soft-capping to prevent logit explosion:
    output = cap * tanh(input / cap)

    This bounds values to [-cap, cap] while maintaining smooth gradients.

    Example:
        >>> softcap = LogitSoftCap(cap=30.0)
        >>> capped_logits = softcap(logits)
    """

    def __init__(self, cap: float = 30.0):
        """
        Initialize LogitSoftCap.

        Args:
            cap: Maximum absolute value for soft-capped outputs
        """
        super().__init__()
        self.cap = cap

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply soft-capping to logits.

        Args:
            logits: Input logits tensor

        Returns:
            Soft-capped logits bounded to [-cap, cap]
        """
        return self.cap * torch.tanh(logits / self.cap)


def apply_soft_cap(
    logits: torch.Tensor,
    cap: float = 30.0,
) -> torch.Tensor:
    """
    Functional interface for logit soft-capping.

    Args:
        logits: Input logits tensor
        cap: Maximum absolute value for soft-capped outputs

    Returns:
        Soft-capped logits
    """
    return cap * torch.tanh(logits / cap)


class SoftCapCrossEntropy(nn.Module):
    """
    Cross-Entropy Loss with built-in soft-capping.

    Combines logit soft-capping with cross-entropy loss computation
    for improved training stability.

    Example:
        >>> loss_fn = SoftCapCrossEntropy(cap=30.0)
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        cap: float = 30.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize SoftCapCrossEntropy.

        Args:
            cap: Soft-capping value for logits
            reduction: Loss reduction mode ("mean", "sum", "none")
            ignore_index: Label value to ignore
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.cap = cap
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with soft-capped logits.

        Args:
            logits: Model logits (B, S, V) or (B, V)
            labels: Target labels (B, S) or (B,)

        Returns:
            Loss tensor
        """
        # Apply soft-capping
        capped_logits = self.cap * torch.tanh(logits / self.cap)

        # Flatten if needed
        if capped_logits.dim() == 3:
            batch_size, seq_len, vocab_size = capped_logits.shape
            capped_logits = capped_logits.view(-1, vocab_size)
            labels = labels.view(-1)

        # Compute cross-entropy
        return F.cross_entropy(
            capped_logits,
            labels,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class AdaptiveSoftCap(nn.Module):
    """
    Adaptive Soft-Capping with learnable cap value.

    The soft-cap value can be learned during training to find
    the optimal balance between expressiveness and stability.

    Example:
        >>> adaptive = AdaptiveSoftCap(initial_cap=30.0, learnable=True)
        >>> capped_logits = adaptive(logits)
        >>> print(f"Current cap: {adaptive.get_cap()}")
    """

    def __init__(
        self,
        initial_cap: float = 30.0,
        learnable: bool = True,
        min_cap: float = 1.0,
        max_cap: float = 100.0,
    ):
        """
        Initialize AdaptiveSoftCap.

        Args:
            initial_cap: Initial soft-cap value
            learnable: Whether to learn the cap value
            min_cap: Minimum allowed cap value
            max_cap: Maximum allowed cap value
        """
        super().__init__()
        self.learnable = learnable
        self.min_cap = min_cap
        self.max_cap = max_cap

        # Store cap in log space for numerical stability
        # cap = min_cap + (max_cap - min_cap) * sigmoid(log_cap)
        if learnable:
            # Initialize such that sigmoid(log_cap) gives desired initial_cap
            target_sigmoid = (initial_cap - min_cap) / (max_cap - min_cap)
            target_sigmoid = max(0.01, min(0.99, target_sigmoid))  # Clamp
            initial_log_cap = torch.log(
                torch.tensor(target_sigmoid / (1 - target_sigmoid))
            )
            self.log_cap = nn.Parameter(
                torch.tensor(initial_log_cap.item())
            )
        else:
            self.register_buffer(
                "log_cap",
                torch.tensor(0.0)  # Not used when not learnable
            )
            self._fixed_cap = initial_cap

    def get_cap(self) -> float:
        """Get the current cap value."""
        if self.learnable:
            sigmoid_val = torch.sigmoid(self.log_cap)
            cap = self.min_cap + (self.max_cap - self.min_cap) * sigmoid_val
            return cap.item()
        else:
            return self._fixed_cap

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive soft-capping to logits.

        Args:
            logits: Input logits tensor

        Returns:
            Soft-capped logits
        """
        if self.learnable:
            sigmoid_val = torch.sigmoid(self.log_cap)
            cap = self.min_cap + (self.max_cap - self.min_cap) * sigmoid_val
        else:
            cap = self._fixed_cap

        return cap * torch.tanh(logits / cap)


class TemperatureSoftCap(nn.Module):
    """
    Temperature-scaled Soft-Capping.

    Combines temperature scaling with soft-capping for fine-grained
    control over logit distribution.

    output = cap * tanh(logits / (cap * temperature))
    """

    def __init__(
        self,
        cap: float = 30.0,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
    ):
        """
        Initialize TemperatureSoftCap.

        Args:
            cap: Soft-cap value
            temperature: Temperature scaling factor
            learnable_temperature: Whether to learn temperature
        """
        super().__init__()
        self.cap = cap

        if learnable_temperature:
            # Store in log space
            self.log_temperature = nn.Parameter(
                torch.tensor(0.0)  # exp(0) = 1
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(0.0)
            )
        self._learnable_temperature = learnable_temperature
        self._fixed_temperature = temperature

    def get_temperature(self) -> float:
        """Get the current temperature value."""
        if self._learnable_temperature:
            return torch.exp(self.log_temperature).item()
        return self._fixed_temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature-scaled soft-capping."""
        if self._learnable_temperature:
            temp = torch.exp(self.log_temperature)
        else:
            temp = self._fixed_temperature

        scaled_cap = self.cap * temp
        return scaled_cap * torch.tanh(logits / scaled_cap)


class GradientClippedSoftCap(nn.Module):
    """
    Soft-Capping with gradient clipping.

    Combines soft-capping with gradient clipping for additional
    training stability.
    """

    def __init__(
        self,
        cap: float = 30.0,
        grad_clip: float = 1.0,
    ):
        """
        Initialize GradientClippedSoftCap.

        Args:
            cap: Soft-cap value
            grad_clip: Maximum gradient norm
        """
        super().__init__()
        self.cap = cap
        self.grad_clip = grad_clip

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply soft-capping with gradient scaling."""
        # Apply soft-cap
        capped = self.cap * torch.tanh(logits / self.cap)

        if self.training and logits.requires_grad:
            # Register hook for gradient clipping
            def clip_grad(grad):
                norm = grad.norm()
                if norm > self.grad_clip:
                    return grad * (self.grad_clip / norm)
                return grad

            capped.register_hook(clip_grad)

        return capped
