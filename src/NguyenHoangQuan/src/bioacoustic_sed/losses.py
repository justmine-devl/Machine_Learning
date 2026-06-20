from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_targets(targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Apply simple label smoothing for multi-label targets."""
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def bce_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    targets = smooth_targets(targets, label_smoothing)
    return F.binary_cross_entropy_with_logits(logits, targets)


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = smooth_targets(targets, label_smoothing)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Weighted Focal BCE with logits.

    This is BCE + optional class weighting + focal modulation.
    """
    targets = smooth_targets(targets, label_smoothing)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal = (1.0 - pt).pow(gamma)
    return (bce * focal).mean()


def soft_label_bce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """BCE where targets are teacher probabilities instead of hard 0/1 labels."""
    return F.binary_cross_entropy_with_logits(logits, soft_targets)


def student_distillation_loss(
    logits: torch.Tensor,
    hard_targets: torch.Tensor,
    soft_targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    hard_weight: float = 0.6,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Combined student loss:

    hard_weight * Weighted Focal BCE(hard labels)
    + (1 - hard_weight) * Soft-label BCE(teacher probabilities)
    """
    hard_loss = focal_bce_loss(
        logits,
        hard_targets,
        pos_weight=pos_weight,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    soft_loss = soft_label_bce_loss(logits, soft_targets)
    return hard_weight * hard_loss + (1.0 - hard_weight) * soft_loss
