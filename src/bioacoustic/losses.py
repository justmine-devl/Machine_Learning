"""BCE-based losses used in the project."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def smooth_targets(targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def bce_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    targets = smooth_targets(targets, label_smoothing)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = smooth_targets(targets, label_smoothing)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="mean")


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Weighted Focal BCE with logits.

    If pos_weight is None, this becomes standard Focal BCE.
    """
    targets = smooth_targets(targets, label_smoothing)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = (1.0 - pt).pow(gamma)
    return (focal * bce).mean()


def soft_label_bce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """BCE with teacher probability targets."""
    return F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="mean")


def student_distillation_loss(
    logits: torch.Tensor,
    hard_targets: torch.Tensor,
    soft_targets: torch.Tensor,
    hard_weight: float = 0.6,
    pos_weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Student loss = hard-label Weighted Focal BCE + soft-label BCE."""
    hard = focal_bce_loss(
        logits,
        hard_targets,
        gamma=gamma,
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
    )
    soft = soft_label_bce_loss(logits, soft_targets)
    return hard_weight * hard + (1.0 - hard_weight) * soft


def compute_pos_weight(target_matrix: torch.Tensor, eps: float = 1e-6, max_weight: float = 20.0) -> torch.Tensor:
    """Compute positive class weights from a binary target matrix [N, C]."""
    positives = target_matrix.sum(dim=0)
    negatives = target_matrix.shape[0] - positives
    weight = negatives / (positives + eps)
    return torch.clamp(weight, min=1.0, max=max_weight)
