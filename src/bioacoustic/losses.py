"""Classification and soft-target losses used in the project."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .dataset import CLASS_ORDER, target_from_row

if TYPE_CHECKING:
    from .training import BirdCLEFTrainingConfig


def smooth_targets(targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    targets = smooth_targets(targets, label_smoothing)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = smooth_targets(targets, label_smoothing)
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="mean"
    )


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Weighted Focal BCE with logits.
    If pos_weight is None, this becomes standard Focal BCE.
    """
    targets = smooth_targets(targets, label_smoothing)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = (1.0 - pt).pow(gamma)
    loss = focal * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()


def soft_label_bce_loss(
    logits: torch.Tensor, soft_targets: torch.Tensor
) -> torch.Tensor:
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


def compute_pos_weight(
    target_matrix: torch.Tensor, eps: float = 1e-6, max_weight: float = 20.0
) -> torch.Tensor:
    """Compute positive class weights from a binary target matrix [N, C]."""
    positives = target_matrix.sum(dim=0)
    negatives = target_matrix.shape[0] - positives
    weight = negatives / (positives + eps)
    return torch.clamp(weight, min=1.0, max=max_weight)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Keep multi-hot targets unnormalized, matching the author's harder-sample weighting.
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def make_pos_weight(
    df: pd.DataFrame, config: BirdCLEFTrainingConfig
) -> Optional[torch.Tensor]:
    if not config.use_pos_weight or df.empty:
        return None
    counts = np.zeros(len(CLASS_ORDER), dtype=np.float64)
    for _, row in df.iterrows():
        counts += target_from_row(row, 1)[0]
    pos = np.maximum(counts, 1.0)
    neg = max(len(df), 1) - counts
    return torch.tensor(neg / pos, dtype=torch.float32)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: BirdCLEFTrainingConfig,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = smooth_targets(targets, config.label_smoothing)
    if config.loss_type == "cross_entropy":
        return soft_cross_entropy(logits, targets)
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    if config.loss_type == "bce":
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
    if config.loss_type == "focal_bce":
        return focal_bce_loss(
            logits,
            targets,
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            pos_weight=pos_weight,
        )
    raise ValueError(f"Unknown loss_type={config.loss_type}")
