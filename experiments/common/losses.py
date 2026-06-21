from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction=reduction)


class WeightedFocalBCELoss(nn.Module):
    """BCE with logits + optional positive class weights + focal modulation.

    This is the main teacher-model objective used in the report.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def smooth_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        targets = self.smooth_targets(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        if self.gamma > 0:
            probs = torch.sigmoid(logits)
            pt = probs * targets + (1 - probs) * (1 - targets)
            bce = bce * (1 - pt).pow(self.gamma)
        if self.reduction == "mean":
            return bce.mean()
        if self.reduction == "sum":
            return bce.sum()
        return bce


class SoftLabelBCELoss(nn.Module):
    """BCE with logits where targets are probabilities from a teacher model."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, soft_targets, reduction=self.reduction)


class StudentDistillationLoss(nn.Module):
    """Hard-label Weighted Focal BCE + soft-label BCE from teacher predictions."""

    def __init__(self, hard_weight: float = 0.6, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.hard_weight = hard_weight
        self.hard_loss = WeightedFocalBCELoss(gamma=gamma, label_smoothing=label_smoothing)
        self.soft_loss = SoftLabelBCELoss()

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hard = self.hard_loss(logits, hard_targets, pos_weight=pos_weight)
        soft = self.soft_loss(logits, soft_targets)
        return self.hard_weight * hard + (1.0 - self.hard_weight) * soft


def compute_pos_weight(targets: torch.Tensor, max_weight: float = 20.0) -> torch.Tensor:
    """Compute positive class weights for multi-label BCE."""
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    weight = negatives / (positives + 1e-6)
    weight = torch.clamp(weight, min=1.0, max=max_weight)
    return weight.float()
