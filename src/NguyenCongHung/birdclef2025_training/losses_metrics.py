from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .class_order import CLASS_ORDER
from .config import TrainingConfig
from .datasets import target_from_row


def smooth_targets(target: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return target
    return target * (1.0 - smoothing) + 0.5 * smoothing


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = bce * ((1.0 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Keep multi-hot targets unnormalized, matching the author's harder-sample weighting.
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def make_pos_weight(df: pd.DataFrame, config: TrainingConfig) -> Optional[torch.Tensor]:
    if not config.use_pos_weight or df.empty:
        return None
    counts = np.zeros(len(CLASS_ORDER), dtype=np.float64)
    for _, row in df.iterrows():
        counts += target_from_row(row, 1)[0]
    pos = np.maximum(counts, 1.0)
    neg = max(len(df), 1) - counts
    return torch.tensor(neg / pos, dtype=torch.float32)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, config: TrainingConfig, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    targets = smooth_targets(targets, config.label_smoothing)
    if config.loss_type == "cross_entropy":
        return soft_cross_entropy(logits, targets)
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    if config.loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    if config.loss_type == "focal_bce":
        return focal_bce_with_logits(logits, targets, gamma=config.focal_gamma, alpha=config.focal_alpha, pos_weight=pos_weight)
    raise ValueError(f"Unknown loss_type={config.loss_type}")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true_flat = y_true.reshape(-1, y_true.shape[-1]).astype(np.int32)
    y_prob_flat = y_prob.reshape(-1, y_prob.shape[-1])
    aucs = []
    per_class_auc = {}
    for i, label in enumerate(CLASS_ORDER):
        if len(np.unique(y_true_flat[:, i])) < 2:
            continue
        try:
            auc = roc_auc_score(y_true_flat[:, i], y_prob_flat[:, i])
            aucs.append(auc)
            per_class_auc[f"auc_{label}"] = float(auc)
        except Exception:
            pass

    y_pred = (y_prob_flat >= threshold).astype(np.int32)
    metrics = {
        "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "valid_auc_classes": int(len(aucs)),
        "macro_f1": float(f1_score(y_true_flat, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true_flat, y_pred, average="micro", zero_division=0)),
        "binary_accuracy": float((y_true_flat == y_pred).mean()),
        "positive_precision": float(precision_score(y_true_flat, y_pred, average="micro", zero_division=0)),
        "positive_recall": float(recall_score(y_true_flat, y_pred, average="micro", zero_division=0)),
    }
    metrics.update(per_class_auc)
    return metrics
