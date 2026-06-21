from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def safe_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        labels = y_true[:, c]
        if len(np.unique(labels)) < 2:
            continue
        scores.append(roc_auc_score(labels, y_pred[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def safe_micro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true.ravel())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true.ravel(), y_pred.ravel()))


def safe_map(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    try:
        return float(average_precision_score(y_true, y_pred, average=average))
    except Exception:
        return float("nan")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, average: str = "macro") -> Dict[str, float]:
    y_bin = (y_pred >= threshold).astype(np.int32)
    return {
        "precision": float(precision_score(y_true, y_bin, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_bin, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_bin, average=average, zero_division=0)),
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    out = {
        "macro_auc": safe_macro_auc(y_true, y_pred),
        "micro_auc": safe_micro_auc(y_true, y_pred),
        "macro_map": safe_map(y_true, y_pred, average="macro"),
        "micro_map": safe_map(y_true, y_pred, average="micro"),
        "threshold": float(threshold),
    }
    out.update({f"macro_{k}": v for k, v in binary_metrics(y_true, y_pred, threshold, "macro").items()})
    out.update({f"micro_{k}": v for k, v in binary_metrics(y_true, y_pred, threshold, "micro").items()})
    return out


def search_best_threshold(y_true: np.ndarray, y_pred: np.ndarray, thresholds: Iterable[float] | None = None) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        f1 = binary_metrics(y_true, y_pred, float(t), average="macro")["f1"]
        if f1 > best_f1:
            best_t = float(t)
            best_f1 = float(f1)
    return best_t, best_f1


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], threshold: float = 0.5):
    rows = []
    y_bin = (y_pred >= threshold).astype(np.int32)
    for i, name in enumerate(class_names):
        labels = y_true[:, i]
        auc = roc_auc_score(labels, y_pred[:, i]) if len(np.unique(labels)) > 1 else np.nan
        ap = average_precision_score(labels, y_pred[:, i]) if labels.sum() > 0 else np.nan
        rows.append({
            "class": name,
            "support": int(labels.sum()),
            "auc": float(auc) if not np.isnan(auc) else np.nan,
            "ap": float(ap) if not np.isnan(ap) else np.nan,
            "precision": float(precision_score(labels, y_bin[:, i], zero_division=0)),
            "recall": float(recall_score(labels, y_bin[:, i], zero_division=0)),
            "f1": float(f1_score(labels, y_bin[:, i], zero_division=0)),
        })
    return rows
