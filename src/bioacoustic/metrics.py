"""Evaluation metrics for multi-label classification."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _valid_auc_classes(y_true: np.ndarray) -> np.ndarray:
    pos = y_true.sum(axis=0)
    neg = y_true.shape[0] - pos
    return (pos > 0) & (neg > 0)


def safe_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = _valid_auc_classes(y_true)
    if not np.any(valid):
        return float("nan")
    return float(roc_auc_score(y_true[:, valid], y_pred[:, valid], average="macro"))


def safe_micro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = _valid_auc_classes(y_true)
    if not np.any(valid):
        return float("nan")
    return float(roc_auc_score(y_true[:, valid], y_pred[:, valid], average="micro"))


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    y_bin = (y_pred >= threshold).astype(int)
    metrics = {
        "macro_auc": safe_macro_auc(y_true, y_pred),
        "micro_auc": safe_micro_auc(y_true, y_pred),
        "macro_map": float(average_precision_score(y_true, y_pred, average="macro")),
        "micro_map": float(average_precision_score(y_true, y_pred, average="micro")),
        "precision_macro": float(precision_score(y_true, y_bin, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_bin, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_bin, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_bin, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_bin, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_bin, average="micro", zero_division=0)),
        "threshold": float(threshold),
    }
    return metrics


def search_best_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Iterable[float] = np.arange(0.01, 0.99, 0.01),
    average: str = "macro",
) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_bin = (y_pred >= t).astype(int)
        score = f1_score(y_true, y_bin, average=average, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None) -> list[dict]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    rows = []
    for c in range(n_classes):
        row = {"class": class_names[c], "support": int(y_true[:, c].sum())}
        if len(np.unique(y_true[:, c])) == 2:
            row["auc"] = float(roc_auc_score(y_true[:, c], y_pred[:, c]))
            row["ap"] = float(average_precision_score(y_true[:, c], y_pred[:, c]))
        else:
            row["auc"] = float("nan")
            row["ap"] = float("nan")
        rows.append(row)
    return rows
