from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_roc_auc(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    try:
        return float(roc_auc_score(y_true, y_pred, average=average))
    except ValueError:
        return float("nan")


def safe_average_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    try:
        return float(average_precision_score(y_true, y_pred, average=average))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard multi-label metrics."""
    y_bin = (y_pred >= threshold).astype(np.int32)
    return {
        "macro_auc": safe_roc_auc(y_true, y_pred, average="macro"),
        "micro_auc": safe_roc_auc(y_true, y_pred, average="micro"),
        "macro_map": safe_average_precision(y_true, y_pred, average="macro"),
        "micro_map": safe_average_precision(y_true, y_pred, average="micro"),
        "macro_precision": float(precision_score(y_true, y_bin, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_bin, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_bin, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_bin, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_bin, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_bin, average="micro", zero_division=0)),
        "threshold": float(threshold),
    }


def search_best_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, pd.DataFrame]:
    """Find the threshold that maximizes macro F1."""
    rows = []
    best_threshold = start
    best_f1 = -1.0

    thresholds = np.arange(start, stop + 1e-9, step)
    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_pred, threshold=float(threshold))
        rows.append(metrics)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame(rows)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Compute per-class AUC and average precision where possible."""
    rows = []
    for i, name in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "class": name,
            "positive_count": int(yt.sum()),
            "auc": safe_roc_auc(yt.reshape(-1, 1), yp.reshape(-1, 1), average="macro") if len(np.unique(yt)) > 1 else np.nan,
            "ap": safe_average_precision(yt.reshape(-1, 1), yp.reshape(-1, 1), average="macro") if yt.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)
