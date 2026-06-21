"""Prediction ensembling, shifted-window blending, and temporal smoothing."""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


def average_predictions(predictions: Iterable[np.ndarray], weights: Optional[Iterable[float]] = None) -> np.ndarray:
    """Average predictions from multiple models."""
    preds = [np.asarray(p, dtype=np.float32) for p in predictions]
    if not preds:
        raise ValueError("No predictions provided.")
    stack = np.stack(preds, axis=0)
    if weights is None:
        return stack.mean(axis=0)
    w = np.asarray(list(weights), dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    return np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)


def blend_regular_shifted(
    regular: np.ndarray,
    shifted: Optional[np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend regular chunk predictions with neighboring shifted-window predictions.

    For middle chunks:
        final[i] = alpha * regular[i] + (1-alpha) * 0.5 * (shifted[i-1] + shifted[i])
    """
    regular = np.asarray(regular, dtype=np.float32).copy()
    if shifted is None or len(shifted) == 0:
        return regular
    shifted = np.asarray(shifted, dtype=np.float32)
    out = regular.copy()
    n = len(regular)
    for i in range(n):
        neighbors = []
        if i - 1 >= 0 and i - 1 < len(shifted):
            neighbors.append(shifted[i - 1])
        if i < len(shifted):
            neighbors.append(shifted[i])
        if neighbors:
            shift_avg = np.mean(neighbors, axis=0)
            out[i] = alpha * regular[i] + (1.0 - alpha) * shift_avg
    return out.astype(np.float32)


def temporal_smoothing(preds: np.ndarray, prev_weight: float = 0.1, cur_weight: float = 0.8, next_weight: float = 0.1) -> np.ndarray:
    """Smooth predictions along time: 0.1 previous + 0.8 current + 0.1 next."""
    preds = np.asarray(preds, dtype=np.float32)
    if len(preds) <= 1:
        return preds
    out = preds.copy()
    for i in range(len(preds)):
        total = cur_weight
        value = cur_weight * preds[i]
        if i > 0:
            value += prev_weight * preds[i - 1]
            total += prev_weight
        if i < len(preds) - 1:
            value += next_weight * preds[i + 1]
            total += next_weight
        out[i] = value / total
    return out.astype(np.float32)


def power_adjust(preds: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply probability power adjustment."""
    preds = np.clip(np.asarray(preds, dtype=np.float32), 0.0, 1.0)
    return np.power(preds, gamma).astype(np.float32)
