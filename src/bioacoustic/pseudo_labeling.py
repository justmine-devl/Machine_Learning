"""Pseudo-label generation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def select_pseudo_labels(
    predictions: np.ndarray,
    row_ids: Optional[List[str]] = None,
    min_max_prob: float = 0.5,
    class_prob_threshold: float = 0.1,
) -> pd.DataFrame:
    """Select pseudo-labeled chunks from model probabilities.

    Keeps chunks whose maximum class probability is at least min_max_prob,
    then zeroes out class probabilities below class_prob_threshold.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    keep = predictions.max(axis=1) >= min_max_prob
    selected = predictions[keep].copy()
    selected[selected < class_prob_threshold] = 0.0
    if row_ids is None:
        row_ids = [f"chunk_{i}" for i in range(len(predictions))]
    selected_ids = np.asarray(row_ids)[keep]
    df = pd.DataFrame(selected)
    df.insert(0, "row_id", selected_ids)
    return df


def save_pseudo_labels(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_pseudo_labels(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def mix_labeled_and_pseudo(
    labeled_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    labeled_ratio: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a mixed training table with labeled and pseudo-labeled rows."""
    rng = np.random.default_rng(seed)
    n_labeled = len(labeled_df)
    n_pseudo = int(n_labeled * (1.0 - labeled_ratio) / max(labeled_ratio, 1e-6))
    pseudo_sample = pseudo_df.sample(n=min(n_pseudo, len(pseudo_df)), random_state=seed) if len(pseudo_df) else pseudo_df
    a = labeled_df.copy()
    a["source"] = "labeled"
    b = pseudo_sample.copy()
    b["source"] = "pseudo"
    return pd.concat([a, b], ignore_index=True)
