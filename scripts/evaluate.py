#!/usr/bin/env python
"""Evaluate saved predictions against target labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_project_src_to_path
add_project_src_to_path()

from bioacoustic.metrics import compute_multilabel_metrics, per_class_metrics, search_best_threshold
from bioacoustic.utils import ensure_dir, save_json


def load_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=["number"])
    return numeric.values.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute multilabel metrics from predictions and targets.")
    parser.add_argument("--pred", type=str, required=True, help="Prediction matrix .npy or .csv")
    parser.add_argument("--target", type=str, required=True, help="Target matrix .npy or .csv")
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--search-threshold", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    y_pred = load_matrix(args.pred)
    y_true = load_matrix(args.target).astype(int)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction shape {y_pred.shape} does not match target shape {y_true.shape}")

    threshold = args.threshold
    if args.search_threshold:
        threshold, best_f1 = search_best_threshold(y_true, y_pred)
        print(f"Best threshold by macro F1: {threshold:.3f} (F1={best_f1:.4f})")

    metrics = compute_multilabel_metrics(y_true, y_pred, threshold=threshold)
    save_json(metrics, out_dir / "metrics_summary.json")
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics_summary.csv", index=False)

    class_names = None
    if args.classes:
        class_names = [x.strip() for x in Path(args.classes).read_text(encoding="utf-8").splitlines() if x.strip()]
    per_class = pd.DataFrame(per_class_metrics(y_true, y_pred, class_names))
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)

    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Saved evaluation files to {out_dir}")


if __name__ == "__main__":
    main()
