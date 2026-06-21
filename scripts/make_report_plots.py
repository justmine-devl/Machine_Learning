#!/usr/bin/env python
"""Create simple plots for the report from experiment CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_src_to_path
add_project_src_to_path()

from bioacoustic.utils import ensure_dir
from bioacoustic.visualization import plot_bar, plot_metric_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report plots.")
    parser.add_argument("--history", type=str, default=None, help="Training history CSV with epoch and metric columns.")
    parser.add_argument("--ablation", type=str, default=None, help="Ablation CSV with method and metric columns.")
    parser.add_argument("--out-dir", type=str, default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    if args.history:
        hist = pd.read_csv(args.history)
        if "epoch" in hist.columns and "macro_auc" in hist.columns:
            plot_metric_curve(hist, "epoch", "macro_auc", out_dir / "macro_auc_curve.png", title="Validation Macro AUC")
        if "epoch" in hist.columns and "train_loss" in hist.columns:
            plot_metric_curve(hist, "epoch", "train_loss", out_dir / "train_loss_curve.png", title="Training Loss")

    if args.ablation:
        abl = pd.read_csv(args.ablation)
        x = "method" if "method" in abl.columns else abl.columns[0]
        if "macro_auc" in abl.columns:
            plot_bar(abl, x, "macro_auc", out_dir / "ablation_macro_auc.png", title="Ablation Study: Macro AUC")
        if "f1_macro" in abl.columns:
            plot_bar(abl, x, "f1_macro", out_dir / "ablation_f1.png", title="Ablation Study: Macro F1")

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
