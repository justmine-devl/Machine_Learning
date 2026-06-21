from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready plots from experiment history CSV")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if "epoch" in df.columns and "train_loss" in df.columns:
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df["train_loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("Training loss curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "training_loss_curve.png", dpi=200)
        plt.close()

    metric_cols = [c for c in ["macro_auc", "macro_map", "macro_f1", "micro_auc"] if c in df.columns]
    if "epoch" in df.columns and metric_cols:
        plt.figure(figsize=(7, 4))
        for col in metric_cols:
            plt.plot(df["epoch"], df[col], marker="o", label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation metric curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "validation_metric_curves.png", dpi=200)
        plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
