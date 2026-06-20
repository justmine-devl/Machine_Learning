from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class ReportPlotter:
    def __init__(self, output_dir: Path, plots_dir: Path | None = None, logs_dir: Path | None = None, pseudo_labels_dir: Path | None = None):
        self.output_dir = output_dir
        self.plots_dir = plots_dir or output_dir / "plots"
        self.logs_dir = logs_dir or output_dir / "logs"
        self.pseudo_labels_dir = pseudo_labels_dir or output_dir / "pseudo_labels"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self, log_path: Optional[Path] = None) -> None:
        log_path = log_path or (self.logs_dir / "train_log.csv")
        if not log_path.exists():
            print(f"No log found for plotting: {log_path}")
            return
        df = pd.read_csv(log_path)
        if df.empty:
            return
        x = df["epoch"].astype(float)
        stage = df["stage"].astype(str)

        plt.figure(figsize=(10, 6))
        for name in ["train_loss", "valid_loss"]:
            if name in df.columns:
                for stage_name, part in df.groupby("stage"):
                    plt.plot(part["epoch"], part[name], marker="o", label=f"{stage_name} {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "loss_curves.png", dpi=160)
        plt.close()

        metric_cols = [c for c in ["macro_auc", "macro_f1", "micro_f1", "binary_accuracy", "positive_precision", "positive_recall"] if c in df.columns]
        if metric_cols:
            plt.figure(figsize=(10, 6))
            for name in metric_cols:
                for stage_name, part in df.groupby("stage"):
                    plt.plot(part["epoch"], part[name], marker="o", label=f"{stage_name} {name}")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.title("Validation Metrics")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "validation_metrics.png", dpi=160)
            plt.close()

        if "lr" in df.columns:
            plt.figure(figsize=(10, 4))
            for stage_name, part in df.groupby("stage"):
                plt.plot(part["epoch"], part["lr"], marker="o", label=stage_name)
            plt.xlabel("Epoch")
            plt.ylabel("Learning rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "lr_curve.png", dpi=160)
            plt.close()

    def plot_class_distribution(self, train_df: pd.DataFrame) -> None:
        if train_df.empty or "primary_label" not in train_df.columns:
            return
        counts = train_df["primary_label"].value_counts().head(40).sort_values()
        plt.figure(figsize=(10, 10))
        counts.plot(kind="barh")
        plt.xlabel("Samples")
        plt.ylabel("Primary label")
        plt.title("Top 40 Primary Label Counts")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "class_distribution_top40.png", dpi=160)
        plt.close()

    def plot_pseudo_stats(self, stats_path: Path) -> None:
        if not stats_path.exists():
            return
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        top = stats.get("top_predicted_classes", {})
        if not top:
            return
        series = pd.Series(top).sort_values()
        plt.figure(figsize=(10, max(5, len(series) * 0.25)))
        series.plot(kind="barh")
        plt.xlabel("Mean pseudo probability")
        plt.ylabel("Class")
        plt.title(f"Top Pseudo-label Classes ({stats_path.stem})")
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{stats_path.stem}_top_classes.png", dpi=160)
        plt.close()

    def plot_all_pseudo_stats(self) -> None:
        for stats_path in sorted(self.pseudo_labels_dir.glob("*.stats.json")):
            self.plot_pseudo_stats(stats_path)

    def render_all(self, train_df: Optional[pd.DataFrame] = None) -> None:
        self.plot_training_curves()
        self.plot_all_pseudo_stats()
        if train_df is not None:
            self.plot_class_distribution(train_df)
        print(f"Plots saved to {self.plots_dir}")


