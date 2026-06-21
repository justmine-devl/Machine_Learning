"""Plotting helpers for experiment reports."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_curve(df: pd.DataFrame, x: str, y: str, out_path: str | Path, group: str | None = None, title: str | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    if group and group in df.columns:
        for name, g in df.groupby(group):
            plt.plot(g[x], g[y], marker="o", label=str(name))
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(df: pd.DataFrame, x: str, y: str, out_path: str | Path, title: str | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar(df[x].astype(str), df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
