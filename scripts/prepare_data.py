#!/usr/bin/env python
"""Prepare metadata, class list, and fold assignment.

This script supports the report's Dataset and Exploratory Analysis section.
It does not move or copy raw audio files. It only creates clean metadata files
that all experiments can share.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_project_src_to_path
add_project_src_to_path()

from bioacoustic.dataset import add_stratified_folds, build_class_list, read_metadata
from bioacoustic.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare metadata and folds.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--metadata", type=str, default=None, help="Override metadata CSV path.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for processed metadata.")
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    metadata_path = Path(args.metadata or data_cfg.get("metadata_path", "data/train_metadata.csv"))
    out_dir = ensure_dir(args.out_dir or data_cfg.get("processed_dir", "outputs/processed"))
    n_splits = int(args.n_splits or train_cfg.get("n_splits", 5))
    seed = int(args.seed or cfg.get("seed", 42))
    primary_col = data_cfg.get("primary_col", "primary_label")

    df = read_metadata(metadata_path)
    if primary_col not in df.columns:
        raise ValueError(f"Column '{primary_col}' was not found in {metadata_path}")

    classes = build_class_list(df, primary_col=primary_col)
    folded = add_stratified_folds(df, n_splits=n_splits, label_col=primary_col, seed=seed)

    folded_path = out_dir / "metadata_with_folds.csv"
    classes_path = out_dir / "classes.txt"
    summary_path = out_dir / "dataset_summary.csv"

    folded.to_csv(folded_path, index=False)
    classes_path.write_text("\n".join(classes) + "\n", encoding="utf-8")

    summary = (
        folded[primary_col]
        .astype(str)
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved fold metadata: {folded_path}")
    print(f"Saved class list:     {classes_path}")
    print(f"Saved summary:        {summary_path}")
    print(f"Rows: {len(folded):,} | Classes: {len(classes):,} | Folds: {n_splits}")


if __name__ == "__main__":
    main()
