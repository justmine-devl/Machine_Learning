#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bioacoustic_sed.audio import AudioConfig, build_regular_and_shifted_specs
from bioacoustic_sed.metrics import compute_metrics, per_class_metrics, search_best_threshold
from bioacoustic_sed.openvino_ensemble import (
    aggregate_chunks,
    load_openvino_models,
    predict_audio_ensemble,
)
from bioacoustic_sed.utils import ensure_dir, find_audio_path, load_config, read_class_list, seed_everything, write_class_list


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenVINO SED ensemble on labeled validation audio.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--metadata", required=True, help="CSV with filename and primary_label columns")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--out-dir", default="experiments/results")
    parser.add_argument("--class-list", default=None, help="Optional class list file. If omitted, inferred from metadata.")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--model-limit", type=int, default=None)
    return parser.parse_args()


def make_targets(labels: list[str], class_names: list[str]) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.zeros((len(labels), len(class_names)), dtype=np.int32)
    for i, label in enumerate(labels):
        if label in class_to_idx:
            y[i, class_to_idx[label]] = 1
    return y


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    out_dir = ensure_dir(args.out_dir)
    metadata = pd.read_csv(args.metadata)
    required_cols = {"filename", "primary_label"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    if args.max_files is not None:
        metadata = metadata.sample(min(args.max_files, len(metadata)), random_state=cfg.get("seed", 42)).reset_index(drop=True)
    elif cfg.get("evaluation", {}).get("max_files") is not None:
        max_files = int(cfg["evaluation"]["max_files"])
        metadata = metadata.sample(min(max_files, len(metadata)), random_state=cfg.get("seed", 42)).reset_index(drop=True)

    if args.class_list:
        class_names = read_class_list(args.class_list)
    else:
        class_names = sorted(metadata["primary_label"].dropna().unique().tolist())
        write_class_list(class_names, out_dir / "classes.txt")

    audio_cfg = AudioConfig(**cfg["audio"])
    ens_cfg = cfg.get("ensemble", {})
    model_limit = args.model_limit if args.model_limit is not None else ens_cfg.get("model_limit")

    print("Loading OpenVINO regular models...")
    regular_models = load_openvino_models(
        args.models_dir,
        prefix=ens_cfg.get("regular_prefix", "model_"),
        device=args.device,
        model_limit=model_limit,
    )

    print("Loading OpenVINO shifted models...")
    shifted_models = load_openvino_models(
        args.models_dir,
        prefix=ens_cfg.get("shifted_prefix", "model2_"),
        device=args.device,
        model_limit=model_limit,
    )

    y_pred_rows = []
    used_rows = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Evaluating"):
        filename = str(row["filename"])
        label = str(row["primary_label"])
        try:
            audio_path = find_audio_path(args.audio_dir, filename, label=label)
            regular_specs, shifted_specs, _, _ = build_regular_and_shifted_specs(audio_path, audio_cfg)
            chunk_preds = predict_audio_ensemble(
                regular_models=regular_models,
                shifted_models=shifted_models,
                regular_specs=regular_specs,
                shifted_specs=shifted_specs,
                regular_weight=ens_cfg.get("regular_weight", 0.50),
                shifted_left_weight=ens_cfg.get("shifted_left_weight", 0.25),
                shifted_right_weight=ens_cfg.get("shifted_right_weight", 0.25),
                smoothing_weight=ens_cfg.get("smoothing_weight", 0.10),
            )
            audio_pred = aggregate_chunks(chunk_preds, method=ens_cfg.get("aggregate_method", "max"))
            y_pred_rows.append(audio_pred)
            used_rows.append(row)
        except Exception as exc:
            print(f"[WARN] Skipped {filename}: {exc}")

    if not y_pred_rows:
        raise RuntimeError("No valid predictions were produced.")

    used = pd.DataFrame(used_rows).reset_index(drop=True)
    y_pred = np.stack(y_pred_rows).astype(np.float32)
    y_true = make_targets(used["primary_label"].astype(str).tolist(), class_names)

    eval_cfg = cfg.get("evaluation", {})
    threshold = float(eval_cfg.get("threshold", 0.5))

    if eval_cfg.get("search_threshold", True):
        best_threshold, threshold_curve = search_best_threshold(y_true, y_pred)
        threshold_curve.to_csv(out_dir / "threshold_curve.csv", index=False)
        threshold = best_threshold

    metrics = compute_metrics(y_true, y_pred, threshold=threshold)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics_summary.csv", index=False)

    pred_df = pd.DataFrame(y_pred, columns=class_names)
    pred_df.insert(0, "filename", used["filename"].values)
    pred_df.insert(1, "primary_label", used["primary_label"].values)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    per_class = per_class_metrics(y_true, y_pred, class_names)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)

    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_pred.npy", y_pred)

    print("\nMetrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
