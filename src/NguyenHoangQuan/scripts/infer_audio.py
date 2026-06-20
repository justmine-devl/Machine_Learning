#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bioacoustic_sed.audio import AudioConfig, build_regular_and_shifted_specs
from bioacoustic_sed.openvino_ensemble import aggregate_chunks, load_openvino_models, predict_audio_ensemble
from bioacoustic_sed.utils import load_config, read_class_list


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenVINO ensemble inference on an unlabeled audio folder.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--class-list", required=True)
    parser.add_argument("--out", default="experiments/results/predictions.csv")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--model-limit", type=int, default=None)
    parser.add_argument("--extensions", default=".ogg,.wav,.mp3,.flac")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    audio_cfg = AudioConfig(**cfg["audio"])
    ens_cfg = cfg.get("ensemble", {})
    class_names = read_class_list(args.class_list)

    model_limit = args.model_limit if args.model_limit is not None else ens_cfg.get("model_limit")
    regular_models = load_openvino_models(args.models_dir, ens_cfg.get("regular_prefix", "model_"), args.device, model_limit)
    shifted_models = load_openvino_models(args.models_dir, ens_cfg.get("shifted_prefix", "model2_"), args.device, model_limit)

    exts = tuple(e.strip().lower() for e in args.extensions.split(","))
    audio_paths = sorted([p for p in Path(args.audio_dir).rglob("*") if p.suffix.lower() in exts])
    if not audio_paths:
        raise FileNotFoundError(f"No audio files found in {args.audio_dir}")

    rows = []
    for path in tqdm(audio_paths, desc="Inference"):
        regular_specs, shifted_specs, _, _ = build_regular_and_shifted_specs(path, audio_cfg)
        chunk_preds = predict_audio_ensemble(
            regular_models,
            shifted_models,
            regular_specs,
            shifted_specs,
            regular_weight=ens_cfg.get("regular_weight", 0.50),
            shifted_left_weight=ens_cfg.get("shifted_left_weight", 0.25),
            shifted_right_weight=ens_cfg.get("shifted_right_weight", 0.25),
            smoothing_weight=ens_cfg.get("smoothing_weight", 0.10),
        )
        audio_pred = aggregate_chunks(chunk_preds, method=ens_cfg.get("aggregate_method", "max"))
        row = {"filename": path.name}
        row.update({cls: float(score) for cls, score in zip(class_names, audio_pred)})
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
