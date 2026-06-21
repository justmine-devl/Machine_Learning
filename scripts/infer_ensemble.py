#!/usr/bin/env python
"""Run OpenVINO ensemble inference on a directory of audio files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from _bootstrap import add_project_src_to_path
add_project_src_to_path()

from bioacoustic.openvino_ensemble import OpenVINOEnsemble, find_openvino_models, infer_audio_file_openvino
from bioacoustic.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVINO ensemble inference.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/inference/predictions.csv")
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    ens_cfg = cfg.get("ensemble", {})
    spec_cfg = cfg.get("spectrogram", {})

    audio_dir = Path(args.audio_dir or data_cfg.get("test_audio_dir", "data/test_audio"))
    models_dir = Path(args.models_dir)
    classes_path = Path(args.classes or data_cfg.get("classes_path", "outputs/processed/classes.txt"))
    classes = [x.strip() for x in classes_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    regular_paths = find_openvino_models(models_dir, prefix=ens_cfg.get("regular_prefix", "model_"))
    shifted_paths = find_openvino_models(models_dir, prefix=ens_cfg.get("shifted_prefix", "model2_"))
    if not regular_paths:
        raise FileNotFoundError(f"No regular OpenVINO model XML files found in {models_dir}")

    regular_models = OpenVINOEnsemble(regular_paths, device=ens_cfg.get("device", "CPU"))
    shifted_models = OpenVINOEnsemble(shifted_paths, device=ens_cfg.get("device", "CPU")) if shifted_paths else None

    files = sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in {".ogg", ".wav", ".mp3", ".flac"}])
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    sample_rate = int(cfg.get("sample_rate", 32000))
    duration = float(cfg.get("clip_duration", 5.0))
    rows = []
    for path in tqdm(files, desc="ensemble inference"):
        probs = infer_audio_file_openvino(
            path,
            regular_models,
            shifted_models,
            sample_rate=sample_rate,
            duration=duration,
            shift=float(ens_cfg.get("shift", 2.5)),
            blend_alpha=float(ens_cfg.get("blend_alpha", 0.5)),
            smooth=bool(ens_cfg.get("temporal_smoothing", True)),
            spectrogram_kwargs=spec_cfg,
        )
        for i, prob in enumerate(probs):
            row = {"row_id": f"{path.stem}_{int((i + 1) * duration)}"}
            row.update({cls: float(prob[j]) for j, cls in enumerate(classes)})
            rows.append(row)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
