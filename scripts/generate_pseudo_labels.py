#!/usr/bin/env python
"""Generate pseudo labels from a trained teacher model.

The output is a CSV containing row_id and soft probabilities for selected chunks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from _bootstrap import add_project_src_to_path
add_project_src_to_path()

from bioacoustic.audio import load_audio, make_regular_windows
from bioacoustic.models import build_model
from bioacoustic.pseudo_labeling import save_pseudo_labels, select_pseudo_labels
from bioacoustic.spectrogram import batch_log_mel
from bioacoustic.training import load_checkpoint
from bioacoustic.utils import ensure_dir, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo labels using a teacher model.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio-dir", type=str, default=None, help="Unlabeled soundscape directory.")
    parser.add_argument("--classes", type=str, default=None, help="classes.txt path.")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    pseudo_cfg = cfg.get("pseudo_labeling", {})
    spec_cfg = cfg.get("spectrogram", {})
    audio_cfg = cfg.get("audio", {})

    audio_dir = Path(args.audio_dir or data_cfg.get("unlabeled_audio_dir", "data/unlabeled_audio"))
    classes_path = Path(args.classes or data_cfg.get("classes_path", "outputs/processed/classes.txt"))
    out_path = Path(args.out or pseudo_cfg.get("output_path", "outputs/pseudo/pseudo_labels.csv"))
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    device = get_device(bool(cfg.get("training", {}).get("use_cuda", True)))
    model = build_model(
        model_cfg.get("type", "sed"),
        num_classes=len(classes),
        backbone=model_cfg.get("backbone", "tf_efficientnet_b0_ns"),
        in_channels=int(model_cfg.get("in_channels", 1)),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    files = sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in {".ogg", ".wav", ".mp3", ".flac"}])
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    all_preds, row_ids, audio_paths, chunk_indices = [], [], [], []
    sample_rate = int(audio_cfg.get("sample_rate", 32000))
    duration = float(audio_cfg.get("clip_duration", 5.0))
    batch_size = int(pseudo_cfg.get("batch_size", 32))

    with torch.no_grad():
        for path in tqdm(files, desc="pseudo-label inference"):
            wav = load_audio(path, sample_rate=sample_rate)
            windows = make_regular_windows(wav, sample_rate=sample_rate, duration=duration)
            specs = batch_log_mel(windows, sample_rate=sample_rate, **spec_cfg)
            preds = []
            for i in range(0, len(specs), batch_size):
                x = torch.tensor(specs[i : i + batch_size], dtype=torch.float32, device=device)
                logits = model(x)["clip_logits"]
                preds.append(torch.sigmoid(logits).cpu().numpy())
            probs = np.concatenate(preds, axis=0)
            all_preds.append(probs)
            stem = path.stem
            for i in range(len(probs)):
                row_ids.append(f"{stem}_{int((i + 1) * duration)}")
                audio_paths.append(str(path.resolve()))
                chunk_indices.append(i)

    predictions = np.concatenate(all_preds, axis=0)
    pseudo_df = select_pseudo_labels(
        predictions,
        row_ids=row_ids,
        min_max_prob=float(pseudo_cfg.get("min_max_prob", 0.5)),
        class_prob_threshold=float(pseudo_cfg.get("class_prob_threshold", 0.1)),
    )
    # Rename class columns for readability.
    rename = {i: cls for i, cls in enumerate(classes)}
    pseudo_df = pseudo_df.rename(columns=rename)
    window_index = pd.DataFrame({
        "row_id": row_ids,
        "audio_path": audio_paths,
        "chunk_index": chunk_indices,
    })
    pseudo_df = window_index.merge(pseudo_df, on="row_id", how="inner", validate="one_to_one")
    save_pseudo_labels(pseudo_df, out_path)
    print(f"Saved pseudo labels: {out_path}")
    print(f"Selected {len(pseudo_df):,} / {len(predictions):,} chunks")


if __name__ == "__main__":
    main()
