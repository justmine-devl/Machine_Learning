#!/usr/bin/env python
"""Train a student model with hard labels and soft teacher targets.

This script supports both Noisy Student and self-distillation experiments.
It expects soft targets aligned with the training metadata rows. If soft targets
are not available, use `generate_pseudo_labels.py` or a custom OOF teacher pipeline first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from _bootstrap import add_project_src_to_path
add_project_src_to_path()

from bioacoustic.dataset import BirdAudioDataset, build_class_list, read_metadata
from bioacoustic.distillation import load_soft_targets
from bioacoustic.losses import student_distillation_loss
from bioacoustic.models import build_model
from bioacoustic.training import save_checkpoint, validate
from bioacoustic.utils import ensure_dir, get_device, load_config, save_json, seed_everything


class StudentDataset(Dataset):
    def __init__(self, base: BirdAudioDataset, soft_targets: np.ndarray) -> None:
        self.base = base
        self.soft_targets = soft_targets.astype(np.float32)
        if len(self.base) != len(self.soft_targets):
            raise ValueError(f"Base dataset length {len(self.base)} != soft target length {len(self.soft_targets)}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, torch.from_numpy(self.soft_targets[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train student model with soft teacher targets.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--soft-targets", type=str, required=True)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    spec_cfg = cfg.get("spectrogram", {})
    distill_cfg = cfg.get("distillation", {})

    metadata_path = Path(args.metadata or data_cfg.get("metadata_path", "outputs/processed/metadata_with_folds.csv"))
    audio_dir = Path(args.audio_dir or data_cfg.get("audio_dir", "data/train_audio"))
    out_dir = ensure_dir(args.out_dir or cfg.get("output_dir", "outputs/student"))
    fold = int(args.fold if args.fold is not None else train_cfg.get("fold", 0))

    df = read_metadata(metadata_path)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    soft_targets = load_soft_targets(args.soft_targets)
    if len(soft_targets) == len(df):
        soft_targets = soft_targets[df["fold"] != fold]

    primary_col = data_cfg.get("primary_col", "primary_label")
    filename_col = data_cfg.get("filename_col", "filename")
    classes = build_class_list(df, primary_col=primary_col)
    dataset_kwargs = dict(
        audio_dir=audio_dir,
        classes=classes,
        filename_col=filename_col,
        primary_col=primary_col,
        secondary_col=data_cfg.get("secondary_col", "secondary_labels"),
        sample_rate=int(cfg.get("sample_rate", 32000)),
        duration=float(cfg.get("clip_duration", 5.0)),
        include_secondary=bool(data_cfg.get("include_secondary", True)),
        spectrogram_kwargs=spec_cfg,
    )
    train_base = BirdAudioDataset(train_df, train=True, **dataset_kwargs)
    valid_ds = BirdAudioDataset(valid_df, train=False, **dataset_kwargs)
    train_ds = StudentDataset(train_base, soft_targets)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(train_cfg.get("valid_batch_size", train_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
    )

    device = get_device(bool(train_cfg.get("use_cuda", True)))
    model = build_model(
        model_cfg.get("type", "sed"),
        num_classes=len(classes),
        backbone=model_cfg.get("backbone", "tf_efficientnet_b0_ns"),
        in_channels=int(model_cfg.get("in_channels", 1)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    def loss_fn(logits, hard, soft):
        return student_distillation_loss(
            logits,
            hard,
            soft,
            hard_weight=float(distill_cfg.get("hard_weight", 0.6)),
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )

    best_auc = -1.0
    history = []
    for epoch in range(1, int(train_cfg.get("epochs", 10)) + 1):
        model.train()
        losses = []
        for x, hard, soft in train_loader:
            x = x.to(device).float()
            hard = hard.to(device).float()
            soft = soft.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)["clip_logits"]
            loss = loss_fn(logits, hard, soft)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_result = validate(model, valid_loader, loss_fn=None, device=device)
        metrics = valid_result["metrics"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(row)
        auc = float(metrics.get("macro_auc", float("nan")))
        if auc == auc and auc > best_auc:
            best_auc = auc
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch=epoch, metrics=metrics)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch=epoch, metrics=metrics)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    save_json({"best_macro_auc": best_auc, "fold": fold}, out_dir / "summary.json")


if __name__ == "__main__":
    main()
