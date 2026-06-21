#!/usr/bin/env python
"""Train the EfficientNet-based SED model with temporal attention pooling."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _bootstrap import add_project_src_to_path
add_project_src_to_path()

from bioacoustic.dataset import BirdAudioDataset, build_class_list, encode_multihot, make_label_map, read_metadata
from bioacoustic.losses import compute_pos_weight, focal_bce_loss, weighted_bce_loss
from bioacoustic.models import build_model
from bioacoustic.training import save_checkpoint, train_one_epoch, validate
from bioacoustic.utils import ensure_dir, get_device, load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet SED model.")
    parser.add_argument("--config", type=str, default="config.yaml")
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

    metadata_path = Path(args.metadata or data_cfg.get("metadata_path", "outputs/processed/metadata_with_folds.csv"))
    audio_dir = Path(args.audio_dir or data_cfg.get("audio_dir", "data/train_audio"))
    out_dir = ensure_dir(args.out_dir or cfg.get("output_dir", "outputs/sed"))
    fold = int(args.fold if args.fold is not None else train_cfg.get("fold", 0))

    df = read_metadata(metadata_path)
    if "fold" not in df.columns:
        raise ValueError("Metadata must contain a 'fold' column. Run scripts/prepare_data.py first.")

    primary_col = data_cfg.get("primary_col", "primary_label")
    secondary_col = data_cfg.get("secondary_col", "secondary_labels")
    filename_col = data_cfg.get("filename_col", "filename")
    classes = build_class_list(df, primary_col=primary_col)
    label_map = make_label_map(classes)

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    dataset_kwargs = dict(
        audio_dir=audio_dir,
        classes=classes,
        filename_col=filename_col,
        primary_col=primary_col,
        secondary_col=secondary_col,
        sample_rate=int(cfg.get("sample_rate", 32000)),
        duration=float(cfg.get("clip_duration", 5.0)),
        include_secondary=bool(data_cfg.get("include_secondary", True)),
        spectrogram_kwargs=spec_cfg,
    )
    train_ds = BirdAudioDataset(train_df, train=True, **dataset_kwargs)
    valid_ds = BirdAudioDataset(valid_df, train=False, **dataset_kwargs)

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
        "sed",
        num_classes=len(classes),
        backbone=model_cfg.get("backbone", "tf_efficientnet_b0_ns"),
        in_channels=int(model_cfg.get("in_channels", 1)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)

    target_rows = [
        encode_multihot(
            row[primary_col],
            row.get(secondary_col, None),
            label_map,
            include_secondary=bool(data_cfg.get("include_secondary", True)),
        )
        for _, row in train_df.iterrows()
    ]
    pos_weight = compute_pos_weight(torch.tensor(target_rows, dtype=torch.float32)).to(device) if train_cfg.get("use_pos_weight", True) else None

    loss_name = train_cfg.get("loss", "weighted_focal_bce").lower()
    if loss_name in {"weighted_bce", "wbce"}:
        loss_fn = lambda logits, targets: weighted_bce_loss(
            logits, targets, pos_weight=pos_weight, label_smoothing=float(train_cfg.get("label_smoothing", 0.0))
        )
    else:
        loss_fn = lambda logits, targets: focal_bce_loss(
            logits,
            targets,
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            pos_weight=pos_weight,
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    history = []
    best_auc = -1.0
    for epoch in range(1, int(train_cfg.get("epochs", 10)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device=device)
        valid_result = validate(model, valid_loader, loss_fn, device=device)
        metrics = valid_result["metrics"]
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_result["loss"], **metrics}
        history.append(row)
        print(row)
        auc = float(metrics.get("macro_auc", float("nan")))
        if auc == auc and auc > best_auc:
            best_auc = auc
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch=epoch, metrics=metrics)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch=epoch, metrics=metrics)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    save_json({"best_macro_auc": best_auc, "fold": fold, "num_classes": len(classes)}, out_dir / "summary.json")


if __name__ == "__main__":
    main()
