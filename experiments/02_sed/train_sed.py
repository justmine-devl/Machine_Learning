from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.config import get_arg_parser, load_config
from experiments.common.dataset import AudioClipDataset, add_stratified_folds, build_class_list, save_classes
from experiments.common.losses import WeightedFocalBCELoss, compute_pos_weight
from experiments.common.models import ModelConfig, build_model
from experiments.common.spectrogram import SpectrogramConfig
from experiments.common.training import run_training_loop
from experiments.common.utils import get_device, seed_everything


def main() -> None:
    args = get_arg_parser("Train EfficientNet-based SED model").parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    df = pd.read_csv(cfg["metadata_csv"])
    if "fold" not in df.columns:
        df = add_stratified_folds(df, cfg.get("n_folds", 5), cfg.get("seed", 42), cfg.get("primary_col", "primary_label"))

    classes = build_class_list(df, cfg.get("primary_col", "primary_label"), cfg.get("secondary_col", "secondary_labels"))
    save_classes(classes, cfg.get("class_file", "outputs/classes.txt"))
    num_classes = len(classes)

    fold = cfg.get("fold", 0)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    spec_cfg = SpectrogramConfig(
        sample_rate=cfg.get("sample_rate", 32000),
        n_fft=cfg.get("n_fft", 2048),
        hop_length=cfg.get("hop_length", 768),
        win_length=cfg.get("win_length", 2048),
        n_mels=cfg.get("n_mels", 192),
        f_min=cfg.get("f_min", 50),
        f_max=cfg.get("f_max", 15000),
    )

    train_ds = AudioClipDataset(
        train_df,
        cfg["audio_dir"],
        classes,
        spec_cfg,
        filename_col=cfg.get("filename_col", "filename"),
        primary_col=cfg.get("primary_col", "primary_label"),
        secondary_col=cfg.get("secondary_col", "secondary_labels"),
        mode="train",
        clip_duration=cfg.get("clip_duration", 5.0),
        use_secondary=cfg.get("use_secondary_labels", True),
    )
    valid_ds = AudioClipDataset(
        valid_df,
        cfg["audio_dir"],
        classes,
        spec_cfg,
        filename_col=cfg.get("filename_col", "filename"),
        primary_col=cfg.get("primary_col", "primary_label"),
        secondary_col=cfg.get("secondary_col", "secondary_labels"),
        mode="valid",
        clip_duration=cfg.get("clip_duration", 5.0),
        use_secondary=cfg.get("use_secondary_labels", True),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16), shuffle=True, num_workers=cfg.get("num_workers", 2), pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.get("batch_size", 16), shuffle=False, num_workers=cfg.get("num_workers", 2), pin_memory=True)

    model = build_model("sed", ModelConfig(
        backbone=cfg.get("backbone", "tf_efficientnet_b0_ns"),
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.2),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 5e-4), weight_decay=cfg.get("weight_decay", 1e-5))
    criterion = WeightedFocalBCELoss(gamma=cfg.get("focal_gamma", 2.0), label_smoothing=cfg.get("label_smoothing", 0.0))

    all_targets = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    pos_weight = compute_pos_weight(all_targets, max_weight=cfg.get("max_pos_weight", 20.0))

    run_training_loop(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=cfg.get("output_dir", "outputs/sed"),
        epochs=cfg.get("epochs", 10),
        pos_weight=pos_weight,
    )


if __name__ == "__main__":
    main()
