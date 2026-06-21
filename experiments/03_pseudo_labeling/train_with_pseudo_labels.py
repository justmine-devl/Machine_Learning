from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.config import get_arg_parser, load_config
from experiments.common.dataset import AudioClipDataset, PseudoLabelDataset, add_stratified_folds, build_class_list, load_classes
from experiments.common.losses import WeightedFocalBCELoss
from experiments.common.models import ModelConfig, build_model
from experiments.common.spectrogram import SpectrogramConfig
from experiments.common.training import run_training_loop
from experiments.common.utils import get_device, seed_everything


def main() -> None:
    args = get_arg_parser("Train model using labeled data and pseudo-labels").parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    df = pd.read_csv(cfg["metadata_csv"])
    if "fold" not in df.columns:
        df = add_stratified_folds(df, cfg.get("n_folds", 5), cfg.get("seed", 42), cfg.get("primary_col", "primary_label"))
    if Path(cfg.get("class_file", "outputs/classes.txt")).exists():
        classes = load_classes(cfg.get("class_file", "outputs/classes.txt"))
    else:
        classes = build_class_list(df, cfg.get("primary_col", "primary_label"), cfg.get("secondary_col", "secondary_labels"))

    fold = cfg.get("fold", 0)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    spec_cfg = SpectrogramConfig(
        sample_rate=cfg.get("sample_rate", 32000), n_fft=cfg.get("n_fft", 2048), hop_length=cfg.get("hop_length", 768),
        win_length=cfg.get("win_length", 2048), n_mels=cfg.get("n_mels", 192), f_min=cfg.get("f_min", 50), f_max=cfg.get("f_max", 15000)
    )

    labeled_ds = AudioClipDataset(train_df, cfg["audio_dir"], classes, spec_cfg, mode="train", clip_duration=cfg.get("clip_duration", 5.0))
    valid_ds = AudioClipDataset(valid_df, cfg["audio_dir"], classes, spec_cfg, mode="valid", clip_duration=cfg.get("clip_duration", 5.0))
    pseudo_ds = PseudoLabelDataset(cfg.get("pseudo_csv", "outputs/pseudo_labeling/pseudo_labels.csv"))
    train_ds = ConcatDataset([labeled_ds, pseudo_ds])

    pseudo_ratio = cfg.get("pseudo_sampling_ratio", 0.4)
    labeled_weight = (1 - pseudo_ratio) / max(1, len(labeled_ds))
    pseudo_weight = pseudo_ratio / max(1, len(pseudo_ds))
    weights = torch.tensor([labeled_weight] * len(labeled_ds) + [pseudo_weight] * len(pseudo_ds), dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 12), sampler=sampler, num_workers=cfg.get("num_workers", 2), pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.get("batch_size", 12), shuffle=False, num_workers=cfg.get("num_workers", 2), pin_memory=True)

    model = build_model(cfg.get("model_type", "sed"), ModelConfig(backbone=cfg.get("backbone", "tf_efficientnetv2_s.in21k"), num_classes=len(classes), pretrained=cfg.get("pretrained", True), dropout=cfg.get("dropout", 0.2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 3e-4), weight_decay=cfg.get("weight_decay", 1e-5))
    criterion = WeightedFocalBCELoss(gamma=cfg.get("focal_gamma", 2.0), label_smoothing=cfg.get("label_smoothing", 0.005))
    run_training_loop(model, train_loader, valid_loader, optimizer, criterion, device, cfg.get("output_dir", "outputs/pseudo_labeling"), epochs=cfg.get("pseudo_epochs", 3), pos_weight=None)


if __name__ == "__main__":
    main()
