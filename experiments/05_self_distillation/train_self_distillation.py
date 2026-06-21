from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.config import get_arg_parser, load_config
from experiments.common.dataset import AudioClipDataset, add_stratified_folds, build_class_list, load_classes
from experiments.common.losses import StudentDistillationLoss
from experiments.common.metrics import compute_all_metrics
from experiments.common.models import ModelConfig, build_model
from experiments.common.spectrogram import SpectrogramConfig
from experiments.common.training import predict_loader
from experiments.common.utils import get_device, save_checkpoint, seed_everything


class DistillationDataset(Dataset):
    def __init__(self, base_ds: AudioClipDataset, soft_targets: np.ndarray) -> None:
        self.base_ds = base_ds
        if len(base_ds) != len(soft_targets):
            raise ValueError(f"base dataset length {len(base_ds)} != soft target length {len(soft_targets)}")
        self.soft_targets = soft_targets.astype(np.float32)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        q = torch.from_numpy(self.soft_targets[idx]).float()
        return x, y, q


def train_one_epoch_distill(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y, q in tqdm(loader, desc="train-distill", leave=False):
        x, y, q = x.to(device), y.to(device), q.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)["clip_logits"]
        loss = criterion(logits, y, q, pos_weight=None)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main() -> None:
    args = get_arg_parser("Train self-distillation student").parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    output_dir = Path(cfg.get("output_dir", "outputs/self_distillation")); output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["metadata_csv"])
    if "fold" not in df.columns:
        df = add_stratified_folds(df, cfg.get("n_folds", 5), cfg.get("seed", 42))
    classes = load_classes(cfg["class_file"]) if Path(cfg["class_file"]).exists() else build_class_list(df)
    fold = cfg.get("fold", 0)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    spec_cfg = SpectrogramConfig(sample_rate=cfg.get("sample_rate", 32000), n_fft=cfg.get("n_fft", 2048), hop_length=cfg.get("hop_length", 768), win_length=cfg.get("win_length", 2048), n_mels=cfg.get("n_mels", 192), f_min=cfg.get("f_min", 50), f_max=cfg.get("f_max", 15000))
    base_train = AudioClipDataset(train_df, cfg["audio_dir"], classes, spec_cfg, mode="train", clip_duration=cfg.get("clip_duration", 5.0))
    valid_ds = AudioClipDataset(valid_df, cfg["audio_dir"], classes, spec_cfg, mode="valid", clip_duration=cfg.get("clip_duration", 5.0))
    soft_targets = np.load(cfg["soft_targets_npy"])
    train_ds = DistillationDataset(base_train, soft_targets)

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 12), shuffle=True, num_workers=cfg.get("num_workers", 2), pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.get("batch_size", 12), shuffle=False, num_workers=cfg.get("num_workers", 2), pin_memory=True)
    model = build_model(cfg.get("model_type", "sed"), ModelConfig(backbone=cfg.get("backbone", "tf_efficientnetv2_s.in21k"), num_classes=len(classes), pretrained=cfg.get("pretrained", True), dropout=cfg.get("dropout", 0.2))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 3e-4), weight_decay=cfg.get("weight_decay", 1e-5))
    criterion = StudentDistillationLoss(hard_weight=cfg.get("hard_loss_weight", 0.6), gamma=cfg.get("focal_gamma", 2.0), label_smoothing=cfg.get("label_smoothing", 0.005))

    rows = []
    best_auc = -1
    for epoch in range(1, cfg.get("epochs", 5) + 1):
        train_loss = train_one_epoch_distill(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = predict_loader(model, valid_loader, device)
        metrics = compute_all_metrics(y_true, y_pred, threshold=0.5)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        rows.append(row)
        print(row)
        save_checkpoint(model, output_dir / "last.pt", epoch=epoch, **metrics)
        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            save_checkpoint(model, output_dir / "best.pt", epoch=epoch, **metrics)
        pd.DataFrame(rows).to_csv(output_dir / "history.csv", index=False)


if __name__ == "__main__":
    main()
