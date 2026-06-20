#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bioacoustic_sed.audio import AudioConfig, batch_log_mel, load_audio, make_windows
from bioacoustic_sed.losses import focal_bce_loss
from bioacoustic_sed.metrics import compute_metrics
from bioacoustic_sed.model import create_model
from bioacoustic_sed.utils import ensure_dir, find_audio_path, load_config, seed_everything, write_class_list


class AudioClipDataset(Dataset):
    """Compact dataset: one random 5-second crop per audio file."""

    def __init__(self, df: pd.DataFrame, audio_dir: str, class_names: list[str], audio_cfg: AudioConfig):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.audio_cfg = audio_cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        label = str(row["primary_label"])
        path = find_audio_path(self.audio_dir, filename, label=label)
        wav = load_audio(path, self.audio_cfg.sample_rate)
        windows, _ = make_windows(wav, self.audio_cfg.sample_rate, self.audio_cfg.window_seconds)
        window = windows[np.random.randint(len(windows))]
        spec = batch_log_mel(window[None, :], self.audio_cfg)[0]
        target = np.zeros(len(self.class_names), dtype=np.float32)
        if label in self.class_to_idx:
            target[self.class_to_idx[label]] = 1.0
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a compact single EfficientNet SED model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out-dir", default="experiments/results/checkpoints")
    parser.add_argument("--valid-frac", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    out_dir = ensure_dir(args.out_dir)

    df = pd.read_csv(args.metadata)
    class_names = sorted(df["primary_label"].dropna().unique().tolist())
    write_class_list(class_names, out_dir / "classes.txt")

    df = df.sample(frac=1, random_state=cfg.get("seed", 42)).reset_index(drop=True)
    n_valid = int(len(df) * args.valid_frac)
    valid_df = df.iloc[:n_valid]
    train_df = df.iloc[n_valid:]

    audio_cfg = AudioConfig(**cfg["audio"])
    train_ds = AudioClipDataset(train_df, args.audio_dir, class_names, audio_cfg)
    valid_ds = AudioClipDataset(valid_df, args.audio_dir, class_names, audio_cfg)

    train_cfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(cfg["model"]["backbone"], len(class_names), pretrained=cfg["model"].get("pretrained", True)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])

    best_f1 = -1.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)["clip_logits"]
            loss = focal_bce_loss(
                logits,
                y,
                gamma=train_cfg.get("focal_gamma", 2.0),
                label_smoothing=train_cfg.get("label_smoothing", 0.0),
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"Epoch {epoch} valid"):
                x = x.to(device)
                logits = model(x)["clip_logits"]
                preds.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(y.numpy())

        y_pred = np.concatenate(preds, axis=0)
        y_true = np.concatenate(targets, axis=0)
        metrics = compute_metrics(y_true, y_pred, threshold=cfg["evaluation"].get("threshold", 0.5))
        mean_loss = float(np.mean(train_losses))
        print(f"Epoch {epoch}: loss={mean_loss:.5f}, macro_f1={metrics['macro_f1']:.5f}, macro_auc={metrics['macro_auc']:.5f}")

        torch.save(model.state_dict(), out_dir / "last_model.pth")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pth")

    print(f"Saved checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
