from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_all_metrics
from .utils import ensure_dir, save_checkpoint


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    losses = []
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = out["clip_logits"]
        try:
            loss = criterion(logits, y, pos_weight=pos_weight)
        except TypeError:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device)
        out = model(x)
        probs = torch.sigmoid(out["clip_logits"]).detach().cpu().numpy()
        preds.append(probs)
        targets.append(y.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)


def run_training_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    output_dir: str | Path,
    epochs: int = 10,
    pos_weight: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    history: List[Dict[str, float]] = []
    best_auc = -1.0
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, pos_weight=pos_weight)
        y_true, y_pred = predict_loader(model, valid_loader, device)
        metrics = compute_all_metrics(y_true, y_pred, threshold=0.5)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(row)

        save_checkpoint(model, output_dir / "last.pt", epoch=epoch, **metrics)
        if metrics["macro_auc"] > best_auc:
            best_auc = metrics["macro_auc"]
            save_checkpoint(model, output_dir / "best.pt", epoch=epoch, **metrics)

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    return pd.DataFrame(history)
