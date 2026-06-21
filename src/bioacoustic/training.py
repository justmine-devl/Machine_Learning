"""Training and validation loops."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_multilabel_metrics


@dataclass
class TrainState:
    epoch: int
    train_loss: float
    valid_loss: float
    metrics: Dict[str, float]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str = "cuda",
    scheduler: Optional[object] = None,
) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out["clip_logits"], y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Optional[Callable] = None,
    device: str = "cuda",
) -> Dict[str, object]:
    model.eval()
    preds, targets, losses = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        out = model(x)
        logits = out["clip_logits"]
        if loss_fn is not None:
            losses.append(float(loss_fn(logits, y).detach().cpu()))
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, 0))
    y_true = np.concatenate(targets, axis=0) if targets else np.zeros((0, 0))
    metrics = compute_multilabel_metrics(y_true, y_pred) if y_pred.size else {}
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, epoch: int = 0, metrics=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
