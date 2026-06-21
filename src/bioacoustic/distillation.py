"""Teacher-student and self-distillation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def generate_teacher_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> np.ndarray:
    """Generate sigmoid probabilities from a trained teacher model."""
    model.eval()
    preds = []
    for x, *_ in loader:
        x = x.to(device).float()
        out = model(x)
        preds.append(torch.sigmoid(out["clip_logits"]).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 0), dtype=np.float32)


def save_soft_targets(path: str | Path, probs: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, probs.astype(np.float32))


def load_soft_targets(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.float32)
