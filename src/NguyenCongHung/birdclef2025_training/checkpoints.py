from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import re

import torch

from .class_order import CLASS_ORDER
from .config import TrainingConfig
from .inference_components import ModelsGroupConfig
from .utils import write_json


EPOCH_CHECKPOINT_PATTERNS = ("epoch_*.training.pt", "epoch_*.pth", "epoch_*.pt")


def extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def checkpoint_epoch_from_path(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def checkpoint_epoch(payload, path: Path) -> int:
    if isinstance(payload, dict) and payload.get("epoch") is not None:
        return int(payload.get("epoch", 0))
    return checkpoint_epoch_from_path(path)


def read_checkpoint_epoch(path: Path) -> int:
    try:
        payload = torch.load(path, map_location="cpu")
        return checkpoint_epoch(payload, path)
    except Exception:
        return checkpoint_epoch_from_path(path)


def find_latest_training_checkpoint(stage_dir: Optional[Path]) -> Optional[Path]:
    if stage_dir is None or not stage_dir.exists():
        return None
    last_path = stage_dir / "last.pth"
    if last_path.exists():
        return last_path
    candidates = []
    for pattern in EPOCH_CHECKPOINT_PATTERNS:
        candidates.extend(stage_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=read_checkpoint_epoch)


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(payload), strict=strict)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def prune_epoch_checkpoints(stage_dir: Path) -> None:
    for pattern in EPOCH_CHECKPOINT_PATTERNS:
        for path in stage_dir.glob(pattern):
            _remove_if_exists(path)


def save_checkpoint_bundle(
    stage_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler,
    config: TrainingConfig,
    model_config: Dict,
    epoch: int,
    metrics: Dict,
    stage: str,
    is_best: bool = False,
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "config": config.to_dict(),
        "class_order": CLASS_ORDER,
        "mel_config": ModelsGroupConfig.spectrogram_params,
        "model_config": model_config,
        "epoch": epoch,
        "metrics": metrics,
        "stage": stage,
    }
    last_path = stage_dir / "last.pth"
    torch.save(payload, last_path)
    if not config.save_best_training_checkpoint:
        _remove_if_exists(stage_dir / "best.pth")
    if is_best:
        best_path = stage_dir / "best.pth"
        if config.save_best_training_checkpoint:
            torch.save(payload, best_path)
            training_checkpoint = str(best_path)
        else:
            _remove_if_exists(best_path)
            training_checkpoint = None
        raw_path = stage_dir / "best_for_inference.pt"
        torch.save(model.state_dict(), raw_path)
        write_json(
            stage_dir / "best_for_inference.config.json",
            {
                "checkpoint_format": "raw_state_dict",
                "training_checkpoint": training_checkpoint,
                "model_key": config.model_key,
                "class_order": CLASS_ORDER,
                "mel_config": ModelsGroupConfig.spectrogram_params,
                "stage": stage,
                "epoch": epoch,
                "metrics": metrics,
            },
        )
    if config.save_every_epochs and epoch % config.save_every_epochs == 0:
        torch.save(payload, stage_dir / f"epoch_{epoch}.pth")
    else:
        prune_epoch_checkpoints(stage_dir)
    return last_path


def load_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler=None,
) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(payload))
    if not isinstance(payload, dict):
        return checkpoint_epoch_from_path(checkpoint_path)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return checkpoint_epoch(payload, checkpoint_path)


