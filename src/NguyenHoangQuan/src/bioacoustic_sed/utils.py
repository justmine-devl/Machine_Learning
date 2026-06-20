from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load YAML config into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """Make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_class_list(path: str | os.PathLike) -> List[str]:
    """Read one class label per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_class_list(labels: Iterable[str], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(str(label) + "\n")


def find_audio_path(audio_dir: str | os.PathLike, filename: str, label: str | None = None) -> Path:
    """Find an audio file from flat or class-subfolder layout."""
    audio_dir = Path(audio_dir)

    candidates = [audio_dir / filename]
    if label is not None:
        candidates.append(audio_dir / label / filename)

    for path in candidates:
        if path.exists():
            return path

    matches = list(audio_dir.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find {filename} under {audio_dir}")
