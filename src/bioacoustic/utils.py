"""General utilities."""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if path.suffix.lower() == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported config format: {path}")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def get_device(prefer_cuda: bool = True) -> str:
    return "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def print_environment() -> None:
    log(f"Python: {sys.version.split()[0]}")
    log(f"Torch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")
    log(f"CPU count: {os.cpu_count()}")


def ensure_output_dirs(
    output_dir: Path,
    logs_dir: Path | None = None,
    plots_dir: Path | None = None,
    pseudo_labels_dir: Path | None = None,
) -> None:
    ensure_dir(output_dir)
    for path in [logs_dir, plots_dir, pseudo_labels_dir]:
        if path is not None:
            ensure_dir(path)


def effective_epochs(debug: bool, debug_epochs: int, full_epochs: int) -> int:
    return debug_epochs if debug else full_epochs


def effective_iterations(
    debug: bool, debug_iterations: int, full_iterations: int
) -> int:
    return debug_iterations if debug else full_iterations
