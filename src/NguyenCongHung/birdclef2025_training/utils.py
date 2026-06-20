from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import os
import random
import sys

import numpy as np
import torch


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def print_environment() -> None:
    log(f"Python: {sys.version.split()[0]}")
    log(f"Torch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")
    log(f"CPU count: {os.cpu_count()}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dirs(output_dir: Path, logs_dir: Path | None = None, plots_dir: Path | None = None, pseudo_labels_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [logs_dir, plots_dir, pseudo_labels_dir]:
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)


def effective_epochs(debug: bool, debug_epochs: int, full_epochs: int) -> int:
    return debug_epochs if debug else full_epochs


def effective_iterations(debug: bool, debug_iterations: int, full_iterations: int) -> int:
    return debug_iterations if debug else full_iterations


