from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import augment_wave, crop_or_pad_wave, load_audio, mixup_focal_pseudo
from .class_order import CLASS_ORDER, LABEL2IDX
from .config import TrainingConfig


def parse_secondary_labels(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return []
    return []


def infer_filename_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["filename", "filepath", "file", "path", "audio_path"]:
        if col in df.columns:
            return col
    return None


def resolve_audio_path(row: pd.Series, train_audio_dir: Optional[Path]) -> Optional[Path]:
    if train_audio_dir is None:
        return None
    filename_col = infer_filename_column(pd.DataFrame([row]))
    primary = str(row.get("primary_label", ""))
    raw_name = str(row.get(filename_col, "")) if filename_col else ""
    candidates = []
    if raw_name:
        raw = Path(raw_name)
        candidates.extend([raw, train_audio_dir / raw_name, train_audio_dir / primary / raw.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if raw_name:
        matches = list(train_audio_dir.rglob(Path(raw_name).name))
        if matches:
            return matches[0]
    return None


def build_focal_dataframe(train_df: pd.DataFrame, train_audio_dir: Optional[Path]) -> pd.DataFrame:
    if train_df.empty:
        return train_df.copy()
    df = train_df.copy()
    df["resolved_path"] = df.apply(lambda row: resolve_audio_path(row, train_audio_dir), axis=1)
    missing = int(df["resolved_path"].isna().sum())
    if missing:
        print(f"Warning: {missing} focal rows could not be resolved to audio files and will be dropped.")
    df = df[df["resolved_path"].notna()].reset_index(drop=True)
    if "secondary_labels" not in df.columns:
        df["secondary_labels"] = "[]"
    return df


def target_from_row(row: pd.Series, frames_per_clip: int) -> np.ndarray:
    y = np.zeros(len(CLASS_ORDER), dtype=np.float32)
    primary = str(row.get("primary_label", ""))
    if primary in LABEL2IDX:
        y[LABEL2IDX[primary]] = 1.0
    for label in parse_secondary_labels(row.get("secondary_labels", "[]")):
        if label in LABEL2IDX:
            y[LABEL2IDX[label]] = 1.0
    return np.repeat(y[None, :], frames_per_clip, axis=0)


class LabeledFocalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: TrainingConfig, mode: str = "train", augment: bool = False, strength: str = "moderate"):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.augment = augment
        self.strength = strength
        self.target_len = int(config.sample_rate * config.train_duration_sec)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        wave = load_audio(Path(row["resolved_path"]), sample_rate=self.config.sample_rate)
        wave = crop_or_pad_wave(wave, self.target_len, mode=self.mode, pad_mode="repeat")
        if self.augment:
            wave = augment_wave(wave, self.config, strength=self.strength)
        target = target_from_row(row, self.config.frames_per_clip)
        return {
            "wave": torch.tensor(wave, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "source": "focal",
            "filename": Path(row["resolved_path"]).name,
        }


def build_soundscape_windows(files: Sequence[Path], config: TrainingConfig) -> pd.DataFrame:
    rows = []
    for path in files:
        try:
            duration = librosa.get_duration(path=str(path))
        except Exception:
            duration = config.train_duration_sec
        max_start = max(0.0, duration - config.train_duration_sec)
        starts = np.arange(0.0, max_start + 1e-6, config.soundscape_stride_sec)
        if len(starts) == 0:
            starts = np.array([0.0])
        for start in starts:
            rows.append(
                {
                    "filepath": str(path),
                    "filename": path.stem,
                    "window_start_sec": float(start),
                    "window_end_sec": float(start + config.train_duration_sec),
                }
            )
    return pd.DataFrame(rows)


class UnlabeledSoundscapeDataset(Dataset):
    def __init__(self, files: Sequence[Path], config: TrainingConfig, augment: bool = False, strength: str = "none"):
        self.index = build_soundscape_windows(files, config)
        self.config = config
        self.augment = augment
        self.strength = strength
        self.target_len = int(config.sample_rate * config.train_duration_sec)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.index.iloc[idx]
        wave = load_audio(
            Path(row["filepath"]),
            sample_rate=self.config.sample_rate,
            offset=float(row["window_start_sec"]),
            duration=self.config.train_duration_sec,
        )
        wave = crop_or_pad_wave(wave, self.target_len, mode="valid", pad_mode="repeat")
        if self.augment:
            wave = augment_wave(wave, self.config, strength=self.strength)
        return {
            "wave": torch.tensor(wave, dtype=torch.float32),
            "filename": row["filename"],
            "filepath": row["filepath"],
            "window_start_sec": float(row["window_start_sec"]),
        }


def read_pseudo_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


class PseudoLabeledSoundscapeDataset(Dataset):
    def __init__(self, pseudo_path: Path, config: TrainingConfig, augment: bool = False, strength: str = "strong"):
        self.pseudo_df = read_pseudo_table(pseudo_path)
        self.config = config
        self.augment = augment
        self.strength = strength
        self.groups = list(self.pseudo_df.groupby(["filename", "window_start_sec"], sort=False))
        self.target_len = int(config.sample_rate * config.train_duration_sec)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        (_, window_start_sec), group = self.groups[idx]
        group = group.sort_values("start_sec")
        filepath = Path(group["filepath"].iloc[0])
        wave = load_audio(filepath, self.config.sample_rate, offset=float(window_start_sec), duration=self.config.train_duration_sec)
        wave = crop_or_pad_wave(wave, self.target_len, mode="valid", pad_mode="repeat")
        if self.augment:
            wave = augment_wave(wave, self.config, strength=self.strength)
        target = np.zeros((self.config.frames_per_clip, len(CLASS_ORDER)), dtype=np.float32)
        values = group[CLASS_ORDER].values.astype(np.float32)
        n = min(len(values), self.config.frames_per_clip)
        target[:n] = values[:n]
        if n and n < self.config.frames_per_clip:
            target[n:] = target[n - 1]
        return {
            "wave": torch.tensor(wave, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "source": "pseudo",
            "filename": str(group["filename"].iloc[0]),
        }


class MixedNoisyStudentDataset(Dataset):
    def __init__(self, focal_ds: Dataset, pseudo_ds: Dataset, config: TrainingConfig):
        self.focal_ds = focal_ds
        self.pseudo_ds = pseudo_ds
        self.config = config

    def __len__(self) -> int:
        return max(len(self.focal_ds), len(self.pseudo_ds))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        focal = self.focal_ds[idx % len(self.focal_ds)]
        pseudo = self.pseudo_ds[np.random.randint(0, len(self.pseudo_ds))]
        if self.config.mixup_focal_pseudo:
            wave, target, lam = mixup_focal_pseudo(
                focal["wave"], focal["target"], pseudo["wave"], pseudo["target"], self.config.mixup_alpha
            )
            return {"wave": wave, "target": target, "source": f"mixup:{lam:.3f}", "filename": focal["filename"]}
        return focal if np.random.rand() < 0.5 else pseudo


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "wave": torch.stack([item["wave"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "source": [item.get("source", "") for item in batch],
        "filename": [item.get("filename", "") for item in batch],
    }



