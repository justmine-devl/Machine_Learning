from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .audio import first_crop_or_pad, load_audio, random_crop_or_pad
from .spectrogram import SpectrogramConfig, log_mel_spectrogram


def parse_secondary_labels(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    value = str(value).strip()
    if value in ["", "[]", "nan", "None"]:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [x.strip() for x in value.split(",") if x.strip()]


def build_class_list(df: pd.DataFrame, primary_col: str = "primary_label", secondary_col: Optional[str] = "secondary_labels") -> List[str]:
    labels = set(df[primary_col].astype(str).tolist())
    if secondary_col and secondary_col in df.columns:
        for v in df[secondary_col].tolist():
            labels.update(parse_secondary_labels(v))
    return sorted(labels)


def save_classes(classes: Sequence[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(classes), encoding="utf-8")


def load_classes(path: str | Path) -> List[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def make_label_vector(row: pd.Series, class_to_idx: Dict[str, int], primary_col: str = "primary_label", secondary_col: Optional[str] = "secondary_labels") -> np.ndarray:
    y = np.zeros(len(class_to_idx), dtype=np.float32)
    primary = str(row[primary_col])
    if primary in class_to_idx:
        y[class_to_idx[primary]] = 1.0
    if secondary_col and secondary_col in row.index:
        for label in parse_secondary_labels(row[secondary_col]):
            if label in class_to_idx:
                y[class_to_idx[label]] = 1.0
    return y


def add_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, label_col: str = "primary_label") -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["fold"] = -1
    counts = df[label_col].value_counts()
    min_count = counts.min()
    n_splits = int(min(n_splits, max(2, min_count))) if min_count > 1 else 2
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(df, df[label_col].astype(str))):
        df.loc[val_idx, "fold"] = fold
    return df


class AudioClipDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str | Path,
        classes: Sequence[str],
        spec_cfg: SpectrogramConfig,
        filename_col: str = "filename",
        primary_col: str = "primary_label",
        secondary_col: Optional[str] = "secondary_labels",
        mode: str = "train",
        clip_duration: float = 5.0,
        use_secondary: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.spec_cfg = spec_cfg
        self.filename_col = filename_col
        self.primary_col = primary_col
        self.secondary_col = secondary_col if use_secondary else None
        self.mode = mode
        self.target_samples = int(spec_cfg.sample_rate * clip_duration)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = self.audio_dir / str(row[self.filename_col])
        waveform = load_audio(audio_path, self.spec_cfg.sample_rate)
        if self.mode == "train":
            if np.random.rand() < 0.5:
                waveform = random_crop_or_pad(waveform, self.target_samples)
            else:
                waveform = first_crop_or_pad(waveform, self.target_samples)
        else:
            waveform = first_crop_or_pad(waveform, self.target_samples)
        spec = log_mel_spectrogram(waveform, self.spec_cfg)
        y = make_label_vector(row, self.class_to_idx, self.primary_col, self.secondary_col)
        return torch.from_numpy(spec[None, :, :]).float(), torch.from_numpy(y).float()


class PseudoLabelDataset(Dataset):
    def __init__(self, pseudo_csv: str | Path, spec_dir: str | Path | None = None) -> None:
        self.df = pd.read_csv(pseudo_csv)
        self.spec_dir = Path(spec_dir) if spec_dir else None
        label_cols = [c for c in self.df.columns if c.startswith("class_")]
        if not label_cols:
            raise ValueError("Pseudo CSV must contain columns named class_0, class_1, ...")
        self.label_cols = label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.spec_dir is None:
            spec_path = Path(row["spec_path"])
        else:
            spec_path = self.spec_dir / str(row["spec_path"])
        spec = np.load(spec_path).astype(np.float32)
        y = row[self.label_cols].values.astype(np.float32)
        return torch.from_numpy(spec).float(), torch.from_numpy(y).float()
