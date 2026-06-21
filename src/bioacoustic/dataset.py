"""Metadata handling and PyTorch datasets."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object

from .audio import crop_first_or_random, load_audio, pad_or_trim
from .spectrogram import log_mel_spectrogram


def read_metadata(path: str | Path) -> pd.DataFrame:
    """Read metadata CSV."""
    return pd.read_csv(path)


def parse_secondary_labels(value) -> List[str]:
    """Parse BirdCLEF-style secondary_labels field safely."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return [] if value.strip() in {"", "[]"} else [value]
    return []


def build_class_list(df: pd.DataFrame, primary_col: str = "primary_label") -> List[str]:
    """Build sorted class list from the primary label column."""
    return sorted(df[primary_col].dropna().astype(str).unique().tolist())


def make_label_map(classes: Sequence[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


def encode_multihot(
    primary_label: str,
    secondary_labels=None,
    label_map: Optional[Dict[str, int]] = None,
    num_classes: Optional[int] = None,
    include_secondary: bool = True,
) -> np.ndarray:
    """Encode primary and optional secondary labels into a multi-hot vector."""
    if label_map is None and num_classes is None:
        raise ValueError("Either label_map or num_classes must be provided.")
    if num_classes is None:
        num_classes = len(label_map)  # type: ignore[arg-type]
    target = np.zeros(num_classes, dtype=np.float32)
    if label_map and str(primary_label) in label_map:
        target[label_map[str(primary_label)]] = 1.0
    if include_secondary and label_map is not None:
        for lab in parse_secondary_labels(secondary_labels):
            if lab in label_map:
                target[label_map[lab]] = 1.0
    return target


def add_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_col: str = "primary_label",
    seed: int = 42,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Assign stratified folds by primary label."""
    from sklearn.model_selection import StratifiedKFold

    out = df.copy().reset_index(drop=True)
    out[fold_col] = -1
    y = out[label_col].astype(str).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(out, y)):
        out.loc[val_idx, fold_col] = fold
    return out


class BirdAudioDataset(Dataset):
    """Simple audio-to-spectrogram dataset for supervised training."""

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str | Path,
        classes: Sequence[str],
        filename_col: str = "filename",
        primary_col: str = "primary_label",
        secondary_col: str = "secondary_labels",
        sample_rate: int = 32000,
        duration: float = 5.0,
        train: bool = True,
        include_secondary: bool = True,
        spectrogram_kwargs: Optional[dict] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for BirdAudioDataset.")
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.classes = list(classes)
        self.label_map = make_label_map(classes)
        self.filename_col = filename_col
        self.primary_col = primary_col
        self.secondary_col = secondary_col
        self.sample_rate = sample_rate
        self.duration = duration
        self.train = train
        self.include_secondary = include_secondary
        self.spectrogram_kwargs = spectrogram_kwargs or {}

    def __len__(self) -> int:
        return len(self.df)

    def _audio_path(self, filename: str) -> Path:
        path = self.audio_dir / filename
        if path.exists():
            return path
        # BirdCLEF train_audio may be organized by label/filename.
        primary = str(self.df.loc[self.df[self.filename_col] == filename, self.primary_col].iloc[0])
        nested = self.audio_dir / primary / filename
        return nested if nested.exists() else path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._audio_path(str(row[self.filename_col]))
        waveform = load_audio(path, sample_rate=self.sample_rate)
        if self.train:
            waveform = crop_first_or_random(waveform, self.sample_rate, self.duration)
        else:
            waveform = pad_or_trim(waveform, int(self.sample_rate * self.duration), random_crop=False)
        spec = log_mel_spectrogram(waveform, sample_rate=self.sample_rate, **self.spectrogram_kwargs)
        target = encode_multihot(
            row[self.primary_col],
            row.get(self.secondary_col, None),
            self.label_map,
            include_secondary=self.include_secondary,
        )
        return torch.from_numpy(spec), torch.from_numpy(target)


class PseudoLabelAudioDataset(Dataset):
    """Load fixed soundscape windows with soft pseudo-label targets.

    The pseudo-label table must contain ``audio_path`` and ``chunk_index``
    columns plus one probability column for every class. This keeps the
    generated teacher output traceable to the exact audio window used during
    student training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        classes: Sequence[str],
        sample_rate: int = 32000,
        duration: float = 5.0,
        spectrogram_kwargs: Optional[dict] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for PseudoLabelAudioDataset.")
        required = {"audio_path", "chunk_index", *classes}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Pseudo-label table is missing columns: {missing}")
        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.spectrogram_kwargs = spectrogram_kwargs or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform = load_audio(Path(str(row["audio_path"])), sample_rate=self.sample_rate)
        start = int(row["chunk_index"]) * self.target_length
        window = pad_or_trim(waveform[start:], self.target_length, random_crop=False)
        spec = log_mel_spectrogram(
            window,
            sample_rate=self.sample_rate,
            **self.spectrogram_kwargs,
        )
        target = row[self.classes].to_numpy(dtype=np.float32)
        return torch.from_numpy(spec), torch.from_numpy(target)
