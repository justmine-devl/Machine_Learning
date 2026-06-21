"""Audio loading and segmentation utilities for bioacoustic experiments."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


def load_audio(path: str | Path, sample_rate: int = 32000, mono: bool = True) -> np.ndarray:
    """Load an audio file as float32 waveform.

    Parameters
    ----------
    path:
        Audio path.
    sample_rate:
        Target sampling rate.
    mono:
        Convert to mono if True.
    """
    if librosa is None:
        raise ImportError("librosa is required for load_audio. Install with `pip install librosa`.")
    wav, _ = librosa.load(str(path), sr=sample_rate, mono=mono)
    return wav.astype(np.float32)


def pad_or_trim(waveform: np.ndarray, target_length: int, random_crop: bool = False) -> np.ndarray:
    """Pad or trim waveform to a fixed number of samples."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) == target_length:
        return waveform
    if len(waveform) < target_length:
        out = np.zeros(target_length, dtype=np.float32)
        out[: len(waveform)] = waveform
        return out
    if random_crop:
        start = np.random.randint(0, len(waveform) - target_length + 1)
    else:
        start = 0
    return waveform[start : start + target_length].astype(np.float32)


def crop_first_or_random(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    duration: float = 5.0,
    p_random: float = 0.5,
) -> np.ndarray:
    """Training crop strategy: random crop with probability p_random, otherwise first crop."""
    target_length = int(sample_rate * duration)
    random_crop = np.random.rand() < p_random
    return pad_or_trim(waveform, target_length, random_crop=random_crop)


def make_regular_windows(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    duration: float = 5.0,
    drop_last: bool = False,
) -> List[np.ndarray]:
    """Split waveform into consecutive non-overlapping windows."""
    win = int(sample_rate * duration)
    waveform = np.asarray(waveform, dtype=np.float32)
    windows: List[np.ndarray] = []
    for start in range(0, len(waveform), win):
        segment = waveform[start : start + win]
        if len(segment) < win and drop_last:
            break
        windows.append(pad_or_trim(segment, win))
    if not windows:
        windows.append(np.zeros(win, dtype=np.float32))
    return windows


def make_shifted_windows(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    duration: float = 5.0,
    shift: float = 2.5,
) -> List[np.ndarray]:
    """Create shifted windows, e.g. 2.5-7.5s, 7.5-12.5s, ..."""
    win = int(sample_rate * duration)
    hop = int(sample_rate * duration)
    offset = int(sample_rate * shift)
    waveform = np.asarray(waveform, dtype=np.float32)
    windows: List[np.ndarray] = []
    for start in range(offset, max(offset, len(waveform) - win + 1), hop):
        segment = waveform[start : start + win]
        if len(segment) == win:
            windows.append(segment.astype(np.float32))
    return windows


def segment_for_inference(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    duration: float = 5.0,
    shift: float = 2.5,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return regular and shifted windows for ensemble inference."""
    regular = make_regular_windows(waveform, sample_rate, duration, drop_last=False)
    shifted = make_shifted_windows(waveform, sample_rate, duration, shift)
    return regular, shifted
