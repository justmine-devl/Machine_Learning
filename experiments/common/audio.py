from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np


def load_audio(path: str | Path, sample_rate: int = 32000, mono: bool = True) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, _ = librosa.load(path, sr=sample_rate, mono=mono)
    waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0).astype(np.float32)
    return waveform


def pad_or_trim(waveform: np.ndarray, target_samples: int, pad_value: float = 0.0) -> np.ndarray:
    if len(waveform) == target_samples:
        return waveform.astype(np.float32)
    if len(waveform) > target_samples:
        return waveform[:target_samples].astype(np.float32)
    out = np.full(target_samples, pad_value, dtype=np.float32)
    out[: len(waveform)] = waveform
    return out


def random_crop_or_pad(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    if len(waveform) <= target_samples:
        return pad_or_trim(waveform, target_samples)
    start = np.random.randint(0, len(waveform) - target_samples + 1)
    return waveform[start : start + target_samples].astype(np.float32)


def first_crop_or_pad(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    return pad_or_trim(waveform, target_samples)


def segment_regular_windows(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    clip_duration: float = 5.0,
    pad_last: bool = True,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    win = int(sample_rate * clip_duration)
    segments = []
    times = []
    if len(waveform) == 0:
        return np.empty((0, win), dtype=np.float32), []

    n_windows = int(np.ceil(len(waveform) / win)) if pad_last else len(waveform) // win
    for i in range(n_windows):
        start = i * win
        end = start + win
        chunk = waveform[start:end]
        if len(chunk) < win:
            if not pad_last:
                continue
            chunk = pad_or_trim(chunk, win)
        segments.append(chunk.astype(np.float32))
        times.append((start / sample_rate, min(end, len(waveform)) / sample_rate))
    return np.stack(segments), times


def segment_shifted_windows(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    clip_duration: float = 5.0,
    shift_seconds: float = 2.5,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    win = int(sample_rate * clip_duration)
    shift = int(sample_rate * shift_seconds)
    segments = []
    times = []
    if len(waveform) <= shift:
        return np.empty((0, win), dtype=np.float32), []

    start = shift
    while start + win <= len(waveform):
        end = start + win
        segments.append(waveform[start:end].astype(np.float32))
        times.append((start / sample_rate, end / sample_rate))
        start += win
    if len(segments) == 0:
        return np.empty((0, win), dtype=np.float32), []
    return np.stack(segments), times


def blend_regular_and_shifted(regular: np.ndarray, shifted: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend regular-window predictions with neighboring shifted-window predictions.

    regular shape: [N, C]
    shifted shape: [N-1, C] usually, where shifted[i] overlaps regular[i] and regular[i+1].
    """
    if shifted.size == 0 or len(regular) == 0:
        return regular
    out = regular.copy()
    n = len(regular)
    if len(shifted) >= 1:
        out[0] = alpha * regular[0] + (1 - alpha) * 0.5 * (regular[0] + shifted[0])
    if n >= 2 and len(shifted) >= n - 1:
        out[-1] = alpha * regular[-1] + (1 - alpha) * 0.5 * (regular[-1] + shifted[n - 2])
    for i in range(1, n - 1):
        if i - 1 < len(shifted) and i < len(shifted):
            out[i] = alpha * regular[i] + (1 - alpha) * 0.5 * (shifted[i - 1] + shifted[i])
    return out


def temporal_smoothing(preds: np.ndarray, prev_weight: float = 0.1, current_weight: float = 0.8, next_weight: float = 0.1) -> np.ndarray:
    if len(preds) <= 1:
        return preds
    out = preds.copy()
    for i in range(len(preds)):
        if i == 0:
            out[i] = (current_weight + prev_weight) * preds[i] + next_weight * preds[i + 1]
        elif i == len(preds) - 1:
            out[i] = prev_weight * preds[i - 1] + (current_weight + next_weight) * preds[i]
        else:
            out[i] = prev_weight * preds[i - 1] + current_weight * preds[i] + next_weight * preds[i + 1]
    return out
