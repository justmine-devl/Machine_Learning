"""Audio loading and segmentation utilities for bioacoustic experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    import torchaudio.transforms as T
except Exception:  # pragma: no cover
    T = None

if TYPE_CHECKING:
    from .training import BirdCLEFTrainingConfig


def load_audio(
    path: str | Path,
    sample_rate: int = 32000,
    mono: bool = True,
    offset: float | None = None,
    duration: float | None = None,
    res_type: str = "kaiser_best",
) -> np.ndarray:
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
        raise ImportError(
            "librosa is required for load_audio. Install with `pip install librosa`."
        )
    wav, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=mono,
        offset=float(offset) if offset is not None else 0.0,
        duration=float(duration) if duration is not None else None,
        res_type=res_type,
    )
    if wav.size == 0:
        wav = np.zeros(int(sample_rate * (duration or 1.0)), dtype=np.float32)
    return wav.astype(np.float32)


def pad_or_trim(
    waveform: np.ndarray, target_length: int, random_crop: bool = False
) -> np.ndarray:
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


def crop_or_pad_wave(
    wave: np.ndarray, target_len: int, mode: str = "train", pad_mode: str = "zero_right"
) -> np.ndarray:
    if len(wave) < target_len:
        missing = target_len - len(wave)
        if pad_mode == "repeat" and len(wave) > 0:
            repeats = math.ceil(target_len / len(wave))
            wave = np.tile(wave, repeats)
        elif pad_mode == "zero_left":
            wave = np.pad(wave, (missing, 0), mode="constant")
        elif pad_mode == "zero_random":
            left = (
                np.random.randint(0, missing + 1) if mode == "train" else missing // 2
            )
            wave = np.pad(wave, (left, missing - left), mode="constant")
        elif pad_mode == "zero_center":
            left = missing // 2
            wave = np.pad(wave, (left, missing - left), mode="constant")
        else:
            wave = np.pad(wave, (0, missing), mode="constant")
    if len(wave) > target_len:
        if mode == "train":
            start = np.random.randint(0, len(wave) - target_len + 1)
        else:
            start = max(0, (len(wave) - target_len) // 2)
        wave = wave[start : start + target_len]
    if len(wave) < target_len:
        wave = np.pad(wave, (0, target_len - len(wave)))
    return wave.astype(np.float32)


def augment_wave(
    wave: np.ndarray, config: BirdCLEFTrainingConfig, strength: str = "moderate"
) -> np.ndarray:
    wave = wave.copy()
    gain_scale = 1.0 if strength == "none" else (1.5 if strength == "strong" else 1.0)
    if config.random_gain_db > 0:
        gain_db = (
            np.random.uniform(-config.random_gain_db, config.random_gain_db)
            * gain_scale
        )
        wave = wave * (10 ** (gain_db / 20.0))
    if config.gaussian_noise_std > 0:
        noise_scale = config.gaussian_noise_std * (2.0 if strength == "strong" else 1.0)
        wave = wave + np.random.normal(0, noise_scale, size=wave.shape).astype(
            np.float32
        )
    if config.time_shift_sec > 0:
        max_shift = int(
            config.time_shift_sec
            * config.sample_rate
            * (2.0 if strength == "strong" else 1.0)
        )
        if max_shift > 0:
            wave = np.roll(wave, np.random.randint(-max_shift, max_shift + 1))
    return wave.astype(np.float32)


class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask_param: int, freq_mask_param: int):
        super().__init__()
        if T is None:
            raise ImportError("torchaudio is required for SpecAugment")
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return self.time_mask(self.freq_mask(x))


def mixup_focal_pseudo(
    x_focal: torch.Tensor,
    y_focal: torch.Tensor,
    x_pseudo: torch.Tensor,
    y_pseudo: torch.Tensor,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    lam = float(np.clip(lam, 0.0, 1.0))
    return (
        lam * x_focal + (1.0 - lam) * x_pseudo,
        lam * y_focal + (1.0 - lam) * y_pseudo,
        float(lam),
    )
