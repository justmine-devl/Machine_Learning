from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import torchaudio.transforms as T

from .config import TrainingConfig


def load_audio(
    path: Path,
    sample_rate: int,
    offset: float | None = None,
    duration: float | None = None,
    res_type: str = "kaiser_best",
) -> np.ndarray:
    kwargs = {"sr": sample_rate, "mono": True, "res_type": res_type}
    if offset is not None:
        kwargs["offset"] = float(offset)
    if duration is not None:
        kwargs["duration"] = float(duration)
    wave, _ = librosa.load(str(path), **kwargs)
    if wave.size == 0:
        wave = np.zeros(int(sample_rate * (duration or 1)), dtype=np.float32)
    return wave.astype(np.float32)


def crop_or_pad_wave(wave: np.ndarray, target_len: int, mode: str = "train", pad_mode: str = "repeat") -> np.ndarray:
    if len(wave) < target_len:
        if pad_mode == "repeat" and len(wave) > 0:
            repeats = math.ceil(target_len / len(wave))
            wave = np.tile(wave, repeats)
        else:
            wave = np.pad(wave, (0, max(0, target_len - len(wave))))
    if len(wave) > target_len:
        if mode == "train":
            start = np.random.randint(0, len(wave) - target_len + 1)
        else:
            start = max(0, (len(wave) - target_len) // 2)
        wave = wave[start : start + target_len]
    if len(wave) < target_len:
        wave = np.pad(wave, (0, target_len - len(wave)))
    return wave.astype(np.float32)


def augment_wave(wave: np.ndarray, config: TrainingConfig, strength: str = "moderate") -> np.ndarray:
    wave = wave.copy()
    gain_scale = 1.0 if strength == "none" else (1.5 if strength == "strong" else 1.0)
    if config.random_gain_db > 0:
        gain_db = np.random.uniform(-config.random_gain_db, config.random_gain_db) * gain_scale
        wave = wave * (10 ** (gain_db / 20.0))
    if config.gaussian_noise_std > 0:
        noise_scale = config.gaussian_noise_std * (2.0 if strength == "strong" else 1.0)
        wave = wave + np.random.normal(0, noise_scale, size=wave.shape).astype(np.float32)
    if config.time_shift_sec > 0:
        max_shift = int(config.time_shift_sec * config.sample_rate * (2.0 if strength == "strong" else 1.0))
        if max_shift > 0:
            wave = np.roll(wave, np.random.randint(-max_shift, max_shift + 1))
    return wave.astype(np.float32)


class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask_param: int, freq_mask_param: int):
        super().__init__()
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
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x_focal, y_focal, 1.0
    lam = np.random.beta(alpha, alpha)
    return lam * x_focal + (1.0 - lam) * x_pseudo, lam * y_focal + (1.0 - lam) * y_pseudo, float(lam)



