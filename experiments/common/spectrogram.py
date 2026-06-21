from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class SpectrogramConfig:
    sample_rate: int = 32000
    n_fft: int = 2048
    hop_length: int = 768
    win_length: int = 2048
    n_mels: int = 192
    f_min: int = 50
    f_max: int = 15000
    eps: float = 1e-6


def log_mel_spectrogram(waveform: np.ndarray, cfg: SpectrogramConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mean = float(logmel.mean())
    std = float(logmel.std())
    logmel = (logmel - mean) / (std + cfg.eps)
    return logmel.astype(np.float32)


def batch_log_mel(waveforms: np.ndarray, cfg: SpectrogramConfig) -> np.ndarray:
    specs = [log_mel_spectrogram(w, cfg) for w in waveforms]
    return np.stack(specs).astype(np.float32)[:, None, :, :]
