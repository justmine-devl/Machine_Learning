from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np


@dataclass
class AudioConfig:
    sample_rate: int = 32000
    window_seconds: float = 5.0
    shift_seconds: float = 2.5
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 768
    n_mels: int = 192
    f_min: int = 50
    f_max: int = 15000


def load_audio(path: str | Path, sample_rate: int = 32000) -> np.ndarray:
    """Load mono audio and resample to the target sample rate."""
    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return wav


def pad_or_trim(wav: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim waveform to exactly target_length samples."""
    if len(wav) >= target_length:
        return wav[:target_length]
    pad = target_length - len(wav)
    return np.pad(wav, (0, pad), mode="constant")


def make_windows(
    wav: np.ndarray,
    sample_rate: int,
    window_seconds: float = 5.0,
    shift_seconds: float | None = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Split waveform into fixed-length windows.

    If shift_seconds is None, non-overlapping windows are used.
    If shift_seconds is provided, windows start at that offset and then repeat every window_seconds.
    """
    win_len = int(sample_rate * window_seconds)
    offset = 0 if shift_seconds is None else int(sample_rate * shift_seconds)
    starts = list(range(offset, max(len(wav) - win_len + 1, offset + 1), win_len))

    if not starts:
        starts = [0]

    windows = []
    times = []
    for start in starts:
        segment = wav[start:start + win_len]
        segment = pad_or_trim(segment, win_len)
        windows.append(segment)
        times.append(start / sample_rate)

    return np.stack(windows).astype(np.float32), times


def log_mel_spectrogram(wav: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Convert waveform segment to normalized log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = logmel.astype(np.float32)

    mean = float(logmel.mean())
    std = float(logmel.std()) + 1e-6
    logmel = (logmel - mean) / std
    return logmel


def batch_log_mel(windows: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """
    Convert a batch of waveform windows into model input.

    Returns shape: [N, 1, n_mels, time]
    """
    specs = [log_mel_spectrogram(w, cfg) for w in windows]
    specs = np.stack(specs).astype(np.float32)
    return specs[:, None, :, :]


def build_regular_and_shifted_specs(path: str | Path, cfg: AudioConfig) -> tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """Load audio and create regular + shifted log-mel batches."""
    wav = load_audio(path, cfg.sample_rate)
    regular_wavs, regular_times = make_windows(
        wav,
        cfg.sample_rate,
        window_seconds=cfg.window_seconds,
        shift_seconds=None,
    )
    shifted_wavs, shifted_times = make_windows(
        wav,
        cfg.sample_rate,
        window_seconds=cfg.window_seconds,
        shift_seconds=cfg.shift_seconds,
    )
    regular_specs = batch_log_mel(regular_wavs, cfg)
    shifted_specs = batch_log_mel(shifted_wavs, cfg)
    return regular_specs, shifted_specs, regular_times, shifted_times
