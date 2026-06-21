"""Spectrogram feature extraction."""
from __future__ import annotations

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 32000,
    n_fft: int = 2048,
    hop_length: int = 768,
    n_mels: int = 192,
    f_min: int = 50,
    f_max: int = 15000,
    eps: float = 1e-8,
    normalize: bool = True,
) -> np.ndarray:
    """Compute a normalized log-mel spectrogram with shape [1, n_mels, time]."""
    if librosa is None:
        raise ImportError("librosa is required for spectrogram extraction.")
    waveform = np.asarray(waveform, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )
    logmel = 10.0 * np.log10(mel + eps)
    if normalize:
        mean = float(logmel.mean())
        std = float(logmel.std())
        logmel = (logmel - mean) / (std + eps)
    return logmel[np.newaxis, :, :].astype(np.float32)


def batch_log_mel(windows: list[np.ndarray], **kwargs) -> np.ndarray:
    """Convert a list of waveform windows into a batch of spectrograms."""
    specs = [log_mel_spectrogram(w, **kwargs) for w in windows]
    return np.stack(specs, axis=0).astype(np.float32)
