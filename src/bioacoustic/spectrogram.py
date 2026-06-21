"""Spectrogram feature extraction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import librosa
except Exception:  
    librosa = None

try:
    import torchaudio.transforms as T
except Exception:  
    T = None


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


class NormalizeMelSpec(nn.Module):
    def __init__(
        self, norm_type: str = "default", eps: float = 1e-6, constant: float = 80
    ):
        super().__init__()
        self.eps = eps
        self.norm_type = norm_type
        self.constant = constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "default":
            mean = x.mean((1, 2), keepdim=True)
            std = x.std((1, 2), keepdim=True)
            xstd = (x - mean) / (std + self.eps)
            norm_max = torch.amax(xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(xstd, dim=(1, 2), keepdim=True)
            return (xstd - norm_min) / (norm_max - norm_min + self.eps)
        if self.norm_type == "top_db":
            return (x + 80) / 80
        if self.norm_type == "constant":
            return x / self.constant
        return x


class SpecFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length=None,
        sample_rate: int = 32000,
        f_max: int = 16000,
        f_min: float = 0,
        n_mels: int = 224,
        top_db: int = 80,
        normalized: bool = True,
        sample_mel_normalize: str | None = "default",
        output_size: Tuple[int, int] | None = None,
    ):
        super().__init__()
        if T is None:
            raise ImportError(
                "torchaudio is required for BirdCLEF spectrogram extraction"
            )
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.append(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                normalized=normalized,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                f_max=f_max,
                n_mels=n_mels,
                f_min=f_min,
            )
        )
        self.feature_extractor.append(T.AmplitudeToDB(top_db=top_db))
        if sample_mel_normalize is not None:
            self.feature_extractor.append(
                NormalizeMelSpec(norm_type=sample_mel_normalize)
            )
        self.resize = (
            nn.UpsamplingBilinear2d(size=output_size)
            if output_size is not None
            else None
        )

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor[-1](x)

    def forward(self, x: torch.Tensor, without_norm: bool = False) -> torch.Tensor:
        if without_norm:
            img = self.feature_extractor[:-1](x)
        else:
            img = self.feature_extractor(x)
        if self.resize is not None:
            img = self.resize(img.unsqueeze(1)).squeeze(1)
        return img
