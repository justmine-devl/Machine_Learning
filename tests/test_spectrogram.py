import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from bioacoustic.spectrogram import batch_log_mel, log_mel_spectrogram


def test_log_mel_shape():
    sr = 32000
    wav = np.random.randn(sr).astype(np.float32)
    spec = log_mel_spectrogram(wav, sample_rate=sr, n_mels=64, n_fft=512, hop_length=256, f_min=50, f_max=8000)
    assert spec.ndim == 3
    assert spec.shape[0] == 1
    assert spec.shape[1] == 64
    assert np.isfinite(spec).all()


def test_batch_log_mel_shape():
    sr = 16000
    windows = [np.random.randn(sr).astype(np.float32) for _ in range(3)]
    batch = batch_log_mel(windows, sample_rate=sr, n_mels=32, n_fft=512, hop_length=256, f_min=50, f_max=7500)
    assert batch.shape[0] == 3
    assert batch.shape[1] == 1
    assert batch.shape[2] == 32
