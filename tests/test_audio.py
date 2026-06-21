import numpy as np

from bioacoustic.audio import make_regular_windows, make_shifted_windows, pad_or_trim


def test_pad_or_trim_padding():
    x = np.ones(5, dtype=np.float32)
    y = pad_or_trim(x, 10)
    assert y.shape == (10,)
    assert np.allclose(y[:5], 1.0)
    assert np.allclose(y[5:], 0.0)


def test_pad_or_trim_trimming():
    x = np.arange(20, dtype=np.float32)
    y = pad_or_trim(x, 10, random_crop=False)
    assert y.shape == (10,)
    assert np.allclose(y, np.arange(10, dtype=np.float32))


def test_regular_windows():
    sr = 10
    wav = np.arange(100, dtype=np.float32)
    windows = make_regular_windows(wav, sample_rate=sr, duration=2.0)
    assert len(windows) == 5
    assert all(w.shape == (20,) for w in windows)


def test_shifted_windows():
    sr = 10
    wav = np.arange(100, dtype=np.float32)
    windows = make_shifted_windows(wav, sample_rate=sr, duration=2.0, shift=1.0)
    assert len(windows) == 4
    assert all(w.shape == (20,) for w in windows)
