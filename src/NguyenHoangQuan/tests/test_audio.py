import numpy as np

from bioacoustic_sed.audio import make_windows, pad_or_trim


def test_pad_or_trim():
    x = np.ones(5)
    assert len(pad_or_trim(x, 10)) == 10
    assert len(pad_or_trim(x, 3)) == 3


def test_make_windows():
    sr = 10
    wav = np.ones(100)
    windows, times = make_windows(wav, sr, window_seconds=2.0)
    assert windows.shape[1] == 20
    assert len(windows) == len(times)
