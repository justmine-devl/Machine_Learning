import numpy as np

from bioacoustic_sed.openvino_ensemble import blend_regular_shifted, temporal_smoothing


def test_blend_regular_shifted():
    regular = np.ones((3, 2))
    shifted = np.zeros((2, 2))
    out = blend_regular_shifted(regular, shifted)
    assert out.shape == regular.shape
    assert np.all(out >= 0)
    assert np.all(out <= 1)


def test_temporal_smoothing():
    preds = np.random.rand(5, 3)
    out = temporal_smoothing(preds)
    assert out.shape == preds.shape
