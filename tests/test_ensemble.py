import numpy as np

from bioacoustic.ensemble import average_predictions, blend_regular_shifted, power_adjust, temporal_smoothing


def test_average_predictions():
    a = np.ones((2, 3), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    out = average_predictions([a, b])
    assert np.allclose(out, 0.5)


def test_weighted_average_predictions():
    a = np.ones((1, 2), dtype=np.float32)
    b = np.zeros((1, 2), dtype=np.float32)
    out = average_predictions([a, b], weights=[0.8, 0.2])
    assert np.allclose(out, 0.8)


def test_blend_regular_shifted_shape():
    regular = np.ones((3, 2), dtype=np.float32)
    shifted = np.zeros((2, 2), dtype=np.float32)
    out = blend_regular_shifted(regular, shifted, alpha=0.5)
    assert out.shape == regular.shape
    assert np.all(out <= 1.0)


def test_temporal_smoothing_shape():
    preds = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    out = temporal_smoothing(preds)
    assert out.shape == preds.shape
    assert out[1, 0] < 1.0


def test_power_adjust():
    preds = np.array([[0.25, 0.5, 1.0]], dtype=np.float32)
    out = power_adjust(preds, gamma=2.0)
    assert np.allclose(out, preds ** 2)
