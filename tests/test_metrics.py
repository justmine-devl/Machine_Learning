import numpy as np

from bioacoustic.metrics import compute_multilabel_metrics, per_class_metrics, search_best_threshold


def test_compute_multilabel_metrics():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    metrics = compute_multilabel_metrics(y_true, y_pred, threshold=0.5)
    assert metrics['macro_auc'] == 1.0
    assert metrics['f1_macro'] == 1.0


def test_search_best_threshold():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    t, f1 = search_best_threshold(y_true, y_pred, thresholds=[0.2, 0.5, 0.8])
    assert 0.0 <= t <= 1.0
    assert f1 >= 0.0


def test_per_class_metrics():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    rows = per_class_metrics(y_true, y_pred, ['a', 'b'])
    assert len(rows) == 2
    assert rows[0]['class'] == 'a'
