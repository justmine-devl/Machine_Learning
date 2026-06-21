import numpy as np
import pandas as pd

from bioacoustic.dataset import build_class_list, encode_multihot, make_label_map, parse_secondary_labels


def test_parse_secondary_labels():
    assert parse_secondary_labels("['a', 'b']") == ['a', 'b']
    assert parse_secondary_labels("[]") == []
    assert parse_secondary_labels(None) == []


def test_encode_multihot_primary_and_secondary():
    classes = ['a', 'b', 'c']
    label_map = make_label_map(classes)
    y = encode_multihot('a', "['c']", label_map=label_map, include_secondary=True)
    assert y.tolist() == [1.0, 0.0, 1.0]


def test_build_class_list():
    df = pd.DataFrame({'primary_label': ['b', 'a', 'b']})
    assert build_class_list(df) == ['a', 'b']
