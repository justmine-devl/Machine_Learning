import pytest
import torch


def test_attention_pooling_shape():
    from bioacoustic.models import AttentionPooling

    pool = AttentionPooling(in_channels=8, num_classes=3)
    x = torch.randn(2, 8, 5)
    out = pool(x)
    assert out['clip_logits'].shape == (2, 3)
    assert out['frame_logits'].shape == (2, 3, 5)
    assert out['attention'].shape == (2, 3, 5)


def test_build_model_requires_timm_if_missing():
    pytest.importorskip("timm")
    from bioacoustic.models import build_model

    model = build_model("baseline", num_classes=5, backbone="tf_efficientnet_b0_ns", pretrained=False)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out['clip_logits'].shape == (2, 5)
