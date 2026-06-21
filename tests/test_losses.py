import torch

from bioacoustic.losses import (
    bce_loss,
    focal_bce_loss,
    soft_label_bce_loss,
    student_distillation_loss,
    weighted_bce_loss,
)


def test_losses_are_finite():
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    pos_weight = torch.tensor([2.0, 1.0], dtype=torch.float32)
    soft = torch.tensor([[0.2, 0.8], [0.6, 0.1]], dtype=torch.float32)

    losses = [
        bce_loss(logits, targets),
        weighted_bce_loss(logits, targets, pos_weight=pos_weight),
        focal_bce_loss(logits, targets, pos_weight=pos_weight),
        soft_label_bce_loss(logits, soft),
        student_distillation_loss(logits, targets, soft, pos_weight=pos_weight),
    ]
    for loss in losses:
        assert torch.isfinite(loss)
        assert loss.item() >= 0
