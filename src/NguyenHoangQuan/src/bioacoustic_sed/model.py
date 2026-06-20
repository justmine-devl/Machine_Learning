from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    timm = None


class TemporalAttentionPooling(nn.Module):
    """Class-wise temporal attention pooling for SED."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        self.attention = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: temporal features with shape [B, C, T]

        Returns:
            frame_logits: [B, num_classes, T]
            clip_logits: [B, num_classes]
            attention: [B, num_classes, T]
        """
        frame_logits = self.classifier(x)
        attn_logits = self.attention(x)
        attn = torch.softmax(torch.clamp(attn_logits, -10, 10), dim=-1)
        clip_logits = torch.sum(frame_logits * attn, dim=-1)
        return {
            "frame_logits": frame_logits,
            "clip_logits": clip_logits,
            "attention": attn,
        }


class EfficientNetSED(nn.Module):
    """
    EfficientNet-based Sound Event Detection model.

    Input shape: [B, 1, n_mels, time]
    Output: class-wise clip logits and frame logits.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        in_chans: int = 1,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for EfficientNetSED. Install with `pip install timm`.")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
        )
        out_channels = self.backbone.feature_info[-1]["num_chs"]
        self.pool = TemporalAttentionPooling(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)[-1]  # [B, C, F, T]
        features = features.mean(dim=2)  # frequency pooling -> [B, C, T]
        return self.pool(features)


def create_model(backbone: str, num_classes: int, pretrained: bool = True) -> EfficientNetSED:
    return EfficientNetSED(backbone_name=backbone, num_classes=num_classes, pretrained=pretrained)


def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return sigmoid probabilities from model input batch."""
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        return torch.sigmoid(outputs["clip_logits"])
