"""Model definitions for spectrogram-based bioacoustic classification."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover
    timm = None


class AttentionPooling(nn.Module):
    """Class-wise temporal attention pooling for SED features.

    Input shape: [B, C, T]
    Output clip logits: [B, num_classes]
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.att = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        self.cla = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        att_logits = self.att(x)
        frame_logits = self.cla(x)
        att_weights = torch.softmax(att_logits, dim=-1)
        clip_logits = torch.sum(att_weights * frame_logits, dim=-1)
        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "attention": att_weights,
        }


class EfficientNetClassifier(nn.Module):
    """Simple EfficientNet spectrogram classifier with global pooling."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "tf_efficientnet_b0_ns",
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for EfficientNetClassifier.")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            global_pool="avg",
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.backbone(x)
        return {"clip_logits": logits}


class EfficientNetSED(nn.Module):
    """EfficientNet-based sound event detection model.

    The model preserves a temporal feature map, averages over frequency,
    and applies class-wise attention pooling.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "tf_efficientnet_b0_ns",
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for EfficientNetSED.")
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(-1,),
        )
        feature_channels = self.encoder.feature_info.channels()[-1]
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPooling(feature_channels, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)[-1]  # [B, C, F, T]
        feat = feat.mean(dim=2)     # [B, C, T]
        feat = self.dropout(feat)
        out = self.pool(feat)
        return out


def build_model(
    model_type: str,
    num_classes: int,
    backbone: str = "tf_efficientnet_b0_ns",
    in_channels: int = 1,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory for baseline and SED models."""
    model_type = model_type.lower()
    if model_type in {"baseline", "classifier", "clip"}:
        return EfficientNetClassifier(num_classes, backbone, in_channels, pretrained, dropout)
    if model_type in {"sed", "attention_sed"}:
        return EfficientNetSED(num_classes, backbone, in_channels, pretrained, dropout)
    raise ValueError(f"Unknown model_type: {model_type}")
