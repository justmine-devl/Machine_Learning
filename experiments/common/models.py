from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

try:
    import timm
except Exception:  # pragma: no cover
    timm = None


@dataclass
class ModelConfig:
    backbone: str = "tf_efficientnet_b0_ns"
    num_classes: int = 206
    in_channels: int = 1
    pretrained: bool = True
    dropout: float = 0.2


class SpectrogramBaseline(nn.Module):
    """Simple spectrogram-as-image multi-label classifier."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for SpectrogramBaseline. Install with `pip install timm`.")
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            num_classes=0,
            global_pool="avg",
        )
        n_features = self.backbone.num_features
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(n_features, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.head(self.dropout(feat))
        return {"logits": logits, "clip_logits": logits}


class AttentionPooling(nn.Module):
    """Class-wise temporal attention pooling for weakly labeled SED."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.att = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        self.cla = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, C, T]
        att_logits = self.att(x)               # [B, K, T]
        frame_logits = self.cla(x)             # [B, K, T]
        att_weights = torch.softmax(att_logits, dim=-1)
        clip_logits = (frame_logits * att_weights).sum(dim=-1)
        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "attention": att_weights,
        }


class SEDModel(nn.Module):
    """EfficientNet-family SED model with temporal attention pooling."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for SEDModel. Install with `pip install timm`.")
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            features_only=True,
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool = AttentionPooling(channels, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        fmap = self.backbone(x)[-1]          # [B, C, F, T]
        temporal = fmap.mean(dim=2)          # frequency pooling -> [B, C, T]
        temporal = self.dropout(temporal)
        return self.pool(temporal)


def build_model(model_type: str, cfg: ModelConfig) -> nn.Module:
    model_type = model_type.lower()
    if model_type in ["baseline", "classifier", "clip"]:
        return SpectrogramBaseline(cfg)
    if model_type in ["sed", "attention_sed"]:
        return SEDModel(cfg)
    raise ValueError(f"Unknown model_type: {model_type}")
