"""Model definitions for spectrogram-based bioacoustic classification."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import CLASS_ORDER, LABEL2IDX
from .spectrogram import SpecFeatureExtractor

try:
    import timm
except Exception:  # pragma: no cover
    timm = None


if TYPE_CHECKING:
    from .training import BirdCLEFTrainingConfig


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
        feat = feat.mean(dim=2)  # [B, C, T]
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
        return EfficientNetClassifier(
            num_classes, backbone, in_channels, pretrained, dropout
        )
    if model_type in {"sed", "attention_sed"}:
        return EfficientNetSED(num_classes, backbone, in_channels, pretrained, dropout)
    raise ValueError(f"Unknown model_type: {model_type}")


def gem_freq(x: torch.Tensor, p=3, eps=1e-6) -> torch.Tensor:
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem_freq(x, p=self.p, eps=self.eps)


class AttHead(nn.Module):
    """SED head copied from the public inference notebook.

    The public code keeps only framewise logits for inference-time overlap
    aggregation. Training pools those dense logits into 5-second targets.
    """

    def __init__(
        self, in_chans: int, p: float = 0.5, num_class: int = 206, hidden_dim: int = 512
    ):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            hidden_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fix_scale = nn.Conv1d(
            hidden_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        framewise_logit = self.fix_scale(feat)
        return {"framewise_logit": framewise_logit}


class CLEFClassifierSED(nn.Module):
    def __init__(self, config: Dict, disable_spectr_generator: bool = False):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for CLEFClassifierSED")
        spectrogram_config = config["spectrogram"]
        backbone_config = config["backbone"]
        head_config = config["head"]

        self.mel_spectr_generator = SpecFeatureExtractor(**spectrogram_config)
        backbone_name = backbone_config["backbone_name"]
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=backbone_config.get("pretrained", False),
                features_only=True,
                drop_path_rate=backbone_config.get("drop_path_rate", 0.0),
            )
        except Exception:
            if backbone_name.startswith("timm/"):
                self.backbone = timm.create_model(
                    backbone_name.replace("timm/", "", 1),
                    pretrained=backbone_config.get("pretrained", False),
                    features_only=True,
                    drop_path_rate=backbone_config.get("drop_path_rate", 0.0),
                )
            else:
                raise

        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(
            in_chans=backbone_dim,
            p=head_config["dropout"],
            num_class=head_config["num_classes"],
        )
        self.infer_duration = head_config["infer_duration"]
        self.duration = head_config["duration"]
        self.inference_type = config["inference_type"]
        self.multilabel_to_train_labels = config.get(
            "multilabel_to_train_labels", "one2one"
        )

    def get_head_preds(
        self, input: torch.Tensor, tta_delta: int = 2, sigmoid: bool = True
    ) -> np.ndarray:
        head_output = self.head(input)
        feat_time = input.shape[-1]
        framewise_prob = head_output["framewise_logit"].sigmoid().detach().cpu().numpy()
        num_segments, num_labels, frames_per_segment = framewise_prob.shape
        step = int(feat_time * (self.infer_duration / self.duration))
        pad_frames = (frames_per_segment - step) // 2
        extra_pad = 1 if (frames_per_segment - step) % 2 != 0 else 0
        framewise_ss_len = frames_per_segment + step * (num_segments - 1)

        framewise_ss_preds = np.zeros(
            (framewise_ss_len, num_segments, num_labels), dtype=np.float16
        )
        framewise_ss_preds_mask = np.zeros((framewise_ss_len, 1), dtype=np.float16)
        for segment_ind in range(num_segments):
            start = segment_ind * step
            framewise_ss_preds[
                start : start + frames_per_segment, segment_ind
            ] += framewise_prob[segment_ind].T
            framewise_ss_preds_mask[start : start + frames_per_segment] += 1

        if self.inference_type in ["overlap_average_max", "overlap_average_max_delta"]:
            framewise_ss_preds = framewise_ss_preds.sum(1) / framewise_ss_preds_mask
        elif self.inference_type == "overlap_max":
            framewise_ss_preds = framewise_ss_preds.max(1)

        framewise_ss_preds = framewise_ss_preds[pad_frames : -pad_frames - extra_pad]
        segmentwise_preds = framewise_ss_preds.reshape(
            num_segments, step, num_labels
        ).max(1)
        if self.inference_type == "overlap_average_max_delta":
            segmentwise_preds *= 0.5
            for segment_ind in range(num_segments):
                if segment_ind == 0:
                    segmentwise_preds[segment_ind] += (
                        framewise_ss_preds[
                            segment_ind * step : (segment_ind + 1) * step
                        ].max(0)
                        * 0.25
                    )
                else:
                    segmentwise_preds[segment_ind] += (
                        framewise_ss_preds[
                            segment_ind * step
                            - tta_delta : (segment_ind + 1) * step
                            - tta_delta
                        ].max(0)
                        * 0.25
                    )
                if segment_ind == (num_segments - 1):
                    segmentwise_preds[segment_ind] += (
                        framewise_ss_preds[
                            segment_ind * step : (segment_ind + 1) * step
                        ].max(0)
                        * 0.25
                    )
                else:
                    segmentwise_preds[segment_ind] += (
                        framewise_ss_preds[
                            segment_ind * step
                            + tta_delta : (segment_ind + 1) * step
                            + tta_delta
                        ].max(0)
                        * 0.25
                    )
        return segmentwise_preds


class InferenceConfig:
    label2ind = LABEL2IDX
    sample_rate = 32000
    num_classes = len(CLASS_ORDER)
    num_segments_sample = 12
    separate_norm = False
    smoothing = True
    preds_weights = np.array([0.133, 0.166, 0.133, 0.133, 0.166, 0.133, 0.133])
    preds_power = 1


class ModelsGroupConfig:
    full_signal_to_spectr = True
    duration = 20
    slice_step_sec = 5
    img_size = (224, 512)
    n_mels = img_size[0]
    hop_length = duration * InferenceConfig.sample_rate // (img_size[1] - 1)
    resize = False
    n_fft = 2048 * 2
    f_min_max = (0, 16000)
    example_dims = (InferenceConfig.num_segments_sample, 3, img_size[-2], img_size[-1])
    spectrogram_params = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "sample_rate": InferenceConfig.sample_rate,
        "n_mels": n_mels,
        "f_min": f_min_max[0],
        "f_max": f_min_max[1],
        "normalized": True,
        "top_db": 80,
        "sample_mel_normalize": "default",
        "output_size": None if not resize else img_size,
    }
    inference_type = "overlap_average_max_delta"

    model_config_1 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {
            "backbone_name": "timm/tf_efficientnet_b3.ns_jft_in1k",
            "pretrained": False,
            "last_layer_hidden_dim": None,
        },
        "head": {
            "dropout": 0.50,
            "num_classes": InferenceConfig.num_classes,
            "infer_duration": 5,
            "duration": duration,
        },
        "inference_type": inference_type,
    }
    model_config_2 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {
            "backbone_name": "timm/regnety_016.tv2_in1k",
            "pretrained": False,
            "last_layer_hidden_dim": None,
        },
        "head": {
            "dropout": 0.50,
            "num_classes": InferenceConfig.num_classes,
            "infer_duration": 5,
            "duration": duration,
        },
        "inference_type": inference_type,
    }
    model_config_3 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {
            "backbone_name": "timm/regnety_008.pycls_in1k",
            "pretrained": False,
            "last_layer_hidden_dim": None,
        },
        "head": {
            "dropout": 0.50,
            "num_classes": InferenceConfig.num_classes,
            "infer_duration": 5,
            "duration": duration,
        },
        "inference_type": inference_type,
    }
    model_config_5 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {
            "backbone_name": "timm/eca_nfnet_l0.ra2_in1k",
            "pretrained": False,
            "last_layer_hidden_dim": None,
        },
        "head": {
            "dropout": 0.50,
            "num_classes": InferenceConfig.num_classes,
            "infer_duration": 5,
            "duration": duration,
        },
        "inference_type": inference_type,
    }
    model_config_6 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {
            "backbone_name": "tf_efficientnet_b4.ns_jft_in1k",
            "pretrained": False,
            "last_layer_hidden_dim": None,
        },
        "head": {
            "dropout": 0.50,
            "num_classes": InferenceConfig.num_classes,
            "infer_duration": 5,
            "duration": duration,
        },
        "inference_type": inference_type,
    }


MODEL_CONFIGS = {
    name: copy.deepcopy(getattr(ModelsGroupConfig, name))
    for name in dir(ModelsGroupConfig)
    if name.startswith("model_config_")
}


def pool_logits_to_label_frames(
    logits: torch.Tensor, frames_per_clip: int
) -> torch.Tensor:
    pooled = F.adaptive_max_pool1d(logits, output_size=frames_per_clip)
    return pooled.permute(0, 2, 1).contiguous()


def forward_sed_logits(
    model: nn.Module,
    waves: torch.Tensor,
    frames_per_clip: int,
    spec_augment: nn.Module | None = None,
) -> torch.Tensor:
    mel = model.mel_spectr_generator(waves)
    if spec_augment is not None:
        mel = spec_augment(mel)
    x = mel.unsqueeze(1).expand(-1, 3, -1, -1)
    features = model.backbone(x)[-1]
    head_output = model.head(features)
    return pool_logits_to_label_frames(head_output["framewise_logit"], frames_per_clip)


def build_birdclef_model(
    config: BirdCLEFTrainingConfig,
    drop_path_rate: float = 0.0,
) -> Tuple[nn.Module, Dict]:
    """Build the inference-compatible SED model without changing the generic factory."""
    if config.model_key not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model_key={config.model_key}. Available: {sorted(MODEL_CONFIGS)}"
        )
    model_config = copy.deepcopy(MODEL_CONFIGS[config.model_key])
    model_config["backbone"]["pretrained"] = config.pretrained_backbone
    model_config["backbone"]["drop_path_rate"] = float(drop_path_rate)
    if model_config["head"]["num_classes"] != len(CLASS_ORDER):
        raise ValueError(
            f"{config.model_key} outputs {model_config['head']['num_classes']} classes, "
            f"expected {len(CLASS_ORDER)}."
        )
    return CLEFClassifierSED(config=model_config), model_config


def load_model_weights(
    model: nn.Module, checkpoint_path: str | Path, strict: bool = True
) -> None:
    """Load either an inference state dict or a full training checkpoint."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get(
            "model_state_dict", payload.get("model", payload.get("state_dict", payload))
        )
    else:
        state = payload
    model.load_state_dict(state, strict=strict)


def freeze_backbone_if_needed(model: nn.Module, config: BirdCLEFTrainingConfig) -> None:
    if config.train_backbone:
        return
    for param in model.backbone.parameters():
        param.requires_grad = False


def make_optimizer(
    model: nn.Module, config: BirdCLEFTrainingConfig
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.learning_rate},
            {"params": head_params, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
