from __future__ import annotations

from typing import Dict, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import timm

from .class_order import CLASS_ORDER, LABEL2IDX


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

    def __init__(self, in_chans: int, p: float = 0.5, num_class: int = 206, hidden_dim: int = 512):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        framewise_logit = self.fix_scale(feat)
        return {"framewise_logit": framewise_logit}


class NormalizeMelSpec(nn.Module):
    def __init__(self, norm_type: str = "default", eps: float = 1e-6, constant: float = 80):
        super().__init__()
        self.eps = eps
        self.norm_type = norm_type
        self.constant = constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "default":
            mean = x.mean((1, 2), keepdim=True)
            std = x.std((1, 2), keepdim=True)
            xstd = (x - mean) / (std + self.eps)
            norm_max = torch.amax(xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(xstd, dim=(1, 2), keepdim=True)
            return (xstd - norm_min) / (norm_max - norm_min + self.eps)
        if self.norm_type == "top_db":
            return (x + 80) / 80
        if self.norm_type == "constant":
            return x / self.constant
        return x


class SpecFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length=None,
        sample_rate: int = 32000,
        f_max: int = 16000,
        f_min: float = 0,
        n_mels: int = 224,
        top_db: int = 80,
        normalized: bool = True,
        sample_mel_normalize: str | None = "default",
        output_size: Tuple[int, int] | None = None,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.append(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                normalized=normalized,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                f_max=f_max,
                n_mels=n_mels,
                f_min=f_min,
            )
        )
        self.feature_extractor.append(T.AmplitudeToDB(top_db=top_db))
        if sample_mel_normalize is not None:
            self.feature_extractor.append(NormalizeMelSpec(norm_type=sample_mel_normalize))
        self.resize = nn.UpsamplingBilinear2d(size=output_size) if output_size is not None else None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor[-1](x)

    def forward(self, x: torch.Tensor, without_norm: bool = False) -> torch.Tensor:
        if without_norm:
            img = self.feature_extractor[:-1](x)
        else:
            img = self.feature_extractor(x)
        if self.resize is not None:
            img = self.resize(img.unsqueeze(1)).squeeze(1)
        return img


class CLEFClassifierSED(nn.Module):
    def __init__(self, config: Dict, disable_spectr_generator: bool = False):
        super().__init__()
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
        self.multilabel_to_train_labels = config.get("multilabel_to_train_labels", "one2one")

    def get_head_preds(self, input: torch.Tensor, tta_delta: int = 2, sigmoid: bool = True) -> np.ndarray:
        head_output = self.head(input)
        feat_time = input.shape[-1]
        framewise_prob = head_output["framewise_logit"].sigmoid().detach().cpu().numpy()
        num_segments, num_labels, frames_per_segment = framewise_prob.shape
        step = int(feat_time * (self.infer_duration / self.duration))
        pad_frames = (frames_per_segment - step) // 2
        extra_pad = 1 if (frames_per_segment - step) % 2 != 0 else 0
        framewise_ss_len = frames_per_segment + step * (num_segments - 1)

        framewise_ss_preds = np.zeros((framewise_ss_len, num_segments, num_labels), dtype=np.float16)
        framewise_ss_preds_mask = np.zeros((framewise_ss_len, 1), dtype=np.float16)
        for segment_ind in range(num_segments):
            start = segment_ind * step
            framewise_ss_preds[start : start + frames_per_segment, segment_ind] += framewise_prob[segment_ind].T
            framewise_ss_preds_mask[start : start + frames_per_segment] += 1

        if self.inference_type in ["overlap_average_max", "overlap_average_max_delta"]:
            framewise_ss_preds = framewise_ss_preds.sum(1) / framewise_ss_preds_mask
        elif self.inference_type == "overlap_max":
            framewise_ss_preds = framewise_ss_preds.max(1)

        framewise_ss_preds = framewise_ss_preds[pad_frames : -pad_frames - extra_pad]
        segmentwise_preds = framewise_ss_preds.reshape(num_segments, step, num_labels).max(1)
        if self.inference_type == "overlap_average_max_delta":
            segmentwise_preds *= 0.5
            for segment_ind in range(num_segments):
                if segment_ind == 0:
                    segmentwise_preds[segment_ind] += framewise_ss_preds[segment_ind * step : (segment_ind + 1) * step].max(0) * 0.25
                else:
                    segmentwise_preds[segment_ind] += framewise_ss_preds[segment_ind * step - tta_delta : (segment_ind + 1) * step - tta_delta].max(0) * 0.25
                if segment_ind == (num_segments - 1):
                    segmentwise_preds[segment_ind] += framewise_ss_preds[segment_ind * step : (segment_ind + 1) * step].max(0) * 0.25
                else:
                    segmentwise_preds[segment_ind] += framewise_ss_preds[segment_ind * step + tta_delta : (segment_ind + 1) * step + tta_delta].max(0) * 0.25
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
        "backbone": {"backbone_name": "timm/tf_efficientnet_b3.ns_jft_in1k", "pretrained": False, "last_layer_hidden_dim": None},
        "head": {"dropout": 0.50, "num_classes": InferenceConfig.num_classes, "infer_duration": 5, "duration": duration},
        "inference_type": inference_type,
    }
    model_config_2 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {"backbone_name": "timm/regnety_016.tv2_in1k", "pretrained": False, "last_layer_hidden_dim": None},
        "head": {"dropout": 0.50, "num_classes": InferenceConfig.num_classes, "infer_duration": 5, "duration": duration},
        "inference_type": inference_type,
    }
    model_config_3 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {"backbone_name": "timm/regnety_008.pycls_in1k", "pretrained": False, "last_layer_hidden_dim": None},
        "head": {"dropout": 0.50, "num_classes": InferenceConfig.num_classes, "infer_duration": 5, "duration": duration},
        "inference_type": inference_type,
    }
    model_config_5 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {"backbone_name": "timm/eca_nfnet_l0.ra2_in1k", "pretrained": False, "last_layer_hidden_dim": None},
        "head": {"dropout": 0.50, "num_classes": InferenceConfig.num_classes, "infer_duration": 5, "duration": duration},
        "inference_type": inference_type,
    }
    model_config_6 = {
        "model_type": "sed",
        "example_dims": example_dims,
        "spectrogram": spectrogram_params,
        "backbone": {"backbone_name": "tf_efficientnet_b4.ns_jft_in1k", "pretrained": False, "last_layer_hidden_dim": None},
        "head": {"dropout": 0.50, "num_classes": InferenceConfig.num_classes, "infer_duration": 5, "duration": duration},
        "inference_type": inference_type,
    }


MODEL_CONFIGS = {
    name: copy.deepcopy(getattr(ModelsGroupConfig, name))
    for name in dir(ModelsGroupConfig)
    if name.startswith("model_config_")
}
