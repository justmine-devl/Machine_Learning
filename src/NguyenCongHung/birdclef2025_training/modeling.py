from __future__ import annotations

from typing import Dict, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .class_order import CLASS_ORDER
from .config import TrainingConfig
from .inference_components import CLEFClassifierSED, MODEL_CONFIGS


def build_model(config: TrainingConfig, drop_path_rate: float = 0.0) -> Tuple[nn.Module, Dict]:
    if config.model_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model_key={config.model_key}. Available: {sorted(MODEL_CONFIGS)}")
    model_config = copy.deepcopy(MODEL_CONFIGS[config.model_key])
    model_config["backbone"]["pretrained"] = config.pretrained_backbone
    model_config["backbone"]["drop_path_rate"] = float(drop_path_rate)
    if model_config["head"]["num_classes"] != len(CLASS_ORDER):
        raise ValueError(f"{config.model_key} outputs {model_config['head']['num_classes']} classes, expected {len(CLASS_ORDER)}.")
    return CLEFClassifierSED(config=model_config), model_config


def pool_logits_to_label_frames(logits: torch.Tensor, frames_per_clip: int) -> torch.Tensor:
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


def freeze_backbone_if_needed(model: nn.Module, config: TrainingConfig) -> None:
    if config.train_backbone:
        return
    for param in model.backbone.parameters():
        param.requires_grad = False


def make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
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
