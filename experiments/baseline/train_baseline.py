# Allow running the experiment script directly from any subfolder.
from __future__ import annotations
import sys
from pathlib import Path


def add_project_src_to_path() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        src = parent / "src"
        if src.exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return parent
    # Script is usually experiments/<method>/file.py, so repo root is two levels up.
    fallback = current.parents[2] if len(current.parents) >= 3 else current.parent
    src = fallback / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return fallback


PROJECT_ROOT = add_project_src_to_path()


import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bioacoustic.dataset import (
    BirdAudioDataset,
    add_stratified_folds,
    build_class_list,
    encode_multihot,
    make_label_map,
    read_metadata,
)
from bioacoustic.losses import compute_pos_weight, focal_bce_loss, student_distillation_loss
from bioacoustic.models import build_model
from bioacoustic.training import train_one_epoch, validate, save_checkpoint, load_checkpoint
from bioacoustic.utils import ensure_dir, get_device, load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML/JSON config file.')
    parser.add_argument('--fold', type=int, default=None, help='Override validation fold.')
    return parser.parse_args()


def get_spectrogram_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    audio = cfg.get('audio', {})
    spec = cfg.get('spectrogram', {})
    return {
        'n_fft': int(spec.get('n_fft', 2048)),
        'hop_length': int(spec.get('hop_length', 768)),
        'n_mels': int(spec.get('n_mels', 192)),
        'f_min': int(spec.get('f_min', 50)),
        'f_max': int(spec.get('f_max', 15000)),
        'normalize': bool(spec.get('normalize', True)),
    }


def prepare_metadata_and_classes(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    data = cfg['data']
    train_csv = Path(data['train_csv'])
    df = read_metadata(train_csv)
    if data.get('classes_path') and Path(data['classes_path']).exists():
        classes = [x.strip() for x in Path(data['classes_path']).read_text(encoding='utf-8').splitlines() if x.strip()]
    else:
        classes = build_class_list(df, primary_col=data.get('primary_col', 'primary_label'))
        if data.get('classes_path'):
            path = Path(data['classes_path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n'.join(classes), encoding='utf-8')
    if data.get('fold_col', 'fold') not in df.columns:
        df = add_stratified_folds(
            df,
            n_splits=int(cfg.get('cv', {}).get('n_splits', 5)),
            label_col=data.get('primary_col', 'primary_label'),
            seed=int(cfg.get('seed', 42)),
            fold_col=data.get('fold_col', 'fold'),
        )
    return df, classes


def make_loaders(cfg: Dict[str, Any], df: pd.DataFrame, classes: List[str], fold: int):
    data = cfg['data']
    train_cfg = cfg.get('training', {})
    audio_cfg = cfg.get('audio', {})
    fold_col = data.get('fold_col', 'fold')
    train_df = df[df[fold_col] != fold].reset_index(drop=True)
    valid_df = df[df[fold_col] == fold].reset_index(drop=True)
    spec_kwargs = get_spectrogram_kwargs(cfg)
    train_ds = BirdAudioDataset(
        train_df,
        audio_dir=data['train_audio_dir'],
        classes=classes,
        filename_col=data.get('filename_col', 'filename'),
        primary_col=data.get('primary_col', 'primary_label'),
        secondary_col=data.get('secondary_col', 'secondary_labels'),
        sample_rate=int(audio_cfg.get('sample_rate', 32000)),
        duration=float(audio_cfg.get('clip_duration', 5.0)),
        train=True,
        include_secondary=bool(data.get('include_secondary', True)),
        spectrogram_kwargs=spec_kwargs,
    )
    valid_ds = BirdAudioDataset(
        valid_df,
        audio_dir=data['train_audio_dir'],
        classes=classes,
        filename_col=data.get('filename_col', 'filename'),
        primary_col=data.get('primary_col', 'primary_label'),
        secondary_col=data.get('secondary_col', 'secondary_labels'),
        sample_rate=int(audio_cfg.get('sample_rate', 32000)),
        duration=float(audio_cfg.get('clip_duration', 5.0)),
        train=False,
        include_secondary=bool(data.get('include_secondary', True)),
        spectrogram_kwargs=spec_kwargs,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get('batch_size', 16)),
        shuffle=True,
        num_workers=int(train_cfg.get('num_workers', 2)),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(train_cfg.get('valid_batch_size', train_cfg.get('batch_size', 16))),
        shuffle=False,
        num_workers=int(train_cfg.get('num_workers', 2)),
        pin_memory=True,
    )
    return train_loader, valid_loader, train_df, valid_df


def build_targets_for_pos_weight(df: pd.DataFrame, classes: List[str], cfg: Dict[str, Any]) -> torch.Tensor:
    data = cfg.get('data', {})
    label_map = make_label_map(classes)
    targets = []
    for _, row in df.iterrows():
        targets.append(
            encode_multihot(
                row[data.get('primary_col', 'primary_label')],
                row.get(data.get('secondary_col', 'secondary_labels'), None),
                label_map=label_map,
                include_secondary=bool(data.get('include_secondary', True)),
            )
        )
    return torch.tensor(np.stack(targets), dtype=torch.float32)


def make_loss_fn(cfg: Dict[str, Any], train_df: pd.DataFrame, classes: List[str], device: str):
    loss_cfg = cfg.get('loss', {})
    loss_name = str(loss_cfg.get('name', 'weighted_focal_bce')).lower()
    pos_weight = None
    if bool(loss_cfg.get('use_pos_weight', True)):
        target_matrix = build_targets_for_pos_weight(train_df, classes, cfg)
        pos_weight = compute_pos_weight(
            target_matrix,
            max_weight=float(loss_cfg.get('max_pos_weight', 20.0)),
        ).to(device)
    gamma = float(loss_cfg.get('gamma', 2.0))
    smoothing = float(loss_cfg.get('label_smoothing', 0.0))

    def loss_fn(logits, targets):
        if loss_name in {'bce', 'plain_bce'}:
            import torch.nn.functional as F
            return F.binary_cross_entropy_with_logits(logits, targets)
        return focal_bce_loss(logits, targets, gamma=gamma, pos_weight=pos_weight, label_smoothing=smoothing)
    return loss_fn


def train_supervised_experiment(cfg: Dict[str, Any], model_type: str, fold: int | None = None) -> Dict[str, Any]:
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device(bool(cfg.get('runtime', {}).get('prefer_cuda', True)))
    df, classes = prepare_metadata_and_classes(cfg)
    fold = int(fold if fold is not None else cfg.get('cv', {}).get('fold', 0))
    train_loader, valid_loader, train_df, _ = make_loaders(cfg, df, classes, fold)

    model_cfg = cfg.get('model', {})
    model = build_model(
        model_type=model_type,
        num_classes=len(classes),
        backbone=model_cfg.get('backbone', 'tf_efficientnet_b0_ns'),
        in_channels=int(model_cfg.get('in_channels', 1)),
        pretrained=bool(model_cfg.get('pretrained', True)),
        dropout=float(model_cfg.get('dropout', 0.2)),
    ).to(device)
    train_cfg = cfg.get('training', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get('learning_rate', 3e-4)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-4)),
    )
    loss_fn = make_loss_fn(cfg, train_df, classes, device)
    output_dir = ensure_dir(cfg.get('output_dir', f'outputs/{model_type}'))
    history = []
    best_auc = -1.0
    epochs = int(train_cfg.get('epochs', 10))
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device=device)
        valid_out = validate(model, valid_loader, loss_fn, device=device)
        metrics = valid_out.get('metrics', {})
        row = {'epoch': epoch, 'fold': fold, 'train_loss': train_loss, 'valid_loss': valid_out.get('loss')}
        row.update(metrics)
        history.append(row)
        print(row)
        auc = float(metrics.get('macro_auc', -1.0)) if metrics else -1.0
        if auc > best_auc:
            best_auc = auc
            save_checkpoint(Path(output_dir) / f'best_fold{fold}.pth', model, optimizer, epoch, metrics)
    pd.DataFrame(history).to_csv(Path(output_dir) / f'history_fold{fold}.csv', index=False)
    save_json({'fold': fold, 'best_macro_auc': best_auc, 'num_classes': len(classes)}, Path(output_dir) / f'summary_fold{fold}.json')
    return {'best_macro_auc': best_auc, 'history': history}



def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_supervised_experiment(cfg, model_type='baseline', fold=args.fold)


if __name__ == '__main__':
    main()
