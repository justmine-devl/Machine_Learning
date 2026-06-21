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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bioacoustic.dataset import BirdAudioDataset, add_stratified_folds, build_class_list, encode_multihot, make_label_map, read_metadata
from bioacoustic.metrics import compute_multilabel_metrics, per_class_metrics, search_best_threshold
from bioacoustic.models import build_model
from bioacoustic.training import load_checkpoint
from bioacoustic.utils import ensure_dir, get_device, load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, default=None)
    return parser.parse_args()


def spec_kwargs(cfg):
    s = cfg.get('spectrogram', {})
    return {'n_fft': int(s.get('n_fft', 2048)), 'hop_length': int(s.get('hop_length', 768)), 'n_mels': int(s.get('n_mels', 192)), 'f_min': int(s.get('f_min', 50)), 'f_max': int(s.get('f_max', 15000)), 'normalize': bool(s.get('normalize', True))}


def main() -> None:
    args = parse_args(); cfg = load_config(args.config); seed_everything(int(cfg.get('seed',42)))
    device = get_device(bool(cfg.get('runtime',{}).get('prefer_cuda', True)))
    data = cfg['data']; cv = cfg.get('cv', {})
    df = read_metadata(data['train_csv'])
    classes = build_class_list(df, primary_col=data.get('primary_col','primary_label'))
    if data.get('fold_col', 'fold') not in df.columns:
        df = add_stratified_folds(df, n_splits=int(cv.get('n_splits',5)), label_col=data.get('primary_col','primary_label'), seed=int(cfg.get('seed',42)), fold_col=data.get('fold_col','fold'))
    fold = int(args.fold if args.fold is not None else cv.get('fold', 0))
    valid_df = df[df[data.get('fold_col','fold')] == fold].reset_index(drop=True)
    ds = BirdAudioDataset(valid_df, data['train_audio_dir'], classes, filename_col=data.get('filename_col','filename'), primary_col=data.get('primary_col','primary_label'), secondary_col=data.get('secondary_col','secondary_labels'), sample_rate=int(cfg.get('audio',{}).get('sample_rate',32000)), duration=float(cfg.get('audio',{}).get('clip_duration',5.0)), train=False, include_secondary=bool(data.get('include_secondary', True)), spectrogram_kwargs=spec_kwargs(cfg))
    loader = DataLoader(ds, batch_size=int(cfg.get('training',{}).get('valid_batch_size', 32)), shuffle=False, num_workers=int(cfg.get('training',{}).get('num_workers',2)))
    mcfg = cfg.get('model', {})
    models=[]
    for ckpt in cfg.get('checkpoints', []):
        model = build_model(mcfg.get('model_type','sed'), len(classes), backbone=mcfg.get('backbone','tf_efficientnetv2_s_in21ft1k'), in_channels=int(mcfg.get('in_channels',1)), pretrained=False, dropout=float(mcfg.get('dropout',0.2))).to(device)
        load_checkpoint(ckpt, model, map_location=device)
        model.eval(); models.append(model)
    if not models:
        raise ValueError('No checkpoints specified in config.')
    preds=[]; targets=[]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            model_probs=[]
            for model in models:
                model_probs.append(torch.sigmoid(model(x)['clip_logits']).cpu().numpy())
            preds.append(np.mean(np.stack(model_probs, axis=0), axis=0))
            targets.append(y.numpy())
    y_pred = np.concatenate(preds, axis=0); y_true = np.concatenate(targets, axis=0)
    best_t, best_f1 = search_best_threshold(y_true, y_pred, average='macro')
    metrics = compute_multilabel_metrics(y_true, y_pred, threshold=best_t)
    metrics['best_threshold'] = best_t; metrics['best_macro_f1'] = best_f1
    out_dir = ensure_dir(cfg.get('output_dir', 'outputs/ensemble'))
    save_json(metrics, Path(out_dir) / f'ensemble_metrics_fold{fold}.json')
    pd.DataFrame(per_class_metrics(y_true, y_pred, classes)).to_csv(Path(out_dir) / f'per_class_metrics_fold{fold}.csv', index=False)
    print(metrics)


if __name__ == '__main__':
    main()
