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
    fallback = current.parents[2] if len(current.parents) >= 3 else current.parent
    src = fallback / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return fallback


PROJECT_ROOT = add_project_src_to_path()


import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bioacoustic.dataset import BirdAudioDataset, add_stratified_folds, build_class_list, read_metadata, encode_multihot, make_label_map
from bioacoustic.distillation import generate_teacher_predictions, save_soft_targets, load_soft_targets
from bioacoustic.losses import compute_pos_weight, student_distillation_loss
from bioacoustic.models import build_model
from bioacoustic.training import load_checkpoint, save_checkpoint, validate
from bioacoustic.utils import ensure_dir, get_device, load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--generate_teacher_targets_only', action='store_true')
    return parser.parse_args()


def spec_kwargs(cfg):
    s = cfg.get('spectrogram', {})
    return {'n_fft': int(s.get('n_fft', 2048)), 'hop_length': int(s.get('hop_length', 768)), 'n_mels': int(s.get('n_mels', 192)), 'f_min': int(s.get('f_min', 50)), 'f_max': int(s.get('f_max', 15000)), 'normalize': bool(s.get('normalize', True))}


def build_dataset(df, cfg, classes, train):
    data = cfg['data']; audio = cfg.get('audio', {})
    return BirdAudioDataset(df, data['train_audio_dir'], classes, filename_col=data.get('filename_col','filename'), primary_col=data.get('primary_col','primary_label'), secondary_col=data.get('secondary_col','secondary_labels'), sample_rate=int(audio.get('sample_rate',32000)), duration=float(audio.get('clip_duration',5.0)), train=train, include_secondary=bool(data.get('include_secondary',True)), spectrogram_kwargs=spec_kwargs(cfg))


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device(bool(cfg.get('runtime', {}).get('prefer_cuda', True)))
    data = cfg['data']; cv = cfg.get('cv', {})
    df = read_metadata(data['train_csv'])
    classes = build_class_list(df, primary_col=data.get('primary_col', 'primary_label'))
    if data.get('fold_col', 'fold') not in df.columns:
        df = add_stratified_folds(df, n_splits=int(cv.get('n_splits', 5)), label_col=data.get('primary_col','primary_label'), seed=int(cfg.get('seed',42)), fold_col=data.get('fold_col','fold'))
    fold = int(args.fold if args.fold is not None else cv.get('fold', 0))
    train_df = df[df[data.get('fold_col','fold')] != fold].reset_index(drop=True)
    valid_df = df[df[data.get('fold_col','fold')] == fold].reset_index(drop=True)
    train_ds = build_dataset(train_df, cfg, classes, train=True)
    valid_ds = build_dataset(valid_df, cfg, classes, train=False)
    train_loader_plain = DataLoader(train_ds, batch_size=int(cfg.get('training',{}).get('valid_batch_size',32)), shuffle=False, num_workers=int(cfg.get('training',{}).get('num_workers',2)))

    model_cfg = cfg.get('model', {})
    teacher = build_model('sed', len(classes), backbone=model_cfg.get('backbone','tf_efficientnetv2_s_in21ft1k'), in_channels=int(model_cfg.get('in_channels',1)), pretrained=False, dropout=float(model_cfg.get('dropout',0.2))).to(device)
    load_checkpoint(cfg['teacher']['checkpoint'], teacher, map_location=device)
    soft_path = Path(data.get('soft_targets_path', 'outputs/self_distillation/teacher_soft_targets.npy'))
    if not soft_path.exists():
        probs = generate_teacher_predictions(teacher, train_loader_plain, device=device)
        save_soft_targets(soft_path, probs)
        print(f'Saved teacher soft targets to {soft_path}')
    if args.generate_teacher_targets_only:
        return
    soft_targets = torch.tensor(load_soft_targets(soft_path), dtype=torch.float32)

    student = build_model('sed', len(classes), backbone=model_cfg.get('backbone','tf_efficientnetv2_s_in21ft1k'), in_channels=int(model_cfg.get('in_channels',1)), pretrained=bool(model_cfg.get('pretrained',False)), dropout=float(model_cfg.get('dropout',0.2))).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg.get('training',{}).get('learning_rate',2e-4)), weight_decay=float(cfg.get('training',{}).get('weight_decay',1e-4)))
    train_loader = DataLoader(IndexedDataset(train_ds), batch_size=int(cfg.get('training',{}).get('batch_size',16)), shuffle=True, num_workers=int(cfg.get('training',{}).get('num_workers',2)))
    valid_loader = DataLoader(valid_ds, batch_size=int(cfg.get('training',{}).get('valid_batch_size',32)), shuffle=False, num_workers=int(cfg.get('training',{}).get('num_workers',2)))

    label_map = make_label_map(classes)
    hard = []
    for _, row in train_df.iterrows():
        hard.append(encode_multihot(row[data.get('primary_col','primary_label')], row.get(data.get('secondary_col','secondary_labels'), None), label_map=label_map, include_secondary=bool(data.get('include_secondary', True))))
    pos_weight = compute_pos_weight(torch.tensor(np.stack(hard), dtype=torch.float32), max_weight=float(cfg.get('loss',{}).get('max_pos_weight',20.0))).to(device)

    out_dir = ensure_dir(cfg.get('output_dir', 'outputs/self_distillation'))
    best_auc = -1.0
    history = []
    for epoch in range(1, int(cfg.get('training',{}).get('epochs',8)) + 1):
        student.train(); losses=[]
        for x, y, idx in train_loader:
            x = x.to(device).float(); y = y.to(device).float(); idx = idx.long()
            soft = soft_targets[idx].to(device).float()
            opt.zero_grad(set_to_none=True)
            logits = student(x)['clip_logits']
            loss = student_distillation_loss(logits, y, soft, hard_weight=float(cfg.get('student',{}).get('hard_weight',0.6)), pos_weight=pos_weight, gamma=float(cfg.get('loss',{}).get('gamma',2.0)))
            loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        valid_out = validate(student, valid_loader, device=device)
        metrics = valid_out['metrics']
        row = {'epoch': epoch, 'fold': fold, 'train_loss': float(np.mean(losses))}; row.update(metrics)
        history.append(row); print(row)
        auc = float(metrics.get('macro_auc', -1.0))
        if auc > best_auc:
            best_auc = auc
            save_checkpoint(Path(out_dir) / f'best_fold{fold}.pth', student, opt, epoch, metrics)
    import pandas as pd
    pd.DataFrame(history).to_csv(Path(out_dir) / f'history_fold{fold}.csv', index=False)
    save_json({'best_macro_auc': best_auc, 'fold': fold}, Path(out_dir) / f'summary_fold{fold}.json')


if __name__ == '__main__':
    main()
