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
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from bioacoustic.audio import load_audio, make_regular_windows
from bioacoustic.dataset import build_class_list, read_metadata
from bioacoustic.models import build_model
from bioacoustic.pseudo_labeling import select_pseudo_labels, save_pseudo_labels
from bioacoustic.spectrogram import log_mel_spectrogram
from bioacoustic.training import load_checkpoint
from bioacoustic.utils import ensure_dir, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    return parser.parse_args()


class UnlabeledWindowDataset(Dataset):
    def __init__(self, audio_files: List[Path], sample_rate: int, duration: float, spectrogram_kwargs: dict):
        self.items = []
        self.sample_rate = sample_rate
        self.duration = duration
        self.spectrogram_kwargs = spectrogram_kwargs
        for audio_path in audio_files:
            wav = load_audio(audio_path, sample_rate=sample_rate)
            windows = make_regular_windows(wav, sample_rate=sample_rate, duration=duration, drop_last=False)
            for i, window in enumerate(windows):
                self.items.append((str(audio_path), i, window))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, chunk_id, window = self.items[idx]
        spec = log_mel_spectrogram(window, sample_rate=self.sample_rate, **self.spectrogram_kwargs)
        row_id = f'{Path(audio_path).stem}_{chunk_id:03d}'
        return torch.from_numpy(spec), row_id, audio_path, chunk_id


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device(bool(cfg.get('runtime', {}).get('prefer_cuda', True)))
    data = cfg['data']
    audio_cfg = cfg.get('audio', {})
    spec_cfg = cfg.get('spectrogram', {})
    pseudo_cfg = cfg.get('pseudo_labeling', {})

    train_df = read_metadata(data['train_csv'])
    classes = build_class_list(train_df, primary_col=data.get('primary_col', 'primary_label'))
    if data.get('classes_path'):
        Path(data['classes_path']).parent.mkdir(parents=True, exist_ok=True)
        Path(data['classes_path']).write_text('\n'.join(classes), encoding='utf-8')

    model_cfg = cfg.get('model', {})
    model = build_model(
        model_type='sed',
        num_classes=len(classes),
        backbone=model_cfg.get('backbone', 'tf_efficientnetv2_s_in21ft1k'),
        in_channels=int(model_cfg.get('in_channels', 1)),
        pretrained=bool(model_cfg.get('pretrained', False)),
        dropout=float(model_cfg.get('dropout', 0.2)),
    ).to(device)
    load_checkpoint(cfg['teacher']['checkpoint'], model, map_location=device)
    model.eval()

    unlabeled_dir = Path(data['unlabeled_audio_dir'])
    exts = tuple(pseudo_cfg.get('audio_extensions', ['.ogg', '.wav', '.mp3', '.flac']))
    audio_files = sorted([p for p in unlabeled_dir.rglob('*') if p.suffix.lower() in exts])
    if not audio_files:
        raise FileNotFoundError(f'No unlabeled audio files found in {unlabeled_dir}')

    spectrogram_kwargs = {
        'n_fft': int(spec_cfg.get('n_fft', 2048)),
        'hop_length': int(spec_cfg.get('hop_length', 768)),
        'n_mels': int(spec_cfg.get('n_mels', 192)),
        'f_min': int(spec_cfg.get('f_min', 50)),
        'f_max': int(spec_cfg.get('f_max', 15000)),
        'normalize': bool(spec_cfg.get('normalize', True)),
    }
    ds = UnlabeledWindowDataset(
        audio_files,
        sample_rate=int(audio_cfg.get('sample_rate', 32000)),
        duration=float(audio_cfg.get('clip_duration', 5.0)),
        spectrogram_kwargs=spectrogram_kwargs,
    )
    loader = DataLoader(ds, batch_size=int(cfg.get('training', {}).get('valid_batch_size', 32)), shuffle=False, num_workers=int(cfg.get('training', {}).get('num_workers', 2)))

    all_probs, row_ids, audio_paths, chunk_indices = [], [], [], []
    with torch.no_grad():
        for x, ids, paths, chunks in loader:
            x = x.to(device).float()
            logits = model(x)['clip_logits']
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            row_ids.extend(list(ids))
            audio_paths.extend(list(paths))
            chunk_indices.extend(int(chunk) for chunk in chunks)
    probs = np.concatenate(all_probs, axis=0)
    pseudo_df = select_pseudo_labels(
        probs,
        row_ids=row_ids,
        min_max_prob=float(pseudo_cfg.get('min_max_prob', 0.55)),
        class_prob_threshold=float(pseudo_cfg.get('class_prob_threshold', 0.10)),
    )
    pseudo_df.columns = ['row_id'] + classes
    window_index = pd.DataFrame({
        'row_id': row_ids,
        'audio_path': audio_paths,
        'chunk_index': chunk_indices,
    })
    pseudo_df = window_index.merge(pseudo_df, on='row_id', how='inner', validate='one_to_one')
    out_path = data.get('pseudo_labels_path', 'outputs/pseudo_labeling/pseudo_labels.csv')
    save_pseudo_labels(pseudo_df, out_path)
    print(f'Saved {len(pseudo_df)} selected pseudo-labeled chunks to {out_path}')


if __name__ == '__main__':
    main()
