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
import pandas as pd
import torch

from bioacoustic.audio import load_audio, segment_for_inference
from bioacoustic.ensemble import average_predictions, blend_regular_shifted, temporal_smoothing, power_adjust
from bioacoustic.models import build_model
from bioacoustic.openvino_ensemble import OpenVINOEnsemble, infer_audio_file_openvino, find_openvino_models
from bioacoustic.spectrogram import batch_log_mel
from bioacoustic.training import load_checkpoint
from bioacoustic.utils import ensure_dir, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--audio_dir', default=None)
    return parser.parse_args()


def spec_kwargs(cfg):
    s = cfg.get('spectrogram', {})
    return {'n_fft': int(s.get('n_fft', 2048)), 'hop_length': int(s.get('hop_length', 768)), 'n_mels': int(s.get('n_mels', 192)), 'f_min': int(s.get('f_min', 50)), 'f_max': int(s.get('f_max', 15000)), 'normalize': bool(s.get('normalize', True))}


def load_classes(path: str | Path):
    return [x.strip() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def infer_torch_file(audio_path, cfg, models, device):
    audio = cfg.get('audio', {}); post = cfg.get('postprocessing', {})
    wav = load_audio(audio_path, sample_rate=int(audio.get('sample_rate', 32000)))
    regular, shifted = segment_for_inference(wav, sample_rate=int(audio.get('sample_rate', 32000)), duration=float(audio.get('clip_duration', 5.0)), shift=float(audio.get('shift', 2.5)))
    reg_x = torch.from_numpy(batch_log_mel(regular, sample_rate=int(audio.get('sample_rate', 32000)), **spec_kwargs(cfg))).to(device).float()
    shift_x = torch.from_numpy(batch_log_mel(shifted, sample_rate=int(audio.get('sample_rate', 32000)), **spec_kwargs(cfg))).to(device).float() if shifted else None
    reg_preds=[]; shift_preds=[]
    with torch.no_grad():
        for model in models:
            model.eval()
            reg_preds.append(torch.sigmoid(model(reg_x)['clip_logits']).cpu().numpy())
            if shift_x is not None:
                shift_preds.append(torch.sigmoid(model(shift_x)['clip_logits']).cpu().numpy())
    reg = average_predictions(reg_preds)
    sh = average_predictions(shift_preds) if shift_preds else None
    pred = blend_regular_shifted(reg, sh, alpha=float(post.get('blend_alpha',0.5)))
    if bool(post.get('temporal_smoothing', True)):
        pred = temporal_smoothing(pred)
    pred = power_adjust(pred, gamma=float(post.get('power_gamma', 1.0)))
    return pred


def main() -> None:
    args = parse_args(); cfg = load_config(args.config); seed_everything(int(cfg.get('seed',42)))
    classes = load_classes(cfg['data']['classes_path'])
    out_dir = ensure_dir(cfg.get('output_dir', 'outputs/ensemble'))
    audio_dir = Path(args.audio_dir or cfg['data'].get('test_audio_dir', 'data/test_soundscapes'))
    audio_files = sorted([p for p in audio_dir.rglob('*') if p.suffix.lower() in {'.ogg','.wav','.mp3','.flac'}])
    if not audio_files:
        raise FileNotFoundError(f'No audio files found in {audio_dir}')

    if bool(cfg.get('openvino', {}).get('enabled', False)):
        ov_cfg = cfg['openvino']
        regular_models = OpenVINOEnsemble(find_openvino_models(ov_cfg['regular_model_dir']), device=ov_cfg.get('device','CPU'))
        shifted_dir = Path(ov_cfg.get('shifted_model_dir', ''))
        shifted_models = OpenVINOEnsemble(find_openvino_models(shifted_dir), device=ov_cfg.get('device','CPU')) if shifted_dir.exists() else None
        all_rows=[]
        for path in audio_files:
            pred = infer_audio_file_openvino(path, regular_models, shifted_models, sample_rate=int(cfg.get('audio',{}).get('sample_rate',32000)), duration=float(cfg.get('audio',{}).get('clip_duration',5.0)), shift=float(cfg.get('audio',{}).get('shift',2.5)), blend_alpha=float(cfg.get('postprocessing',{}).get('blend_alpha',0.5)), smooth=bool(cfg.get('postprocessing',{}).get('temporal_smoothing', True)), spectrogram_kwargs=spec_kwargs(cfg))
            for i, row in enumerate(pred):
                all_rows.append({'row_id': f'{path.stem}_{i:03d}', **{c: float(v) for c, v in zip(classes, row)}})
    else:
        device = get_device(bool(cfg.get('runtime',{}).get('prefer_cuda', True)))
        mcfg = cfg.get('model', {})
        models=[]
        for ckpt in cfg.get('checkpoints', []):
            model = build_model(mcfg.get('model_type','sed'), len(classes), backbone=mcfg.get('backbone','tf_efficientnetv2_s_in21ft1k'), in_channels=int(mcfg.get('in_channels',1)), pretrained=False, dropout=float(mcfg.get('dropout',0.2))).to(device)
            load_checkpoint(ckpt, model, map_location=device)
            models.append(model)
        if not models:
            raise ValueError('No checkpoints specified in config.')
        all_rows=[]
        for path in audio_files:
            pred = infer_torch_file(path, cfg, models, device)
            for i, row in enumerate(pred):
                all_rows.append({'row_id': f'{path.stem}_{i:03d}', **{c: float(v) for c, v in zip(classes, row)}})
    out_path = Path(out_dir) / 'ensemble_predictions.csv'
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f'Saved predictions to {out_path}')


if __name__ == '__main__':
    main()
