from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

from .audio import blend_regular_and_shifted, load_audio, segment_regular_windows, segment_shifted_windows, temporal_smoothing
from .spectrogram import SpectrogramConfig, batch_log_mel


@torch.no_grad()
def predict_chunks_torch_models(
    models: Sequence[torch.nn.Module],
    chunks: np.ndarray,
    spec_cfg: SpectrogramConfig,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    if len(chunks) == 0:
        return np.empty((0, 0), dtype=np.float32)
    model_preds = []
    for model in models:
        model.eval().to(device)
        preds = []
        for i in range(0, len(chunks), batch_size):
            specs = batch_log_mel(chunks[i : i + batch_size], spec_cfg)
            x = torch.from_numpy(specs).float().to(device)
            out = model(x)
            probs = torch.sigmoid(out["clip_logits"]).cpu().numpy()
            preds.append(probs)
        model_preds.append(np.concatenate(preds, axis=0))
    return np.mean(model_preds, axis=0)


def ensemble_predict_audio(
    models_regular: Sequence[torch.nn.Module],
    audio_path: str | Path,
    spec_cfg: SpectrogramConfig,
    device: torch.device,
    models_shifted: Sequence[torch.nn.Module] | None = None,
    clip_duration: float = 5.0,
    shift_seconds: float = 2.5,
    batch_size: int = 16,
    blend_alpha: float = 0.5,
    smooth: bool = True,
) -> np.ndarray:
    waveform = load_audio(audio_path, spec_cfg.sample_rate)
    regular_chunks, _ = segment_regular_windows(waveform, spec_cfg.sample_rate, clip_duration, pad_last=True)
    shifted_chunks, _ = segment_shifted_windows(waveform, spec_cfg.sample_rate, clip_duration, shift_seconds)

    regular_pred = predict_chunks_torch_models(models_regular, regular_chunks, spec_cfg, device, batch_size)
    if models_shifted is not None and len(models_shifted) > 0 and len(shifted_chunks) > 0:
        shifted_pred = predict_chunks_torch_models(models_shifted, shifted_chunks, spec_cfg, device, batch_size)
        pred = blend_regular_and_shifted(regular_pred, shifted_pred, alpha=blend_alpha)
    else:
        pred = regular_pred
    if smooth:
        pred = temporal_smoothing(pred)
    return pred


def average_prediction_files(paths: Iterable[str | Path], output_path: str | Path | None = None) -> np.ndarray:
    arrays = [np.load(p) for p in paths]
    pred = np.mean(arrays, axis=0)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, pred)
    return pred
