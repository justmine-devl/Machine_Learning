from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .audio import load_audio, segment_regular_windows
from .spectrogram import SpectrogramConfig, batch_log_mel


@torch.no_grad()
def predict_audio_chunks(
    model: torch.nn.Module,
    audio_path: str | Path,
    spec_cfg: SpectrogramConfig,
    device: torch.device,
    clip_duration: float = 5.0,
    batch_size: int = 16,
) -> np.ndarray:
    waveform = load_audio(audio_path, spec_cfg.sample_rate)
    chunks, _ = segment_regular_windows(waveform, spec_cfg.sample_rate, clip_duration, pad_last=True)
    if len(chunks) == 0:
        return np.empty((0, 0), dtype=np.float32)
    preds = []
    model.eval()
    for i in range(0, len(chunks), batch_size):
        specs = batch_log_mel(chunks[i : i + batch_size], spec_cfg)
        x = torch.from_numpy(specs).float().to(device)
        out = model(x)
        probs = torch.sigmoid(out["clip_logits"]).cpu().numpy()
        preds.append(probs)
    return np.concatenate(preds, axis=0)


def select_pseudo_labels(
    probs: np.ndarray,
    min_max_probability: float = 0.5,
    class_probability_floor: float = 0.1,
) -> np.ndarray:
    selected = probs.copy()
    keep = selected.max(axis=1) >= min_max_probability
    selected[~keep] = 0.0
    selected[selected < class_probability_floor] = 0.0
    return selected


def generate_pseudo_label_csv(
    model: torch.nn.Module,
    soundscape_files: Sequence[str | Path],
    spec_cfg: SpectrogramConfig,
    device: torch.device,
    output_csv: str | Path,
    class_names: List[str],
    min_max_probability: float = 0.5,
    class_probability_floor: float = 0.1,
    clip_duration: float = 5.0,
    batch_size: int = 16,
) -> pd.DataFrame:
    rows = []
    for file_path in tqdm(soundscape_files, desc="pseudo-label soundscapes"):
        probs = predict_audio_chunks(model, file_path, spec_cfg, device, clip_duration, batch_size)
        selected = select_pseudo_labels(probs, min_max_probability, class_probability_floor)
        for chunk_idx, vec in enumerate(selected):
            if vec.max() <= 0:
                continue
            row = {"filename": str(file_path), "chunk_index": chunk_idx, "start_sec": chunk_idx * clip_duration, "end_sec": (chunk_idx + 1) * clip_duration}
            for i, _ in enumerate(class_names):
                row[f"class_{i}"] = float(vec[i])
            rows.append(row)
    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
