from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    import openvino as ov
except Exception:  # pragma: no cover
    ov = None

from .audio import blend_regular_and_shifted, load_audio, segment_regular_windows, segment_shifted_windows, temporal_smoothing
from .spectrogram import SpectrogramConfig, batch_log_mel


def load_openvino_models(models_dir: str | Path, prefix: str = "model_", limit: int | None = None, device: str = "CPU"):
    if ov is None:
        raise ImportError("openvino is required. Install OpenVINO or use the PyTorch ensemble path.")
    models_dir = Path(models_dir)
    xml_files = sorted(models_dir.glob(f"{prefix}*.xml"))
    if limit is not None:
        xml_files = xml_files[:limit]
    if not xml_files:
        raise FileNotFoundError(f"No OpenVINO XML files found in {models_dir} with prefix {prefix}")
    core = ov.Core()
    compiled = []
    for xml in xml_files:
        model = core.read_model(model=str(xml))
        compiled.append(core.compile_model(model, device))
    return compiled


def predict_chunks_openvino(compiled_models: Sequence, chunks: np.ndarray, spec_cfg: SpectrogramConfig, batch_size: int = 16) -> np.ndarray:
    if len(chunks) == 0:
        return np.empty((0, 0), dtype=np.float32)
    specs = batch_log_mel(chunks, spec_cfg)
    model_preds = []
    for cmodel in compiled_models:
        out_batches = []
        output_key = cmodel.outputs[0]
        for i in range(0, len(specs), batch_size):
            x = specs[i : i + batch_size].astype(np.float32)
            logits = cmodel([x])[output_key]
            probs = 1.0 / (1.0 + np.exp(-logits))
            out_batches.append(probs.astype(np.float32))
        model_preds.append(np.concatenate(out_batches, axis=0))
    return np.mean(model_preds, axis=0)


def predict_audio_openvino_ensemble(
    audio_path: str | Path,
    regular_models: Sequence,
    shifted_models: Sequence | None,
    spec_cfg: SpectrogramConfig,
    clip_duration: float = 5.0,
    shift_seconds: float = 2.5,
    batch_size: int = 16,
    blend_alpha: float = 0.5,
    smooth: bool = True,
) -> np.ndarray:
    waveform = load_audio(audio_path, spec_cfg.sample_rate)
    regular_chunks, _ = segment_regular_windows(waveform, spec_cfg.sample_rate, clip_duration, pad_last=True)
    shifted_chunks, _ = segment_shifted_windows(waveform, spec_cfg.sample_rate, clip_duration, shift_seconds)
    regular_pred = predict_chunks_openvino(regular_models, regular_chunks, spec_cfg, batch_size)
    if shifted_models and len(shifted_chunks) > 0:
        shifted_pred = predict_chunks_openvino(shifted_models, shifted_chunks, spec_cfg, batch_size)
        pred = blend_regular_and_shifted(regular_pred, shifted_pred, alpha=blend_alpha)
    else:
        pred = regular_pred
    if smooth:
        pred = temporal_smoothing(pred)
    return pred
